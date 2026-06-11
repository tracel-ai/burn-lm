use super::super::*;
use super::fakes::*;
use crate::{
    batching::{step_round, ActiveSeq, PrefillBudget, StepOutcome},
    job::{GenerationParams, InferenceJob, InferenceTask},
    sampler::Sampler,
    TextGenerationListener,
};
use std::sync::{Arc, Mutex};

/// Capacity >= 2: both jobs are admitted concurrently and their emission streams INTERLEAVE.
/// Each sequence's first emission precedes the other's last (structural overlap).
#[test]
fn capacity_two_admits_concurrently_and_interleaves() {
    let log = submit_two(2);
    assert!(
        log.contains(&0) && log.contains(&1),
        "both sequences should produce output: {log:?}"
    );
    let first0 = log.iter().position(|&x| x == 0).unwrap();
    let last0 = log.iter().rposition(|&x| x == 0).unwrap();
    let first1 = log.iter().position(|&x| x == 1).unwrap();
    let last1 = log.iter().rposition(|&x| x == 1).unwrap();
    assert!(
        first1 < last0 && first0 < last1,
        "sequences did not interleave (ran serially): {log:?}"
    );
}

/// Capacity == 1: admission is one-at-a-time, so the two sequences run SERIALLY (no overlap).
/// This proves capacity-based admission/backpressure.
#[test]
fn capacity_one_serializes() {
    let log = submit_two(1);
    assert!(
        log.contains(&0) && log.contains(&1),
        "both sequences should produce output: {log:?}"
    );
    let first0 = log.iter().position(|&x| x == 0).unwrap();
    let last0 = log.iter().rposition(|&x| x == 0).unwrap();
    let first1 = log.iter().position(|&x| x == 1).unwrap();
    let last1 = log.iter().rposition(|&x| x == 1).unwrap();
    // Serial ⇒ one sequence fully precedes the other.
    assert!(
        last0 < first1 || last1 < first0,
        "sequences interleaved but capacity==1 should serialize them: {log:?}"
    );
}

/// SLOT REUSE, no residue: with capacity 1 every job runs in slot 0, so the second job reuses
/// the first one's slot. The fake decoder's per-slot step counter is its stand-in cache: if the
/// worker skipped `release` on retire (or the decoder kept slot state across `release`), the
/// second job would resume the first's counter past `emit` and stop immediately, producing
/// different output. Both jobs must produce identical, full output.
#[test]
fn a_retired_sequences_slot_is_reused_with_no_residue() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
        1,
        Arc::new(Mutex::new(Vec::new())),
    ));

    let run = || {
        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        channel.submit(job).unwrap().recv().unwrap().unwrap();
        handle.join()
    };

    let first = run();
    let second = run();
    assert_eq!(first, "10101010", "the first job runs a full generation");
    assert_eq!(
        second, first,
        "a reused slot must start fresh — residue from the previous sequence changed the output"
    );
}

/// PREFILL BUDGET: while another sequence is mid-decode, at most ONE prompt prefills per round;
/// the deferred prompt's tail is untouched and it prefills the next round. This drives
/// `step_round` exactly like the serving worker does — one call per sequence, all sharing the
/// round's single [`PrefillBudget`] — so a per-call budget regression (where a single-sequence
/// slice never sees the others decoding) fails here.
#[test]
fn at_most_one_prompt_prefills_per_round_while_another_sequence_decodes() {
    fn seq(slot: usize, tokens: Vec<u32>) -> ActiveSeq<()> {
        ActiveSeq {
            slot,
            tokens,
            processed: 0,
            generated: 0,
            max_gen: 8,
            finished: false,
            extra: (),
        }
    }

    // `emit` is high so nothing stops mid-test.
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    let mut sampler = Sampler::default();
    let stop_ids = [0u32];

    // Slot 0 is GENUINELY mid-decode: its prompt was already prefilled (`processed > 0`) and it
    // owes one new token. (A fresh one-token prompt would NOT count — `PrefillBudget` requires
    // `processed > 0`, so a brand-new batch never defers its own prompts.) Slots 1 and 2 hold
    // fresh multi-token prompts.
    let mut mid_decode = seq(0, vec![10, 10]);
    mid_decode.processed = 1;
    mid_decode.generated = 1;
    let mut active = vec![mid_decode, seq(1, vec![11, 11]), seq(2, vec![12, 12])];

    // Round 1: one budget for the round, one `step_round` call per sequence (the worker's shape).
    let mut budget = PrefillBudget::for_round(&active);
    let outcomes: Vec<StepOutcome> = (0..active.len())
        .map(|i| {
            step_round(
                &mut decoder,
                &mut active[i..i + 1],
                &mut sampler,
                &stop_ids,
                &mut budget,
            )
            .pop()
            .expect("one sequence in yields exactly one outcome")
        })
        .collect();

    assert!(
        matches!(outcomes[0], StepOutcome::Stepped { .. }),
        "the decoding sequence advances"
    );
    assert!(
        matches!(outcomes[1], StepOutcome::Stepped { .. }),
        "exactly one prompt prefills this round"
    );
    assert!(
        matches!(outcomes[2], StepOutcome::Skipped),
        "the second prompt must defer while another sequence is decoding"
    );
    assert_eq!(
        active[2].processed, 0,
        "a deferred prompt's tail is untouched"
    );

    // Round 2: a fresh budget admits the deferred prompt.
    let mut budget = PrefillBudget::for_round(&active);
    let outcome = step_round(
        &mut decoder,
        &mut active[2..3],
        &mut sampler,
        &stop_ids,
        &mut budget,
    )
    .pop()
    .expect("one sequence in yields exactly one outcome");
    assert!(
        matches!(outcome, StepOutcome::Stepped { .. }),
        "the deferred prompt prefills on the next round"
    );
}

/// Two CONCURRENT jobs with different per-request params must sample with independently
/// configured samplers — the request params merged over config at admission, not a shared
/// mutated server config (which would make one request clobber the other's temperature).
#[test]
fn concurrent_jobs_sample_with_their_own_request_params() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
        2,
        Arc::new(Mutex::new(Vec::new())),
    ));

    // The fake's `next_token_sampler` turns a per-job temperature into a fixed token, so each
    // job's whole output reveals which params built its sampler.
    let hot = GenerationParams {
        temperature: Some(7.0),
        ..Default::default()
    };
    let cold = GenerationParams {
        temperature: Some(9.0),
        ..Default::default()
    };
    let (job_a, ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        hot,
        TextGenerationListener::default(),
    );
    let (job_b, hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        cold,
        TextGenerationListener::default(),
    );

    // Both in flight together (capacity 2), interleaving round-robin.
    let rx_a = channel.submit(job_a).unwrap();
    let rx_b = channel.submit(job_b).unwrap();
    rx_a.recv().unwrap().unwrap();
    rx_b.recv().unwrap().unwrap();

    assert_eq!(ha.join(), "7".repeat(16), "job A must use its own params");
    assert_eq!(hb.join(), "9".repeat(16), "job B must use its own params");
}

/// A request's `max_tokens` can LOWER the server's generation cap but never RAISE it: the
/// operator-set server cap stays authoritative.
#[test]
fn request_max_tokens_lowers_but_cannot_raise_the_server_cap() {
    let channel = BatchingChannel::<FakeServer>::with_server(
        FakeServer::new(1, Arc::new(Mutex::new(Vec::new()))).with_fixed_sampler(7),
    );

    // Below the server cap (16): the request wins.
    let (job, handle) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams {
            max_tokens: Some(4),
            ..Default::default()
        },
        TextGenerationListener::default(),
    );
    channel.submit(job).unwrap().recv().unwrap().unwrap();
    assert_eq!(handle.join(), "7".repeat(4));

    // Above the server cap: clamped to the cap.
    let (job, handle) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams {
            max_tokens: Some(100),
            ..Default::default()
        },
        TextGenerationListener::default(),
    );
    channel.submit(job).unwrap().recv().unwrap().unwrap();
    assert_eq!(handle.join(), "7".repeat(16));
}

/// The worker must sample with the server-configured sampler (the `next_token_sampler`
/// primitive), not a hard-coded argmax. The fixed sampler always picks token 7 regardless of
/// logits; argmax over the fake decoder's logits would instead echo the identity token (10)
/// and then stop. Since 7 is never a stop id, the sequence runs to `max_gen_tokens` (16) and
/// streams sixteen "7"s — unmistakably the configured sampler's output.
#[test]
fn worker_uses_the_server_configured_sampler() {
    let channel = BatchingChannel::<FakeServer>::with_server(
        FakeServer::new(1, Arc::new(Mutex::new(Vec::new()))).with_fixed_sampler(7),
    );

    let (job, handle) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    channel.submit(job).unwrap().recv().unwrap().unwrap();

    assert_eq!(
        handle.join(),
        "7".repeat(16),
        "emitted text must come from the server's configured sampler, not argmax"
    );
}

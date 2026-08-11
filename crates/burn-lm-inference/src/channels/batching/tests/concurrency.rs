use super::super::*;
use super::fakes::*;
use crate::{
    batching::{step_round, ActiveSeq, PrefillBudget, StepOutcome},
    job::{GenerationParams, InferenceJob, InferenceTask},
    sampler::{Argmax, Sampler},
    TextGenerationListener,
};
use std::sync::{Arc, Mutex};

/// Argmax over each row of `[n, vocab]` logits, returning one id per row — the batched-closure
/// shape `step_round` expects. The direct `step_round` tests below use this as their `sample`
/// closure, standing in for the worker's call into the shared `Argmax` sampler.
fn argmax_rows(logits: burn::tensor::Tensor<2>) -> InferenceResult<Vec<u32>> {
    Argmax.sample(logits)
}

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

/// KV BUDGET GATE: two free slots, but a block budget that fits only one worst-case reservation —
/// the jobs run SERIALLY. Each job reserves `blocks_for(prompt + max_gen) = 1 + 16 = 17` blocks
/// (block_size 1); the 20-block budget fits one, so the second waits in the queue until the first
/// retires and its reservation frees. Nothing errors — running short of blocks is backpressure,
/// not failure — and both jobs complete in full. With an unlimited budget the same two slots run
/// these jobs interleaved (`capacity_two_admits_concurrently_and_interleaves`), so serialization
/// here is attributable to the KV gate alone.
#[test]
fn kv_budget_serializes_admission_when_blocks_run_short() {
    let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
    let server = FakeServer::new(2, log.clone()).with_kv_budget(1, 20);
    let out = submit_two_on(server, log);
    assert!(
        out.contains(&0) && out.contains(&1),
        "both sequences should produce output: {out:?}"
    );
    let first0 = out.iter().position(|&x| x == 0).unwrap();
    let last0 = out.iter().rposition(|&x| x == 0).unwrap();
    let first1 = out.iter().position(|&x| x == 1).unwrap();
    let last1 = out.iter().rposition(|&x| x == 1).unwrap();
    assert!(
        last0 < first1 || last1 < first0,
        "a budget with room for one reservation must serialize the jobs: {out:?}"
    );
}

/// A request whose worst case exceeds the WHOLE pool can never run: it is rejected at admission
/// with `KvPoolExhausted` instead of blocking the queue head forever, and the job behind it — which
/// fits — runs normally.
#[test]
fn a_request_larger_than_the_whole_pool_is_rejected_not_queued() {
    let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
    // Worst case per job: 1 prompt token + 16 max_gen = 17 blocks at block_size 1. A 10-block pool
    // can never fit it... for job A. Job B lowers max_tokens to 5 -> 6 blocks -> fits.
    let server = FakeServer::new(2, log.clone()).with_kv_budget(1, 10);
    let channel = BatchingChannel::<FakeServer>::with_server(server);

    let (job_a, _ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(), // max_tokens unset -> server cap 16 -> needs 17 blocks
        NullListener,
    );
    let (job_b, _hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams {
            max_tokens: Some(5),
            ..GenerationParams::default()
        },
        NullListener,
    );

    let rx_a = channel.submit(job_a).unwrap();
    let rx_b = channel.submit(job_b).unwrap();
    let err = match rx_a.recv().unwrap() {
        Err(err) => err,
        Ok(_) => panic!("a job needing 17 blocks must be rejected by a 10-block pool"),
    };
    assert!(
        matches!(err, InferenceError::KvPoolExhausted(7)),
        "17 needed vs 10 in the pool is 7 short: {err:?}"
    );
    rx_b.recv().unwrap().unwrap();
    let out = log.lock().unwrap().clone();
    assert!(out.contains(&1), "the job behind the rejected one still runs: {out:?}");
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
/// `step_round` exactly like the serving worker does — one fused call for the whole round, sharing
/// the round's single [`PrefillBudget`] — so a budget regression (a prompt prefilling while another
/// sequence is decoding) fails here.
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
            kv_reservation: 0,
            extra: (),
        }
    }

    // `emit` is high so nothing stops mid-test.
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    let stop_ids = [0u32];

    // Slot 0 is GENUINELY mid-decode: its prompt was already prefilled (`processed > 0`) and it
    // owes one new token. (A fresh one-token prompt would NOT count — `PrefillBudget` requires
    // `processed > 0`, so a brand-new batch never defers its own prompts.) Slots 1 and 2 hold
    // fresh multi-token prompts.
    let mut mid_decode = seq(0, vec![10, 10]);
    mid_decode.processed = 1;
    mid_decode.generated = 1;
    let mut active = vec![mid_decode, seq(1, vec![11, 11]), seq(2, vec![12, 12])];

    // Round 1: one fused `step_round` call for the whole round (the worker's shape). The shared
    // sampler closure ignores the row index here.
    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(
        &mut decoder,
        &mut active,
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    );

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
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    )
    .pop()
    .expect("one sequence in yields exactly one outcome");
    assert!(
        matches!(outcome, StepOutcome::Stepped { .. }),
        "the deferred prompt prefills on the next round"
    );
}

/// FUSION (the S7b throughput win): a round with several decoding sequences must issue ONE `decode`
/// call carrying every row — not one single-row call per sequence. A regression to per-row decode
/// would make `decode_calls` `[1, 1, 1]` instead of `[3]`.
#[test]
fn decoding_sequences_share_one_fused_decode_call() {
    // Mid-decode: prompt already processed, owes exactly one new token (a decode row, not prefill).
    fn decoding(slot: usize, last: u32) -> ActiveSeq<()> {
        ActiveSeq {
            slot,
            tokens: vec![last, last],
            processed: 1,
            generated: 1,
            max_gen: 8,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }

    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    let stop_ids = [0u32];
    let mut active = vec![decoding(0, 10), decoding(1, 11), decoding(2, 12)];

    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(
        &mut decoder,
        &mut active,
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    );

    assert!(
        outcomes
            .iter()
            .all(|o| matches!(o, StepOutcome::Stepped { .. })),
        "every decoding sequence advances this round"
    );
    assert_eq!(
        decoder.decode_calls,
        vec![3],
        "the round must fuse all decode rows into a single decode call, got {:?}",
        decoder.decode_calls
    );
}

/// FUSION FAILURE SEMANTICS: a fused decode is all-or-nothing, so a single decode error retires
/// EVERY decode row that round — a per-row regression that retired only one would not turn this red.
/// A prompt admitted the same round prefills BEFORE the decode call, so it survives: prefill errors
/// stay per-sequence, decode errors are batch-wide.
#[test]
fn a_fused_decode_error_retires_every_decode_row_but_not_a_concurrent_prefill() {
    fn decoding(slot: usize, last: u32) -> ActiveSeq<()> {
        ActiveSeq {
            slot,
            tokens: vec![last, last],
            processed: 1,
            generated: 1,
            max_gen: 8,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }
    fn prompt(slot: usize) -> ActiveSeq<()> {
        ActiveSeq {
            slot,
            tokens: vec![20, 20],
            processed: 0,
            generated: 0,
            max_gen: 8,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }

    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    decoder.fail_decodes = 1;
    let stop_ids = [0u32];
    // Three decode rows + one fresh prompt (admitted because the others are decoding).
    let mut active = vec![decoding(0, 10), decoding(1, 11), decoding(2, 12), prompt(3)];

    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(
        &mut decoder,
        &mut active,
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    );

    for i in 0..3 {
        assert!(
            matches!(outcomes[i], StepOutcome::Failed(_)),
            "decode row {i} must be retired with the error"
        );
        assert!(active[i].finished, "decode row {i} must be marked finished");
    }
    assert!(
        matches!(outcomes[3], StepOutcome::Stepped { .. }),
        "the prompt prefilled before the decode and survives the decode failure"
    );
    assert!(
        !active[3].finished,
        "the prefilled sequence is not retired by the decode failure"
    );
}

/// MIXED-ROUND ALIGNMENT: a round where one prefill row AND the fused decode rows all STEP must put
/// each sampled token on the RIGHT sequence. The sampler closure returns a token keyed to the row's
/// active index, so a prefill/decode index mixup (e.g. using the fused row index instead of the
/// sequence index) would land the wrong token on the wrong sequence. No existing test has a prefill
/// AND a decode both stepping in one call.
#[test]
fn a_mixed_round_aligns_each_sampled_token_to_its_sequence() {
    fn decoding(slot: usize, last: u32) -> ActiveSeq<()> {
        ActiveSeq {
            slot,
            tokens: vec![last, last],
            processed: 1,
            generated: 1,
            max_gen: 8,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }
    fn prompt(slot: usize) -> ActiveSeq<()> {
        ActiveSeq {
            slot,
            tokens: vec![30, 30],
            processed: 0,
            generated: 0,
            max_gen: 8,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }

    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    let stop_ids = [0u32];
    // seq1 is a fresh prompt (a prefill row); seq0 and seq2 are mid-decode (fused decode rows).
    let mut active = vec![decoding(0, 10), prompt(1), decoding(2, 12)];

    let mut budget = PrefillBudget::for_round(&active);
    // Argmax over the decoder's one-hot logits: the `FakeDecoder` echoes each row's identity token as
    // a one-hot row, so argmax recovers a token unique to that row — the two fused-decode rows their
    // last tokens (10, 12) and the prefill row its prompt token (30). A prefill/decode index mixup in
    // `step_round` would land one of these on the wrong sequence, which the distinct values catch.
    let outcomes = step_round(
        &mut decoder,
        &mut active,
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    );

    assert_eq!(*active[0].tokens.last().unwrap(), 10, "decode row 0 token");
    assert_eq!(*active[1].tokens.last().unwrap(), 30, "prefill row 1 token");
    assert_eq!(*active[2].tokens.last().unwrap(), 12, "decode row 2 token");
    for (i, expected) in [(0usize, 10u32), (1, 30), (2, 12)] {
        assert!(
            matches!(outcomes[i], StepOutcome::Stepped { token, .. } if token == expected),
            "outcome {i} must carry its own sequence's token"
        );
    }
}

/// A lane that would overflow its context window retires ALONE with `ContextLengthExceeded`, in the
/// classification sweep, before the fused decode — so the lanes batched with it decode normally
/// instead of being failed too. Lane mode does not evict, so the model's `prepare_lanes` would error
/// the whole all-or-nothing decode if an over-long lane reached it; the engine retires that one lane
/// up front so its batch-mates are never poisoned.
#[test]
fn a_lane_over_the_context_limit_retires_alone_without_failing_its_batch_mates() {
    // A decode-ready sequence: one unprocessed token at the tail, so this round forwards exactly one
    // token and the lane's length becomes `tokens.len()`.
    fn decoding(slot: usize, tokens: Vec<u32>) -> ActiveSeq<()> {
        let processed = tokens.len() - 1;
        ActiveSeq {
            slot,
            tokens,
            processed,
            generated: 1,
            max_gen: 100,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }

    // A lane can hold at most 3 tokens.
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    decoder.context_len = 3;
    let stop_ids = [0u32];
    // seq0 reaches length 4 this round (> 3) — over the limit. seq1 reaches 2 (<= 3) — it fits.
    let mut active = vec![decoding(0, vec![10, 10, 10, 10]), decoding(1, vec![11, 11])];

    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(
        &mut decoder,
        &mut active,
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    );

    assert!(
        matches!(
            outcomes[0],
            StepOutcome::Failed(crate::InferenceError::ContextLengthExceeded(4, 3))
        ),
        "the over-limit lane retires with ContextLengthExceeded(4, 3): {:?}",
        outcomes[0]
    );
    assert!(active[0].finished, "the over-limit lane is finished");
    assert!(
        matches!(outcomes[1], StepOutcome::Stepped { .. }),
        "the in-limit lane decodes normally, not poisoned by its batch-mate: {:?}",
        outcomes[1]
    );
    // Exactly one row entered the fused decode (seq1), proving seq0 was excluded before it — the
    // over-limit lane never reached the decoder.
    assert_eq!(
        decoder.decode_calls,
        vec![1],
        "only the in-limit lane should enter the fused decode"
    );
}

/// A prompt longer than the context window is rejected in the classification sweep, with
/// `ContextLengthExceeded`, before any prefill is attempted — so no forward runs for a prompt that
/// could never fit a (non-evicting) lane.
#[test]
fn a_prompt_longer_than_the_context_window_is_rejected_before_prefill() {
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    decoder.context_len = 3;
    let stop_ids = [0u32];
    // A five-token prompt cannot fit a three-token window.
    let mut active = vec![ActiveSeq {
        slot: 0,
        tokens: vec![7, 7, 7, 7, 7],
        processed: 0,
        generated: 0,
        max_gen: 8,
        finished: false,
        kv_reservation: 0,
        extra: (),
    }];

    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(
        &mut decoder,
        &mut active,
        &stop_ids,
        &mut budget,
        0,
        argmax_rows,
    );

    assert!(
        matches!(
            outcomes[0],
            StepOutcome::Failed(crate::InferenceError::ContextLengthExceeded(5, 3))
        ),
        "the over-long prompt is rejected with ContextLengthExceeded(5, 3): {:?}",
        outcomes[0]
    );
    assert!(active[0].finished, "the rejected prompt is finished");
    // The prefill never ran: the decoder's per-slot step counter for slot 0 was never touched.
    assert!(
        !decoder.steps.contains_key(&0),
        "no prefill forward should run for a prompt that cannot fit the window"
    );
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

/// The worker must sample with the server-configured sampler (the `sampler` primitive), not a
/// hard-coded argmax. The fixed sampler always picks token 7 regardless of logits; argmax over the
/// fake decoder's logits would instead echo the identity token (10) and then stop. Since 7 is never
/// a stop id, the sequence runs to `max_gen_tokens` (16) and streams sixteen "7"s — unmistakably the
/// configured sampler's output.
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

/// Chunked prefill scheduling: a prompt longer than `chunk_size` prefills one chunk per round, each
/// intermediate chunk reporting `Prefilling` and advancing `processed` by the chunk width WITHOUT
/// sampling. Only the final chunk produces a token — either directly, when the last chunk still has
/// more than one token, or via the fused decode that classification routes a lone trailing token to.
/// Both endings are exercised. (The decode MATH of chunked prefill is gated separately against the
/// real model in `burn-lm-llama`; this is the engine-side routing.)
#[test]
fn chunked_prefill_defers_sampling_to_the_final_chunk() {
    fn seq(tokens: Vec<u32>, max_gen: usize) -> ActiveSeq<()> {
        ActiveSeq {
            slot: 0,
            tokens,
            processed: 0,
            generated: 0,
            max_gen,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }
    // `emit` high so the fake never stops on its own; no stop ids, so only `max_gen` ends a sequence.
    let chunk = 4;

    // 10-token prompt, chunk 4: chunks [0,4) [4,8) [8,10). The final chunk holds two tokens, so the
    // prefill final-chunk branch itself samples.
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    let mut active = vec![seq((1..=10).collect(), 2)];
    for (round, expected) in [(1usize, 4usize), (2, 8)] {
        let mut budget = PrefillBudget::for_round(&active);
        let outcomes = step_round(&mut decoder, &mut active, &[], &mut budget, chunk, argmax_rows);
        assert!(
            matches!(outcomes[0], StepOutcome::Prefilling),
            "round {round}: an intermediate chunk should report Prefilling"
        );
        assert_eq!(active[0].processed, expected, "round {round}: cursor advances by the chunk width");
        assert_eq!(active[0].generated, 0, "round {round}: no token before the prompt is fully in");
    }
    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(&mut decoder, &mut active, &[], &mut budget, chunk, argmax_rows);
    assert!(
        matches!(outcomes[0], StepOutcome::Stepped { .. }),
        "the final prefill chunk (two tokens left) samples the first generated token"
    );
    assert_eq!(active[0].processed, 10);
    assert_eq!(active[0].generated, 1);

    // 9-token prompt, chunk 4: chunks [0,4) [4,8), then a single token (index 8) is left, which
    // classification routes to the fused decode rather than a final prefill chunk.
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    let mut active = vec![seq((1..=9).collect(), 2)];
    for expected in [4usize, 8] {
        let mut budget = PrefillBudget::for_round(&active);
        let outcomes = step_round(&mut decoder, &mut active, &[], &mut budget, chunk, argmax_rows);
        assert!(matches!(outcomes[0], StepOutcome::Prefilling));
        assert_eq!(active[0].processed, expected);
    }
    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(&mut decoder, &mut active, &[], &mut budget, chunk, argmax_rows);
    assert!(
        matches!(outcomes[0], StepOutcome::Stepped { .. }),
        "a lone trailing prompt token is decoded, sampling the first generated token"
    );
    assert_eq!(active[0].generated, 1);
}

/// A forward error on an INTERMEDIATE prefill chunk retires the sequence with that error, just as a
/// one-shot prefill failure does — it must not linger half-prefilled, pinning a slot. This covers the
/// chunked path's error branch, which a non-final chunk reaches (the final chunk fails through the
/// shared `advance_or_fail` path instead).
#[test]
fn a_failed_prefill_chunk_retires_the_sequence() {
    fn seq(tokens: Vec<u32>, max_gen: usize) -> ActiveSeq<()> {
        ActiveSeq {
            slot: 0,
            tokens,
            processed: 0,
            generated: 0,
            max_gen,
            finished: false,
            kv_reservation: 0,
            extra: (),
        }
    }
    // Fail the first prefill call. For a 10-token prompt at chunk 4 that first call is chunk [0,4) —
    // an intermediate chunk, still short of the full prompt — so it exercises the chunked error path.
    let mut decoder = FakeDecoder::new(Arc::new(Mutex::new(Vec::new())), 100);
    decoder.fail_prefills = 1;
    let mut active = vec![seq((1..=10).collect(), 8)];

    let mut budget = PrefillBudget::for_round(&active);
    let outcomes = step_round(&mut decoder, &mut active, &[], &mut budget, 4, argmax_rows);

    assert!(
        matches!(outcomes[0], StepOutcome::Failed(_)),
        "a failed intermediate chunk retires the sequence: {:?}",
        outcomes[0]
    );
    assert!(
        active[0].finished,
        "the sequence is marked finished so the retire sweep frees its slot"
    );
    assert_eq!(
        active[0].generated, 0,
        "a prompt that failed before its final chunk produced no token"
    );
}

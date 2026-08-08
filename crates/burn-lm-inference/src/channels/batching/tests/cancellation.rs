use super::super::*;
use super::fakes::*;
use crate::{
    errors::InferenceError,
    job::{GenerationParams, InferenceJob, InferenceTask},
    StatEntry, TextGenerationListener,
};
use std::sync::{Arc, Mutex};

/// A cancel fired MID-FLIGHT must retire the sequence within one round: held-back detok bytes
/// are flushed before completion, the reply is `Ok` (the client already got real tokens) with
/// the tokens generated so far and a finish-reason stat.
#[test]
fn cancel_mid_flight_retires_within_one_round_and_flushes_detok() {
    // 🦀 is 4 bytes, one per token; the cancel fires while the 2nd byte is produced, so the
    // detok cursor holds a partial character at retire time.
    let mut script: Vec<u32> = "🦀".bytes().map(u32::from).collect();
    script.extend(std::iter::repeat(b'x' as u32).take(12));

    let (job, handle) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    let channel = BatchingChannel::<ByteServer>::with_server(
        ByteServer::new(script, 16).with_cancel_after(2, handle.cancel_signal()),
    );

    let stats = channel.submit(job).unwrap().recv().unwrap().unwrap();

    // Retired within ONE round of the cancel: the 2nd token's round still streams normally,
    // the next round's cancel sweep retires before any 3rd forward.
    assert!(
        stats.entries.contains(&StatEntry::TokensCount(2)),
        "expected exactly the 2 tokens generated before the cancel: {:?}",
        stats.entries
    );
    assert!(
        stats.entries.contains(&StatEntry::Named(
            FINISH_REASON_STAT_NAME.to_string(),
            "Cancelled".to_string()
        )),
        "an in-flight cancel must report its finish reason: {:?}",
        stats.entries
    );
    // The held-back partial character was flushed (lossily, as at any true end of stream)
    // BEFORE completion fired — not silently dropped.
    assert_eq!(handle.join(), "\u{FFFD}");
}

/// A job cancelled while still QUEUED must never be admitted: no prefill (observable as zero
/// forwards for its identity in the order log), no slot, and an `Err(Cancelled)` reply — the
/// caller never received a token, so an empty `Ok` would be misleading.
#[test]
fn cancelled_while_queued_is_never_admitted_and_replies_cancelled() {
    let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
    // One slot + a slow decoder: job A occupies the slot long enough for B to sit queued.
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_slow(1, log.clone()));

    let (job_a, _ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let rx_a = channel.submit(job_a).unwrap();

    // B is cancelled BEFORE it is submitted, so the admission check must catch it.
    let (job_b, hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    hb.cancel();
    let rx_b = channel.submit(job_b).unwrap();

    assert!(
        matches!(rx_b.recv().unwrap(), Err(InferenceError::Cancelled)),
        "a job cancelled while queued must reply Err(Cancelled)"
    );
    rx_a.recv().unwrap().unwrap();

    // B (identity 1 in the log) never reached the decoder: it was dropped before prefill.
    assert!(
        !log.lock().unwrap().contains(&1),
        "cancelled-while-queued job must never be prefilled"
    );
}

/// REGRESSION (the live panic-class bug): a multi-byte character split across tokens must
/// stream as the complete character — exactly once, no U+FFFD, no panic — and the worker must
/// survive to serve a second job. Before S1 the worker decoded each token to TEXT in
/// isolation; the first half of the emoji failed that decode (`ByteServer::detokenize`
/// panics, like `Tiktoken::decode`'s `.expect`) and the panic killed the channel permanently.
#[test]
fn split_multibyte_character_streams_intact_and_worker_survives() {
    // 🦀 is 4 bytes, one per token, followed by an ASCII '!'.
    let mut script: Vec<u32> = "🦀".bytes().map(u32::from).collect();
    script.push(b'!' as u32);
    let channel = BatchingChannel::<ByteServer>::with_server(ByteServer::new(script, 16));

    let (job, handle) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    channel.submit(job).unwrap().recv().unwrap().unwrap();

    let text = handle.join();
    assert_eq!(text, "🦀!", "split emoji must reassemble exactly once");
    assert!(
        !text.contains('\u{FFFD}'),
        "no mid-stream replacement chars"
    );

    // The worker survived (no panic on the partial-character tokens): a second job completes.
    let (job2, h2) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    channel
        .submit(job2)
        .expect("worker must still accept jobs")
        .recv()
        .expect("worker must survive a split multi-byte character")
        .unwrap();
    assert_eq!(h2.join(), "🦀!");
}

/// A sequence retired with held-back bytes (here: the `max_gen` cap lands mid-character) must
/// FLUSH its detok cursor — the trailing bytes reach the listener (lossily, U+FFFD is
/// permitted at true end of stream) instead of being silently dropped, and completion fires.
#[test]
fn retire_flushes_trailing_partial_character() {
    // Only the first 2 of 🦀's 4 bytes fit under max_gen == 2.
    let script: Vec<u32> = "🦀".bytes().map(u32::from).collect();
    let channel = BatchingChannel::<ByteServer>::with_server(ByteServer::new(script, 2));

    let (job, handle) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    channel.submit(job).unwrap().recv().unwrap().unwrap();

    assert_eq!(
        handle.join(),
        "\u{FFFD}",
        "held-back bytes must flush on retire (lossy replacement at end of stream)"
    );
}

/// The queue-wait stat: every completed job reports how long it sat queued before admission,
/// in the same fixed-seconds rendering as the other duration stats.
#[test]
fn completion_stats_include_the_queue_wait() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
        1,
        Arc::new(Mutex::new(Vec::new())),
    ));

    let (job, _h) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let stats = channel.submit(job).unwrap().recv().unwrap().unwrap();

    assert!(
        stats.entries.iter().any(|entry| matches!(
            entry,
            StatEntry::Named(name, value)
                if name == QUEUE_WAIT_STAT_NAME && value.ends_with('s')
        )),
        "completion stats must carry a fixed-seconds queue-wait entry: {:?}",
        stats.entries
    );
}

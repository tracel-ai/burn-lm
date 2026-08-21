use super::super::*;
use super::fakes::*;
use crate::{
    errors::InferenceError,
    job::{GenerationParams, InferenceJob, InferenceTask},
    TextGenerationListener,
};
use std::sync::{Arc, Mutex};

/// A client whose stream errors mid-generation must NOT brick the channel for everyone else.
/// Before the fix, the worker's `emitter.completed()` `.unwrap()` panicked when that listener
/// died, permanently killing the single worker thread. (The stderr panic from `PanicOnText` is
/// the simulated disconnect and is expected.)
#[test]
fn worker_survives_a_client_stream_panic() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
        1,
        Arc::new(Mutex::new(Vec::new())),
    ));

    // Job A: its listener panics on the first emitted token (broken pipe).
    let (job_a, _ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        PanicOnText,
    );
    let rx_a = channel.submit(job_a).unwrap();
    let _ = rx_a.recv(); // A's own outcome is irrelevant; its listener died.

    // Job B: a healthy client must still be served — proving the worker survived A.
    let (job_b, _hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    let rx_b = channel.submit(job_b).unwrap();
    rx_b.recv()
        .expect("channel must survive a client panic")
        .expect("job B should still complete");
}

/// A decoder that breaks the forward rows-in==rows-out contract must retire that sequence with
/// an error, NOT panic the worker — and the channel must keep serving.
#[test]
fn worker_survives_a_decoder_contract_violation() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_bad(
        1,
        Arc::new(Mutex::new(Vec::new())),
    ));

    let (job1, _h1) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let out1 = channel
        .submit(job1)
        .unwrap()
        .recv()
        .expect("worker must survive");
    assert!(
        matches!(out1, Err(crate::InferenceError::BatchContractViolation(_))),
        "contract violation should retire the sequence with a BatchContractViolation error"
    );

    // The worker is still alive: a second job is accepted and processed (likewise retired).
    let (job2, _h2) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    let out2 = channel
        .submit(job2)
        .expect("worker must still accept jobs")
        .recv()
        .expect("worker must survive");
    assert!(out2.is_err(), "second job should also retire with an error");
}

/// The rows-in==rows-out check guards the PREFILL call site too, not just decode. A multi-token
/// prompt makes the sequence's first step genuine prefill work, so the misbehaving decoder's
/// extra logits row is caught there: the sequence retires with `BatchContractViolation` and the
/// worker keeps serving.
#[test]
fn worker_survives_a_prefill_contract_violation() {
    let channel = BatchingChannel::<FakeServer>::with_server(
        FakeServer::new_bad(1, Arc::new(Mutex::new(Vec::new()))).with_prompt_tokens(3),
    );

    let (job1, _h1) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let out1 = channel
        .submit(job1)
        .unwrap()
        .recv()
        .expect("worker must survive");
    assert!(
        matches!(out1, Err(InferenceError::BatchContractViolation(_))),
        "a prefill returning the wrong row count should retire the sequence with a \
         BatchContractViolation error"
    );

    // The worker is still alive: a second job is accepted and processed (likewise retired).
    let (job2, _h2) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    let out2 = channel
        .submit(job2)
        .expect("worker must still accept jobs")
        .recv()
        .expect("worker must survive");
    assert!(out2.is_err(), "second job should also retire with an error");
}

/// A failing `prefill` must retire ONLY its own sequence — with the prefill error, no streamed
/// text — and leave the slot clean: the next job admitted into the same slot (capacity 1) runs a
/// full, fresh generation. This pins the `prefill` contract ("an erring prefill leaves the slot
/// as if it had never been used") together with the worker's release-on-every-retire-path.
#[test]
fn a_failed_prefill_retires_its_sequence_and_leaves_the_slot_clean() {
    let channel = BatchingChannel::<FakeServer>::with_server(
        FakeServer::new(1, Arc::new(Mutex::new(Vec::new())))
            .with_prompt_tokens(2) // multi-token prompt ⇒ the first step is a real prefill
            .with_failing_prefills(1),
    );

    let (job1, h1) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    let out1 = channel
        .submit(job1)
        .unwrap()
        .recv()
        .expect("worker must survive a failed prefill");
    assert!(
        matches!(out1, Err(InferenceError::ContextLengthExceeded(..))),
        "the sequence should retire with the prefill's own error"
    );
    assert_eq!(h1.join(), "", "a failed prefill must not stream any text");

    // Capacity 1 ⇒ the second job reuses slot 0; it must see a pristine slot and complete fully.
    let (job2, h2) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    channel
        .submit(job2)
        .unwrap()
        .recv()
        .expect("worker must keep serving")
        .expect("the slot must be reusable after a failed prefill");
    assert_eq!(
        h2.join(),
        "10101010",
        "the reused slot must start fresh after the failed prefill"
    );
}

/// Completion must be sent EXACTLY ONCE per job. Before the fix, a failed forward sent `Err`
/// on the completion channel (filling the bounded one-shot) and the retire sweep then sent
/// `Ok` AGAIN for the same sequence — blocking the worker until the caller drained the first
/// message. We prove the worker stays live by NOT recv-ing the first job's completion and
/// asserting a second job still completes.
#[test]
fn failed_sequence_completes_exactly_once_without_blocking_the_worker() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_bad(
        1,
        Arc::new(Mutex::new(Vec::new())),
    ));

    // Job 1 fails its forward; its `Err` sits buffered in the one-shot because we don't recv.
    // A double send would now block the worker on the buffered channel.
    let (job1, _h1) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let rx1 = channel.submit(job1).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Job 2 must still be served — proving the worker did not block on a second send.
    let (job2, _h2) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    let rx2 = channel.submit(job2).unwrap();
    let _ = rx2
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("worker blocked: completion was sent twice for the failed sequence");

    // The first receiver yields exactly one message (the `Err`), then disconnects — no
    // buffered second message.
    assert!(
        rx1.recv().expect("first completion must arrive").is_err(),
        "failed sequence should complete with the forward error"
    );
    assert!(
        rx1.recv().is_err(),
        "completion channel should disconnect after exactly one message"
    );
}

/// FAILURE LADDER, top rung: a panic inside the worker loop must reply `WorkerDied` — exactly
/// once — to the active job AND every queued job (text streamed before the panic still reaches
/// the listener via the detok flush), release their depth slots, and the NEXT submission must
/// lazily respawn a fresh worker. One panic never bricks the channel. (The scripted panic's
/// stderr backtrace is expected.)
#[test]
fn worker_panic_fails_active_and_queued_with_workerdied_then_respawns() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_panicky(
        1,
        Arc::new(Mutex::new(Vec::new())),
        2, // panic on the 3rd step: 2 tokens stream first, ~20ms apart
    ));

    // A is admitted (one slot); B is surely queued behind it before the ~60ms panic.
    let (job_a, ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        TextGenerationListener::default(),
    );
    let (job_b, _hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    let rx_a = channel.submit(job_a).unwrap();
    let rx_b = channel.submit(job_b).unwrap();

    assert!(
        matches!(rx_a.recv().unwrap(), Err(InferenceError::WorkerDied)),
        "the in-flight job must observe WorkerDied"
    );
    assert!(
        matches!(rx_b.recv().unwrap(), Err(InferenceError::WorkerDied)),
        "queued jobs must observe WorkerDied too"
    );
    // Exactly one reply each: the channels disconnect after the WorkerDied.
    assert!(rx_a.recv().is_err());
    assert!(rx_b.recv().is_err());
    // The two tokens generated before the panic were streamed (and the detok cursor flushed)
    // before A's completion fired.
    assert_eq!(ha.join(), "1010");
    // Both queue permits were released in the panic fallout.
    assert_eq!(channel.queue_depth(), 0);

    // LAZY RESPAWN: the next submit detects the finished worker thread and spawns a fresh one
    // around a fresh `Server::default()` (unloaded; admission lazy-loads it).
    let (job_c, _hc) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    channel
        .submit(job_c)
        .expect("a dead worker must be respawned on the next submit")
        .recv()
        .expect("the respawned worker must serve")
        .expect("the post-panic job should complete normally");
}

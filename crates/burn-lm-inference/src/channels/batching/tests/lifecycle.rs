use super::super::*;
use super::fakes::*;
use crate::{
    errors::InferenceError,
    job::{GenerationParams, InferenceJob, InferenceTask},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[test]
fn lifecycle_round_trip() {
    let channel = BatchingChannel::<FakeServer>::new();
    assert!(!channel.is_spawned());
    // A pre-spawn `is_loaded` answers without spawning the worker.
    assert!(!channel.is_loaded());
    assert!(!channel.is_spawned());

    channel.load().unwrap();
    assert!(channel.is_spawned());
    assert!(channel.is_loaded());

    channel.unload().unwrap();
    assert!(!channel.is_loaded());

    // Lifecycle traffic on an idle worker must not (re)load the model: `step` skips the
    // decoder when nothing is active (the fake's `decoder()` lazy-loads like the real ones).
    let _ = channel.is_downloaded();
    assert!(!channel.is_loaded());
}

/// BACKPRESSURE: at the queue bound, `submit` must shed synchronously with `Overloaded` — and
/// a rejected submit must not leak its depth-counter slot. `max_slots == 0` keeps the queued
/// job pinned (nothing is ever admitted), so the bound is exercised without timing races.
#[test]
fn submit_sheds_with_overloaded_at_the_queue_bound() {
    let channel = BatchingChannel::<FakeServer>::with_server_and_depth(
        FakeServer::with_capacity_probe(0, Arc::new(AtomicUsize::new(0))),
        1,
    );
    assert_eq!(channel.queue_depth(), 0, "queue starts empty");
    assert!(!channel.is_overloaded());

    let (job_a, _ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let _rx_a = channel
        .submit(job_a)
        .expect("first job fits the depth-1 queue");
    assert_eq!(channel.queue_depth(), 1);
    assert!(
        channel.is_overloaded(),
        "advisory probe must report a full queue"
    );

    let (job_b, _hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );
    assert!(
        matches!(channel.submit(job_b), Err(InferenceError::Overloaded)),
        "a submit beyond the bound must shed synchronously"
    );
    assert_eq!(
        channel.queue_depth(),
        1,
        "a shed submit must release its counter slot"
    );
}

/// A server reporting `max_slots == 0` with a job queued must PARK the worker, not busy-spin a
/// core. We detect a spin via the `batch_capacity` call count: parked ⇒ a couple of calls;
/// spinning ⇒ thousands over the same window.
#[test]
fn max_slots_zero_parks_instead_of_spinning() {
    let calls = Arc::new(AtomicUsize::new(0));
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::with_capacity_probe(
        0,
        calls.clone(),
    ));

    // Queued, but never admittable while max_slots == 0.
    let (job, _h) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let _rx = channel.submit(job).unwrap();

    std::thread::sleep(std::time::Duration::from_millis(50));
    let n = calls.load(Ordering::Relaxed);
    assert!(
        n < 100,
        "worker busy-spun on max_slots==0 (batch_capacity called {n} times in 50ms); it should park"
    );
}

/// Unload (and clear-state) while work is in flight must be REJECTED, not silently reload the
/// model under in-flight per-seq caches. Once the job retires, unload succeeds.
#[test]
fn unload_is_rejected_while_a_job_is_in_flight() {
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_slow(
        1,
        Arc::new(Mutex::new(Vec::new())),
    ));

    // ~16 steps × 20ms ⇒ the job is comfortably still running 50ms in.
    let (job, _h) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let rx = channel.submit(job).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    assert!(
        matches!(channel.unload(), Err(InferenceError::Busy(_, _))),
        "unload must be rejected while a sequence is active"
    );

    // After the job completes the active set is empty, so unload succeeds.
    rx.recv().unwrap().unwrap();
    channel.unload().expect("unload should succeed once idle");
}

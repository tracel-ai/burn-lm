use super::super::*;
use super::fakes::*;
use crate::{
    errors::InferenceError,
    job::{GenerationParams, InferenceJob, InferenceTask},
};
use std::sync::{atomic::AtomicUsize, Arc, Mutex};

/// STRESS, exactly-one-reply invariant: a concurrent burst against a depth-2 queue produces
/// mixed outcomes — completed, shed with a synchronous `Overloaded`, cancelled while queued —
/// and EVERY submission resolves exactly once (no hang, no second reply). The mix is seeded
/// deterministically; the actual interleaving comes from real thread scheduling.
#[test]
fn every_submission_resolves_exactly_once_under_overload() {
    let server = FakeServer {
        loaded: false,
        slots: 1,
        decoder: FakeDecoder {
            log: Arc::new(Mutex::new(Vec::new())),
            emit: 2,
            extra_rows: 0,
            step_delay_ms: 2, // slow enough that submissions genuinely pile up
            panic_at_step: None,
        },
        capacity_calls: Arc::new(AtomicUsize::new(0)),
        fixed_token: None,
    };
    let channel = BatchingChannel::<FakeServer>::with_server_and_depth(server, 2);

    let handles: Vec<_> = (0..16)
        .map(|i| {
            let channel = channel.clone();
            std::thread::spawn(move || {
                let (job, handle) = InferenceJob::create(
                    InferenceTask::Prompt(if i % 2 == 0 { "a" } else { "b" }.into()),
                    GenerationParams::default(),
                    NullListener,
                );
                if i % 3 == 0 {
                    handle.cancel(); // a third of the jobs cancel before/while queued
                }
                match channel.submit(job) {
                    // Shed: resolved synchronously, nothing to wait on.
                    Err(InferenceError::Overloaded) => {}
                    Err(other) => panic!("unexpected submit error: {other:?}"),
                    Ok(rx) => {
                        match rx.recv().expect("every accepted job must get a reply") {
                            Ok(_) | Err(InferenceError::Cancelled) => {}
                            Err(other) => panic!("unexpected completion: {other:?}"),
                        }
                        assert!(rx.recv().is_err(), "a job must reply exactly once");
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("no submitter may hang or panic");
    }

    // Every permit was released on its job's way out, however it resolved.
    assert_eq!(channel.queue_depth(), 0);

    // And the channel is still healthy: a final job completes normally.
    let (job, _h) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    channel
        .submit(job)
        .expect("queue must be empty again")
        .recv()
        .unwrap()
        .unwrap();
}

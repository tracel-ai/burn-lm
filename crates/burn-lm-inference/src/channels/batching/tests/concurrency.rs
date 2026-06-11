use super::super::*;
use super::fakes::*;
use crate::{
    job::{GenerationParams, InferenceJob, InferenceTask},
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

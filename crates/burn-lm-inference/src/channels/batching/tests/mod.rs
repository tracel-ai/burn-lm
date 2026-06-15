//! Orchestration tests for the batching channel: admission, backpressure and shedding,
//! cancellation, the exactly-one-reply discipline, and worker-death recovery. They run against the
//! fake server in `fakes`, so they exercise the engine's plumbing without a real model. The
//! companion suite that checks the decode math — that batched output matches solo runs
//! byte-for-byte — lives with the model in `burn-lm-llama` (`generation::batched_equivalence`),
//! because only a real model produces logits to compare.

mod fakes;

mod backpressure;
mod cancellation;
mod concurrency;
mod failure;
mod lifecycle;
mod stress;

//! Next-token sampling, owned by the framework rather than the model.
//!
//! The continuous-batching engine owns sampling instead of delegating it to the model, which keeps
//! the model boundary to forward and tokenizer primitives only. A sampler is config-driven: the
//! server builds one from its own sampling config (the same one whether a request runs solo or
//! batched), and every in-flight sequence shares that one sampler. What varies per sequence is only
//! the RNG, which a stateful strategy (top-p with a seed) has to advance independently for each
//! sequence so two concurrent requests don't draw off one shared stream. That per-sequence RNG is
//! the whole of `SamplingState`.
//!
//! The seam is one method on the server (`sampler`) returning a `Box<dyn Sampler>`, so the worker
//! grabs the sampler as an owned value before it borrows the decoder — the owned box doesn't borrow
//! the server, so the two borrows don't collide. The framework's own default is deterministic
//! argmax; a model with a real sampling config (temperature, top-p, seed) supplies its own sampler.

use burn::tensor::{Int, Tensor};
use rand::SeedableRng;

use crate::errors::{InferenceError, InferenceResult};

/// One sequence's per-sequence sampling state. With config-driven sampling the only thing that
/// genuinely differs between two in-flight sequences is the RNG: a seeded top-p draw has to advance
/// its own stream across that sequence's tokens, independently of every other sequence, so two
/// concurrent requests never share one RNG. The shared sampler holds everything else (temperature,
/// top-p threshold), so this is just the RNG.
pub struct SamplingState {
    /// The sequence's own RNG. A stateful strategy advances it across the sequence's tokens; a
    /// stateless one (argmax) ignores it. Seeded per sequence by the sampler's `fresh_state`.
    pub rng: rand::rngs::StdRng,
}

/// Picks the next token id for a whole batch of sequences from a `[batch, vocab]` tensor of logits.
///
/// One sampler is shared by every in-flight sequence (it carries the config: temperature, top-p),
/// and the per-sequence variation rides in `states`. The trait is object-safe — no associated
/// types — so the worker can hold it as `Box<dyn Sampler>`, grabbed as an owned value before the
/// decoder borrow so the two never collide. `states[i]` is the per-sequence state for row `i` of
/// `logits`, and the returned `Vec` has one token id per row, in row order.
pub trait Sampler: Send {
    /// Build a fresh per-sequence state for one admitted sequence. A seeded strategy seeds the RNG
    /// here (so each sequence advances its own stream); a stateless one returns any seeded RNG,
    /// since it ignores it.
    fn fresh_state(&self) -> SamplingState;

    /// Sample one token id per row of a `[batch, vocab]` logits tensor. `states[i]` is row `i`'s
    /// per-sequence state, and the returned ids are in row order, one per row.
    fn sample(&self, logits: Tensor<2>, states: &mut [SamplingState]) -> InferenceResult<Vec<u32>>;
}

/// The framework's default sampler: deterministic greedy argmax. It ignores the per-sequence state
/// entirely — there is nothing stochastic to advance — so a model with no sampling config needs
/// nothing beyond this default.
pub struct Argmax;

impl Sampler for Argmax {
    fn fresh_state(&self) -> SamplingState {
        // The RNG is never read, but `SamplingState` always carries one. Any seed is fine, since
        // argmax's output does not depend on it; we use a fixed one to keep `fresh_state` total and
        // free of entropy calls.
        SamplingState {
            rng: rand::rngs::StdRng::seed_from_u64(0),
        }
    }

    fn sample(
        &self,
        logits: Tensor<2>,
        _states: &mut [SamplingState],
    ) -> InferenceResult<Vec<u32>> {
        ids_to_host(logits.argmax(1))
    }
}

/// Read a `[batch, 1]` tensor of sampled token ids back to host `u32`s — the single device-to-host
/// readback every sampler funnels through. A conversion failure is a batch-contract violation
/// rather than a panic, because on the worker thread a panic here would unwind and brick the channel.
pub fn ids_to_host(ids: Tensor<2, Int>) -> InferenceResult<Vec<u32>> {
    ids.into_data()
        .convert::<u32>()
        .into_vec::<u32>()
        .map_err(|_| {
            InferenceError::BatchContractViolation(
                "sampled token tensor did not convert to u32".to_string(),
            )
        })
}

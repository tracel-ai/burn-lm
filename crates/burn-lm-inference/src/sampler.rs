//! Next-token sampling, owned by the framework rather than the model.
//!
//! The continuous-batching engine owns sampling instead of delegating it to the model, which keeps
//! the model boundary to forward and tokenizer primitives only. A sampler is config-driven: the
//! server builds one from its own sampling config, and that one sampler serves every in-flight
//! sequence in a round. It carries only its configuration — there is no per-sequence state to thread
//! through the engine, because any randomness a strategy needs is drawn on the device from the
//! tensor backend's own RNG, not from a host RNG owned per sequence.
//!
//! So the seam is small: one method on the server (`sampler`) returns a `Box<dyn Sampler>`, and one
//! method on the sampler (`sample`) turns a `[batch, vocab]` logits tensor into one token id per row.
//! The framework's own default is deterministic argmax; a model with a real sampling config
//! (temperature, top-p) supplies its own sampler. Reproducibility, when a deployment or a test wants
//! it, comes from seeding the backend RNG (`Device::seed`), which makes the backend's random draws
//! deterministic for that run.

use burn::tensor::{Int, Tensor};

use crate::errors::{InferenceError, InferenceResult};

/// Picks the next token id for a whole batch of sequences from a `[batch, vocab]` tensor of logits.
///
/// One sampler is shared by every in-flight sequence in a round: it carries the configuration
/// (temperature, top-p) and nothing sequence-specific. The trait is object-safe — no associated
/// types, no generics — so the worker can hold it as `Box<dyn Sampler>`, grabbed as an owned value
/// before it borrows the decoder so the two borrows never collide. Any randomness a strategy needs
/// is drawn from the tensor backend's RNG inside `sample`, so there is no per-sequence state to pass
/// in; the returned `Vec` has one token id per row of `logits`, in row order.
pub trait Sampler: Send {
    /// Sample one token id per row of a `[batch, vocab]` logits tensor, returned in row order, one
    /// per row.
    fn sample(&self, logits: Tensor<2>) -> InferenceResult<Vec<u32>>;
}

/// The framework's default sampler: deterministic greedy argmax. It draws no randomness at all, so a
/// model with no sampling config needs nothing beyond this default.
pub struct Argmax;

impl Sampler for Argmax {
    fn sample(&self, logits: Tensor<2>) -> InferenceResult<Vec<u32>> {
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

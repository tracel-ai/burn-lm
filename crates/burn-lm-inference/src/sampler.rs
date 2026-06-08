//! Minimal next-token sampling owned by the framework continuous engine.
//!
//! The continuous-batching engine in [`BatchingChannel`](crate::channels::batching::BatchingChannel)
//! owns sampling rather than delegating it to the model: the model seam is forward + tokenizer
//! primitives only. Phase 1 only needs deterministic argmax; richer strategies (top-p, temperature)
//! can be ported here later without touching the model trait.

use burn::tensor::{Int, Tensor};

/// Strategy for picking the next token id from a row of logits.
#[derive(Debug, Clone, Copy, Default)]
pub enum Sampler {
    /// Greedy: pick the highest-logit token. Deterministic.
    #[default]
    Argmax,
}

impl Sampler {
    /// Sample one token id per row of a `[batch, vocab]` logits tensor, returning `[batch, 1]`.
    pub fn sample(&mut self, logits: Tensor<2>) -> Tensor<2, Int> {
        match self {
            Self::Argmax => logits.argmax(1),
        }
    }
}

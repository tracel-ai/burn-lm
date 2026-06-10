//! Minimal next-token sampling owned by the framework continuous engine.
//!
//! The continuous-batching engine in [`BatchingChannel`](crate::channels::batching::BatchingChannel)
//! owns sampling rather than delegating it to the model: the model seam is forward + tokenizer
//! primitives only. Phase 1 only needs deterministic argmax; richer strategies (top-p, temperature)
//! can be ported here later without touching the model trait.

use burn::tensor::{Int, Tensor};

/// Picks the next token id from a `[batch, vocab]` row of logits.
///
/// The generic decode core ([`step_round`](crate::batching::step_round)) is parameterized over this
/// trait rather than a concrete sampler so that callers can plug in their own strategy — the
/// framework's argmax [`Sampler`], or (on the library side) a temperature-/top-p-aware sampler —
/// without the core knowing which. Implementations return one token id per input row, shaped
/// `[batch, 1]`.
pub trait NextTokenSampler {
    /// Sample one token id per row of a `[batch, vocab]` logits tensor, returning `[batch, 1]`.
    fn sample_next(&mut self, logits: Tensor<2>) -> Tensor<2, Int>;
}

/// Boxed samplers delegate, so the worker can hold the server-built
/// `Box<dyn NextTokenSampler + Send>` (see
/// [`BatchedInferenceServer::next_token_sampler`](crate::batching::BatchedInferenceServer::next_token_sampler))
/// while [`step_round`](crate::batching::step_round) keeps its plain `S: NextTokenSampler` bound.
impl NextTokenSampler for Box<dyn NextTokenSampler + Send> {
    fn sample_next(&mut self, logits: Tensor<2>) -> Tensor<2, Int> {
        (**self).sample_next(logits)
    }
}

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

impl NextTokenSampler for Sampler {
    fn sample_next(&mut self, logits: Tensor<2>) -> Tensor<2, Int> {
        self.sample(logits)
    }
}

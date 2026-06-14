//! Next-token sampling, owned by the framework rather than the model.
//!
//! The continuous-batching engine owns sampling instead of delegating it to the model, which keeps
//! the model boundary to forward and tokenizer primitives only. Today the framework's own sampler
//! only needs deterministic argmax; richer strategies like top-p and temperature can be added here
//! later without touching the model trait. A model that has its own sampling config supplies its
//! own sampler through `next_token_sampler` instead.

use burn::tensor::{Int, Tensor};

/// Picks the next token id from a `[batch, vocab]` tensor of logits.
///
/// The generic decode core `step_round` is parameterized over this trait rather than a concrete
/// sampler so callers can plug in their own strategy — the framework's argmax `Sampler`, or a
/// temperature- or top-p-aware sampler on the library side — without the core knowing which.
/// Implementations return one token id per input row, shaped `[batch, 1]`.
pub trait NextTokenSampler {
    /// Sample one token id per row of a `[batch, vocab]` logits tensor, returning `[batch, 1]`.
    fn sample_next(&mut self, logits: Tensor<2>) -> Tensor<2, Int>;
}

/// A boxed sampler is itself a `NextTokenSampler`, forwarding to its inner one. The server hands out
/// per-request samplers as `Box<dyn NextTokenSampler + Send>` (see `next_token_sampler`), and this
/// impl lets that box be used anywhere the trait is expected.
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

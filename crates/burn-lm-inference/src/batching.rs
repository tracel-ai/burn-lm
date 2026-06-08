//! Opt-in abstractions for request batching.
//!
//! These traits let a model expose a low-level decoder primitive that can be driven by either a
//! single-request engine (today's path) or, in the future, a continuous-batching engine owned by
//! [`BatchingChannel`](crate::channels::batching::BatchingChannel).
//!
//! This module currently only defines the *shape* of the abstraction. No engine consumes it yet:
//! the first `BatchingChannel` is a worker/queue skeleton that still drives jobs through
//! [`InferenceServer::run_job`](crate::InferenceServer). The decoder traits are introduced now so
//! the seam is stable before the batched engine lands.

use burn::tensor::{Int, Tensor};

use crate::{errors::InferenceResult, job::InferenceTask, server::InferenceServer};

/// Spare capacity a batched decoder currently has for admitting more work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchCapacity {
    /// Number of additional sequences that can be admitted.
    pub free_slots: usize,
    /// Number of additional KV-cache tokens that can be stored across all slots.
    pub free_kv_tokens: usize,
}

/// A ragged batch of token rows to forward through a decoder.
///
/// `input_tokens` is `[batch, seq]`. Each row carries its own starting position (`positions`) and
/// physical cache slot (`cache_slots`) so prefill (long `seq`) and decode (`seq == 1`) work for
/// sequences admitted at different times.
#[derive(Debug, Clone)]
pub struct ForwardBatch {
    /// Token ids for each active row, shaped `[batch, seq]`.
    pub input_tokens: Tensor<2, Int>,
    /// Absolute position of the first token of each row (one per batch row).
    pub positions: Vec<usize>,
    /// Physical KV-cache slot assigned to each row (one per batch row).
    pub cache_slots: Vec<usize>,
}

/// Output of a single decoder forward pass.
///
/// Contract enforced by the engine: `logits` must have exactly one row per input row
/// (`logits.dims()[0] == batch.input_tokens.dims()[0]`) and at least one position. A decoder that
/// violates this retires the offending sequence with a [`BatchContractViolation`] error rather than
/// silently sampling the wrong sequence or panicking the worker.
///
/// [`BatchContractViolation`]: crate::InferenceError::BatchContractViolation
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Logits for every position, shaped `[batch, seq, vocab]`.
    pub logits: Tensor<3>,
}

/// A reusable, batch-capable decoder primitive.
///
/// Model authors implement this once; both the single-request and (future) continuous-batching
/// engines call [`forward`](BatchedDecoder::forward) with different shapes.
pub trait BatchedDecoder {
    /// Decoder-managed KV/state cache.
    type Cache;

    /// Allocate a cache sized for the given capacity.
    fn allocate_cache(&self, capacity: BatchCapacity) -> Self::Cache;

    /// Forward a ragged batch, returning logits and mutating the cache in place.
    fn forward(
        &mut self,
        batch: ForwardBatch,
        cache: &mut Self::Cache,
    ) -> InferenceResult<ForwardOutput>;
}

/// A server that can expose a [`BatchedDecoder`] for batched serving.
///
/// Implementing this trait is what makes a model eligible for the batching channel. The existing
/// [`InferenceServer`] surface (lifecycle, config, single-job `run_job`) is unchanged.
pub trait BatchedInferenceServer: InferenceServer {
    /// The decoder primitive this server drives.
    type Decoder: BatchedDecoder;

    /// Mutably borrow the loaded decoder, loading the model first if needed.
    ///
    /// The framework continuous loop holds the server with exclusive access (the worker thread owns
    /// it), so a plain `&mut` borrow is enough — no lock or callback is required.
    fn decoder(&mut self) -> InferenceResult<&mut Self::Decoder>;

    /// Current spare capacity for admitting more sequences.
    fn batch_capacity(&self) -> BatchCapacity;

    /// Tokenize a submitted task into the token ids the decoder consumes.
    ///
    /// This is a thin wrapper over the model's own tokenizer; it is a *primitive* the framework
    /// continuous loop calls during admission, not a loop the model owns.
    fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>>;

    /// Detokenize generated token ids back to text. Called by the framework loop per emitted token.
    fn detokenize(&self, tokens: &[u32]) -> String;

    /// Token ids that, when generated, end a sequence (EOS/EOT/EOM, …).
    fn stop_ids(&self) -> Vec<u32>;

    /// Maximum number of tokens to generate per sequence before forcibly retiring it.
    ///
    /// A capacity/config primitive (like [`batch_capacity`](Self::batch_capacity)): it bounds the
    /// loop without exposing scheduling vocabulary. The framework loop stops a sequence at the
    /// first stop id or once it has generated this many tokens, whichever comes first.
    fn max_gen_tokens(&self) -> usize;

    /// Allocate a fresh per-sequence cache for a newly admitted sequence.
    ///
    /// Default implementation routes through [`decoder`](Self::decoder) +
    /// [`BatchedDecoder::allocate_cache`]. The cache is then OWNED BY THE FRAMEWORK engine, which
    /// passes it back into [`BatchedDecoder::forward`] on every step for that sequence.
    fn allocate_cache(
        &mut self,
        capacity: BatchCapacity,
    ) -> InferenceResult<<Self::Decoder as BatchedDecoder>::Cache> {
        Ok(self.decoder()?.allocate_cache(capacity))
    }
}

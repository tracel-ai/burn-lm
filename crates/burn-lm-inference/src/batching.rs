//! Opt-in abstractions for request batching.
//!
//! These traits let a model expose a low-level decoder primitive that any engine can drive,
//! batched or not — single-request generation is just the batch=1 degenerate case.
//!
//! Two drivers consume this seam today, both through the shared decode core [`step_round`]: the
//! serving `BatchingChannel` worker, whose continuous loop admits queued jobs and streams tokens
//! per round, and the library's `Llama::generate_batch`, which drives a fixed set of prompts to
//! completion. Per-sequence state lives in [`ActiveSeq`]; the model only exposes `forward`,
//! tokenizer primitives and capacity.

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    errors::{InferenceError, InferenceResult},
    job::InferenceTask,
    sampler::{NextTokenSampler, Sampler},
    server::InferenceServer,
};

/// The decoder's STATIC capacity limits, declared by the model.
///
/// These are maxima the model can handle (a function of the model + hardware), not a live
/// "free right now" figure: the engine owns the active set and derives the actual free capacity as
/// `max - in-use`. (If a future phase moves KV accounting model-side, these could become live
/// figures; today they are fixed after the model loads.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchCapacity {
    /// Maximum number of sequences the decoder can run concurrently. The engine admits while
    /// `active.len() < max_slots`, so the actual free slots are `max_slots - active.len()`.
    pub max_slots: usize,
    /// Maximum number of KV-cache tokens across all slots. Declared but NOT yet enforced by
    /// admission — enforcement pairs with real KV management in Phase 2.
    pub max_kv_tokens: usize,
}

/// A ragged batch of token rows to forward through a decoder.
///
/// `input_tokens` is `[batch, seq]`. Each row carries its own starting position (`positions`) and
/// physical cache slot (`cache_slots`) so prefill (long `seq`) and decode (`seq == 1`) work for
/// sequences admitted at different times.
///
/// PHASE-1 SHAPE: the rectangular `[batch, seq]` tensor cannot express a truly ragged round that
/// fuses a prefill row (long) with decode rows (len 1) without padding — today that never arises
/// because every batch is a single row. Phase 2 must revisit this layout once the ragged encoding
/// is decided (padded-rectangular vs a flattened `[total_tokens]` tensor plus per-row lengths,
/// vLLM-style). Carrying raw token ids here instead of a pre-built tensor would also let
/// [`BatchedDecoder::device`] be deleted, since the decoder would build its own input tensor.
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

    /// The device the model lives on. The generic decode core ([`step_round`]) builds each round's
    /// input-token tensor on this device so it matches the model's weights — building on the global
    /// inference device instead would put inputs on the wrong device for a model loaded elsewhere
    /// (e.g. a unit test on the default device).
    fn device(&self) -> Device;

    /// Allocate a cache sized for the given capacity.
    fn allocate_cache(&self, capacity: BatchCapacity) -> Self::Cache;

    /// Forward a ragged batch, returning logits and mutating the cache in place.
    fn forward(
        &mut self,
        batch: ForwardBatch,
        cache: &mut Self::Cache,
    ) -> InferenceResult<ForwardOutput>;
}

/// The cache type of a server's decoder.
pub type CacheOf<S> = <<S as BatchedInferenceServer>::Decoder as BatchedDecoder>::Cache;

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

    /// The decoder's static capacity maxima (see [`BatchCapacity`]). Not a live "free right now"
    /// figure: the engine owns the active set and derives free capacity as `max - in-use`.
    fn batch_capacity(&self) -> BatchCapacity;

    /// Tokenize a submitted task into the token ids the decoder consumes.
    ///
    /// This is a thin wrapper over the model's own tokenizer; it is a *primitive* the framework
    /// continuous loop calls during admission, not a loop the model owns.
    fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>>;

    /// Detokenize generated token ids back to text. Called by the framework loop per emitted token.
    fn detokenize(&self, tokens: &[u32]) -> String;

    /// Detokenize generated token ids to RAW BYTES, not guaranteed to be valid UTF-8 on their
    /// own. This is the primitive the framework loop actually streams through: byte-level BPE
    /// tokenizers routinely split a multi-byte character across tokens, so per-token text decode
    /// can fail (or emit U+FFFD) mid-character — the loop instead feeds these bytes through a
    /// per-sequence [`Utf8Buffer`](crate::utf8::Utf8Buffer) and emits only complete text.
    ///
    /// The default round-trips through [`detokenize`](Self::detokenize), so a model whose
    /// per-token decode is already total needs nothing extra; models with a byte-level tokenizer
    /// (e.g. Tiktoken) override this with their tokenizer's infallible byte decode.
    fn detokenize_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        self.detokenize(tokens).into_bytes()
    }

    /// Token ids that, when generated, end a sequence (EOS/EOT/EOM, …).
    fn stop_ids(&self) -> Vec<u32>;

    /// Maximum number of tokens to generate per sequence before forcibly retiring it.
    ///
    /// A capacity/config primitive (like [`batch_capacity`](Self::batch_capacity)): it bounds the
    /// loop without exposing scheduling vocabulary. The framework loop stops a sequence at the
    /// first stop id or once it has generated this many tokens, whichever comes first.
    fn max_gen_tokens(&self) -> usize;

    /// Build a fresh next-token sampler from the server's CURRENT sampling config.
    ///
    /// A config primitive (like [`max_gen_tokens`](Self::max_gen_tokens)): the engine asks for a
    /// new sampler per ADMITTED sequence and keeps it for that sequence's whole generation, so a
    /// seeded RNG advances across the sequence's tokens (matching the single-request path) while
    /// config changes (e.g. via `ParseJsonConfig`) take effect for later-admitted sequences. The
    /// default is the framework's deterministic argmax [`Sampler`]; servers with sampling config
    /// (temperature, top-p, seed, …) override this to honor it.
    fn next_token_sampler(&self) -> Box<dyn NextTokenSampler + Send> {
        Box::new(Sampler::default())
    }

    /// Allocate a fresh per-sequence cache for a newly admitted sequence.
    ///
    /// Default implementation routes through [`decoder`](Self::decoder) +
    /// [`BatchedDecoder::allocate_cache`]. The cache is then OWNED BY THE FRAMEWORK engine, which
    /// passes it back into [`BatchedDecoder::forward`] on every step for that sequence.
    fn allocate_cache(&mut self, capacity: BatchCapacity) -> InferenceResult<CacheOf<Self>> {
        Ok(self.decoder()?.allocate_cache(capacity))
    }
}

/// Per-sequence state advanced by [`step_round`].
///
/// This is the *generic* core of an in-flight sequence: its engine-owned [`cache`](Self::cache),
/// token buffer, position cursor, generated-token counter and `finished` flag. Driver-specific
/// bookkeeping (a serving job's emitter + completion sender, or the library's
/// `GenerationContext`) rides along in [`extra`](Self::extra), which `step_round` never touches —
/// each driver chooses what to attach and how to act on the per-round results.
pub struct ActiveSeq<Cache, Extra = ()> {
    /// The model's per-sequence cache, allocated by the driver and passed into every `forward`.
    pub cache: Cache,
    /// Full token buffer (prompt + generated). The next forward consumes the unprocessed tail.
    pub tokens: Vec<u32>,
    /// Number of tokens already pushed through the decoder (== absolute position of next input).
    pub processed: usize,
    /// Generated-token count (excludes the prompt).
    pub generated: usize,
    /// Hard cap on generated tokens for this sequence.
    pub max_gen: usize,
    /// Set once a stop id or the cap is hit; the next sweep retires it.
    pub finished: bool,
    /// Driver-owned payload (emitter/completion, generation context, …). Opaque to `step_round`.
    pub extra: Extra,
}

/// Per-sequence outcome of one [`step_round`] sweep, aligned position-for-position with the slice
/// passed in. A sequence that was already `finished` (or empty) at the start of the round yields
/// [`StepOutcome::Skipped`]; an advanced sequence yields [`StepOutcome::Stepped`] (or
/// [`StepOutcome::Failed`] if its `forward` violated the batch contract).
pub enum StepOutcome {
    /// The sequence was not advanced this round (already finished, or had no unprocessed tokens).
    Skipped,
    /// The sequence advanced by one token.
    Stepped {
        /// The token id sampled this round (already appended to the sequence's `tokens`).
        token: u32,
        /// Whether this token is a stop id. A stop token ends the sequence and (by convention) is
        /// not streamed as text; drivers use this to decide emission.
        is_stop: bool,
        /// Whether the sequence is now finished (stop id reached, or the `max_gen` cap hit). When
        /// set, the sequence's `finished` flag is also set, so the next round skips it.
        finished: bool,
    },
    /// The sequence's forward failed; the driver should retire it with this error.
    Failed(InferenceError),
}

/// Advance every active sequence by exactly one token: build input from the per-seq cache →
/// [`forward`](BatchedDecoder::forward) → enforce the rows-in==rows-out contract → slice the last
/// position's logits → sample → synchronous stop check → advance the cursor / counters.
///
/// This is the single, generic decode core shared by both drivers (the serving
/// [`BatchingChannel`](crate::channels::batching::BatchingChannel) worker and the library's
/// `Llama::generate_batch`). It is deliberately *pure per-round*: no queue, no admission, no
/// completion signalling, no emitter ownership, no detokenization — those belong to the driver,
/// which inspects the returned [`StepOutcome`]s (one per input sequence) to stream/retire.
///
/// Stop detection is **synchronous**: a sampled token that is in `stop_ids` finishes its sequence
/// in the same round it is produced, so no extra token is generated past a stop id. The `max_gen`
/// cap is likewise checked in-round.
pub fn step_round<D: BatchedDecoder, X, S: NextTokenSampler>(
    decoder: &mut D,
    active: &mut [ActiveSeq<D::Cache, X>],
    sampler: &mut S,
    stop_ids: &[u32],
) -> Vec<StepOutcome> {
    let mut outcomes = Vec::with_capacity(active.len());
    // Build each round's input tensor on the model's own device (see `BatchedDecoder::device`).
    let device = decoder.device();

    for seq in active.iter_mut() {
        if seq.finished {
            outcomes.push(StepOutcome::Skipped);
            continue;
        }

        // No generation budget left (notably `max_gen == 0`): finish without a forward. The cap
        // bounds *generated* tokens, so enforcing it before prefill keeps a zero-token request a
        // true no-op instead of prefilling and producing one token.
        if seq.generated >= seq.max_gen {
            seq.finished = true;
            outcomes.push(StepOutcome::Skipped);
            continue;
        }

        // The unprocessed tail: on the first step this is the whole prompt (prefill); afterwards a
        // single token (decode). An empty prompt cannot generate, so retire it immediately.
        if seq.processed >= seq.tokens.len() {
            seq.finished = true;
            outcomes.push(StepOutcome::Skipped);
            continue;
        }

        let input_ids: Vec<i32> = seq.tokens[seq.processed..].iter().map(|&t| t as i32).collect();
        let seq_len = input_ids.len();
        let position = seq.processed;

        let input_tokens =
            Tensor::<2, Int>::from_data(TensorData::new(input_ids, [1, seq_len]), &device);
        let batch = ForwardBatch {
            input_tokens,
            positions: vec![position],
            cache_slots: vec![0],
        };

        let next_token = match forward_one(decoder, &mut seq.cache, batch, sampler) {
            Ok(token) => token,
            Err(err) => {
                seq.finished = true;
                outcomes.push(StepOutcome::Failed(err));
                continue;
            }
        };

        // Everything up to here is now processed; the next step consumes only the new token.
        seq.processed = seq.tokens.len();
        seq.tokens.push(next_token);
        seq.generated += 1;

        let is_stop = stop_ids.contains(&next_token);
        if is_stop || seq.generated >= seq.max_gen {
            seq.finished = true;
        }

        outcomes.push(StepOutcome::Stepped {
            token: next_token,
            is_stop,
            finished: seq.finished,
        });
    }

    outcomes
}

/// One batch-1 forward + sample. Takes the engine-owned cache for this sequence and runs the
/// model's `forward` against it, then framework-samples the next token id from the last position.
fn forward_one<D: BatchedDecoder, S: NextTokenSampler>(
    decoder: &mut D,
    cache: &mut D::Cache,
    batch: ForwardBatch,
    sampler: &mut S,
) -> InferenceResult<u32> {
    let in_rows = batch.input_tokens.dims()[0];
    let output = decoder.forward(batch, cache)?;

    // Forward contract: the decoder must return exactly one logits row per input row, with at least
    // one position. Enforce it as a per-sequence error (never a panic): on the worker thread a
    // panic here would unwind and brick the whole channel, and a wrong row count would otherwise
    // silently sample the wrong sequence. The caller's per-sequence error path retires just this
    // one.
    let [batch_size, seq_len, vocab_size] = output.logits.dims();
    if batch_size != in_rows || seq_len == 0 {
        return Err(InferenceError::BatchContractViolation(format!(
            "forward returned logits {:?} for {in_rows} input row(s); expected [{in_rows}, >=1, vocab]",
            [batch_size, seq_len, vocab_size]
        )));
    }

    let next_token_logits = output
        .logits
        .slice([0..batch_size, seq_len - 1..seq_len])
        .reshape([batch_size, vocab_size]);

    let token = sampler.sample_next(next_token_logits);
    let ids = token
        .into_data()
        .convert::<u32>()
        .into_vec::<u32>()
        .map_err(|_| {
            InferenceError::BatchContractViolation(
                "sampled token tensor did not convert to u32".to_string(),
            )
        })?;
    let id = *ids
        .first()
        .ok_or_else(|| InferenceError::BatchContractViolation("sampler produced no token".to_string()))?;
    Ok(id)
}

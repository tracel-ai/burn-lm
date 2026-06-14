//! Opt-in abstractions for request batching.
//!
//! These traits let a model expose a low-level decoder primitive that any engine can drive,
//! batched or not — single-request generation is just the batch=1 degenerate case.
//!
//! Two drivers consume this seam today, both through the shared decode core [`step_round`]: the
//! serving `BatchingChannel` worker, whose continuous loop admits queued jobs and streams tokens
//! per round, and the library's `Llama::generate_batch`, which drives a fixed set of prompts to
//! completion. Per-sequence state lives in [`ActiveSeq`]; the model only exposes
//! [`prefill`](BatchedDecoder::prefill)/[`decode`](BatchedDecoder::decode)/
//! [`release`](BatchedDecoder::release), tokenizer primitives and capacity. Token ids cross the
//! seam as host data (`&[u32]`, [`DecodeRow`]); the decoder builds its own tensors and owns its
//! caches internally, keyed by the slot numbers the engine hands it.

use burn::tensor::Tensor;

use crate::{
    errors::{InferenceError, InferenceResult},
    job::{GenerationParams, InferenceTask},
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
    /// admission — enforcement arrives together with real KV management.
    pub max_kv_tokens: usize,
}

/// One decoding sequence's next input: the token to feed a slot and the absolute position it
/// sits at. Plain host data — the decoder builds whatever tensors it needs from this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRow {
    /// The slot whose sequence this token advances.
    pub slot: usize,
    /// The token id to feed (the one sampled last round).
    pub token: u32,
    /// Absolute position of this token in its sequence (== tokens already in the slot).
    pub position: usize,
}

/// A reusable, batch-capable decoder primitive.
///
/// Model authors implement this once; both the single-request and continuous-batching engines
/// drive it the same way. The decoder owns its per-sequence caches internally, behind the slot
/// numbers the engine passes in: the engine assigns each admitted sequence a free slot in
/// `0..max_slots` (see [`BatchedInferenceServer::batch_capacity`]), prefills the prompt into it,
/// decodes one token at a time, and releases the slot when the sequence retires. Token ids cross
/// this seam as host data; the decoder builds its own input tensors on its own device.
pub trait BatchedDecoder {
    /// Run a whole prompt into one slot, returning the last position's logits, shaped
    /// `[1, vocab]`.
    ///
    /// `position` is the absolute position of `tokens[0]` in the slot's sequence (0 for a fresh
    /// prompt). A prefill that returns `Err` should leave the slot as if it had never been used.
    ///
    /// The engine does not rely on that rollback alone: after a safety review it also calls
    /// [`release`](Self::release) on a slot right before giving it to a new sequence, so a
    /// forgotten rollback cannot leak one prompt's state into another. What isolation ultimately
    /// rests on is `release` actually dropping the slot's state.
    fn prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
        position: usize,
    ) -> InferenceResult<Tensor<2>>;

    /// Advance every row by one token, returning logits shaped `[rows.len(), vocab]`, where row
    /// `i` belongs to `rows[i]`.
    ///
    /// `decode` returns exactly one logits row per input row — the framework checks this at every
    /// call site and retires a violating sequence with a `BatchContractViolation` error, never a
    /// panic. `decode` is never called with an empty `rows` slice.
    fn decode(&mut self, rows: &[DecodeRow]) -> InferenceResult<Tensor<2>>;

    /// Free the slot, dropping whatever cached state it held; the slot must keep zero residue,
    /// so the next sequence assigned to it starts fresh. `release` on an already-free slot is a
    /// no-op.
    ///
    /// Cross-sequence isolation ultimately rests on this method: the engine calls it on every
    /// retire path AND again right before reusing a slot (calling it twice is fine — it is a
    /// no-op on a free slot), so as long as `release` drops the slot's state, no other mistake
    /// can leak one sequence's state into another.
    fn release(&mut self, slot: usize);
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

    /// The decoder's static capacity maxima (see [`BatchCapacity`]). Not a live "free right now"
    /// figure: the engine owns the active set and derives free capacity as `max - in-use`. The
    /// slot numbers the engine passes to [`prefill`](BatchedDecoder::prefill)/
    /// [`decode`](BatchedDecoder::decode)/[`release`](BatchedDecoder::release) are `0..max_slots`.
    ///
    /// DECISION: capacity stays here on the server rather than becoming a `max_slots` method on
    /// [`BatchedDecoder`] — it sits next to the other operator-facing limits (`max_kv_tokens`,
    /// [`max_gen_tokens`](Self::max_gen_tokens)), it keeps the decoder trait pure data plane, and
    /// admission can ask for it without borrowing (and so lazily loading) the model.
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

    /// Build a fresh next-token sampler for one admitted sequence: the REQUEST's
    /// [`GenerationParams`] merged over the server's CURRENT sampling config.
    ///
    /// A config primitive (like [`max_gen_tokens`](Self::max_gen_tokens)): the engine asks for a
    /// new sampler per ADMITTED sequence and keeps it for that sequence's whole generation, so a
    /// seeded RNG advances across the sequence's tokens (matching the single-request path).
    /// Building it from the job's params instead of mutated shared config is what keeps two
    /// concurrent requests with different temperatures from clobbering each other. The default is
    /// the framework's deterministic argmax [`Sampler`], which ignores params — a model with no
    /// sampling config needs nothing extra; servers with sampling config (temperature, top-p,
    /// seed, …) override this to merge request over config.
    fn next_token_sampler(&self, params: &GenerationParams) -> Box<dyn NextTokenSampler + Send> {
        let _ = params;
        Box::new(Sampler::default())
    }
}

/// Per-sequence state advanced by [`step_round`].
///
/// This is the *generic* core of an in-flight sequence: its decoder [`slot`](Self::slot), token
/// buffer, position cursor, generated-token counter and `finished` flag. Driver-specific
/// bookkeeping (a serving job's emitter + completion sender, or the library's
/// `GenerationContext`) rides along in [`extra`](Self::extra), which `step_round` never touches —
/// each driver chooses what to attach and how to act on the per-round results.
pub struct ActiveSeq<Extra = ()> {
    /// The decoder slot this sequence occupies. The engine owns the free-slot list (slots are
    /// `0..max_slots`), assigns one at admission and releases it on every retire path; the
    /// decoder keeps the sequence's cache behind this number.
    pub slot: usize,
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

/// The one-prompt-per-round prefill budget, computed over the FULL active set at the start of a
/// round and threaded through every [`step_round`] call of that round.
///
/// While any sequence is mid-decode, at most ONE prompt may prefill per round, so a long prompt
/// cannot stall the decoders for more than one round; with no decoders to stall, prompts run
/// freely. The budget is computed once over the full active set and passed into the single
/// [`step_round`] call for the round, which honours it across its prefill pass.
pub struct PrefillBudget {
    /// Whether any live sequence's unprocessed tail is exactly one token this round (i.e. it is
    /// decoding, and a long prefill would stall it).
    any_decoding: bool,
    /// Set once a prompt has prefilled this round; with `any_decoding`, later prompts defer.
    prefilled: bool,
}

impl PrefillBudget {
    /// Compute the budget for one round from the FULL active set (every sequence the driver will
    /// step this round, however many `step_round` calls that takes).
    pub fn for_round<X>(active: &[ActiveSeq<X>]) -> Self {
        // "Decoding" means the sequence has been through the decoder before (`processed > 0`)
        // and now owes exactly one new token. Without the `processed > 0` clause a never-run
        // one-token prompt would count as decoding, making a fresh batch that mixes one-token
        // and longer prompts defer the longer ones — different from running them all up front.
        let any_decoding = active.iter().any(|seq| {
            !seq.finished
                && seq.generated < seq.max_gen
                && seq.processed > 0
                && seq.tokens.len() == seq.processed + 1
        });
        Self {
            any_decoding,
            prefilled: false,
        }
    }

    /// May one more prompt prefill this round? Claims the budget when it answers yes.
    fn admit_prefill(&mut self) -> bool {
        if self.prefilled && self.any_decoding {
            return false;
        }
        self.prefilled = true;
        true
    }
}

/// Advance every active sequence by exactly one token: route prompt work through
/// [`prefill`](BatchedDecoder::prefill) and single-token work through
/// [`decode`](BatchedDecoder::decode) → enforce the rows-in==rows-out contract → sample from the
/// returned last-position logits → synchronous stop check → advance the cursor / counters.
///
/// This is the single, generic decode core shared by both drivers (the serving
/// [`BatchingChannel`](crate::channels::batching::BatchingChannel) worker and the library's
/// `Llama::generate_batch`). It is deliberately *pure per-round*: no queue, no admission, no
/// completion signalling, no emitter ownership, no detokenization — those belong to the driver,
/// which inspects the returned [`StepOutcome`]s (one per input sequence) to stream/retire.
///
/// A sequence whose unprocessed tail is more than one token is prompt work and prefills, subject
/// to the caller-supplied [`PrefillBudget`] (at most one prompt per round while others decode); a
/// deferred prompt yields [`StepOutcome::Skipped`] and stays prompt work for the next round. Every
/// sequence with exactly one new token decodes through a SINGLE fused [`decode`](BatchedDecoder::decode)
/// call for the whole round (`[n, vocab]`), and each returned row is sampled by its own sampler via
/// the `sample_row` closure.
///
/// `sample_row(i, logits)` samples the next token for active sequence `i` from its last-position
/// logits (`[1, vocab]`). The closure — not a single shared sampler — is what lets the serving
/// worker give each in-flight request its own (possibly seeded) sampler while the library driver
/// passes one shared sampler that ignores the index. Keeping the samplers OUTSIDE `active` (the
/// driver owns them) is what lets this borrow `active` mutably and still sample per row.
///
/// Stop detection is **synchronous**: a sampled token that is in `stop_ids` finishes its sequence
/// in the same round it is produced, so no extra token is generated past a stop id. The `max_gen`
/// cap is likewise checked in-round.
pub fn step_round<D: BatchedDecoder, X>(
    decoder: &mut D,
    active: &mut [ActiveSeq<X>],
    stop_ids: &[u32],
    budget: &mut PrefillBudget,
    mut sample_row: impl FnMut(usize, Tensor<2>) -> InferenceResult<u32>,
) -> Vec<StepOutcome> {
    let mut outcomes: Vec<StepOutcome> = (0..active.len()).map(|_| StepOutcome::Skipped).collect();

    // CLASSIFY: retire the no-ops (already finished, `max_gen` reached, empty prompt) and split the
    // rest into prefill candidates and decode rows. `max_gen` is checked before prefill so a
    // zero-token request is a true no-op instead of prefilling and producing one token.
    let mut prefills: Vec<usize> = Vec::new();
    let mut decode_rows: Vec<(usize, DecodeRow)> = Vec::new();
    for (i, seq) in active.iter_mut().enumerate() {
        if seq.finished {
            continue;
        }
        if seq.generated >= seq.max_gen || seq.processed >= seq.tokens.len() {
            seq.finished = true;
            continue;
        }
        if seq.tokens.len() - seq.processed > 1 {
            prefills.push(i);
        } else {
            decode_rows.push((
                i,
                DecodeRow {
                    slot: seq.slot,
                    token: seq.tokens[seq.processed],
                    position: seq.processed,
                },
            ));
        }
    }

    // PREFILL pass: at most one prompt per round while anything decodes (budget). Prompts are
    // ragged, so each is its own call; a deferred prompt stays Skipped and remains prompt work.
    for i in prefills {
        if !budget.admit_prefill() {
            continue;
        }
        let position = active[i].processed;
        let sampled = decoder
            .prefill(active[i].slot, &active[i].tokens[position..], position)
            .and_then(|logits| expect_rows(logits, 1))
            .and_then(|logits| sample_row(i, logits));
        outcomes[i] = advance_or_fail(&mut active[i], sampled, stop_ids);
    }

    // DECODE pass: ONE fused call advances every decode-ready row, then each row is sampled by its
    // own sampler. A fused decode is all-or-nothing — it cannot fail for one row and succeed for
    // another — so a decode error retires every decode row this round (the `expect_rows` contract
    // still guards against a silently misaligned row count).
    if !decode_rows.is_empty() {
        let rows: Vec<DecodeRow> = decode_rows.iter().map(|(_, row)| *row).collect();
        match decoder
            .decode(&rows)
            .and_then(|logits| expect_rows(logits, rows.len()))
        {
            Ok(logits) => {
                let [_, vocab] = logits.dims();
                // Per-row sampling: each row goes to ITS OWN sampler, so concurrent requests can use
                // different params. This samples one row at a time, and `sample_token` reads each
                // chosen id back to the host — so a round of N decoders costs N device→host syncs.
                //
                // FUTURE (D4 — batched sampler): collapse this into one pass over the whole
                // `[n, vocab]` that applies per-row params (a temperature vector, a fused top-p) and
                // returns all `n` ids in a single readback. That removes the per-row sync, the main
                // sampling cost at higher batch. It needs a richer seam than `sample_row(i, logits)`
                // (the per-request params must cross as batched data). NOTE: `logits.clone()` below
                // is a cheap handle/refcount bump that shares the buffer — NOT a `[n, vocab]` copy —
                // so the clone is not the cost; the per-row host readback is.
                for (row_index, (seq_index, _)) in decode_rows.iter().enumerate() {
                    let row_logits = logits.clone().slice([row_index..row_index + 1, 0..vocab]);
                    let sampled = sample_row(*seq_index, row_logits);
                    outcomes[*seq_index] =
                        advance_or_fail(&mut active[*seq_index], sampled, stop_ids);
                }
            }
            Err(err) => {
                // Route every decode row through the shared fail path so the retire semantics live
                // in one place (`advance_or_fail` on `Err` marks the sequence finished and yields
                // `Failed`).
                for (seq_index, _) in &decode_rows {
                    outcomes[*seq_index] =
                        advance_or_fail(&mut active[*seq_index], Err(err.clone()), stop_ids);
                }
            }
        }
    }

    outcomes
}

/// Apply a sampled-token result to one sequence: on success advance its cursor/counters and run the
/// synchronous stop / `max_gen` check; on error mark it finished. Returns the per-sequence outcome.
/// Shared by the prefill and decode passes so both advance identically.
fn advance_or_fail<X>(
    seq: &mut ActiveSeq<X>,
    sampled: InferenceResult<u32>,
    stop_ids: &[u32],
) -> StepOutcome {
    let next_token = match sampled {
        Ok(token) => token,
        Err(err) => {
            seq.finished = true;
            return StepOutcome::Failed(err);
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

    StepOutcome::Stepped {
        token: next_token,
        is_stop,
        finished: seq.finished,
    }
}

/// The framework's rows-in==rows-out check, applied at every
/// [`prefill`](BatchedDecoder::prefill)/[`decode`](BatchedDecoder::decode) call site: the decoder
/// must return exactly one logits row per input row. Enforced as a per-sequence error (never a
/// panic): on the worker thread a panic here would unwind and brick the whole channel, and a
/// wrong row count would otherwise silently sample the wrong sequence. The caller's per-sequence
/// error path retires just this one.
fn expect_rows(logits: Tensor<2>, in_rows: usize) -> InferenceResult<Tensor<2>> {
    let [rows, vocab] = logits.dims();
    if rows != in_rows {
        return Err(InferenceError::BatchContractViolation(format!(
            "decoder returned logits {:?} for {in_rows} input row(s); expected [{in_rows}, vocab]",
            [rows, vocab]
        )));
    }
    Ok(logits)
}

/// Framework-samples the next token id from a single sequence's last-position logits
/// (`[1, vocab]`). Takes the sampler as a trait object so a driver can pass either one shared
/// sampler (the library path) or a per-sequence boxed sampler (the serving worker) from inside its
/// [`step_round`] `sample_row` closure.
pub fn sample_token(logits: Tensor<2>, sampler: &mut dyn NextTokenSampler) -> InferenceResult<u32> {
    let token = sampler.sample_next(logits);
    let ids = token
        .into_data()
        .convert::<u32>()
        .into_vec::<u32>()
        .map_err(|_| {
            InferenceError::BatchContractViolation(
                "sampled token tensor did not convert to u32".to_string(),
            )
        })?;
    let id = *ids.first().ok_or_else(|| {
        InferenceError::BatchContractViolation("sampler produced no token".to_string())
    })?;
    Ok(id)
}

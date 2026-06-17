//! The shared decode core: the model-agnostic machinery that turns a batch of in-flight sequences
//! into one token each per round. This is where the batching story begins. This module defines the
//! decoder seam — a small trait a model implements in its own crate (`burn-lm-llama` implements it
//! for the Llama decoder) — and any engine can then drive that decoder, batched or not, since
//! single-request generation is just the degenerate batch-of-one case.
//!
//! Two drivers consume this seam today, both through `step_round`. The serving `BatchingChannel`
//! worker runs a continuous loop that admits queued jobs and streams tokens out per round; the
//! library's `Llama::generate_batch` drives a fixed set of prompts to completion. They share this
//! core precisely because it owns nothing driver-specific: no queue, no streaming, no completion
//! signalling. It only advances tokens and reports back what happened to each sequence.
//!
//! The split that makes that work runs along the model boundary. Per-sequence decode state lives in
//! `ActiveSeq`, owned by the engine. The model exposes only `prefill`, `decode` and `release`, plus
//! tokenizer and capacity primitives. Token ids cross the boundary as plain host data (`&[u32]`,
//! `DecodeRow`); the decoder builds its own tensors and owns its per-sequence caches internally,
//! keyed by the slot numbers the engine hands it.

use burn::tensor::Tensor;

use crate::{
    errors::{InferenceError, InferenceResult},
    job::InferenceTask,
    sampler::{Argmax, Sampler},
    server::InferenceServer,
};

/// The decoder's static capacity limits, declared by the model.
///
/// These are the maxima the model can handle, a function of the model and the hardware. They are
/// not a live "free right now" figure: the engine owns the active set, so it derives the actual
/// free capacity itself as `max - in-use`. Today these are fixed once the model loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchCapacity {
    /// The maximum number of sequences the decoder can run at once. The engine admits new work
    /// while `active.len() < max_slots`, so the free slots are `max_slots - active.len()`.
    pub max_slots: usize,
}

/// One decoding sequence's next input for a round: the token to feed into a slot. This is plain
/// host data, so the decoder builds whatever tensors it needs from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRow {
    /// The slot whose sequence this token advances.
    pub slot: usize,
    /// The token id to feed, which is the one this sequence sampled last round.
    pub token: u32,
}

/// A reusable, batch-capable decoder primitive.
///
/// This is the whole model side of the batching seam. A model author implements it once, and both
/// the single-request and continuous-batching engines drive it the same way. The decoder owns its
/// per-sequence caches internally, behind the slot numbers the engine passes in. The engine assigns
/// each admitted sequence a free slot in `0..max_slots`, prefills the prompt into it, decodes one
/// token at a time, and releases the slot when the sequence retires. Token ids cross as host data;
/// the decoder builds its own input tensors on its own device.
pub trait BatchedDecoder {
    /// Run a whole prompt into one slot, returning the last position's logits, shaped `[1, vocab]`.
    ///
    /// `position` is the absolute position of `tokens[0]` in the slot's sequence, which is 0 for a
    /// fresh prompt. A prefill that returns `Err` must leave the slot as if it had never been used.
    ///
    /// The engine does not rely on that rollback alone. It also calls `release` on a slot right
    /// before handing it to a new sequence, so a forgotten rollback cannot leak one prompt's state
    /// into another. Isolation ultimately depends on `release` actually dropping the slot's state.
    fn prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
        position: usize,
    ) -> InferenceResult<Tensor<2>>;

    /// Advance every row by one token, returning logits shaped `[rows.len(), vocab]`, where row
    /// `i` belongs to `rows[i]`.
    ///
    /// The decoder must return exactly one logits row per input row. The engine checks this at
    /// every call site and retires a violating sequence with a `BatchContractViolation` error
    /// rather than panicking. `decode` is never called with an empty `rows` slice.
    fn decode(&mut self, rows: &[DecodeRow]) -> InferenceResult<Tensor<2>>;

    /// Free the slot, dropping whatever cached state it held. The slot must keep zero residue, so
    /// the next sequence assigned to it starts fresh. Releasing an already-free slot is a no-op.
    ///
    /// Cross-sequence isolation ultimately depends on this method. The engine calls it on every
    /// retire path, and again right before reusing a slot — calling it twice is fine, since it is a
    /// no-op on a free slot. So as long as `release` drops the slot's state, no other mistake can
    /// leak one sequence's state into another.
    fn release(&mut self, slot: usize);
}

/// A server that can expose a `BatchedDecoder` for batched serving.
///
/// Implementing this trait is what makes a model eligible for the batching channel. The existing
/// `InferenceServer` surface (lifecycle, config, single-job `run_job`) is unchanged; this trait
/// adds the decoder and the few tokenizer and capacity primitives the engine needs around it.
pub trait BatchedInferenceServer: InferenceServer {
    /// The decoder primitive this server drives.
    type Decoder: BatchedDecoder;

    /// Mutably borrow the loaded decoder, loading the model first if needed.
    ///
    /// The continuous loop holds the server with exclusive access, since the worker thread owns it,
    /// so a plain `&mut` borrow is enough and no lock or callback is required.
    fn decoder(&mut self) -> InferenceResult<&mut Self::Decoder>;

    /// The decoder's static capacity maxima. This is not a live "free right now" figure: the engine
    /// owns the active set and derives free capacity itself as `max - in-use`. The slot numbers the
    /// engine passes to `prefill`/`decode`/`release` are `0..max_slots`.
    ///
    /// Capacity lives here on the server rather than as a `max_slots` method on `BatchedDecoder`.
    /// That keeps it next to the other operator-facing limits like `max_gen_tokens`, keeps the
    /// decoder trait a pure data plane, and lets admission ask for it without borrowing — and so
    /// lazily loading — the model.
    fn batch_capacity(&self) -> BatchCapacity;

    /// Tokenize a submitted task into the token ids the decoder consumes.
    ///
    /// This is a thin wrapper over the model's own tokenizer. It is a primitive the continuous loop
    /// calls during admission, not a loop the model owns.
    fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>>;

    /// Detokenize generated token ids back to text.
    fn detokenize(&self, tokens: &[u32]) -> String;

    /// Detokenize generated token ids to raw bytes, which are not guaranteed to be valid UTF-8 on
    /// their own. This is the primitive the loop actually streams through. Byte-level BPE tokenizers
    /// routinely split a multi-byte character across tokens, so a per-token text decode can fail or
    /// emit U+FFFD mid-character. The loop instead feeds these bytes through a per-sequence
    /// `Utf8Buffer` and emits only complete text.
    ///
    /// The default round-trips through `detokenize`, so a model whose per-token decode is already
    /// total needs nothing extra. A model with a byte-level tokenizer (e.g. Tiktoken) overrides
    /// this with its tokenizer's infallible byte decode.
    fn detokenize_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        self.detokenize(tokens).into_bytes()
    }

    /// Token ids that, when generated, end a sequence (EOS, EOT, EOM, and so on).
    fn stop_ids(&self) -> Vec<u32>;

    /// Maximum number of tokens to generate per sequence before forcibly retiring it.
    ///
    /// This is a config primitive alongside `batch_capacity`: it bounds the loop without exposing
    /// scheduling vocabulary. The loop stops a sequence at the first stop id or once it has
    /// generated this many tokens, whichever comes first.
    fn max_gen_tokens(&self) -> usize;

    /// The sampler this server uses to turn logits into token ids, built from the server's own
    /// sampling config.
    ///
    /// Sampling is config-driven: one sampler serves every in-flight sequence and carries only the
    /// config (temperature, top-p), with no per-sequence state — any randomness a strategy needs is
    /// drawn from the tensor backend's RNG. The worker grabs this as an owned `Box<dyn Sampler>`
    /// before it borrows the decoder, so the owned box — which does not borrow the server — cannot
    /// collide with the `&mut` decoder borrow. The default is the framework's deterministic argmax,
    /// so a model with no sampling config needs nothing extra; a server with sampling config
    /// overrides this with its own sampler.
    fn sampler(&self) -> Box<dyn Sampler> {
        Box::new(Argmax)
    }
}

/// The per-sequence state that `step_round` advances.
///
/// This is the generic core of one in-flight sequence: its decoder slot, token buffer, position
/// cursor, generated-token counter, and `finished` flag. Anything driver-specific — a serving
/// job's emitter and completion sender, or the library's generation context — rides along in
/// `extra`, which `step_round` never touches. Each driver chooses what to attach there and how to
/// act on the per-round results.
pub struct ActiveSeq<Extra = ()> {
    /// The decoder slot this sequence occupies. The engine owns the free-slot list (slots are
    /// `0..max_slots`), assigns one at admission, and releases it on every retire path; the decoder
    /// keeps the sequence's cache behind this number.
    pub slot: usize,
    /// The full token buffer (prompt plus generated). The next forward consumes the unprocessed
    /// tail.
    pub tokens: Vec<u32>,
    /// The number of tokens already pushed through the decoder, which equals the absolute position
    /// of the next input.
    pub processed: usize,
    /// The count of generated tokens, excluding the prompt.
    pub generated: usize,
    /// The hard cap on generated tokens for this sequence.
    pub max_gen: usize,
    /// Set once a stop id or the cap is hit; the next sweep retires the sequence.
    pub finished: bool,
    /// The driver-owned payload (emitter and completion, generation context). Opaque to
    /// `step_round`.
    pub extra: Extra,
}

/// The per-sequence outcome of one `step_round` sweep, aligned position-for-position with the slice
/// passed in. A sequence that was already finished (or empty) at the start of the round yields
/// `Skipped`; an advanced sequence yields `Stepped`, or `Failed` if its forward violated the batch
/// contract.
pub enum StepOutcome {
    /// The sequence was not advanced this round, because it was already finished or had no
    /// unprocessed tokens.
    Skipped,
    /// The sequence advanced by one token.
    Stepped {
        /// The token id sampled this round, already appended to the sequence's `tokens`.
        token: u32,
        /// Whether this token is a stop id. A stop token ends the sequence and by convention is not
        /// streamed as text; drivers use this to decide whether to emit it.
        is_stop: bool,
        /// Whether the sequence is now finished, because a stop id was reached or the `max_gen` cap
        /// was hit. When set, the sequence's `finished` flag is also set, so the next round skips
        /// it.
        finished: bool,
    },
    /// The sequence's forward failed; the driver should retire it with this error.
    Failed(InferenceError),
}

/// The one-prompt-per-round prefill budget, computed over the full active set at the start of a
/// round and threaded through the round's `step_round` call.
///
/// A prompt prefill is a large forward pass; a decode is a single token. So while any sequence is
/// mid-decode, at most one prompt may prefill per round, which keeps a long prompt from stalling
/// the in-flight decoders for more than that one round. With no decoders to stall, prompts run
/// freely. The budget is computed once and honoured across the prefill pass.
pub struct PrefillBudget {
    /// Whether any live sequence is mid-decode this round — it has been through the decoder before
    /// and now owes exactly one new token — and so would be stalled by a long prefill. (A never-run
    /// one-token prompt does not count; see `for_round` for why that distinction matters.)
    any_decoding: bool,
    /// Set once a prompt has prefilled this round; combined with `any_decoding`, later prompts
    /// defer.
    prefilled: bool,
}

impl PrefillBudget {
    /// Compute the budget for one round from the full active set — every sequence the driver will
    /// step this round.
    pub fn for_round<X>(active: &[ActiveSeq<X>]) -> Self {
        // A sequence counts as decoding when it has been through the decoder before (`processed >
        // 0`) and now owes exactly one new token. The `processed > 0` clause is what correctness
        // depends on here: without it a never-run one-token prompt would count as decoding, so a
        // fresh batch mixing one-token and longer prompts would defer the longer ones instead of
        // running them all up front.
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

    /// Whether one more prompt may prefill this round. Claims the budget when it answers yes.
    fn admit_prefill(&mut self) -> bool {
        if self.prefilled && self.any_decoding {
            return false;
        }
        self.prefilled = true;
        true
    }
}

/// Advance every active sequence by exactly one token. This is the heart of the batching engine.
/// For each round it routes prompt work through `prefill` and single-token work through `decode`,
/// enforces the rows-in-equals-rows-out contract, samples from the returned last-position logits,
/// runs a synchronous stop check, and advances each sequence's cursor and counters.
///
/// This is the single generic decode core both drivers share: the serving `BatchingChannel` worker
/// and the library's `Llama::generate_batch`. It is deliberately pure per-round. It has no queue,
/// no admission, no completion signalling, no emitter ownership, and no detokenization. Those all
/// belong to the driver, which inspects the returned `StepOutcome`s — one per input sequence — to
/// decide what to stream and what to retire.
///
/// A sequence whose unprocessed tail is more than one token is prompt work and prefills, subject to
/// the caller-supplied `PrefillBudget` (at most one prompt per round while others decode); a
/// deferred prompt yields `Skipped` and stays prompt work for the next round. Every sequence with
/// exactly one new token decodes through a single fused `decode` call for the whole round
/// (`[n, vocab]`), and the whole round's rows are sampled in one batched `sample` call.
///
/// `sample(logits)` samples the next token for a batch of rows at once: `logits` is `[rows, vocab]`
/// and the returned `Vec` has one token id per row, in row order. A prefill samples its single row;
/// the whole decode pass samples all its rows in one call. The sampler carries no per-sequence
/// state — any randomness comes from the backend RNG — so nothing sequence-specific needs to cross
/// this seam, which is what lets `step_round` borrow `active` mutably and still sample.
///
/// Stop detection is synchronous: a sampled token that is in `stop_ids` finishes its sequence in
/// the same round it is produced, so no extra token is generated past a stop id. The `max_gen` cap
/// is checked in the same round.
pub fn step_round<D: BatchedDecoder, X>(
    decoder: &mut D,
    active: &mut [ActiveSeq<X>],
    stop_ids: &[u32],
    budget: &mut PrefillBudget,
    mut sample: impl FnMut(Tensor<2>) -> InferenceResult<Vec<u32>>,
) -> Vec<StepOutcome> {
    let mut outcomes: Vec<StepOutcome> = (0..active.len()).map(|_| StepOutcome::Skipped).collect();

    // Classify each sequence: retire the no-ops (already finished, `max_gen` reached, empty prompt)
    // and split the rest into prefill candidates and decode rows. The `max_gen` check comes before
    // prefill so a zero-token request is a true no-op instead of prefilling and producing one token.
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
                },
            ));
        }
    }

    // Announce only when the fused-decode width CHANGES to more than one row — the signal that we're
    // parallelizing sequences — instead of a line every round. The width is a coarse process-global
    // gauge (a function-local static), enough for an "are we actually batching?" sanity check.
    static LAST_DECODE_WIDTH: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    let decode_width = decode_rows.len();
    let prev_width = LAST_DECODE_WIDTH.swap(decode_width, std::sync::atomic::Ordering::Relaxed);
    if decode_width > 1 && decode_width != prev_width {
        tracing::info!(decode_width, "fused decode parallelizing across sequences");
    }

    // Prefill pass: the budget allows at most one prompt per round while anything decodes. Prompts
    // have different lengths, so each is its own call; a deferred prompt stays Skipped and remains
    // prompt work for a later round.
    for i in prefills {
        if !budget.admit_prefill() {
            continue;
        }
        let position = active[i].processed;
        // A prefill produces a single `[1, vocab]` row, so `sample(logits)` returns a one-element
        // `Vec`, and we take its single id.
        let sampled = decoder
            .prefill(active[i].slot, &active[i].tokens[position..], position)
            .and_then(|logits| expect_rows(logits, 1))
            .and_then(|logits| sample(logits))
            .and_then(|ids| {
                ids.into_iter().next().ok_or_else(|| {
                    InferenceError::BatchContractViolation(
                        "sampler produced no token for the prefill row".to_string(),
                    )
                })
            });
        outcomes[i] = advance_or_fail(&mut active[i], sampled, stop_ids);
    }

    // Decode pass: one fused call advances every decode-ready row, then the whole round's rows are
    // sampled in one batched `sample` call. A fused decode is all-or-nothing — it cannot fail for one
    // row and succeed for another — so a decode error retires every decode row this round. The
    // `expect_rows` contract still guards against a silently misaligned row count.
    if !decode_rows.is_empty() {
        let rows: Vec<DecodeRow> = decode_rows.iter().map(|(_, row)| *row).collect();
        let sampled = decoder
            .decode(&rows)
            .and_then(|logits| expect_rows(logits, rows.len()))
            .and_then(|logits| sample(logits));
        match sampled {
            Ok(ids) if ids.len() == decode_rows.len() => {
                // Fan each sampled id back to its sequence, in row order, through the shared advance
                // path so prefill and decode advance identically.
                for ((seq_index, _), id) in decode_rows.iter().zip(ids) {
                    outcomes[*seq_index] =
                        advance_or_fail(&mut active[*seq_index], Ok(id), stop_ids);
                }
            }
            Ok(_) => {
                // The sampler returned the wrong number of ids for the round's rows — a batch
                // contract violation. Retire every decode row with it, same all-or-nothing semantics
                // as a decode error.
                let err = InferenceError::BatchContractViolation(format!(
                    "sampler returned a different number of ids than the {} decode row(s) sampled",
                    decode_rows.len()
                ));
                for (seq_index, _) in &decode_rows {
                    outcomes[*seq_index] =
                        advance_or_fail(&mut active[*seq_index], Err(err.clone()), stop_ids);
                }
            }
            Err(err) => {
                // Route every decode row through the shared fail path so the retire semantics live
                // in one place: `advance_or_fail` on an `Err` marks the sequence finished and yields
                // `Failed`.
                for (seq_index, _) in &decode_rows {
                    outcomes[*seq_index] =
                        advance_or_fail(&mut active[*seq_index], Err(err.clone()), stop_ids);
                }
            }
        }
    }

    outcomes
}

/// Apply a sampled-token result to one sequence. On success it advances the cursor and counters and
/// runs the synchronous stop and `max_gen` check; on error it marks the sequence finished. Returns
/// the per-sequence outcome. The prefill and decode passes share this so both advance identically.
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

    // Everything up to here is now processed, so the next step consumes only the new token.
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

/// The rows-in-equals-rows-out check, applied at every `prefill` and `decode` call site: the
/// decoder must return exactly one logits row per input row. It is enforced as a per-sequence error
/// rather than a panic, because on the worker thread a panic here would unwind and brick the whole
/// channel, and a wrong row count would otherwise silently sample the wrong sequence. The caller's
/// per-sequence error path retires just the offending one.
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

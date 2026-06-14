use super::super::*;
use crate::{
    batching::{BatchCapacity, BatchedDecoder, DecodeRow},
    errors::InferenceError,
    job::{CancelSignal, GenerationParams, InferenceJob, InferenceTask},
    sampler::NextTokenSampler,
    server::{InferenceServer, ServerConfigParsing},
    InferenceServerConfig, Stats, INFERENCE_DEVICE,
};
use burn::tensor::{Int, Tensor, TensorData};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Debug, Default, Clone, serde::Deserialize, clap::Parser)]
pub(super) struct FakeConfig {}
impl InferenceServerConfig for FakeConfig {}

/// Shared, synchronous record of emission order. The fake decoder appends to it on the worker
/// thread at the moment a (non-stop) token is produced, so the recorded order is exactly the
/// engine's generation interleaving — independent of how the async emitter threads later drain.
pub(super) type OrderLog = Arc<Mutex<Vec<usize>>>;

/// A trivial decoder. It keeps a per-slot step counter (its only "cache"). It echoes the
/// sequence's identity token (the prompt's first token, which it then re-receives every decode
/// step) for a few steps, recording the emission order, then emits the stop id (0).
#[derive(Debug, Clone)]
pub(super) struct FakeDecoder {
    pub(super) log: OrderLog,
    /// How many tokens each sequence emits before stopping.
    pub(super) emit: usize,
    /// Extra logits rows beyond the input rows — simulates a decoder that violates the
    /// rows-in==rows-out contract. 0 = well-behaved.
    pub(super) extra_rows: usize,
    /// Per-step sleep — simulates a slow model so a test can observe a job in flight.
    pub(super) step_delay_ms: u64,
    /// When set, the decoder PANICS at this per-sequence step — simulates a model bug that
    /// unwinds the worker iteration (the failure-ladder rung above per-sequence errors).
    pub(super) panic_at_step: Option<usize>,
    /// How many upcoming `prefill` calls fail (e.g. a prompt past the context window). Per the
    /// [`BatchedDecoder::prefill`] contract the failing call leaves the slot untouched.
    pub(super) fail_prefills: usize,
    /// How many upcoming `decode` calls fail — a fused decode is all-or-nothing, so this lets a test
    /// prove a single decode error retires every row of the round.
    pub(super) fail_decodes: usize,
    /// Per-slot step counters: the decoder-owned stand-in for a real per-slot KV cache.
    pub(super) steps: HashMap<usize, usize>,
    /// Row count of each `decode` call, in order. Lets a test prove the round FUSES — one call with
    /// N rows, not N calls with one. (Only read by direct `step_round` tests that own the decoder;
    /// the worker moves it onto its own thread.)
    pub(super) decode_calls: Vec<usize>,
}

pub(super) const VOCAB: usize = 64;

impl FakeDecoder {
    pub(super) fn new(log: OrderLog, emit: usize) -> Self {
        Self {
            log,
            emit,
            extra_rows: 0,
            step_delay_ms: 0,
            panic_at_step: None,
            fail_prefills: 0,
            fail_decodes: 0,
            steps: HashMap::new(),
            decode_calls: Vec::new(),
        }
    }

    /// One step for `slot` with `last_token` as the identity: bump the slot's counter, maybe
    /// sleep/panic, log the emission, and pick the next token (the stop id once `emit` is spent).
    fn step_slot(&mut self, slot: usize, last_token: u32) -> usize {
        if self.step_delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.step_delay_ms));
        }

        let counter = self.steps.entry(slot).or_insert(0);
        let step = *counter;
        *counter += 1;

        if self.panic_at_step == Some(step) {
            panic!("scripted decoder panic at step {step}");
        }

        let identity = last_token as usize;
        if step < self.emit {
            // Record which sequence emitted, synchronously, in true generation order.
            self.log.lock().unwrap().push(identity % 2);
            identity % VOCAB
        } else {
            0 // stop id
        }
    }

    /// One-hot logits, one row per produced token plus `extra_rows` zero rows (the contract
    /// violation a misbehaving decoder would produce).
    fn logits(&self, tokens: &[usize]) -> Tensor<2> {
        let rows = tokens.len() + self.extra_rows;
        let mut data = vec![0.0f32; rows * VOCAB];
        for (row, &token) in tokens.iter().enumerate() {
            data[row * VOCAB + token] = 1.0;
        }
        Tensor::<2>::from_data(TensorData::new(data, [rows, VOCAB]), &*INFERENCE_DEVICE)
    }
}

impl BatchedDecoder for FakeDecoder {
    fn prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
        _position: usize,
    ) -> InferenceResult<Tensor<2>> {
        if self.fail_prefills > 0 {
            self.fail_prefills -= 1;
            // Contract: a failing prefill leaves the slot as if it had never been used — so the
            // slot's step counter is NOT bumped.
            return Err(InferenceError::ContextLengthExceeded(tokens.len(), 0));
        }
        let token = self.step_slot(slot, *tokens.last().unwrap());
        Ok(self.logits(&[token]))
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> InferenceResult<Tensor<2>> {
        self.decode_calls.push(rows.len());
        if self.fail_decodes > 0 {
            self.fail_decodes -= 1;
            return Err(InferenceError::ContextLengthExceeded(rows.len(), 0));
        }
        let tokens: Vec<usize> = rows
            .iter()
            .map(|row| self.step_slot(row.slot, row.token))
            .collect();
        Ok(self.logits(&tokens))
    }

    fn release(&mut self, slot: usize) {
        self.steps.remove(&slot);
    }
}

/// A sampler that ignores the logits and always returns the same token id — observably NOT
/// argmax, so a test can prove the worker uses the server-configured sampler.
pub(super) struct FixedSampler(pub(super) u32);

impl NextTokenSampler for FixedSampler {
    fn sample_next(&mut self, _logits: Tensor<2>) -> Tensor<2, Int> {
        Tensor::from_data(
            TensorData::new(vec![self.0 as i32], [1, 1]),
            &*INFERENCE_DEVICE,
        )
    }
}

#[derive(Debug, Clone)]
pub(super) struct FakeServer {
    pub(super) loaded: bool,
    pub(super) slots: usize,
    pub(super) decoder: FakeDecoder,
    /// Counts `batch_capacity` calls — lets the `max_slots == 0` test detect a busy-spin.
    pub(super) capacity_calls: Arc<AtomicUsize>,
    /// When set, `next_token_sampler` returns a [`FixedSampler`] for this token instead of the
    /// default argmax — stands in for a server with non-greedy sampling config. A job whose
    /// params carry a temperature overrides this with `temperature as u32` (the merge-over-
    /// config behavior a real server implements), so tests can observe per-request samplers.
    pub(super) fixed_token: Option<u32>,
    /// How many tokens `tokenize` produces per prompt (the identity token, repeated). The default
    /// 1 makes a sequence's first step decode work; tests that must exercise the PREFILL path
    /// (multi-token prompt work) raise it via [`with_prompt_tokens`](Self::with_prompt_tokens).
    pub(super) prompt_tokens: usize,
}

impl Default for FakeServer {
    fn default() -> Self {
        Self::new(1, Arc::new(Mutex::new(Vec::new())))
    }
}

impl FakeServer {
    pub(super) fn new(slots: usize, log: OrderLog) -> Self {
        Self {
            loaded: false,
            slots,
            decoder: FakeDecoder::new(log, 4),
            capacity_calls: Arc::new(AtomicUsize::new(0)),
            fixed_token: None,
            prompt_tokens: 1,
        }
    }

    /// A server whose decoder returns 2 logits rows for a 1-row input — violates the forward
    /// rows-in==rows-out contract.
    pub(super) fn new_bad(slots: usize, log: OrderLog) -> Self {
        let mut server = Self::new(slots, log);
        server.decoder.extra_rows = 1;
        server
    }

    /// A server whose decoder emits many tokens, each after a small sleep — a long-running job
    /// a test can interrogate (e.g. unload) while it is demonstrably still in flight.
    pub(super) fn new_slow(slots: usize, log: OrderLog) -> Self {
        let mut server = Self::new(slots, log);
        server.decoder.emit = 1000; // effectively capped by `max_gen_tokens` (16)
        server.decoder.step_delay_ms = 20;
        server
    }

    /// A server whose decoder PANICS at the given per-sequence step, after a small per-step
    /// delay (the delay makes "a second job is queued behind the panicking one" a sure thing
    /// rather than a race) — the failure-ladder rung ABOVE per-sequence errors: the panic
    /// unwinds the whole worker iteration.
    pub(super) fn new_panicky(slots: usize, log: OrderLog, panic_at_step: usize) -> Self {
        let mut server = Self::new(slots, log);
        server.decoder.step_delay_ms = 20;
        server.decoder.panic_at_step = Some(panic_at_step);
        server
    }

    /// A server whose `next_token_sampler` always picks `token`, regardless of logits —
    /// observably different from the default argmax (which would echo the identity token).
    pub(super) fn with_fixed_sampler(mut self, token: u32) -> Self {
        self.fixed_token = Some(token);
        self
    }

    /// A server reporting `slots` free slots that records every `batch_capacity` call into
    /// `calls`, so a test can tell whether the worker busy-spins when nothing is admittable.
    pub(super) fn with_capacity_probe(slots: usize, calls: Arc<AtomicUsize>) -> Self {
        let mut server = Self::new(slots, Arc::new(Mutex::new(Vec::new())));
        server.capacity_calls = calls;
        server
    }

    /// A server whose prompts tokenize to `n` tokens, so a sequence's first step is genuine
    /// PREFILL work (the default single-token prompt goes straight to `decode`).
    pub(super) fn with_prompt_tokens(mut self, n: usize) -> Self {
        self.prompt_tokens = n;
        self
    }

    /// A server whose decoder fails its next `n` `prefill` calls (leaving the slot untouched,
    /// per the `prefill` contract) and behaves normally afterwards.
    pub(super) fn with_failing_prefills(mut self, n: usize) -> Self {
        self.decoder.fail_prefills = n;
        self
    }
}

impl ServerConfigParsing for FakeServer {
    type Config = FakeConfig;
    fn parse_cli_config(&mut self, _args: &clap::ArgMatches) {}
    fn parse_json_config(&mut self, _json: &str) {}
}

impl InferenceServer for FakeServer {
    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        self.loaded = true;
        Ok(None)
    }
    fn is_loaded(&mut self) -> bool {
        self.loaded
    }
    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.loaded = false;
        Ok(None)
    }
    fn run_job(&mut self, _job: InferenceJob) -> InferenceResult<Stats> {
        Ok(Stats::new())
    }
    fn clear_state(&mut self) -> InferenceResult<()> {
        Ok(())
    }
}

impl BatchedInferenceServer for FakeServer {
    type Decoder = FakeDecoder;

    fn decoder(&mut self) -> InferenceResult<&mut FakeDecoder> {
        // Mirror the real servers (`Llama3BaseServer::decoder`): borrowing the decoder
        // lazy-loads the model, so an idle worker touching `decoder()` is observable as a
        // spurious load (e.g. an immediate reload after `Unload`).
        self.loaded = true;
        Ok(&mut self.decoder)
    }

    fn batch_capacity(&self) -> BatchCapacity {
        self.capacity_calls.fetch_add(1, Ordering::Relaxed);
        BatchCapacity {
            max_slots: self.slots,
        }
    }

    fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>> {
        // Map each prompt to a distinct identity token so sequences are distinguishable.
        let id = match task {
            InferenceTask::Prompt(p) if p == "a" => 10u32,
            InferenceTask::Prompt(p) if p == "b" => 11u32,
            _ => 12u32,
        };
        Ok(vec![id; self.prompt_tokens.max(1)])
    }

    fn detokenize(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|t| t.to_string()).collect()
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![0]
    }

    fn max_gen_tokens(&self) -> usize {
        16
    }

    fn next_token_sampler(&self, params: &GenerationParams) -> Box<dyn NextTokenSampler + Send> {
        // Request params merged over server config, like the real servers: a per-job
        // temperature wins over the server-level `fixed_token`.
        match params.temperature.map(|t| t as u32).or(self.fixed_token) {
            Some(token) => Box::new(FixedSampler(token)),
            None => Box::new(crate::sampler::Sampler::default()),
        }
    }
}

/// A no-op listener (text is recorded synchronously in the decoder, not here).
pub(super) struct NullListener;
impl crate::job::InferenceJobListener for NullListener {
    type CompletedItem = ();
    fn on_text(&mut self, _text: String) {}
    fn on_finished(self) {}
}

/// A listener that panics on the first emitted token — stands in for a client whose stream
/// broke (a dropped SSE connection makes `WriteListener::on_text`'s write `.unwrap()` panic),
/// which kills the listener thread out from under the worker.
pub(super) struct PanicOnText;
impl crate::job::InferenceJobListener for PanicOnText {
    type CompletedItem = ();
    fn on_text(&mut self, _text: String) {
        panic!("simulated broken pipe: client dropped its stream");
    }
    fn on_finished(self) {}
}

/// A decoder that emits a fixed token script (one token per step), then the stop id. Used by
/// the byte-level detok tests, where tokens ARE byte values and the script deliberately
/// splits a multi-byte UTF-8 character across steps.
#[derive(Debug, Clone, Default)]
pub(super) struct ScriptedDecoder {
    pub(super) script: Vec<u32>,
    /// When set to `(n, signal)`, fires `signal` while producing the n-th generated token —
    /// a deterministic "client disconnected mid-generation" event, with no sleeps to race.
    pub(super) cancel_after: Option<(usize, CancelSignal)>,
    /// Per-slot step counters: the decoder-owned stand-in for a real per-slot KV cache.
    pub(super) steps: HashMap<usize, usize>,
}

/// Token ids 0..=255 are the raw bytes; 256 is the stop id.
pub(super) const BYTE_STOP: u32 = 256;
pub(super) const BYTE_VOCAB: usize = 257;

impl ScriptedDecoder {
    /// One scripted step for `slot`: emit the script's next token (the stop id past the end),
    /// as one-hot `[1, vocab]` logits.
    fn step_slot(&mut self, slot: usize) -> InferenceResult<Tensor<2>> {
        let counter = self.steps.entry(slot).or_insert(0);
        let step = *counter;
        *counter += 1;

        if let Some((after, cancel)) = &self.cancel_after {
            if step + 1 == *after {
                cancel.cancel();
            }
        }

        let token = self.script.get(step).copied().unwrap_or(BYTE_STOP) as usize;

        let mut data = vec![0.0f32; BYTE_VOCAB];
        data[token] = 1.0;
        Ok(Tensor::<2>::from_data(
            TensorData::new(data, [1, BYTE_VOCAB]),
            &*INFERENCE_DEVICE,
        ))
    }
}

impl BatchedDecoder for ScriptedDecoder {
    fn prefill(
        &mut self,
        slot: usize,
        _tokens: &[u32],
        _position: usize,
    ) -> InferenceResult<Tensor<2>> {
        self.step_slot(slot)
    }

    fn decode(&mut self, rows: &[DecodeRow]) -> InferenceResult<Tensor<2>> {
        let outputs = rows
            .iter()
            .map(|row| self.step_slot(row.slot))
            .collect::<InferenceResult<Vec<_>>>()?;
        Ok(Tensor::cat(outputs, 0))
    }

    fn release(&mut self, slot: usize) {
        self.steps.remove(&slot);
    }
}

/// A server with a BYTE-LEVEL vocab: token ids are raw byte values, so a multi-byte UTF-8
/// character genuinely splits across tokens (like Llama-3's Tiktoken). Its `detokenize` is
/// deliberately STRICT — it panics on a partial character, exactly like
/// `Tiktoken::decode`'s `.expect` — so these tests prove the worker streams through the
/// byte-level `detokenize_bytes` path instead.
#[derive(Debug, Clone)]
pub(super) struct ByteServer {
    pub(super) loaded: bool,
    pub(super) decoder: ScriptedDecoder,
    pub(super) max_gen: usize,
}

impl Default for ByteServer {
    fn default() -> Self {
        Self {
            loaded: false,
            decoder: ScriptedDecoder::default(),
            max_gen: 16,
        }
    }
}

impl ByteServer {
    pub(super) fn new(script: Vec<u32>, max_gen: usize) -> Self {
        Self {
            loaded: false,
            decoder: ScriptedDecoder {
                script,
                cancel_after: None,
                steps: HashMap::new(),
            },
            max_gen,
        }
    }

    /// Fire `cancel` while producing the `after`-th generated token (see
    /// [`ScriptedDecoder::cancel_after`]).
    pub(super) fn with_cancel_after(mut self, after: usize, cancel: CancelSignal) -> Self {
        self.decoder.cancel_after = Some((after, cancel));
        self
    }
}

impl ServerConfigParsing for ByteServer {
    type Config = FakeConfig;
    fn parse_cli_config(&mut self, _args: &clap::ArgMatches) {}
    fn parse_json_config(&mut self, _json: &str) {}
}

impl InferenceServer for ByteServer {
    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        self.loaded = true;
        Ok(None)
    }
    fn is_loaded(&mut self) -> bool {
        self.loaded
    }
    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.loaded = false;
        Ok(None)
    }
    fn run_job(&mut self, _job: InferenceJob) -> InferenceResult<Stats> {
        Ok(Stats::new())
    }
    fn clear_state(&mut self) -> InferenceResult<()> {
        Ok(())
    }
}

impl BatchedInferenceServer for ByteServer {
    type Decoder = ScriptedDecoder;

    fn decoder(&mut self) -> InferenceResult<&mut ScriptedDecoder> {
        self.loaded = true;
        Ok(&mut self.decoder)
    }

    fn batch_capacity(&self) -> BatchCapacity {
        BatchCapacity { max_slots: 1 }
    }

    fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>> {
        // The prompt's bytes are its tokens (the decoder ignores them anyway).
        let InferenceTask::Prompt(prompt) = task else {
            return Ok(vec![b'?' as u32]);
        };
        Ok(prompt.bytes().map(u32::from).collect())
    }

    /// STRICT per-token text decode, like `Tiktoken::decode`: panics on a partial character.
    /// The worker must never call this per generated token — that is the bug S1 fixes.
    fn detokenize(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8(bytes).expect("partial UTF-8: per-token text decode is unsound")
    }

    fn detokenize_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        tokens.iter().map(|&t| t as u8).collect()
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![BYTE_STOP]
    }

    fn max_gen_tokens(&self) -> usize {
        self.max_gen
    }
}

pub(super) fn submit_two(slots: usize) -> Vec<usize> {
    let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
    let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(slots, log.clone()));

    let (job_a, _ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        NullListener,
    );
    let (job_b, _hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );

    // Enqueue BOTH before either completes (non-blocking submit), then wait for both.
    let rx_a = channel.submit(job_a).unwrap();
    let rx_b = channel.submit(job_b).unwrap();
    rx_a.recv().unwrap().unwrap();
    rx_b.recv().unwrap().unwrap();

    let out = log.lock().unwrap().clone();
    out
}

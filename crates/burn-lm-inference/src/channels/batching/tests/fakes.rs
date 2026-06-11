use super::super::*;
use crate::{
    batching::{BatchCapacity, BatchedDecoder, ForwardBatch, ForwardOutput},
    job::{CancelSignal, GenerationParams, InferenceJob, InferenceTask},
    sampler::NextTokenSampler,
    server::{InferenceServer, ServerConfigParsing},
    InferenceServerConfig, Stats, TextGenerationListener, INFERENCE_DEVICE,
};
use burn::tensor::{Int, Tensor, TensorData};
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

/// A trivial decoder. Its cache is a per-sequence step counter (owned by the engine). It echoes
/// the sequence's identity token (the prompt's first token, which it then re-receives every
/// decode step) for a few steps, recording the emission order, then emits the stop id (0).
#[derive(Debug, Clone)]
pub(super) struct FakeDecoder {
    pub(super) log: OrderLog,
    /// How many tokens each sequence emits before stopping.
    pub(super) emit: usize,
    /// Extra logits rows beyond the single input row — simulates a decoder that violates the
    /// rows-in==rows-out contract. 0 = well-behaved.
    pub(super) extra_rows: usize,
    /// Per-forward sleep — simulates a slow model so a test can observe a job in flight.
    pub(super) step_delay_ms: u64,
    /// When set, `forward` PANICS at this per-sequence step — simulates a model bug that
    /// unwinds the worker iteration (the failure-ladder rung above per-sequence errors).
    pub(super) panic_at_step: Option<usize>,
}

pub(super) const VOCAB: usize = 64;

impl BatchedDecoder for FakeDecoder {
    type Cache = usize;

    fn device(&self) -> burn::tensor::Device {
        INFERENCE_DEVICE.clone()
    }

    fn allocate_cache(&self, _capacity: BatchCapacity) -> usize {
        0
    }

    fn forward(
        &mut self,
        batch: ForwardBatch,
        cache: &mut usize,
    ) -> InferenceResult<ForwardOutput> {
        if self.step_delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.step_delay_ms));
        }

        let step = *cache;
        *cache += 1;

        if self.panic_at_step == Some(step) {
            panic!("scripted decoder panic at step {step}");
        }

        // Identity = last input token (the prompt token on prefill, the echoed token after).
        let ids = batch
            .input_tokens
            .into_data()
            .convert::<u32>()
            .into_vec::<u32>()
            .unwrap();
        let identity = *ids.last().unwrap() as usize;

        let token = if step < self.emit {
            // Record which sequence emitted, synchronously, in true generation order.
            self.log.lock().unwrap().push(identity % 2);
            identity % VOCAB
        } else {
            0 // stop id
        };

        let rows = 1 + self.extra_rows;
        let mut data = vec![0.0f32; rows * VOCAB];
        data[token] = 1.0;
        let logits =
            Tensor::<3>::from_data(TensorData::new(data, [rows, 1, VOCAB]), &*INFERENCE_DEVICE);
        Ok(ForwardOutput { logits })
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
            decoder: FakeDecoder {
                log,
                emit: 4,
                extra_rows: 0,
                step_delay_ms: 0,
                panic_at_step: None,
            },
            capacity_calls: Arc::new(AtomicUsize::new(0)),
            fixed_token: None,
        }
    }

    /// A server whose decoder returns 2 logits rows for a 1-row input — violates the forward
    /// rows-in==rows-out contract.
    pub(super) fn new_bad(slots: usize, log: OrderLog) -> Self {
        Self {
            loaded: false,
            slots,
            decoder: FakeDecoder {
                log,
                emit: 4,
                extra_rows: 1,
                step_delay_ms: 0,
                panic_at_step: None,
            },
            capacity_calls: Arc::new(AtomicUsize::new(0)),
            fixed_token: None,
        }
    }

    /// A server whose decoder emits many tokens, each after a small sleep — a long-running job
    /// a test can interrogate (e.g. unload) while it is demonstrably still in flight.
    pub(super) fn new_slow(slots: usize, log: OrderLog) -> Self {
        Self {
            loaded: false,
            slots,
            decoder: FakeDecoder {
                log,
                emit: 1000, // effectively capped by `max_gen_tokens` (16)
                extra_rows: 0,
                step_delay_ms: 20,
                panic_at_step: None,
            },
            capacity_calls: Arc::new(AtomicUsize::new(0)),
            fixed_token: None,
        }
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
        Self {
            loaded: false,
            slots,
            decoder: FakeDecoder {
                log: Arc::new(Mutex::new(Vec::new())),
                emit: 4,
                extra_rows: 0,
                step_delay_ms: 0,
                panic_at_step: None,
            },
            capacity_calls: calls,
            fixed_token: None,
        }
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
            max_kv_tokens: 1024,
        }
    }

    fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>> {
        // Map each prompt to a distinct identity token so sequences are distinguishable.
        let id = match task {
            InferenceTask::Prompt(p) if p == "a" => 10u32,
            InferenceTask::Prompt(p) if p == "b" => 11u32,
            _ => 12u32,
        };
        Ok(vec![id])
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
}

/// Token ids 0..=255 are the raw bytes; 256 is the stop id.
pub(super) const BYTE_STOP: u32 = 256;
pub(super) const BYTE_VOCAB: usize = 257;

impl BatchedDecoder for ScriptedDecoder {
    type Cache = usize;

    fn device(&self) -> burn::tensor::Device {
        INFERENCE_DEVICE.clone()
    }

    fn allocate_cache(&self, _capacity: BatchCapacity) -> usize {
        0
    }

    fn forward(
        &mut self,
        _batch: ForwardBatch,
        cache: &mut usize,
    ) -> InferenceResult<ForwardOutput> {
        let step = *cache;
        *cache += 1;

        if let Some((after, cancel)) = &self.cancel_after {
            if step + 1 == *after {
                cancel.cancel();
            }
        }

        let token = self.script.get(step).copied().unwrap_or(BYTE_STOP) as usize;

        let mut data = vec![0.0f32; BYTE_VOCAB];
        data[token] = 1.0;
        let logits = Tensor::<3>::from_data(
            TensorData::new(data, [1, 1, BYTE_VOCAB]),
            &*INFERENCE_DEVICE,
        );
        Ok(ForwardOutput { logits })
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
        BatchCapacity {
            max_slots: 1,
            max_kv_tokens: 1024,
        }
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

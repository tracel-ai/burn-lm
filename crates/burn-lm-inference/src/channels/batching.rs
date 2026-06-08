//! A channel that drives a model from a single long-lived worker thread running a *continuous*
//! generation loop.
//!
//! Unlike [`MutexChannel`](crate::channels::mutex::MutexChannel), which locks the server for the
//! whole of every call, `BatchingChannel` hands exclusive ownership of the server to one worker
//! thread and communicates with it over a command queue. Callers block only on the reply to their
//! own command.
//!
//! The worker owns the whole continuous-batching loop: an inbound queue of submitted jobs plus an
//! active set of in-flight sequences. All per-sequence state (the model's
//! [`BatchedDecoder::Cache`], token buffer, position, emitter, completion sender, detok cursor,
//! generated count, finished flag) lives HERE in the framework — the model only exposes `forward`,
//! tokenizer primitives and capacity (see [`BatchedInferenceServer`]).
//!
//! Phase 1 STUB: the decode step is round-robin — each active sequence gets its own batch-1
//! [`forward`](BatchedDecoder::forward) call against its own engine-owned cache. This proves the
//! scheduling/admission/streaming plumbing end to end. Phase 2 will replace only the forward body
//! with a fused multi-row GPU call.

use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    sync::{
        mpsc::{Receiver, Sender, SyncSender},
        Arc, Mutex,
    },
    thread::JoinHandle,
};

use burn::tensor::{Int, Tensor, TensorData};

use crate::{
    batching::{BatchedDecoder, BatchedInferenceServer, ForwardBatch},
    errors::{InferenceError, InferenceResult},
    sampler::Sampler,
    GeneratedItem, GeneratedItemEmitter, InferenceJob, Stats, INFERENCE_DEVICE,
};

use super::InferenceChannel;

type DownloaderFn = fn() -> InferenceResult<Option<Stats>>;

/// Commands sent from channel handles to the worker that owns the server.
///
/// Each variant carries a one-shot reply sender; the calling method blocks on the matching
/// receiver. `Submit`'s `completion` reply *is* the per-job completion signal, sent by the loop
/// when the sequence retires (not when the command is received).
enum Command {
    Downloader(SyncSender<Option<DownloaderFn>>),
    IsDownloaded(SyncSender<bool>),
    Deleter(SyncSender<Option<DownloaderFn>>),
    ParseCliConfig(clap::ArgMatches, SyncSender<()>),
    ParseJsonConfig(String, SyncSender<()>),
    Load(SyncSender<InferenceResult<Option<Stats>>>),
    IsLoaded(SyncSender<bool>),
    Unload(SyncSender<InferenceResult<Option<Stats>>>),
    Submit {
        job: InferenceJob,
        completion: SyncSender<InferenceResult<Stats>>,
    },
    ClearState(SyncSender<InferenceResult<()>>),
    Shutdown,
}

/// A running worker: its command sender and thread handle.
struct WorkerInner {
    sender: Sender<Command>,
    handle: JoinHandle<()>,
}

/// Holds the worker, which is spawned lazily on first use rather than at channel construction.
/// The registry builds a channel per registered model, but most are never used; deferring the
/// thread until the first command means only models that are actually exercised spawn one.
struct Worker {
    inner: Mutex<Option<WorkerInner>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Ask the worker to stop and wait for it so the owned server is dropped cleanly.
        if let Some(inner) = self.inner.lock().unwrap().take() {
            let _ = inner.sender.send(Command::Shutdown);
            let _ = inner.handle.join();
        }
    }
}

/// Channel that owns a batched model on a dedicated worker thread running a continuous loop.
pub struct BatchingChannel<Server: BatchedInferenceServer> {
    worker: Arc<Worker>,
    _server: PhantomData<Server>,
}

impl<Server: BatchedInferenceServer> Clone for BatchingChannel<Server> {
    fn clone(&self) -> Self {
        Self {
            worker: self.worker.clone(),
            _server: PhantomData,
        }
    }
}

impl<Server: BatchedInferenceServer> fmt::Debug for BatchingChannel<Server> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchingChannel").finish_non_exhaustive()
    }
}

impl<Server: BatchedInferenceServer + 'static> Default for BatchingChannel<Server> {
    fn default() -> Self {
        Self::new()
    }
}

/// Framework-owned state for a single in-flight sequence.
///
/// This is the per-sequence active state the engine drives. The cache is the model's, but it is
/// OWNED HERE and handed back to the decoder on each step; everything else (buffers, cursors,
/// counters, the job's emitter and completion sender) is pure framework bookkeeping.
struct ActiveSeq<S: BatchedInferenceServer> {
    /// The model's per-sequence cache, allocated by the engine and passed into every `forward`.
    cache: <S::Decoder as BatchedDecoder>::Cache,
    /// Full token buffer (prompt + generated). The next forward consumes the unprocessed tail.
    tokens: Vec<u32>,
    /// Number of tokens already pushed through the decoder (== absolute position of next input).
    processed: usize,
    /// Where this sequence's text is streamed.
    emitter: GeneratedItemEmitter,
    /// One-shot completion signal for the submitting caller, fired when the sequence retires.
    completion: SyncSender<InferenceResult<Stats>>,
    /// Generated-token count (excludes the prompt).
    generated: usize,
    /// Hard cap on generated tokens for this sequence.
    max_gen: usize,
    /// Set once a stop id or the cap is hit; the next sweep retires it.
    finished: bool,
}

impl<Server: BatchedInferenceServer + 'static> BatchingChannel<Server> {
    pub fn new() -> Self {
        Self {
            worker: Arc::new(Worker {
                inner: Mutex::new(None),
            }),
            _server: PhantomData,
        }
    }

    /// Build a channel whose worker is spawned immediately around a specific server instance.
    /// Test-only: lets tests configure capacity and decoder behavior (production uses the lazy
    /// `Server::default()` path via [`new`](Self::new)).
    #[cfg(test)]
    fn with_server(server: Server) -> Self {
        let channel = Self::new();
        *channel.worker.inner.lock().unwrap() = Some(Self::spawn_worker_with(server));
        channel
    }

    /// Spawn the worker thread that owns the server and runs the continuous loop.
    fn spawn_worker() -> WorkerInner {
        Self::spawn_worker_with(Server::default())
    }

    /// Spawn the worker around a specific server instance. Production uses `Server::default()`;
    /// tests seed a configured server so capacity and behavior are controllable.
    fn spawn_worker_with(seed: Server) -> WorkerInner {
        let (sender, receiver) = std::sync::mpsc::channel::<Command>();

        let handle = std::thread::spawn(move || {
            let mut server = seed;
            let mut queue: VecDeque<(InferenceJob, SyncSender<InferenceResult<Stats>>)> =
                VecDeque::new();
            let mut active: Vec<ActiveSeq<Server>> = Vec::new();
            let mut sampler = Sampler::default();

            loop {
                // Park (block for the next command) only when there is genuinely nothing to do:
                // nothing active to advance AND nothing admittable right now. A queued job is
                // admittable when the queue is non-empty and the server has a free slot. Parking
                // while such a job waits would HANG it: a slot frees when a sequence retires (which
                // can leave `active` empty) and no new command arrives to wake `recv()`. Parking
                // when there is no free slot (or the queue is empty) still avoids busy-spinning.
                let mut shutdown = false;
                let can_admit =
                    !queue.is_empty() && server.batch_capacity().free_slots > active.len();
                if active.is_empty() && !can_admit {
                    match receiver.recv() {
                        Ok(command) => {
                            shutdown = handle_command(&mut server, &mut queue, command);
                        }
                        Err(_) => break, // all senders dropped
                    }
                }
                // Drain any further pending commands without blocking, so a burst of submissions is
                // fully enqueued before the next admit/step sweep. Admission then sees all ready
                // jobs together (deterministic batching) rather than one per iteration.
                if !shutdown {
                    while let Ok(command) = receiver.try_recv() {
                        if handle_command(&mut server, &mut queue, command) {
                            shutdown = true;
                            break;
                        }
                    }
                }
                if shutdown {
                    break;
                }

                // ADMISSION (backpressure): admit queued jobs while there is free capacity. A job
                // that does not fit stays at the front of the queue for a later iteration.
                admit(&mut server, &mut queue, &mut active);

                // STEP (round-robin stub): advance every active sequence by one token, then retire
                // any that finished. Retiring frees a slot so the next iteration admits more.
                step(&mut server, &mut active, &mut sampler);
            }
        });

        WorkerInner { sender, handle }
    }

    /// Return a sender to the worker, spawning it on first use.
    fn sender(&self) -> Sender<Command> {
        let mut guard = self.worker.inner.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Self::spawn_worker());
        }
        guard.as_ref().unwrap().sender.clone()
    }

    /// Whether the worker has been spawned, i.e. the channel has been used at least once.
    fn is_spawned(&self) -> bool {
        self.worker.inner.lock().unwrap().is_some()
    }

    /// Send a command (spawning the worker if needed) and block on its reply, mapping a dead
    /// worker to an error.
    fn request<T>(&self, make: impl FnOnce(SyncSender<T>) -> Command) -> Result<T, ()> {
        let (reply, rx) = std::sync::mpsc::sync_channel::<T>(1);
        self.sender().send(make(reply)).map_err(|_| ())?;
        rx.recv().map_err(|_| ())
    }

    /// Enqueue a job without waiting for it to complete. Returns the completion receiver so the
    /// caller can wait later (or drop it to fire-and-forget). This is the non-blocking entry point.
    pub fn submit(&self, job: InferenceJob) -> Result<Receiver<InferenceResult<Stats>>, ()> {
        let (completion, rx) = std::sync::mpsc::sync_channel::<InferenceResult<Stats>>(1);
        self.sender()
            .send(Command::Submit { job, completion })
            .map_err(|_| ())?;
        Ok(rx)
    }
}

/// Handle a lifecycle/config command between loop iterations. Returns `true` on shutdown.
///
/// `Submit` is special: it does not run the job, it just enqueues it; the completion reply is sent
/// later by [`step`] when the sequence retires.
fn handle_command<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<(InferenceJob, SyncSender<InferenceResult<Stats>>)>,
    command: Command,
) -> bool {
    match command {
        Command::Downloader(reply) => {
            let _ = reply.send(server.downloader());
        }
        Command::IsDownloaded(reply) => {
            let _ = reply.send(server.is_downloaded());
        }
        Command::Deleter(reply) => {
            let _ = reply.send(server.deleter());
        }
        Command::ParseCliConfig(args, reply) => {
            server.parse_cli_config(&args);
            let _ = reply.send(());
        }
        Command::ParseJsonConfig(json, reply) => {
            server.parse_json_config(&json);
            let _ = reply.send(());
        }
        Command::Load(reply) => {
            let _ = reply.send(server.load());
        }
        Command::IsLoaded(reply) => {
            let _ = reply.send(server.is_loaded());
        }
        Command::Unload(reply) => {
            let _ = reply.send(server.unload());
        }
        Command::Submit { job, completion } => {
            queue.push_back((job, completion));
        }
        Command::ClearState(reply) => {
            let _ = reply.send(server.clear_state());
        }
        Command::Shutdown => return true,
    }
    false
}

/// Admit queued jobs into the active set while there is free capacity (backpressure).
///
/// `batch_capacity().free_slots` is the server's reported concurrent-sequence budget. The engine
/// owns the active set, so "free" = that budget minus what is already active: a job is admitted
/// only while `active.len() < free_slots`. A job that does not fit stays queued for a later sweep
/// (which runs after a retire frees a slot), making admission continuous.
fn admit<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<(InferenceJob, SyncSender<InferenceResult<Stats>>)>,
    active: &mut Vec<ActiveSeq<S>>,
) {
    while active.len() < server.batch_capacity().free_slots {
        let Some((job, completion)) = queue.pop_front() else {
            break;
        };

        // Allocate the cache first: this loads the model if needed (via `decoder`), so the
        // subsequent `tokenize`/`detokenize` primitives can rely on the tokenizer being available.
        let capacity = server.batch_capacity();
        let cache = match server.allocate_cache(capacity) {
            Ok(cache) => cache,
            Err(err) => {
                let _ = completion.send(Err(err));
                continue;
            }
        };

        let tokens = match server.tokenize(&job.task) {
            Ok(tokens) => tokens,
            Err(err) => {
                let _ = completion.send(Err(err));
                continue;
            }
        };

        active.push(ActiveSeq {
            cache,
            tokens,
            processed: 0,
            emitter: job.emitter,
            completion,
            generated: 0,
            max_gen: server.max_gen_tokens(),
            finished: false,
        });
    }
}

/// Advance every active sequence by one token (round-robin), then retire finished ones.
fn step<S: BatchedInferenceServer>(
    server: &mut S,
    active: &mut Vec<ActiveSeq<S>>,
    sampler: &mut Sampler,
) {
    let stop_ids = server.stop_ids();

    for seq in active.iter_mut() {
        if seq.finished {
            continue;
        }

        // The unprocessed tail: on the first step this is the whole prompt (prefill); afterwards a
        // single token (decode). Empty prompts cannot generate, so retire immediately.
        if seq.processed >= seq.tokens.len() {
            seq.finished = true;
            continue;
        }
        let input_ids: Vec<i32> = seq.tokens[seq.processed..].iter().map(|&t| t as i32).collect();
        let seq_len = input_ids.len();
        let position = seq.processed;

        let input_tokens = Tensor::<2, Int>::from_data(
            TensorData::new(input_ids, [1, seq_len]),
            &*INFERENCE_DEVICE,
        );
        let batch = ForwardBatch {
            input_tokens,
            positions: vec![position],
            cache_slots: vec![0],
        };

        let next = forward_one(server, &mut seq.cache, batch, sampler);
        let next_token = match next {
            Ok(token) => token,
            Err(err) => {
                let _ = seq.completion.send(Err(err));
                seq.finished = true;
                continue;
            }
        };

        seq.processed = seq.tokens.len();
        seq.tokens.push(next_token);
        seq.generated += 1;

        if stop_ids.contains(&next_token) {
            seq.finished = true;
        } else {
            // Stream the new token's text to the job emitter.
            let text = server.detokenize(&[next_token]);
            if !text.is_empty() {
                seq.emitter.completed(GeneratedItem::Text(text));
            }
        }

        if seq.generated >= seq.max_gen {
            seq.finished = true;
        }
    }

    // RETIRE: drop finished sequences and signal completion, freeing capacity for admission.
    active.retain(|seq| {
        if seq.finished {
            let mut stats = Stats::new();
            stats
                .entries
                .insert(crate::stats::StatEntry::TokensCount(seq.generated));
            let _ = seq.completion.send(Ok(stats));
            false
        } else {
            true
        }
    });
}

/// One batch-1 forward + sample. Takes the engine-owned cache for this sequence and runs the
/// model's `forward` against it, then framework-samples the next token id from the last position.
fn forward_one<S: BatchedInferenceServer>(
    server: &mut S,
    cache: &mut <S::Decoder as BatchedDecoder>::Cache,
    batch: ForwardBatch,
    sampler: &mut Sampler,
) -> InferenceResult<u32> {
    let in_rows = batch.input_tokens.dims()[0];
    let output = server.decoder()?.forward(batch, cache)?;

    // Forward contract: the decoder must return exactly one logits row per input row, with at least
    // one position. Enforce it as a per-sequence error (never a panic): on the worker thread a
    // panic here would unwind and brick the whole channel, and a wrong row count would otherwise
    // silently sample the wrong sequence. step()'s per-sequence error path retires just this one.
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

    let token = sampler.sample(next_token_logits);
    let ids = token
        .into_data()
        .convert::<u32>()
        .into_vec::<u32>()
        .map_err(|_| {
            InferenceError::BatchContractViolation("sampled token tensor did not convert to u32".to_string())
        })?;
    let id = *ids
        .first()
        .ok_or_else(|| InferenceError::BatchContractViolation("sampler produced no token".to_string()))?;
    Ok(id)
}

fn worker_gone() -> InferenceError {
    InferenceError::LoadError("batching worker thread is not available".to_string())
}

impl<Server: BatchedInferenceServer + 'static> InferenceChannel<Server> for BatchingChannel<Server> {
    fn downloader(&self) -> Option<DownloaderFn> {
        self.request(Command::Downloader).unwrap_or(None)
    }

    fn is_downloaded(&self) -> bool {
        self.request(Command::IsDownloaded).unwrap_or(false)
    }

    fn deleter(&self) -> Option<DownloaderFn> {
        self.request(Command::Deleter).unwrap_or(None)
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        let _ = self.request(|reply| Command::ParseCliConfig(args.clone(), reply));
    }

    fn parse_json_config(&self, json: &str) {
        let json = json.to_string();
        let _ = self.request(|reply| Command::ParseJsonConfig(json, reply));
    }

    fn load(&self) -> InferenceResult<Option<Stats>> {
        self.request(Command::Load).map_err(|_| worker_gone())?
    }

    fn is_loaded(&self) -> bool {
        // Nothing can be loaded before the worker is even spawned; avoid spawning just to answer.
        if !self.is_spawned() {
            return false;
        }
        self.request(Command::IsLoaded).unwrap_or(false)
    }

    fn unload(&self) -> InferenceResult<Option<Stats>> {
        self.request(Command::Unload).map_err(|_| worker_gone())?
    }

    /// Blocking job entry point preserved for the `InferenceChannel`/`InferencePlugin` contract:
    /// submit (enqueue) then wait on the per-sequence completion signal.
    fn run_job(&self, job: InferenceJob) -> InferenceResult<Stats> {
        let rx = self.submit(job).map_err(|_| worker_gone())?;
        rx.recv().map_err(|_| worker_gone())?
    }

    fn clear_state(&self) -> InferenceResult<()> {
        self.request(Command::ClearState).map_err(|_| worker_gone())?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batching::{BatchCapacity, ForwardOutput},
        job::{InferenceJob, InferenceTask},
        server::{InferenceServer, ServerConfigParsing},
        InferenceServerConfig,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    #[derive(Debug, Default, Clone, serde::Deserialize, clap::Parser)]
    struct FakeConfig {}
    impl InferenceServerConfig for FakeConfig {}

    /// Shared, synchronous record of emission order. The fake decoder appends to it on the worker
    /// thread at the moment a (non-stop) token is produced, so the recorded order is exactly the
    /// engine's generation interleaving — independent of how the async emitter threads later drain.
    type OrderLog = Arc<Mutex<Vec<usize>>>;

    /// A trivial decoder. Its cache is a per-sequence step counter (owned by the engine). It echoes
    /// the sequence's identity token (the prompt's first token, which it then re-receives every
    /// decode step) for a few steps, recording the emission order, then emits the stop id (0).
    #[derive(Debug, Clone)]
    struct FakeDecoder {
        log: OrderLog,
        /// How many tokens each sequence emits before stopping.
        emit: usize,
        /// Extra logits rows beyond the single input row — simulates a decoder that violates the
        /// rows-in==rows-out contract. 0 = well-behaved.
        extra_rows: usize,
    }

    const VOCAB: usize = 64;

    impl BatchedDecoder for FakeDecoder {
        type Cache = usize;

        fn allocate_cache(&self, _capacity: BatchCapacity) -> usize {
            0
        }

        fn forward(
            &mut self,
            batch: ForwardBatch,
            cache: &mut usize,
        ) -> InferenceResult<ForwardOutput> {
            let step = *cache;
            *cache += 1;

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
            let logits = Tensor::<3>::from_data(
                TensorData::new(data, [rows, 1, VOCAB]),
                &*INFERENCE_DEVICE,
            );
            Ok(ForwardOutput { logits })
        }
    }

    #[derive(Debug, Clone)]
    struct FakeServer {
        loaded: bool,
        slots: usize,
        decoder: FakeDecoder,
        /// Counts `batch_capacity` calls — lets the `free_slots == 0` test detect a busy-spin.
        capacity_calls: Arc<AtomicUsize>,
    }

    impl Default for FakeServer {
        fn default() -> Self {
            Self::new(1, Arc::new(Mutex::new(Vec::new())))
        }
    }

    impl FakeServer {
        fn new(slots: usize, log: OrderLog) -> Self {
            Self {
                loaded: false,
                slots,
                decoder: FakeDecoder {
                    log,
                    emit: 4,
                    extra_rows: 0,
                },
                capacity_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// A server whose decoder returns 2 logits rows for a 1-row input — violates the forward
        /// rows-in==rows-out contract.
        fn new_bad(slots: usize, log: OrderLog) -> Self {
            Self {
                loaded: false,
                slots,
                decoder: FakeDecoder {
                    log,
                    emit: 4,
                    extra_rows: 1,
                },
                capacity_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// A server reporting `slots` free slots that records every `batch_capacity` call into
        /// `calls`, so a test can tell whether the worker busy-spins when nothing is admittable.
        fn with_capacity_probe(slots: usize, calls: Arc<AtomicUsize>) -> Self {
            Self {
                loaded: false,
                slots,
                decoder: FakeDecoder {
                    log: Arc::new(Mutex::new(Vec::new())),
                    emit: 4,
                    extra_rows: 0,
                },
                capacity_calls: calls,
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
            Ok(&mut self.decoder)
        }

        fn batch_capacity(&self) -> BatchCapacity {
            self.capacity_calls.fetch_add(1, Ordering::Relaxed);
            BatchCapacity {
                free_slots: self.slots,
                free_kv_tokens: 1024,
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
    }

    /// A no-op listener (text is recorded synchronously in the decoder, not here).
    struct NullListener;
    impl crate::job::InferenceJobListener for NullListener {
        type CompletedItem = ();
        fn on_text(&mut self, _text: String) {}
        fn on_finished(self) {}
    }

    /// A listener that panics on the first emitted token — stands in for a client whose stream
    /// broke (a dropped SSE connection makes `WriteListener::on_text`'s write `.unwrap()` panic),
    /// which kills the listener thread out from under the worker.
    struct PanicOnText;
    impl crate::job::InferenceJobListener for PanicOnText {
        type CompletedItem = ();
        fn on_text(&mut self, _text: String) {
            panic!("simulated broken pipe: client dropped its stream");
        }
        fn on_finished(self) {}
    }

    /// A client whose stream errors mid-generation must NOT brick the channel for everyone else.
    /// Before the fix, the worker's `emitter.completed()` `.unwrap()` panicked when that listener
    /// died, permanently killing the single worker thread. (The stderr panic from `PanicOnText` is
    /// the simulated disconnect and is expected.)
    #[test]
    fn worker_survives_a_client_stream_panic() {
        let channel =
            BatchingChannel::<FakeServer>::with_server(FakeServer::new(1, Arc::new(Mutex::new(Vec::new()))));

        // Job A: its listener panics on the first emitted token (broken pipe).
        let (job_a, _ha) = InferenceJob::create(InferenceTask::Prompt("a".into()), PanicOnText);
        let rx_a = channel.submit(job_a).unwrap();
        let _ = rx_a.recv(); // A's own outcome is irrelevant; its listener died.

        // Job B: a healthy client must still be served — proving the worker survived A.
        let (job_b, _hb) = InferenceJob::create(InferenceTask::Prompt("b".into()), NullListener);
        let rx_b = channel.submit(job_b).unwrap();
        rx_b
            .recv()
            .expect("channel must survive a client panic")
            .expect("job B should still complete");
    }

    /// A decoder that breaks the forward rows-in==rows-out contract must retire that sequence with
    /// an error, NOT panic the worker — and the channel must keep serving.
    #[test]
    fn worker_survives_a_decoder_contract_violation() {
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_bad(
            1,
            Arc::new(Mutex::new(Vec::new())),
        ));

        let (job1, _h1) = InferenceJob::create(InferenceTask::Prompt("a".into()), NullListener);
        let out1 = channel.submit(job1).unwrap().recv().expect("worker must survive");
        assert!(
            matches!(out1, Err(crate::InferenceError::BatchContractViolation(_))),
            "contract violation should retire the sequence with a BatchContractViolation error"
        );

        // The worker is still alive: a second job is accepted and processed (likewise retired).
        let (job2, _h2) = InferenceJob::create(InferenceTask::Prompt("b".into()), NullListener);
        let out2 = channel
            .submit(job2)
            .expect("worker must still accept jobs")
            .recv()
            .expect("worker must survive");
        assert!(out2.is_err(), "second job should also retire with an error");
    }

    /// A server reporting `free_slots == 0` with a job queued must PARK the worker, not busy-spin a
    /// core. We detect a spin via the `batch_capacity` call count: parked ⇒ a couple of calls;
    /// spinning ⇒ thousands over the same window.
    #[test]
    fn free_slots_zero_parks_instead_of_spinning() {
        let calls = Arc::new(AtomicUsize::new(0));
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::with_capacity_probe(
            0,
            calls.clone(),
        ));

        // Queued, but never admittable while free_slots == 0.
        let (job, _h) = InferenceJob::create(InferenceTask::Prompt("a".into()), NullListener);
        let _rx = channel.submit(job).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        let n = calls.load(Ordering::Relaxed);
        assert!(
            n < 100,
            "worker busy-spun on free_slots==0 (batch_capacity called {n} times in 50ms); it should park"
        );
    }

    fn submit_two(slots: usize) -> Vec<usize> {
        let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(slots, log.clone()));

        let (job_a, _ha) = InferenceJob::create(InferenceTask::Prompt("a".into()), NullListener);
        let (job_b, _hb) = InferenceJob::create(InferenceTask::Prompt("b".into()), NullListener);

        // Enqueue BOTH before either completes (non-blocking submit), then wait for both.
        let rx_a = channel.submit(job_a).unwrap();
        let rx_b = channel.submit(job_b).unwrap();
        rx_a.recv().unwrap().unwrap();
        rx_b.recv().unwrap().unwrap();

        let out = log.lock().unwrap().clone();
        out
    }

    /// Capacity >= 2: both jobs are admitted concurrently and their emission streams INTERLEAVE.
    /// Each sequence's first emission precedes the other's last (structural overlap).
    #[test]
    fn capacity_two_admits_concurrently_and_interleaves() {
        let log = submit_two(2);
        assert!(
            log.contains(&0) && log.contains(&1),
            "both sequences should produce output: {log:?}"
        );
        let first0 = log.iter().position(|&x| x == 0).unwrap();
        let last0 = log.iter().rposition(|&x| x == 0).unwrap();
        let first1 = log.iter().position(|&x| x == 1).unwrap();
        let last1 = log.iter().rposition(|&x| x == 1).unwrap();
        assert!(
            first1 < last0 && first0 < last1,
            "sequences did not interleave (ran serially): {log:?}"
        );
    }

    /// Capacity == 1: admission is one-at-a-time, so the two sequences run SERIALLY (no overlap).
    /// This proves capacity-based admission/backpressure.
    #[test]
    fn capacity_one_serializes() {
        let log = submit_two(1);
        assert!(
            log.contains(&0) && log.contains(&1),
            "both sequences should produce output: {log:?}"
        );
        let first0 = log.iter().position(|&x| x == 0).unwrap();
        let last0 = log.iter().rposition(|&x| x == 0).unwrap();
        let first1 = log.iter().position(|&x| x == 1).unwrap();
        let last1 = log.iter().rposition(|&x| x == 1).unwrap();
        // Serial ⇒ one sequence fully precedes the other.
        assert!(
            last0 < first1 || last1 < first0,
            "sequences interleaved but capacity==1 should serialize them: {log:?}"
        );
    }

    #[test]
    fn lifecycle_round_trip() {
        let channel = BatchingChannel::<FakeServer>::new();
        assert!(!channel.is_spawned());
        // A pre-spawn `is_loaded` answers without spawning the worker.
        assert!(!channel.is_loaded());
        assert!(!channel.is_spawned());

        channel.load().unwrap();
        assert!(channel.is_spawned());
        assert!(channel.is_loaded());

        channel.unload().unwrap();
        assert!(!channel.is_loaded());
    }
}

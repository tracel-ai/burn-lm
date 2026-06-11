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
//!
//! BACKPRESSURE: the job queue is bounded ([`DEFAULT_MAX_QUEUE_DEPTH`], settable via
//! [`with_queue_depth`](BatchingChannel::with_queue_depth)). [`submit`](BatchingChannel::submit)
//! sheds synchronously with [`InferenceError::Overloaded`] when the bound is reached, so sustained
//! overload turns into immediate 4xx-class rejections instead of an ever-growing queue of parked
//! caller threads.
//!
//! FAILURE LADDER: per-sequence faults (forward errors, contract violations) retire just that
//! sequence; a panic anywhere in a worker iteration is caught, every in-flight and queued job is
//! answered with [`InferenceError::WorkerDied`] (after flushing its detok cursor), and the worker
//! thread exits — the next command lazily respawns a FRESH worker around a fresh
//! `Server::default()`. Callers can never park forever on a dead worker. Caller-visible
//! consequence: the fresh server starts UNLOADED, so a previously observed `load()` success does
//! not survive a panic — jobs still work because admission lazy-loads through
//! [`decoder`](BatchedInferenceServer::decoder)/`allocate_cache`, the first post-panic job just
//! pays the load again.

use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, SyncSender},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::{
    batching::{step_round, ActiveSeq, BatchedInferenceServer, CacheOf, StepOutcome},
    errors::{InferenceError, InferenceResult},
    job::CancelSignal,
    sampler::NextTokenSampler,
    utf8::Utf8Buffer,
    GeneratedItem, GeneratedItemEmitter, InferenceJob, StatEntry, Stats,
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
    Submit(QueuedJob),
    ClearState(SyncSender<InferenceResult<()>>),
    Shutdown,
}

/// Default bound on the number of submitted-but-not-yet-admitted jobs (see
/// [`BatchingChannel::with_queue_depth`]). Beyond it, [`submit`](BatchingChannel::submit) sheds
/// synchronously with [`InferenceError::Overloaded`].
pub const DEFAULT_MAX_QUEUE_DEPTH: usize = 32;

/// Stat name for the time a job spent queued before admission (enqueue → admission). Admission is
/// the endpoint (rather than completion) because the stat exists to expose QUEUEING delay — the
/// thing the queue bound trades off — while generation time is already covered by the token-count
/// and duration stats.
pub const QUEUE_WAIT_STAT_NAME: &str = "Queue Wait";

/// RAII permit for one job's slot in the queue's budget — issued by the queue's hand-rolled
/// semaphore (the shared `AtomicUsize` depth counter), in the same shape as tokio's
/// `OwnedSemaphorePermit`.
///
/// The counter is incremented by `submit` BEFORE the job is sent to the worker; this permit rides
/// with the job and decrements on drop, which happens exactly when the job "leaves the queue" on
/// ANY path: admitted into the active set, rejected at admission (cancel/tokenize/cache errors),
/// drained by a worker panic — or dropped wholesale with a dying worker's command channel. Tying
/// the decrement to `Drop` means no path (present or future) can leak the counter upward.
struct QueuePermit(Arc<AtomicUsize>);

impl Drop for QueuePermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// One submitted-but-not-yet-admitted job: the job itself, its one-shot completion sender, the
/// enqueue timestamp (for the queue-wait stat) and its queue permit.
struct QueuedJob {
    job: InferenceJob,
    completion: SyncSender<InferenceResult<Stats>>,
    /// Queue-wait clock, started at submit and read at admission (see [`QUEUE_WAIT_STAT_NAME`]).
    enqueued_at: Instant,
    /// Dropped when the job leaves the queue; see [`QueuePermit`].
    permit: QueuePermit,
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
    /// Count of submitted-but-not-yet-admitted jobs, shared between submitters (increment) and
    /// the per-job [`QueuePermit`]s (decrement on drop). The bound this enforces is APPROXIMATE
    /// by design: the check-and-increment in `submit` is atomic, but a job admitted a microsecond
    /// after a shed decision means the queue briefly had room — being off by a few under
    /// contention is acceptable; the point is that overload sheds instead of growing without
    /// bound. Plain counter, no other data synchronized through it, hence `Relaxed` everywhere.
    pending: Arc<AtomicUsize>,
    /// Backpressure bound on `pending` (see [`DEFAULT_MAX_QUEUE_DEPTH`]).
    max_queue_depth: usize,
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

/// Serving-driver payload attached to a generic [`ActiveSeq`]: where a sequence's text is streamed
/// and the one-shot completion signal fired when it retires. The generic decode core
/// ([`step_round`]) never touches this — it advances the [`ActiveSeq`]'s cache/tokens/counters and
/// reports back; the worker uses this payload to stream tokens and signal completion.
struct JobMeta {
    /// Where this sequence's text is streamed.
    emitter: GeneratedItemEmitter,
    /// Per-sequence detok cursor: byte-level BPE can split a multi-byte UTF-8 character across
    /// tokens, so the loop streams each token's RAW BYTES
    /// ([`detokenize_bytes`](BatchedInferenceServer::detokenize_bytes)) through this buffer and
    /// emits only complete text — never a panic, never mid-stream U+FFFD. Every retire path
    /// flushes it ([`flush_detok`]) so trailing held-back bytes are not silently dropped.
    detok: Utf8Buffer,
    /// Per-sequence sampler, built from the server's sampling config at admission and persisted
    /// for the sequence's whole generation — same as the single-request path, where one (possibly
    /// seeded) sampler's RNG advances across all of a request's tokens. Rebuilding per round would
    /// reset a seeded RNG before every token. `Option` so [`step`] can `.take()` it while the
    /// sequence itself is mutably borrowed for the round.
    sampler: Option<Box<dyn NextTokenSampler + Send>>,
    /// One-shot completion signal for the submitting caller, fired when the sequence retires.
    /// `Option` so every send site `.take()`s it: completion fires exactly once (first send wins).
    /// The channel is a bounded one-shot, so a second send with the first still buffered would
    /// block the worker thread — `take()` makes that impossible by construction.
    completion: Option<SyncSender<InferenceResult<Stats>>>,
    /// The job's cancellation signal, observed once per round by [`step`]'s cancel sweep.
    cancel: CancelSignal,
    /// How long the job sat queued before admission, reported as the queue-wait stat on
    /// completion (see [`QUEUE_WAIT_STAT_NAME`]).
    queue_wait: Duration,
    /// Set by the cancel sweep so the retire sweep can report the right finish reason. (Reading
    /// `cancel` again at retire would do, but a signal fired between sweep and retire would then
    /// mislabel a normally-finished sequence.)
    cancelled: bool,
}

/// Stat name reported when a sequence is retired by cancellation rather than finishing.
pub const FINISH_REASON_STAT_NAME: &str = "Finish Reason";

/// The framework's in-flight sequence: the generic per-seq decode state plus the serving payload.
type JobSeq<S> = ActiveSeq<CacheOf<S>, JobMeta>;

impl<Server: BatchedInferenceServer + 'static> BatchingChannel<Server> {
    pub fn new() -> Self {
        Self::with_queue_depth(DEFAULT_MAX_QUEUE_DEPTH)
    }

    /// Build a channel with a specific backpressure bound (see [`DEFAULT_MAX_QUEUE_DEPTH`]).
    /// Production uses the default via [`new`](Self::new); tests use small depths to exercise
    /// shedding deterministically.
    pub fn with_queue_depth(max_queue_depth: usize) -> Self {
        Self {
            worker: Arc::new(Worker {
                inner: Mutex::new(None),
                pending: Arc::new(AtomicUsize::new(0)),
                max_queue_depth,
            }),
            _server: PhantomData,
        }
    }

    /// Build a channel whose worker is spawned immediately around a specific server instance.
    /// Test-only: lets tests configure capacity and decoder behavior (production uses the lazy
    /// `Server::default()` path via [`new`](Self::new)).
    #[cfg(test)]
    fn with_server(server: Server) -> Self {
        Self::with_server_and_depth(server, DEFAULT_MAX_QUEUE_DEPTH)
    }

    /// Test-only: a specific server instance AND a specific queue bound.
    #[cfg(test)]
    fn with_server_and_depth(server: Server, max_queue_depth: usize) -> Self {
        let channel = Self::with_queue_depth(max_queue_depth);
        *channel.worker.inner.lock().unwrap() =
            Some(Self::spawn_worker_with(server).expect("test worker should spawn"));
        channel
    }

    /// Number of jobs currently submitted but not yet admitted (observability; approximate under
    /// contention, see [`Worker::pending`]).
    pub fn queue_depth(&self) -> usize {
        self.worker.pending.load(Ordering::Relaxed)
    }

    /// Spawn the worker thread that owns the server and runs the continuous loop.
    fn spawn_worker() -> InferenceResult<WorkerInner> {
        Self::spawn_worker_with(Server::default())
    }

    /// Spawn the worker around a specific server instance. Production uses `Server::default()`;
    /// tests seed a configured server so capacity and behavior are controllable.
    ///
    /// Spawn failure is returned as an error so the caller fails synchronously — setting up
    /// channel state as if a worker existed and then panicking would leave every later caller
    /// parked forever on a worker that was never born.
    fn spawn_worker_with(seed: Server) -> InferenceResult<WorkerInner> {
        let (sender, receiver) = std::sync::mpsc::channel::<Command>();

        let handle = std::thread::Builder::new()
            .name("burn-lm-batching-worker".to_string())
            .spawn(move || {
                let mut server = seed;
                let mut queue: VecDeque<QueuedJob> = VecDeque::new();
                let mut active: Vec<JobSeq<Server>> = Vec::new();

                loop {
                    // PANIC BOUNDARY (failure ladder): one `catch_unwind` per loop ITERATION — not
                    // per server call (an iteration is the unit of state consistency: a panic
                    // anywhere mid-iteration leaves queue/active partially advanced, and we discard
                    // rather than repair), and not around the whole loop (the boundary must return
                    // control HERE so the intact `queue`/`active` locals can be failed over to
                    // their callers).
                    //
                    // `AssertUnwindSafe` is justified because nothing that crossed the boundary is
                    // reused after a panic: the server (the only state model code can have left
                    // half-mutated) is dropped when the thread exits, and `queue`/`active` are only
                    // read to send `WorkerDied` replies and then cleared. A fresh worker gets a
                    // fresh `Server::default()`.
                    let flow = catch_unwind(AssertUnwindSafe(|| {
                        worker_iteration(&mut server, &mut queue, &mut active, &receiver)
                    }));
                    match flow {
                        Ok(Flow::Continue) => {}
                        Ok(Flow::Shutdown) => break,
                        Err(payload) => {
                            // Panic: log the payload (the default panic hook's stderr line has no
                            // correlation to the `WorkerDied` replies callers are about to see),
                            // answer everyone (active AND queued) with `WorkerDied`, then let the
                            // thread die. The next command lazily respawns a fresh worker (see
                            // `sender`).
                            let message = payload
                                .downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| payload.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "<non-string panic payload>".to_string());
                            tracing::error!(
                                "batching worker panicked ({message}); failing {} active and {} \
                                 queued job(s) with WorkerDied",
                                active.len(),
                                queue.len()
                            );
                            fail_everything::<Server>(&mut queue, &mut active);
                            break;
                        }
                    }
                }
            })
            .map_err(|_| InferenceError::WorkerDied)?;

        Ok(WorkerInner { sender, handle })
    }

    /// Return a sender to the worker, spawning it on first use — and RESPAWNING it if the previous
    /// worker died (panicked iteration). A dead worker is detected by its finished thread handle;
    /// it is reaped (join is instant on a finished thread) and replaced with a fresh worker around
    /// a fresh `Server::default()`, so one panic never bricks the channel.
    fn sender(&self) -> InferenceResult<Sender<Command>> {
        let mut guard = self.worker.inner.lock().unwrap();
        if guard
            .as_ref()
            .is_some_and(|inner| inner.handle.is_finished())
        {
            if let Some(inner) = guard.take() {
                let _ = inner.handle.join();
            }
        }
        if guard.is_none() {
            *guard = Some(Self::spawn_worker()?);
        }
        Ok(guard.as_ref().expect("just spawned").sender.clone())
    }

    /// Whether the worker has been spawned, i.e. the channel has been used at least once.
    fn is_spawned(&self) -> bool {
        self.worker.inner.lock().unwrap().is_some()
    }

    /// Send a command (spawning the worker if needed) and block on its reply, mapping a dead
    /// worker to an error.
    fn request<T>(&self, make: impl FnOnce(SyncSender<T>) -> Command) -> Result<T, ()> {
        let (reply, rx) = std::sync::mpsc::sync_channel::<T>(1);
        self.sender()
            .map_err(|_| ())?
            .send(make(reply))
            .map_err(|_| ())?;
        rx.recv().map_err(|_| ())
    }

    /// Enqueue a job without waiting for it to complete. Returns the completion receiver so the
    /// caller can wait later (or drop it to fire-and-forget). This is the non-blocking entry point.
    ///
    /// BACKPRESSURE: sheds synchronously with [`InferenceError::Overloaded`] when the queue is at
    /// its bound — before spawning or waking anything — so an overloaded channel costs a rejected
    /// caller one atomic increment, not a parked thread.
    pub fn submit(&self, job: InferenceJob) -> InferenceResult<Receiver<InferenceResult<Stats>>> {
        // Increment-then-check: the previous value decides admission, so concurrent submitters
        // race for the remaining slots atomically. The bound stays approximate in a different
        // way: a job admitted by the worker right after this check briefly frees a slot the shed
        // decision didn't see (documented on `Worker::pending`).
        let pending = &self.worker.pending;
        if pending.fetch_add(1, Ordering::Relaxed) >= self.worker.max_queue_depth {
            pending.fetch_sub(1, Ordering::Relaxed);
            return Err(InferenceError::Overloaded);
        }
        // From here the slot is owned by the permit: every exit (send failure, worker drain,
        // admission) releases it on drop.
        let permit = QueuePermit(pending.clone());
        let (completion, rx) = std::sync::mpsc::sync_channel::<InferenceResult<Stats>>(1);
        self.sender()?
            .send(Command::Submit(QueuedJob {
                job,
                completion,
                enqueued_at: Instant::now(),
                permit,
            }))
            .map_err(|_| InferenceError::WorkerDied)?;
        Ok(rx)
    }
}

/// Loop-control outcome of one worker iteration.
enum Flow {
    Continue,
    Shutdown,
}

/// One iteration of the worker's continuous loop: park/drain commands, admit, step. Runs inside
/// the per-iteration panic boundary (see `spawn_worker_with`).
fn worker_iteration<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<QueuedJob>,
    active: &mut Vec<JobSeq<S>>,
    receiver: &Receiver<Command>,
) -> Flow {
    // Block for the next command only when there is genuinely no progress to make:
    // nothing active to advance AND nothing admittable right now. "Admittable" means a
    // job is queued and the server reports a free slot. Parking otherwise (e.g. while a
    // job is queued and a slot just freed up after a retire) would dead-lock: the queued
    // job's `Submit` was already drained into `queue`, so no further command is coming to
    // wake the `recv()` — the worker would sleep forever with admittable work in hand.
    // (Pre-existing latent bug: the old guard parked on `active.is_empty()` alone, which
    // only stayed live because a job's submit usually raced in after the previous one
    // retired; with a freed slot and a job already queued it hangs.) When `max_slots ==
    // 0` with a job queued, `can_admit` is false, so we still park instead of busy-spin.
    let can_admit = !queue.is_empty() && server.batch_capacity().max_slots > active.len();
    if active.is_empty() && !can_admit {
        match receiver.recv() {
            Ok(command) => {
                if handle_command(server, queue, active, command) {
                    return Flow::Shutdown;
                }
            }
            Err(_) => return Flow::Shutdown, // all senders dropped
        }
    }
    // Drain any further pending commands without blocking, so a burst of submissions is
    // fully enqueued before the next admit/step sweep. Admission then sees all ready
    // jobs together (deterministic batching) rather than one per iteration.
    while let Ok(command) = receiver.try_recv() {
        if handle_command(server, queue, active, command) {
            return Flow::Shutdown;
        }
    }

    // ADMISSION (backpressure): admit queued jobs while there is free capacity. A job
    // that does not fit stays at the front of the queue for a later iteration.
    admit(server, queue, active);

    // STEP (round-robin stub): advance every active sequence by one token, then retire
    // any that finished. Retiring frees a slot so the next iteration admits more.
    step(server, active);

    Flow::Continue
}

/// The panic fallout path: every active sequence gets its detok cursor flushed (trailing
/// held-back bytes reach the emitter) and a `WorkerDied` reply via the usual send-once `.take()`
/// discipline; every queued job is answered `WorkerDied` too, its queue permit released on drop.
/// Commands still buffered in the mpsc when the thread exits are dropped with the receiver, which
/// disconnects their reply senders — those callers also observe `WorkerDied`, and their permits
/// drop with the commands, so the depth counter cannot leak.
fn fail_everything<S: BatchedInferenceServer>(
    queue: &mut VecDeque<QueuedJob>,
    active: &mut Vec<JobSeq<S>>,
) {
    for seq in active.iter_mut() {
        flush_detok(&mut seq.extra);
        if let Some(completion) = seq.extra.completion.take() {
            let _ = completion.send(Err(InferenceError::WorkerDied));
        }
    }
    active.clear();
    for queued in queue.drain(..) {
        let _ = queued.completion.send(Err(InferenceError::WorkerDied));
    }
}

/// Handle a lifecycle/config command between loop iterations. Returns `true` on shutdown.
///
/// `Submit` is special: it does not run the job, it just enqueues it; the completion reply is sent
/// later by [`step`] when the sequence retires.
///
/// `active` is read-only context: `Unload` and `ClearState` are rejected while work is in flight
/// (see the policy comments on those arms).
fn handle_command<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<QueuedJob>,
    active: &[JobSeq<S>],
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
            // POLICY: unload is REJECTED while work is in flight (active sequences or queued
            // jobs). Commands drain between rounds, so an unload could otherwise land
            // mid-generation; the next round's `decoder()` would then silently reload the model
            // and resume in-flight sequences with per-seq caches built against the previous
            // instance — accidental semantics. Callers must wait out (or drain) in-flight work.
            let result = if active.is_empty() && queue.is_empty() {
                server.unload()
            } else {
                Err(InferenceError::Busy(active.len(), queue.len()))
            };
            let _ = reply.send(result);
        }
        Command::Submit(queued) => {
            queue.push_back(queued);
        }
        Command::ClearState(reply) => {
            // Same hazard as `Unload`: clearing model state under in-flight sequences would yank
            // shared state out from under their per-seq caches mid-generation, so it is likewise
            // rejected while work is in flight.
            let result = if active.is_empty() && queue.is_empty() {
                server.clear_state()
            } else {
                Err(InferenceError::Busy(active.len(), queue.len()))
            };
            let _ = reply.send(result);
        }
        Command::Shutdown => return true,
    }
    false
}

/// Admit queued jobs into the active set while there is free capacity (backpressure).
///
/// `batch_capacity().max_slots` is the server's reported concurrent-sequence budget. The engine
/// owns the active set, so "free" = that budget minus what is already active: a job is admitted
/// only while `active.len() < max_slots`. A job that does not fit stays queued for a later sweep
/// (which runs after a retire frees a slot), making admission continuous.
fn admit<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<QueuedJob>,
    active: &mut Vec<JobSeq<S>>,
) {
    while active.len() < server.batch_capacity().max_slots {
        let Some(QueuedJob {
            job,
            completion,
            enqueued_at,
            permit,
        }) = queue.pop_front()
        else {
            break;
        };
        // Whatever happens next — admitted or rejected — the job has left the queue: release its
        // depth slot so submitters see the freed capacity.
        drop(permit);
        let queue_wait = enqueued_at.elapsed();

        // Cancelled while queued: reply WITHOUT touching the model — no prefill, no slot. The
        // caller never received a token, so the reply is an error (unlike an in-flight cancel,
        // which retires with the stats of what was already streamed).
        if job.cancel.is_cancelled() {
            let _ = completion.send(Err(InferenceError::Cancelled));
            continue;
        }

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

        // The request's `max_tokens` can LOWER the server's generation cap but never raise it:
        // the server cap is a capacity/config bound the operator set, so a request asking for
        // more is clamped rather than trusted.
        let max_gen = match job.params.max_tokens {
            Some(requested) => requested.min(server.max_gen_tokens()),
            None => server.max_gen_tokens(),
        };

        active.push(ActiveSeq {
            cache,
            tokens,
            processed: 0,
            generated: 0,
            max_gen,
            finished: false,
            extra: JobMeta {
                emitter: job.emitter,
                detok: Utf8Buffer::new(),
                // Built once at admission, from THIS job's params merged over the server config,
                // so the sampler (and its RNG) persists across the sequence's rounds and two
                // concurrent jobs with different params sample independently.
                sampler: Some(server.next_token_sampler(&job.params)),
                completion: Some(completion),
                cancel: job.cancel,
                queue_wait,
                cancelled: false,
            },
        });
    }
}

/// Advance every active sequence by one token (round-robin) via the generic [`step_round`] core,
/// stream the new tokens to their job emitters, then retire finished sequences.
///
/// This is the serving driver's thin wrapper around the shared decode core: `step_round` owns the
/// forward → contract-check → sample → stop-check dance; the framework-specific work that stays
/// here is detokenizing/streaming each new token to its job emitter and signalling per-job
/// completion on retire.
fn step<S: BatchedInferenceServer>(server: &mut S, active: &mut Vec<JobSeq<S>>) {
    // Idle fast-path: nothing to advance, so do not touch the model. `decoder()` lazy-loads, so
    // borrowing it on an empty round would force-load the model from any lifecycle command — and
    // reload it in the very same loop iteration after a successful `Unload`.
    if active.is_empty() {
        return;
    }

    // CANCEL SWEEP (once per round, so a fired signal retires its sequence within one round): mark
    // cancelled sequences finished BEFORE stepping. `step_round` then skips them — no further
    // forward is spent on a client that is gone — and the retire sweep below handles them like any
    // other finished sequence: detok flushed first, completion fired exactly once.
    for seq in active.iter_mut() {
        if !seq.finished && seq.extra.cancel.is_cancelled() {
            seq.finished = true;
            seq.extra.cancelled = true;
        }
    }

    let stop_ids = server.stop_ids();

    // Borrow the decoder for the whole round. If the model is not loaded, retire every active
    // sequence with that error rather than panicking the worker.
    let outcomes = match server.decoder() {
        Ok(decoder) => {
            // Advance each sequence with ITS OWN sampler, taken out of `extra` for the duration of
            // the call (`step_round` borrows the sequence and the sampler separately).
            let mut outcomes = Vec::with_capacity(active.len());
            for index in 0..active.len() {
                let mut sampler = active[index]
                    .extra
                    .sampler
                    .take()
                    .expect("sampler is only taken for the duration of a step");
                let outcome = step_round(
                    decoder,
                    &mut active[index..index + 1],
                    &mut sampler,
                    &stop_ids,
                )
                .pop()
                .expect("one sequence in yields exactly one outcome");
                active[index].extra.sampler = Some(sampler);
                outcomes.push(outcome);
            }
            outcomes
        }
        Err(err) => {
            for seq in active.iter_mut() {
                flush_detok(&mut seq.extra);
                if let Some(completion) = seq.extra.completion.take() {
                    let _ = completion.send(Err(err.clone()));
                }
            }
            active.clear();
            return;
        }
    };

    // STREAM: for each advanced sequence, push the new token's raw bytes through its detok
    // cursor and emit whatever text is now complete (a stop token is not streamed; bytes that end
    // mid-character are held back for the next round). A per-sequence forward error goes to its
    // completion sender instead.
    for (seq, outcome) in active.iter_mut().zip(outcomes) {
        match outcome {
            StepOutcome::Stepped { token, is_stop, .. } => {
                if !is_stop {
                    let bytes = server.detokenize_bytes(&[token]);
                    if let Some(text) = seq.extra.detok.push(&bytes) {
                        seq.extra.emitter.completed(GeneratedItem::Text(text));
                    }
                }
            }
            StepOutcome::Failed(err) => {
                // Flush held-back bytes BEFORE the completion fires (mirrors the
                // `decoder()`-error path above), so the RETIRE invariant — trailing text always
                // reaches the emitter before completion — holds on this path too.
                flush_detok(&mut seq.extra);
                if let Some(completion) = seq.extra.completion.take() {
                    let _ = completion.send(Err(err));
                }
            }
            StepOutcome::Skipped => {}
        }
    }

    // RETIRE: drop finished sequences and signal completion, freeing capacity for admission.
    // Flushing the detok cursor FIRST covers every retire path through this sweep — stop token,
    // `max_gen` cap, empty prompt and forward failure (`step_round` marks failed sequences
    // finished) — so held-back trailing bytes always reach the emitter before completion fires.

    // The completion sender is `.take()`n at every send site, so a sequence retired by a forward
    // error (its `Err` already sent above) sends nothing here. That matters: the channel is a
    // bounded one-shot, so a second send with the `Err` still buffered would block the worker —
    // and the public `submit()` lets callers hold the receiver and recv much later — stalling
    // every other active sequence.
    active.retain_mut(|seq| {
        if seq.finished {
            flush_detok(&mut seq.extra);
            if let Some(completion) = seq.extra.completion.take() {
                let mut stats = Stats::new();
                stats
                    .entries
                    .insert(crate::stats::StatEntry::TokensCount(seq.generated));
                // Queue-wait observability: how long this job sat queued before admission.
                // Rendered as fixed seconds to match every other duration stat (see
                // `Stats::display_stats`); a `Named` entry rather than a new `StatEntry` variant
                // because nothing needs the raw `Duration` back out.
                stats.entries.insert(StatEntry::Named(
                    QUEUE_WAIT_STAT_NAME.to_string(),
                    format!("{:.2}s", seq.extra.queue_wait.as_secs_f64()),
                ));
                // An in-flight cancel still replies `Ok`: the caller already received real tokens,
                // and the finish-reason stat says why the stream stopped short. (Only the
                // cancelled path carries a finish reason today, so normally-finished sequences
                // keep their existing, byte-identical stats output.)
                if seq.extra.cancelled {
                    stats.entries.insert(StatEntry::Named(
                        FINISH_REASON_STAT_NAME.to_string(),
                        "Cancelled".to_string(),
                    ));
                }
                let _ = completion.send(Ok(stats));
            }
            false
        } else {
            true
        }
    });
}

/// Drain a retiring sequence's detok cursor into its emitter. At true end of stream a held-back
/// partial character can never complete, so lossy U+FFFD replacement is permitted here (and only
/// here) rather than silently dropping the bytes. Called on EVERY retire path: the [`step`]
/// retire sweep (stop token, `max_gen` cap, empty prompt, forward failure) and the
/// `decoder()`-error mass-retire.
fn flush_detok(meta: &mut JobMeta) {
    if let Some(text) = meta.detok.finish() {
        meta.emitter.completed(GeneratedItem::Text(text));
    }
}

/// The error every caller of a dead (or unspawnable) worker observes: the command channel or the
/// completion channel disconnected. A dedicated variant — not a repurposed `LoadError` — so HTTP
/// and CLI callers can tell "the worker died, retry" apart from a genuine model-loading failure.
fn worker_gone() -> InferenceError {
    InferenceError::WorkerDied
}

impl<Server: BatchedInferenceServer + 'static> InferenceChannel<Server>
    for BatchingChannel<Server>
{
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
    /// submit (enqueue) then wait on the per-sequence completion signal. `Overloaded` (the queue
    /// bound) propagates synchronously from `submit`, before this blocks.
    fn run_job(&self, job: InferenceJob) -> InferenceResult<Stats> {
        let rx = self.submit(job)?;
        rx.recv().map_err(|_| worker_gone())?
    }

    fn clear_state(&self) -> InferenceResult<()> {
        self.request(Command::ClearState)
            .map_err(|_| worker_gone())?
    }

    /// Advisory: whether a submit right now would shed with `Overloaded`. Used by HTTP streaming
    /// as a pre-flight check before committing SSE headers (a 429 can only be sent before the 200).
    fn is_overloaded(&self) -> bool {
        self.queue_depth() >= self.worker.max_queue_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batching::{BatchCapacity, BatchedDecoder, ForwardBatch, ForwardOutput},
        job::{GenerationParams, InferenceJob, InferenceTask},
        sampler::NextTokenSampler,
        server::{InferenceServer, ServerConfigParsing},
        InferenceServerConfig, TextGenerationListener, INFERENCE_DEVICE,
    };
    use burn::tensor::{Int, Tensor, TensorData};
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
        /// Per-forward sleep — simulates a slow model so a test can observe a job in flight.
        step_delay_ms: u64,
        /// When set, `forward` PANICS at this per-sequence step — simulates a model bug that
        /// unwinds the worker iteration (the failure-ladder rung above per-sequence errors).
        panic_at_step: Option<usize>,
    }

    const VOCAB: usize = 64;

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
    struct FixedSampler(u32);

    impl NextTokenSampler for FixedSampler {
        fn sample_next(&mut self, _logits: Tensor<2>) -> Tensor<2, Int> {
            Tensor::from_data(
                TensorData::new(vec![self.0 as i32], [1, 1]),
                &*INFERENCE_DEVICE,
            )
        }
    }

    #[derive(Debug, Clone)]
    struct FakeServer {
        loaded: bool,
        slots: usize,
        decoder: FakeDecoder,
        /// Counts `batch_capacity` calls — lets the `max_slots == 0` test detect a busy-spin.
        capacity_calls: Arc<AtomicUsize>,
        /// When set, `next_token_sampler` returns a [`FixedSampler`] for this token instead of the
        /// default argmax — stands in for a server with non-greedy sampling config. A job whose
        /// params carry a temperature overrides this with `temperature as u32` (the merge-over-
        /// config behavior a real server implements), so tests can observe per-request samplers.
        fixed_token: Option<u32>,
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
                    step_delay_ms: 0,
                    panic_at_step: None,
                },
                capacity_calls: Arc::new(AtomicUsize::new(0)),
                fixed_token: None,
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
                    step_delay_ms: 0,
                    panic_at_step: None,
                },
                capacity_calls: Arc::new(AtomicUsize::new(0)),
                fixed_token: None,
            }
        }

        /// A server whose decoder emits many tokens, each after a small sleep — a long-running job
        /// a test can interrogate (e.g. unload) while it is demonstrably still in flight.
        fn new_slow(slots: usize, log: OrderLog) -> Self {
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
        fn new_panicky(slots: usize, log: OrderLog, panic_at_step: usize) -> Self {
            let mut server = Self::new(slots, log);
            server.decoder.step_delay_ms = 20;
            server.decoder.panic_at_step = Some(panic_at_step);
            server
        }

        /// A server whose `next_token_sampler` always picks `token`, regardless of logits —
        /// observably different from the default argmax (which would echo the identity token).
        fn with_fixed_sampler(mut self, token: u32) -> Self {
            self.fixed_token = Some(token);
            self
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

        fn next_token_sampler(
            &self,
            params: &GenerationParams,
        ) -> Box<dyn NextTokenSampler + Send> {
            // Request params merged over server config, like the real servers: a per-job
            // temperature wins over the server-level `fixed_token`.
            match params.temperature.map(|t| t as u32).or(self.fixed_token) {
                Some(token) => Box::new(FixedSampler(token)),
                None => Box::new(crate::sampler::Sampler::default()),
            }
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
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
            1,
            Arc::new(Mutex::new(Vec::new())),
        ));

        // Job A: its listener panics on the first emitted token (broken pipe).
        let (job_a, _ha) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            PanicOnText,
        );
        let rx_a = channel.submit(job_a).unwrap();
        let _ = rx_a.recv(); // A's own outcome is irrelevant; its listener died.

        // Job B: a healthy client must still be served — proving the worker survived A.
        let (job_b, _hb) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            NullListener,
        );
        let rx_b = channel.submit(job_b).unwrap();
        rx_b.recv()
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

        let (job1, _h1) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let out1 = channel
            .submit(job1)
            .unwrap()
            .recv()
            .expect("worker must survive");
        assert!(
            matches!(out1, Err(crate::InferenceError::BatchContractViolation(_))),
            "contract violation should retire the sequence with a BatchContractViolation error"
        );

        // The worker is still alive: a second job is accepted and processed (likewise retired).
        let (job2, _h2) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            NullListener,
        );
        let out2 = channel
            .submit(job2)
            .expect("worker must still accept jobs")
            .recv()
            .expect("worker must survive");
        assert!(out2.is_err(), "second job should also retire with an error");
    }

    /// Completion must be sent EXACTLY ONCE per job. Before the fix, a failed forward sent `Err`
    /// on the completion channel (filling the bounded one-shot) and the retire sweep then sent
    /// `Ok` AGAIN for the same sequence — blocking the worker until the caller drained the first
    /// message. We prove the worker stays live by NOT recv-ing the first job's completion and
    /// asserting a second job still completes.
    #[test]
    fn failed_sequence_completes_exactly_once_without_blocking_the_worker() {
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_bad(
            1,
            Arc::new(Mutex::new(Vec::new())),
        ));

        // Job 1 fails its forward; its `Err` sits buffered in the one-shot because we don't recv.
        // A double send would now block the worker on the buffered channel.
        let (job1, _h1) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let rx1 = channel.submit(job1).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Job 2 must still be served — proving the worker did not block on a second send.
        let (job2, _h2) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            NullListener,
        );
        let rx2 = channel.submit(job2).unwrap();
        let _ = rx2
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("worker blocked: completion was sent twice for the failed sequence");

        // The first receiver yields exactly one message (the `Err`), then disconnects — no
        // buffered second message.
        assert!(
            rx1.recv().expect("first completion must arrive").is_err(),
            "failed sequence should complete with the forward error"
        );
        assert!(
            rx1.recv().is_err(),
            "completion channel should disconnect after exactly one message"
        );
    }

    /// Unload (and clear-state) while work is in flight must be REJECTED, not silently reload the
    /// model under in-flight per-seq caches. Once the job retires, unload succeeds.
    #[test]
    fn unload_is_rejected_while_a_job_is_in_flight() {
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_slow(
            1,
            Arc::new(Mutex::new(Vec::new())),
        ));

        // ~16 steps × 20ms ⇒ the job is comfortably still running 50ms in.
        let (job, _h) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let rx = channel.submit(job).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));

        assert!(
            matches!(channel.unload(), Err(InferenceError::Busy(_, _))),
            "unload must be rejected while a sequence is active"
        );

        // After the job completes the active set is empty, so unload succeeds.
        rx.recv().unwrap().unwrap();
        channel.unload().expect("unload should succeed once idle");
    }

    /// A server reporting `max_slots == 0` with a job queued must PARK the worker, not busy-spin a
    /// core. We detect a spin via the `batch_capacity` call count: parked ⇒ a couple of calls;
    /// spinning ⇒ thousands over the same window.
    #[test]
    fn max_slots_zero_parks_instead_of_spinning() {
        let calls = Arc::new(AtomicUsize::new(0));
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::with_capacity_probe(
            0,
            calls.clone(),
        ));

        // Queued, but never admittable while max_slots == 0.
        let (job, _h) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let _rx = channel.submit(job).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        let n = calls.load(Ordering::Relaxed);
        assert!(
            n < 100,
            "worker busy-spun on max_slots==0 (batch_capacity called {n} times in 50ms); it should park"
        );
    }

    fn submit_two(slots: usize) -> Vec<usize> {
        let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
        let channel =
            BatchingChannel::<FakeServer>::with_server(FakeServer::new(slots, log.clone()));

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

    /// The worker must sample with the server-configured sampler (the `next_token_sampler`
    /// primitive), not a hard-coded argmax. The fixed sampler always picks token 7 regardless of
    /// logits; argmax over the fake decoder's logits would instead echo the identity token (10)
    /// and then stop. Since 7 is never a stop id, the sequence runs to `max_gen_tokens` (16) and
    /// streams sixteen "7"s — unmistakably the configured sampler's output.
    #[test]
    fn worker_uses_the_server_configured_sampler() {
        let channel = BatchingChannel::<FakeServer>::with_server(
            FakeServer::new(1, Arc::new(Mutex::new(Vec::new()))).with_fixed_sampler(7),
        );

        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        channel.submit(job).unwrap().recv().unwrap().unwrap();

        assert_eq!(
            handle.join(),
            "7".repeat(16),
            "emitted text must come from the server's configured sampler, not argmax"
        );
    }

    /// A decoder that emits a fixed token script (one token per step), then the stop id. Used by
    /// the byte-level detok tests, where tokens ARE byte values and the script deliberately
    /// splits a multi-byte UTF-8 character across steps.
    #[derive(Debug, Clone, Default)]
    struct ScriptedDecoder {
        script: Vec<u32>,
        /// When set to `(n, signal)`, fires `signal` while producing the n-th generated token —
        /// a deterministic "client disconnected mid-generation" event, with no sleeps to race.
        cancel_after: Option<(usize, CancelSignal)>,
    }

    /// Token ids 0..=255 are the raw bytes; 256 is the stop id.
    const BYTE_STOP: u32 = 256;
    const BYTE_VOCAB: usize = 257;

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
    struct ByteServer {
        loaded: bool,
        decoder: ScriptedDecoder,
        max_gen: usize,
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
        fn new(script: Vec<u32>, max_gen: usize) -> Self {
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
        fn with_cancel_after(mut self, after: usize, cancel: CancelSignal) -> Self {
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

    /// REGRESSION (the live panic-class bug): a multi-byte character split across tokens must
    /// stream as the complete character — exactly once, no U+FFFD, no panic — and the worker must
    /// survive to serve a second job. Before S1 the worker decoded each token to TEXT in
    /// isolation; the first half of the emoji failed that decode (`ByteServer::detokenize`
    /// panics, like `Tiktoken::decode`'s `.expect`) and the panic killed the channel permanently.
    #[test]
    fn split_multibyte_character_streams_intact_and_worker_survives() {
        // 🦀 is 4 bytes, one per token, followed by an ASCII '!'.
        let mut script: Vec<u32> = "🦀".bytes().map(u32::from).collect();
        script.push(b'!' as u32);
        let channel = BatchingChannel::<ByteServer>::with_server(ByteServer::new(script, 16));

        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        channel.submit(job).unwrap().recv().unwrap().unwrap();

        let text = handle.join();
        assert_eq!(text, "🦀!", "split emoji must reassemble exactly once");
        assert!(
            !text.contains('\u{FFFD}'),
            "no mid-stream replacement chars"
        );

        // The worker survived (no panic on the partial-character tokens): a second job completes.
        let (job2, h2) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        channel
            .submit(job2)
            .expect("worker must still accept jobs")
            .recv()
            .expect("worker must survive a split multi-byte character")
            .unwrap();
        assert_eq!(h2.join(), "🦀!");
    }

    /// A sequence retired with held-back bytes (here: the `max_gen` cap lands mid-character) must
    /// FLUSH its detok cursor — the trailing bytes reach the listener (lossily, U+FFFD is
    /// permitted at true end of stream) instead of being silently dropped, and completion fires.
    #[test]
    fn retire_flushes_trailing_partial_character() {
        // Only the first 2 of 🦀's 4 bytes fit under max_gen == 2.
        let script: Vec<u32> = "🦀".bytes().map(u32::from).collect();
        let channel = BatchingChannel::<ByteServer>::with_server(ByteServer::new(script, 2));

        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        channel.submit(job).unwrap().recv().unwrap().unwrap();

        assert_eq!(
            handle.join(),
            "\u{FFFD}",
            "held-back bytes must flush on retire (lossy replacement at end of stream)"
        );
    }

    /// A job cancelled while still QUEUED must never be admitted: no prefill (observable as zero
    /// forwards for its identity in the order log), no slot, and an `Err(Cancelled)` reply — the
    /// caller never received a token, so an empty `Ok` would be misleading.
    #[test]
    fn cancelled_while_queued_is_never_admitted_and_replies_cancelled() {
        let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
        // One slot + a slow decoder: job A occupies the slot long enough for B to sit queued.
        let channel =
            BatchingChannel::<FakeServer>::with_server(FakeServer::new_slow(1, log.clone()));

        let (job_a, _ha) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let rx_a = channel.submit(job_a).unwrap();

        // B is cancelled BEFORE it is submitted, so the admission check must catch it.
        let (job_b, hb) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            NullListener,
        );
        hb.cancel();
        let rx_b = channel.submit(job_b).unwrap();

        assert!(
            matches!(rx_b.recv().unwrap(), Err(InferenceError::Cancelled)),
            "a job cancelled while queued must reply Err(Cancelled)"
        );
        rx_a.recv().unwrap().unwrap();

        // B (identity 1 in the log) never reached the decoder: it was dropped before prefill.
        assert!(
            !log.lock().unwrap().contains(&1),
            "cancelled-while-queued job must never be prefilled"
        );
    }

    /// A cancel fired MID-FLIGHT must retire the sequence within one round: held-back detok bytes
    /// are flushed before completion, the reply is `Ok` (the client already got real tokens) with
    /// the tokens generated so far and a finish-reason stat.
    #[test]
    fn cancel_mid_flight_retires_within_one_round_and_flushes_detok() {
        // 🦀 is 4 bytes, one per token; the cancel fires while the 2nd byte is produced, so the
        // detok cursor holds a partial character at retire time.
        let mut script: Vec<u32> = "🦀".bytes().map(u32::from).collect();
        script.extend(std::iter::repeat(b'x' as u32).take(12));

        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        let channel = BatchingChannel::<ByteServer>::with_server(
            ByteServer::new(script, 16).with_cancel_after(2, handle.cancel_signal()),
        );

        let stats = channel.submit(job).unwrap().recv().unwrap().unwrap();

        // Retired within ONE round of the cancel: the 2nd token's round still streams normally,
        // the next round's cancel sweep retires before any 3rd forward.
        assert!(
            stats.entries.contains(&StatEntry::TokensCount(2)),
            "expected exactly the 2 tokens generated before the cancel: {:?}",
            stats.entries
        );
        assert!(
            stats.entries.contains(&StatEntry::Named(
                FINISH_REASON_STAT_NAME.to_string(),
                "Cancelled".to_string()
            )),
            "an in-flight cancel must report its finish reason: {:?}",
            stats.entries
        );
        // The held-back partial character was flushed (lossily, as at any true end of stream)
        // BEFORE completion fired — not silently dropped.
        assert_eq!(handle.join(), "\u{FFFD}");
    }

    /// Two CONCURRENT jobs with different per-request params must sample with independently
    /// configured samplers — the request params merged over config at admission, not a shared
    /// mutated server config (which would make one request clobber the other's temperature).
    #[test]
    fn concurrent_jobs_sample_with_their_own_request_params() {
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
            2,
            Arc::new(Mutex::new(Vec::new())),
        ));

        // The fake's `next_token_sampler` turns a per-job temperature into a fixed token, so each
        // job's whole output reveals which params built its sampler.
        let hot = GenerationParams {
            temperature: Some(7.0),
            ..Default::default()
        };
        let cold = GenerationParams {
            temperature: Some(9.0),
            ..Default::default()
        };
        let (job_a, ha) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            hot,
            TextGenerationListener::default(),
        );
        let (job_b, hb) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            cold,
            TextGenerationListener::default(),
        );

        // Both in flight together (capacity 2), interleaving round-robin.
        let rx_a = channel.submit(job_a).unwrap();
        let rx_b = channel.submit(job_b).unwrap();
        rx_a.recv().unwrap().unwrap();
        rx_b.recv().unwrap().unwrap();

        assert_eq!(ha.join(), "7".repeat(16), "job A must use its own params");
        assert_eq!(hb.join(), "9".repeat(16), "job B must use its own params");
    }

    /// A request's `max_tokens` can LOWER the server's generation cap but never RAISE it: the
    /// operator-set server cap stays authoritative.
    #[test]
    fn request_max_tokens_lowers_but_cannot_raise_the_server_cap() {
        let channel = BatchingChannel::<FakeServer>::with_server(
            FakeServer::new(1, Arc::new(Mutex::new(Vec::new()))).with_fixed_sampler(7),
        );

        // Below the server cap (16): the request wins.
        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams {
                max_tokens: Some(4),
                ..Default::default()
            },
            TextGenerationListener::default(),
        );
        channel.submit(job).unwrap().recv().unwrap().unwrap();
        assert_eq!(handle.join(), "7".repeat(4));

        // Above the server cap: clamped to the cap.
        let (job, handle) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams {
                max_tokens: Some(100),
                ..Default::default()
            },
            TextGenerationListener::default(),
        );
        channel.submit(job).unwrap().recv().unwrap().unwrap();
        assert_eq!(handle.join(), "7".repeat(16));
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

        // Lifecycle traffic on an idle worker must not (re)load the model: `step` skips the
        // decoder when nothing is active (the fake's `decoder()` lazy-loads like the real ones).
        let _ = channel.is_downloaded();
        assert!(!channel.is_loaded());
    }

    /// BACKPRESSURE: at the queue bound, `submit` must shed synchronously with `Overloaded` — and
    /// a rejected submit must not leak its depth-counter slot. `max_slots == 0` keeps the queued
    /// job pinned (nothing is ever admitted), so the bound is exercised without timing races.
    #[test]
    fn submit_sheds_with_overloaded_at_the_queue_bound() {
        let channel = BatchingChannel::<FakeServer>::with_server_and_depth(
            FakeServer::with_capacity_probe(0, Arc::new(AtomicUsize::new(0))),
            1,
        );
        assert_eq!(channel.queue_depth(), 0, "queue starts empty");
        assert!(!channel.is_overloaded());

        let (job_a, _ha) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let _rx_a = channel
            .submit(job_a)
            .expect("first job fits the depth-1 queue");
        assert_eq!(channel.queue_depth(), 1);
        assert!(
            channel.is_overloaded(),
            "advisory probe must report a full queue"
        );

        let (job_b, _hb) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            NullListener,
        );
        assert!(
            matches!(channel.submit(job_b), Err(InferenceError::Overloaded)),
            "a submit beyond the bound must shed synchronously"
        );
        assert_eq!(
            channel.queue_depth(),
            1,
            "a shed submit must release its counter slot"
        );
    }

    /// FAILURE LADDER, top rung: a panic inside the worker loop must reply `WorkerDied` — exactly
    /// once — to the active job AND every queued job (text streamed before the panic still reaches
    /// the listener via the detok flush), release their depth slots, and the NEXT submission must
    /// lazily respawn a fresh worker. One panic never bricks the channel. (The scripted panic's
    /// stderr backtrace is expected.)
    #[test]
    fn worker_panic_fails_active_and_queued_with_workerdied_then_respawns() {
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new_panicky(
            1,
            Arc::new(Mutex::new(Vec::new())),
            2, // panic on the 3rd step: 2 tokens stream first, ~20ms apart
        ));

        // A is admitted (one slot); B is surely queued behind it before the ~60ms panic.
        let (job_a, ha) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            TextGenerationListener::default(),
        );
        let (job_b, _hb) = InferenceJob::create(
            InferenceTask::Prompt("b".into()),
            GenerationParams::default(),
            NullListener,
        );
        let rx_a = channel.submit(job_a).unwrap();
        let rx_b = channel.submit(job_b).unwrap();

        assert!(
            matches!(rx_a.recv().unwrap(), Err(InferenceError::WorkerDied)),
            "the in-flight job must observe WorkerDied"
        );
        assert!(
            matches!(rx_b.recv().unwrap(), Err(InferenceError::WorkerDied)),
            "queued jobs must observe WorkerDied too"
        );
        // Exactly one reply each: the channels disconnect after the WorkerDied.
        assert!(rx_a.recv().is_err());
        assert!(rx_b.recv().is_err());
        // The two tokens generated before the panic were streamed (and the detok cursor flushed)
        // before A's completion fired.
        assert_eq!(ha.join(), "1010");
        // Both queue permits were released in the panic fallout.
        assert_eq!(channel.queue_depth(), 0);

        // LAZY RESPAWN: the next submit detects the finished worker thread and spawns a fresh one
        // around a fresh `Server::default()` (unloaded; admission lazy-loads it).
        let (job_c, _hc) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        channel
            .submit(job_c)
            .expect("a dead worker must be respawned on the next submit")
            .recv()
            .expect("the respawned worker must serve")
            .expect("the post-panic job should complete normally");
    }

    /// The queue-wait stat: every completed job reports how long it sat queued before admission,
    /// in the same fixed-seconds rendering as the other duration stats.
    #[test]
    fn completion_stats_include_the_queue_wait() {
        let channel = BatchingChannel::<FakeServer>::with_server(FakeServer::new(
            1,
            Arc::new(Mutex::new(Vec::new())),
        ));

        let (job, _h) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        let stats = channel.submit(job).unwrap().recv().unwrap().unwrap();

        assert!(
            stats.entries.iter().any(|entry| matches!(
                entry,
                StatEntry::Named(name, value)
                    if name == QUEUE_WAIT_STAT_NAME && value.ends_with('s')
            )),
            "completion stats must carry a fixed-seconds queue-wait entry: {:?}",
            stats.entries
        );
    }

    /// STRESS, exactly-one-reply invariant: a concurrent burst against a depth-2 queue produces
    /// mixed outcomes — completed, shed with a synchronous `Overloaded`, cancelled while queued —
    /// and EVERY submission resolves exactly once (no hang, no second reply). The mix is seeded
    /// deterministically; the actual interleaving comes from real thread scheduling.
    #[test]
    fn every_submission_resolves_exactly_once_under_overload() {
        let server = FakeServer {
            loaded: false,
            slots: 1,
            decoder: FakeDecoder {
                log: Arc::new(Mutex::new(Vec::new())),
                emit: 2,
                extra_rows: 0,
                step_delay_ms: 2, // slow enough that submissions genuinely pile up
                panic_at_step: None,
            },
            capacity_calls: Arc::new(AtomicUsize::new(0)),
            fixed_token: None,
        };
        let channel = BatchingChannel::<FakeServer>::with_server_and_depth(server, 2);

        let handles: Vec<_> = (0..16)
            .map(|i| {
                let channel = channel.clone();
                std::thread::spawn(move || {
                    let (job, handle) = InferenceJob::create(
                        InferenceTask::Prompt(if i % 2 == 0 { "a" } else { "b" }.into()),
                        GenerationParams::default(),
                        NullListener,
                    );
                    if i % 3 == 0 {
                        handle.cancel(); // a third of the jobs cancel before/while queued
                    }
                    match channel.submit(job) {
                        // Shed: resolved synchronously, nothing to wait on.
                        Err(InferenceError::Overloaded) => {}
                        Err(other) => panic!("unexpected submit error: {other:?}"),
                        Ok(rx) => {
                            match rx.recv().expect("every accepted job must get a reply") {
                                Ok(_) | Err(InferenceError::Cancelled) => {}
                                Err(other) => panic!("unexpected completion: {other:?}"),
                            }
                            assert!(rx.recv().is_err(), "a job must reply exactly once");
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no submitter may hang or panic");
        }

        // Every permit was released on its job's way out, however it resolved.
        assert_eq!(channel.queue_depth(), 0);

        // And the channel is still healthy: a final job completes normally.
        let (job, _h) = InferenceJob::create(
            InferenceTask::Prompt("a".into()),
            GenerationParams::default(),
            NullListener,
        );
        channel
            .submit(job)
            .expect("queue must be empty again")
            .recv()
            .unwrap()
            .unwrap();
    }
}

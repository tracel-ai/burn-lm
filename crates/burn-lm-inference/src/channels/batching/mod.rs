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
//!
//! Layout: `mod.rs` = caller-side facade and protocol types (`BatchingChannel`, `Command`,
//! queue/permit/worker structs); `worker.rs` = the engine (spawn, loop, admit, step, retire);
//! `tests.rs` = the test suite.

use std::{
    fmt,
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, Sender, SyncSender},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::Instant,
};

use crate::{
    batching::BatchedInferenceServer,
    errors::{InferenceError, InferenceResult},
    InferenceJob, Stats,
};

use super::InferenceChannel;

mod worker;
pub use worker::FINISH_REASON_STAT_NAME;

#[cfg(test)]
mod tests;

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
            Some(worker::spawn(server).expect("test worker should spawn"));
        channel
    }

    /// Number of jobs currently submitted but not yet admitted (observability; approximate under
    /// contention, see [`Worker::pending`]).
    pub fn queue_depth(&self) -> usize {
        self.worker.pending.load(Ordering::Relaxed)
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
            *guard = Some(worker::spawn(Server::default())?);
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

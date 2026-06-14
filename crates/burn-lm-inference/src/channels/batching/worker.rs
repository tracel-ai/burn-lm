//! The worker engine: the long-lived thread that owns the server and runs the continuous
//! batching loop. See `mod.rs` for the caller-side facade and protocol types.

use std::{
    collections::VecDeque,
    panic::{catch_unwind, AssertUnwindSafe},
    time::Duration,
};

use crate::{
    batching::{
        sample_token, step_round, ActiveSeq, BatchedDecoder, BatchedInferenceServer, PrefillBudget,
        StepOutcome,
    },
    errors::{InferenceError, InferenceResult},
    job::CancelSignal,
    sampler::NextTokenSampler,
    utf8::Utf8Buffer,
    GeneratedItem, GeneratedItemEmitter, Stats,
};

use super::{Command, QueuedJob, WorkerInner};

/// Serving-driver payload attached to a generic [`ActiveSeq`]: where a sequence's text is streamed
/// and the one-shot completion signal fired when it retires. The generic decode core
/// ([`step_round`]) never touches this — it advances the [`ActiveSeq`]'s tokens/counters and
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
    completion: Option<std::sync::mpsc::SyncSender<InferenceResult<Stats>>>,
    /// The job's cancellation signal, observed once per round by [`step`]'s cancel sweep.
    cancel: CancelSignal,
    /// How long the job sat queued before admission, reported as the queue-wait stat on
    /// completion (see [`super::QUEUE_WAIT_STAT_NAME`]).
    queue_wait: Duration,
    /// Set by the cancel sweep so the retire sweep can report the right finish reason. (Reading
    /// `cancel` again at retire would do, but a signal fired between sweep and retire would then
    /// mislabel a normally-finished sequence.)
    cancelled: bool,
}

/// Stat name reported when a sequence is retired by cancellation rather than finishing.
pub const FINISH_REASON_STAT_NAME: &str = "Finish Reason";

/// The framework's in-flight sequence: the generic per-seq decode state plus the serving payload.
type JobSeq = ActiveSeq<JobMeta>;

/// Loop-control outcome of one worker iteration.
enum Flow {
    Continue,
    Shutdown,
}

/// Spawn the worker thread that owns the server and runs the continuous loop around `seed`.
/// Production calls this with `Server::default()`; tests seed a configured server so capacity
/// and behavior are controllable.
///
/// Spawn failure is returned as an error so the caller fails synchronously — setting up
/// channel state as if a worker existed and then panicking would leave every later caller
/// parked forever on a worker that was never born.
pub(super) fn spawn<S: BatchedInferenceServer + 'static>(seed: S) -> InferenceResult<WorkerInner> {
    let (sender, receiver) = std::sync::mpsc::channel::<Command>();

    let handle = std::thread::Builder::new()
        .name("burn-lm-batching-worker".to_string())
        .spawn(move || {
            let mut server = seed;
            let mut queue: VecDeque<QueuedJob> = VecDeque::new();
            let mut active: Vec<JobSeq> = Vec::new();

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
                        fail_everything(&mut queue, &mut active);
                        break;
                    }
                }
            }
        })
        .map_err(|_| InferenceError::WorkerDied)?;

    Ok(WorkerInner { sender, handle })
}

/// One iteration of the worker's continuous loop: park/drain commands, admit, step. Runs inside
/// the per-iteration panic boundary (see `spawn`).
fn worker_iteration<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<QueuedJob>,
    active: &mut Vec<JobSeq>,
    receiver: &std::sync::mpsc::Receiver<Command>,
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

    // STEP: advance the round — ≤1 prefill plus one fused decode over every decoding sequence —
    // then retire any that finished. Retiring frees a slot so the next iteration admits more.
    step(server, active);

    Flow::Continue
}

/// The panic fallout path: every active sequence gets its detok cursor flushed (trailing
/// held-back bytes reach the emitter) and a `WorkerDied` reply via the usual send-once `.take()`
/// discipline; every queued job is answered `WorkerDied` too, its queue permit released on drop.
/// Commands still buffered in the mpsc when the thread exits are dropped with the receiver, which
/// disconnects their reply senders — those callers also observe `WorkerDied`, and their permits
/// drop with the commands, so the depth counter cannot leak. No slots are released here: the
/// server (and with it every slot's cache inside the decoder) is dropped when the thread exits,
/// and the respawned worker starts with a fresh server.
fn fail_everything(queue: &mut VecDeque<QueuedJob>, active: &mut Vec<JobSeq>) {
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
    active: &[JobSeq],
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
            // and resume in-flight sequences whose slot caches died with the previous instance —
            // accidental semantics. Callers must wait out (or drain) in-flight work.
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
            // their slot caches out from under them mid-generation, so it is likewise
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
    active: &mut Vec<JobSeq>,
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

        // Borrow the decoder first: this loads the model if needed, so the subsequent
        // `tokenize`/`detokenize` primitives can rely on the tokenizer being available.
        if let Err(err) = server.decoder().map(|_| ()) {
            let _ = completion.send(Err(err));
            continue;
        }

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

        // Assign the lowest free slot. The engine owns the free-slot list: slots are
        // `0..max_slots`, and a slot is free iff no active sequence holds it. The admission
        // guard above guarantees one is free.
        let slot = (0..server.batch_capacity().max_slots)
            .find(|candidate| active.iter().all(|seq| seq.slot != *candidate))
            .expect("admission only runs while a slot is free");

        // Release the slot before handing it to the new sequence (added after a safety review).
        // The retire sweep normally already did this — releasing twice is a documented no-op —
        // but doing it here too means one prompt's leftovers can never be resumed by the next,
        // even if some retire path or a model author's prefill-error rollback forgot. The one
        // thing isolation rests on is the simplest method in the trait: `release(slot)` drops
        // that slot's state. (The decoder borrow cannot fail here; it was checked just above.)
        if let Ok(decoder) = server.decoder() {
            decoder.release(slot);
        }

        active.push(ActiveSeq {
            slot,
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

/// Advance the round (one fused decode over all decoding sequences, plus ≤1 prefill) via the generic
/// [`step_round`] core, stream the new tokens to their job emitters, then retire finished sequences.
///
/// This is the serving driver's thin wrapper around the shared decode core: `step_round` owns the
/// forward → contract-check → sample → stop-check dance; the framework-specific work that stays
/// here is detokenizing/streaming each new token to its job emitter and signalling per-job
/// completion on retire.
fn step<S: BatchedInferenceServer>(server: &mut S, active: &mut Vec<JobSeq>) {
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
            // ONE prefill budget for the whole round, computed over the FULL active set, so a long
            // prompt cannot stall the in-flight decoders for more than one round.
            let mut budget = PrefillBudget::for_round(active);
            // Each in-flight request has its OWN sampler (a seeded RNG persists across its tokens).
            // Take them all out of `extra` into a parallel Vec so the per-row sampling closure can
            // index them while `step_round` holds `active` borrowed; restore them after the round.
            let mut samplers: Vec<Box<dyn NextTokenSampler + Send>> = active
                .iter_mut()
                .map(|seq| {
                    seq.extra
                        .sampler
                        .take()
                        .expect("sampler is only taken for the duration of a round")
                })
                .collect();
            let outcomes = step_round(decoder, active, &stop_ids, &mut budget, |index, logits| {
                sample_token(logits, samplers[index].as_mut())
            });
            for (seq, sampler) in active.iter_mut().zip(samplers) {
                seq.extra.sampler = Some(sampler);
            }
            outcomes
        }
        Err(err) => {
            // Mass-retire: the decoder could not even be borrowed, so there is nothing to
            // `release` slots into — the caches live inside the decoder, and a later fresh load
            // starts with every slot empty. Clearing `active` is what frees the slots here.
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
    //
    // Every retire through this sweep — stop token, `max_gen` cap, empty prompt, forward failure,
    // cancellation — frees its decoder slot: the slot numbers are collected here and released
    // below, after the sweep, so the slot's cache is dropped and the next admitted sequence
    // starts clean.
    let mut freed_slots = Vec::new();
    active.retain_mut(|seq| {
        if seq.finished {
            freed_slots.push(seq.slot);
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
                stats.entries.insert(crate::StatEntry::Named(
                    super::QUEUE_WAIT_STAT_NAME.to_string(),
                    format!("{:.2}s", seq.extra.queue_wait.as_secs_f64()),
                ));
                // An in-flight cancel still replies `Ok`: the caller already received real tokens,
                // and the finish-reason stat says why the stream stopped short. (Only the
                // cancelled path carries a finish reason today, so normally-finished sequences
                // keep their existing, byte-identical stats output.)
                if seq.extra.cancelled {
                    stats.entries.insert(crate::StatEntry::Named(
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

    // Release the retired sequences' slots so the decoder drops their caches. The decoder was
    // borrowable at the top of this round, so a failure here is unreachable in practice; if it
    // does fail, the slots are free engine-side anyway and a fresh decoder load starts empty.
    if !freed_slots.is_empty() {
        if let Ok(decoder) = server.decoder() {
            for slot in freed_slots {
                decoder.release(slot);
            }
        }
    }
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

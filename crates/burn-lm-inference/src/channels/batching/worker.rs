//! The worker engine: the background thread that owns the model server and runs the continuous
//! batching loop. Requests reach it as commands over a channel; this thread admits them, drives
//! every in-flight sequence forward one token per round, and streams the results back out. The
//! caller-side facade and the command/protocol types it speaks live in `mod.rs`.

use std::{
    collections::VecDeque,
    panic::{catch_unwind, AssertUnwindSafe},
    time::Duration,
};

use crate::{
    batching::{
        step_round, ActiveSeq, BatchedDecoder, BatchedInferenceServer, PrefillBudget, StepOutcome,
    },
    errors::{InferenceError, InferenceResult},
    job::CancelSignal,
    utf8::Utf8Buffer,
    Stats,
};

use super::emission;
use super::{Command, QueuedJob, WorkerInner};

/// Everything the serving worker needs to remember about one in-flight request, kept next to the
/// generic decode state in its `ActiveSeq`. The shared decode core (`step_round`) is deliberately
/// model-agnostic — it only advances tokens and reports back — so all the serving-specific
/// machinery for turning those tokens into a streamed response lives here instead. A request picks
/// up a `JobMeta` when it is admitted, streams through it each round, and is retired from it when
/// it finishes.
struct JobMeta {
    /// The worker's monotonic key for this request. Slots are reused between requests; ids are
    /// not — the emission thread files this request's delivery state (emitter, detok cursor,
    /// completion) under this id from admission to retirement.
    id: u64,
    /// Whether a stop token ended this sequence — the distinction the finish reason reports:
    /// a sequence that stopped itself finished with `stop`; one cut off by its token cap finished
    /// with `length`, and the client is expected to act on that (continue, or raise `max_tokens`).
    hit_stop: bool,
    /// The caller's cancellation signal. `step` checks it once per round, so a cancelled request
    /// stops promptly instead of running all the way to its natural end.
    cancel: CancelSignal,
    /// How long this request waited in the queue before it was admitted. We hold on to it so we can
    /// report it as a stat when the request completes.
    queue_wait: Duration,
    /// Set by the cancel sweep when it retires a sequence, so the later retire step knows to label
    /// the finish reason as cancelled. We record it here rather than re-reading `cancel` at retire,
    /// because a signal arriving in the gap between the two would otherwise mislabel a request that
    /// had already finished normally.
    cancelled: bool,
}

/// The stat key a retiring sequence reports its stop reason under (set today only when a request is
/// cancelled mid-flight).
pub const FINISH_REASON_STAT_NAME: &str = "Finish Reason";

/// One in-flight request as the worker sees it: the generic per-sequence decode state, plus the
/// serving payload above.
type JobSeq = ActiveSeq<JobMeta>;

/// What one turn of the worker loop decided to do next: keep going, or shut the thread down.
enum Flow {
    Continue,
    Shutdown,
}

/// Start the worker thread. It takes ownership of `seed` (the model server) and runs the continuous
/// batching loop for the rest of the process's life. Production passes a fresh `Server::default()`;
/// tests pass a server they have pre-configured so they can control its capacity and behaviour.
///
/// If the thread fails to spawn we hand the error straight back, so the caller finds out
/// synchronously. The alternative — wiring up the channel as if a worker existed and only then
/// discovering it never started — would leave every future caller blocked forever on a worker that
/// was never born.
pub(super) fn spawn<S: BatchedInferenceServer + 'static>(seed: S) -> InferenceResult<WorkerInner> {
    let (sender, receiver) = std::sync::mpsc::channel::<Command>();

    let handle = std::thread::Builder::new()
        .name("burn-lm-batching-worker".to_string())
        .spawn(move || {
            let mut server = seed;
            let mut queue: VecDeque<QueuedJob> = VecDeque::new();
            let mut active: Vec<JobSeq> = Vec::new();
            // The delivery side of this worker (see `emission.rs`): owned by this thread so a
            // fresh worker always gets a fresh emission thread, and dropping our sender on ANY
            // exit path — shutdown or panic — is what tells it to fail whatever is still live.
            let Ok((emission, _emission_handle)) = emission::spawn() else {
                fail_everything(&mut queue, &mut active);
                return;
            };
            // Monotonic request ids for the emission thread's ledger; slots are reused, ids never.
            let mut next_id: u64 = 0;

            loop {
                // One panic boundary per loop turn — a turn is the unit we keep consistent. If model
                // code panics mid-turn we don't repair the half-updated `queue`/`active`; we catch
                // here and fail the affected callers before the thread dies. `AssertUnwindSafe` holds
                // because nothing crossing the boundary is reused: the server is dropped on exit, and
                // `queue`/`active` are only read to send `WorkerDied` replies, then cleared.
                let flow = catch_unwind(AssertUnwindSafe(|| {
                    worker_iteration(&mut server, &mut queue, &mut active, &receiver, &emission, &mut next_id)
                }));
                match flow {
                    Ok(Flow::Continue) => {}
                    Ok(Flow::Shutdown) => break,
                    Err(payload) => {
                        // The worker panicked. We log the payload ourselves because the default
                        // panic hook's line on stderr has nothing tying it to the `WorkerDied`
                        // replies the callers are about to get. Then we answer everyone waiting —
                        // both the active sequences and the still-queued jobs — with `WorkerDied`
                        // and let the thread end. The next command to arrive lazily spawns a fresh
                        // worker (see `sender`).
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

/// One turn of the worker loop: take in any pending commands, admit what fits, and advance the
/// round by one token. Runs inside the per-turn panic boundary set up in `spawn`.
fn worker_iteration<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<QueuedJob>,
    active: &mut Vec<JobSeq>,
    receiver: &std::sync::mpsc::Receiver<Command>,
    emission: &std::sync::mpsc::Sender<emission::EmissionEvent>,
    next_id: &mut u64,
) -> Flow {
    // Only block for a command when there's genuinely nothing to do: nothing active, and nothing
    // admittable (a queued job with a free slot). We only wake from `recv` when a new command
    // arrives — but an admittable job is already in `queue`, with no further command coming to wake
    // us — so blocking now would sleep forever on work we could have run.
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
    // Drain whatever else is already waiting, without blocking, so a burst of submissions is fully
    // enqueued before we decide what to admit. That way admission sees the whole batch of ready
    // jobs at once and batches them deterministically, instead of letting them in one per turn.
    while let Ok(command) = receiver.try_recv() {
        if handle_command(server, queue, active, command) {
            return Flow::Shutdown;
        }
    }

    // Admit queued jobs while there is room for them. Anything that doesn't fit stays at the front
    // of the queue for a later turn — this is where backpressure happens.
    admit(server, queue, active, emission, next_id);

    // Advance every decoding sequence by one token (plus at most one prefill), then retire whatever
    // just finished. Each retire frees a slot, so the next turn can admit more.
    step(server, active, emission);

    Flow::Continue
}

/// Tell everyone the worker has died. Active sequences' delivery state (emitter, cursor,
/// completion) lives on the emission thread, and this worker exiting right after this call drops
/// its event sender — the emission thread then flushes and answers every still-live request with
/// `WorkerDied`. Queued jobs never had delivery state transferred, so they are answered here, each
/// releasing its queue permit as it drops. Commands still buffered in the channel disconnect when
/// the receiver drops, so those callers also see `WorkerDied` and their permits drop too — the
/// in-flight counter can't leak. No slots are released: the server, and every slot's cache with
/// it, dies with the thread, and the replacement worker starts empty.
fn fail_everything(queue: &mut VecDeque<QueuedJob>, active: &mut Vec<JobSeq>) {
    active.clear();
    for queued in queue.drain(..) {
        let _ = queued.completion.send(Err(InferenceError::WorkerDied));
    }
}

/// Handle one lifecycle or config command, in between rounds. Returns `true` if it was a shutdown.
///
/// `Submit` is the exception: it doesn't run anything, it just puts the job on the queue — the
/// completion reply is sent later, by `step`, when the sequence finally retires. `active` is passed
/// in read-only, used only to refuse `Unload` and `ClearState` while work is still in flight (see
/// the reasons on those two arms).
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
            // Refuse to unload while work is in flight. An unload slipping in between rounds would
            // be quietly destructive: the next round lazily reloads the model and carries on with
            // sequences whose KV caches died with the old instance. Callers must drain first.
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
            // The same hazard as `Unload`: wiping model state out from under in-flight sequences
            // would pull their KV caches away mid-generation, so we refuse it the same way.
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

/// Move queued jobs into the active set for as long as there is free capacity. This is the one
/// place new work enters a round.
///
/// `batch_capacity().max_slots` is how many sequences the server can run at once. The engine owns
/// the active set, so "free" is simply that limit minus what is already running, and we keep
/// admitting while `active.len()` is below it. A job that doesn't fit stays on the queue until a
/// later sweep — and since that sweep runs right after each retire frees a slot, admission is in
/// effect continuous.
fn admit<S: BatchedInferenceServer>(
    server: &mut S,
    queue: &mut VecDeque<QueuedJob>,
    active: &mut Vec<JobSeq>,
    emission: &std::sync::mpsc::Sender<emission::EmissionEvent>,
    next_id: &mut u64,
) {
    while active.len() < server.batch_capacity().max_slots {
        // Decide whether the FRONT job fits before taking it off the queue, so a job that must
        // wait for KV blocks keeps both its place in line and its queue permit. Jobs that will
        // never run (cancelled, failed tokenize) are popped and answered; only "doesn't fit yet"
        // leaves the queue untouched.
        if queue.front().is_none() {
            break;
        }

        // Cancelled while still queued: reply without touching the model — no prefill, no slot.
        // It never produced a token, so unlike an in-flight cancel this comes back as an error.
        if queue.front().is_some_and(|front| front.job.cancel.is_cancelled()) {
            let cancelled = queue.pop_front().expect("checked above");
            let _ = cancelled.completion.send(Err(InferenceError::Cancelled));
            continue;
        }

        // Borrow the decoder first, since that's what loads the model if needed — so the
        // `tokenize` call below can count on the tokenizer being ready — and read the context
        // ceiling while we hold it: it clamps the KV reservation below.
        let max_context_len = match server.decoder() {
            Ok(decoder) => decoder.max_context_len(),
            Err(err) => {
                let failed = queue.pop_front().expect("checked above");
                let _ = failed.completion.send(Err(err));
                continue;
            }
        };

        let front = queue.front().expect("checked above");
        let tokens = match server.tokenize(&front.job.task) {
            Ok(tokens) => tokens,
            Err(err) => {
                let failed = queue.pop_front().expect("checked above");
                let _ = failed.completion.send(Err(err));
                continue;
            }
        };

        // A request may ask for fewer tokens than the server's cap, but never more — the cap is an
        // operator-set limit, so anything larger is clamped down to it.
        let max_gen = match front.job.params.max_tokens {
            Some(requested) => requested.min(server.max_gen_tokens()),
            None => server.max_gen_tokens(),
        };

        // The KV half of the admission gate: reserve this sequence's worst case — every prompt
        // token plus every token it may generate, clamped by the context ceiling — against the
        // decoder's block budget. The outstanding total is derived from the active set, so retiring
        // a sequence frees its reservation by construction. Since the pool only ever allocates up
        // to a sequence's actual length, a reservation that fits here can never find the pool dry
        // mid-flight. A front job that doesn't fit YET stays queued (with its permit) and admission
        // stops — first come, first served, drained again after the next retire.
        let kv = server.batch_capacity().kv;
        let need = kv.blocks_for((tokens.len() + max_gen).min(max_context_len));
        // A request whose worst case exceeds the WHOLE pool can never run — waiting would block the
        // queue head forever. Reject it now, like an over-long prompt, so the jobs behind it keep
        // flowing. (Lowering `max_tokens` shrinks the worst case, so the client can retry smaller.)
        if need > kv.total_blocks {
            let failed = queue.pop_front().expect("checked above");
            let _ = failed
                .completion
                .send(Err(InferenceError::KvPoolExhausted(need - kv.total_blocks)));
            continue;
        }
        let reserved: usize = active.iter().map(|seq| seq.kv_reservation).sum();
        if need > kv.total_blocks.saturating_sub(reserved) {
            break;
        }

        let QueuedJob {
            job,
            completion,
            enqueued_at,
            permit,
        } = queue.pop_front().expect("checked above");
        // The job has now left the queue for good, so release its queue permit; that is what lets
        // waiting submitters see capacity free up.
        drop(permit);
        let queue_wait = enqueued_at.elapsed();

        // Give the new sequence the lowest-numbered free slot. Slots are just `0..max_slots`, and a
        // slot is free when no active sequence is using it; the admission check above guarantees at
        // least one is.
        let slot = (0..server.batch_capacity().max_slots)
            .find(|candidate| active.iter().all(|seq| seq.slot != *candidate))
            .expect("admission only runs while a slot is free");

        // Lifecycle trace, at debug level (off by default): a sequence just entered this lane. Paired
        // with the retire log below, it bounds each request's life in the log so its overlap with other
        // in-flight requests is readable. `in_flight` is the count after this admission.
        tracing::debug!(
            target: "batching",
            slot,
            in_flight = active.len() + 1,
            kv_blocks_reserved = need,
            kv_blocks_outstanding = reserved + need,
            kv_blocks_total = kv.total_blocks,
            "admitted a sequence to a decode lane"
        );

        // Hand this request's delivery state to the emission thread under a fresh id. From here on
        // the worker never touches the emitter, cursor, or completion — it schedules; the emission
        // thread delivers (see `emission.rs`). A send can only fail if the emission thread died,
        // which only happens when this thread is already exiting.
        let id = *next_id;
        *next_id += 1;
        let _ = emission.send(emission::EmissionEvent::Admitted {
            id,
            emitter: job.emitter,
            detok: Utf8Buffer::new(),
            completion,
        });

        // Release the slot before handing it over, even though retire normally already did. A safety
        // net added after a review: re-releasing a free slot is a no-op, but it guarantees one
        // prompt's leftover state can never bleed into the next if some retire or failed-prefill
        // rollback forgot to. Per-request isolation rests on this method actually dropping the state.
        // (The decoder borrow can't fail — we just checked it.)
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
            kv_reservation: need,
            extra: JobMeta {
                id,
                hit_stop: false,
                cancel: job.cancel,
                queue_wait,
                cancelled: false,
            },
        });
    }
}

/// Logged once, the first time a round successfully borrows the decoder, so the deployment's real
/// admission cap (the loaded KV-slab lane count) is unambiguous in the logs. This answers "is
/// `max_slots` what I set?" without needing debug logging.
static CAP_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run one round and deal with its results. The shared `step_round` core does the actual work — one
/// fused decode over every decoding sequence, plus at most one prefill — and this function handles
/// the serving-specific part around it: stream each new token out to its caller, and retire the
/// sequences that just finished.
fn step<S: BatchedInferenceServer>(
    server: &mut S,
    active: &mut Vec<JobSeq>,
    emission: &std::sync::mpsc::Sender<emission::EmissionEvent>,
) {
    // Nothing active, so don't touch the model. This guard matters because `decoder()` loads
    // lazily: borrowing it on an empty round would force a load — even reloading right after a
    // successful `Unload` in the same turn.
    if active.is_empty() {
        return;
    }

    // Sweep for cancellations once per round so a fired signal retires its sequence this round, not
    // later. Marking them finished before stepping makes `step_round` skip them — no forward pass
    // wasted on a caller who left — and the retire sweep below handles them like any other.
    for seq in active.iter_mut() {
        if !seq.finished && seq.extra.cancel.is_cancelled() {
            seq.finished = true;
            seq.extra.cancelled = true;
        }
    }

    let stop_ids = server.stop_ids();

    // Grab the sampler as an OWNED value first, while we still only hold `&self`. It carries the
    // server's sampling config and is shared by every row this round. Taking it owned is what lets us
    // borrow the decoder mutably next: the owned box doesn't borrow `server`, so it cannot collide
    // with the `&mut` decoder borrow below — no raw pointer, no unsafe.
    let sampler = server.sampler();

    // Read the chunked-prefill width now, while we still hold only `&self` — the same reason the
    // sampler is taken owned above: it must not collide with the `&mut` decoder borrow below.
    let chunk_size = server.prefill_chunk_size();

    // Borrow the decoder for the whole round. If the model isn't loaded we can't run anything, so
    // rather than panic the worker we retire every active sequence with that error.
    let outcomes = match server.decoder() {
        Ok(decoder) => {
            // One prefill budget covers the whole round and is computed across the full active set,
            // so a single long prompt can't hold up the in-flight decoders for more than one round.
            let mut budget = PrefillBudget::for_round(active);
            // The sampler carries no per-sequence state — any randomness it needs is drawn from the
            // backend RNG — so the round's whole job is to hand `step_round` the one shared sampler.
            step_round(decoder, active, &stop_ids, &mut budget, chunk_size, |logits| {
                sampler.sample(logits)
            })
        }
        Err(err) => {
            // Couldn't borrow the decoder, so retire every sequence with this error. There's nothing
            // to release slots into (the caches live inside the decoder, and a fresh load starts
            // empty), so clearing `active` is what frees them.
            for seq in active.iter() {
                let _ = emission.send(emission::EmissionEvent::Retire {
                    id: seq.extra.id,
                    reply: Err(err.clone()),
                });
            }
            active.clear();
            return;
        }
    };

    // Once the model is loaded, log the real admission cap so the deployment's effective lane count
    // (not just the configured `max_slots`) is unambiguous in the logs.
    if !CAP_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            target: "batching",
            max_slots = server.batch_capacity().max_slots,
            "effective batching capacity (loaded lane count)"
        );
    }

    // Ship the round's output — ONE event for the whole round, whatever the width. The worker only
    // does the cheap part here (the per-token byte lookup, which needs the server-owned tokenizer);
    // the per-request delivery work — UTF-8 cursoring, listener wakeups, stream writes — happens on
    // the emission thread, overlapping the next forward instead of lengthening this round. A stop
    // token isn't streamed; a failed forward retires its sequence through the same FIFO, so its
    // trailing bytes still flush before its completion fires.
    let mut round_tokens: Vec<(u64, Vec<u8>)> = Vec::new();
    for (seq, outcome) in active.iter_mut().zip(outcomes) {
        match outcome {
            StepOutcome::Stepped { token, is_stop, .. } => {
                if is_stop {
                    seq.extra.hit_stop = true;
                } else {
                    round_tokens.push((seq.extra.id, server.detokenize_bytes(&[token])));
                }
            }
            StepOutcome::Failed(err) => {
                let _ = emission.send(emission::EmissionEvent::Retire {
                    id: seq.extra.id,
                    reply: Err(err),
                });
            }
            StepOutcome::Skipped => {}
            // An intermediate prefill chunk advanced its lane's KV but sampled no token, so there is
            // nothing to stream and nothing to retire — the sequence keeps prefilling next round.
            StepOutcome::Prefilling => {}
        }
    }
    if !round_tokens.is_empty() {
        let _ = emission.send(emission::EmissionEvent::Tokens(round_tokens));
    }

    // Retire finished sequences: drop them and signal completion, freeing capacity to admit.
    // `retire` flushes the detok cursor first, so trailing bytes reach the caller before completion
    // on every path here (stop token, `max_gen` cap, empty prompt, forward failure). A sequence
    // already failed above sent its `Err` and `took` its sender, so it won't send twice — which
    // matters because the completion channel is a bounded one-shot a caller may read much later, so a
    // second send would block the worker. Slots are collected now and released after the sweep.
    let mut freed_slots = Vec::new();
    active.retain_mut(|seq| {
        if seq.finished {
            freed_slots.push(seq.slot);
            // Lifecycle trace (debug): this request is leaving its lane, freeing it for the next
            // admission. Pairs with the admit log to delimit the request's run in the log stream.
            tracing::debug!(
                target: "batching",
                slot = seq.slot,
                generated = seq.generated,
                kv_blocks_freed = seq.kv_reservation,
                "retired a sequence from its decode lane"
            );
            let mut stats = Stats::new();
            stats
                .entries
                .insert(crate::stats::StatEntry::TokensCount(seq.generated));
            // How long this request waited in the queue before admission. Rendered as fixed seconds
            // to match the other duration stats, and a named entry rather than its own `StatEntry`
            // variant because nothing needs the raw `Duration` back.
            stats.entries.insert(crate::StatEntry::Named(
                super::QUEUE_WAIT_STAT_NAME.to_string(),
                format!("{:.2}s", seq.extra.queue_wait.as_secs_f64()),
            ));
            // Every retirement reports WHY it ended, so the client can act on it: `Stop` — the
            // model stopped itself (a stop token, or a degenerate no-op job); `Length` — cut off by
            // the token cap (the request's `max_tokens`, or the server's `sample_len` default), the
            // signal to continue with a follow-up request or raise the cap; `Cancelled` — the
            // caller walked away mid-flight. The HTTP layer maps these onto the OpenAI
            // `finish_reason` field (`stop` / `length`).
            let reason = if seq.extra.cancelled {
                "Cancelled"
            } else if seq.extra.hit_stop {
                "Stop"
            } else if seq.generated >= seq.max_gen {
                "Length"
            } else {
                "Stop"
            };
            stats.entries.insert(crate::StatEntry::Named(
                FINISH_REASON_STAT_NAME.to_string(),
                reason.to_string(),
            ));
            let _ = emission.send(emission::EmissionEvent::Retire {
                id: seq.extra.id,
                reply: Ok(stats),
            });
            false
        } else {
            true
        }
    });

    // Release the retired slots so the decoder drops their caches. The borrow succeeded at the top
    // of the round, so failing here is effectively impossible — and harmless if it did, since the
    // slots are already free engine-side.
    if !freed_slots.is_empty() {
        if let Ok(decoder) = server.decoder() {
            for slot in freed_slots {
                decoder.release(slot);
            }
        }
    }
}



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
    sampler::SamplingState,
    utf8::Utf8Buffer,
    GeneratedItem, GeneratedItemEmitter, Stats,
};

use super::{Command, QueuedJob, WorkerInner};

/// Everything the serving worker needs to remember about one in-flight request, kept next to the
/// generic decode state in its `ActiveSeq`. The shared decode core (`step_round`) is deliberately
/// model-agnostic — it only advances tokens and reports back — so all the serving-specific
/// machinery for turning those tokens into a streamed response lives here instead. A request picks
/// up a `JobMeta` when it is admitted, streams through it each round, and is retired from it when
/// it finishes.
struct JobMeta {
    /// The channel we stream this request's generated text back to the caller on.
    emitter: GeneratedItemEmitter,
    /// The detokenizer cursor for this request. A token is just bytes, and byte-level BPE can split
    /// a single UTF-8 character across two tokens, so we can't simply decode tokens to text one at
    /// a time. Each round we push the new token's raw bytes in here and emit only the text that is
    /// now complete, holding a trailing partial character back until its next byte arrives. Without
    /// this the stream could surface a broken U+FFFD character, or the decoder could panic on
    /// invalid UTF-8.
    detok: Utf8Buffer,
    /// This request's per-sequence sampling state, built once when the request is admitted and kept
    /// for its whole life. Sampling can be stateful — a seeded RNG has to advance across the
    /// request's tokens rather than restart every round — so we never rebuild it mid-flight, and two
    /// concurrent requests each get their own RNG so they never draw off one shared stream. The
    /// sampler itself (the config) is shared across the batch and owned by the round; only this
    /// per-sequence state lives here. It is an `Option` only so `step` can briefly `take` it out
    /// while the sequence is borrowed for the round, then put it back.
    sample_state: Option<SamplingState>,
    /// The one-shot channel that tells the caller their request is done, with stats or an error. It
    /// is an `Option` so every place that might send must first `take`, to guarantee we reply
    /// exactly once: the first send wins. This matters because the channel is a bounded one-shot —
    /// a second send, with the first still sitting unread, would block this whole worker thread.
    completion: Option<std::sync::mpsc::SyncSender<InferenceResult<Stats>>>,
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

            loop {
                // One panic boundary per loop turn — a turn is the unit we keep consistent. If model
                // code panics mid-turn we don't repair the half-updated `queue`/`active`; we catch
                // here and fail the affected callers before the thread dies. `AssertUnwindSafe` holds
                // because nothing crossing the boundary is reused: the server is dropped on exit, and
                // `queue`/`active` are only read to send `WorkerDied` replies, then cleared.
                let flow = catch_unwind(AssertUnwindSafe(|| {
                    worker_iteration(&mut server, &mut queue, &mut active, &receiver)
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
    admit(server, queue, active);

    // Advance every decoding sequence by one token (plus at most one prefill), then retire whatever
    // just finished. Each retire frees a slot, so the next turn can admit more.
    step(server, active);

    Flow::Continue
}

/// Tell everyone the worker has died. Active sequences are retired with `WorkerDied` (same send-once
/// discipline as a normal retire) and queued jobs answered the same way, each releasing its queue
/// permit as it drops. Commands still buffered in the channel disconnect when the receiver drops, so
/// those callers also see `WorkerDied` and their permits drop too — the in-flight counter can't
/// leak. No slots are released: the server, and every slot's cache with it, dies with the thread,
/// and the replacement worker starts empty.
fn fail_everything(queue: &mut VecDeque<QueuedJob>, active: &mut Vec<JobSeq>) {
    for seq in active.iter_mut() {
        retire(&mut seq.extra, Err(InferenceError::WorkerDied));
    }
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
        // The job has now left the queue whatever we decide next, so release its queue permit right
        // away; that is what lets waiting submitters see capacity free up.
        drop(permit);
        let queue_wait = enqueued_at.elapsed();

        // Cancelled while still queued: reply without touching the model — no prefill, no slot.
        // It never produced a token, so unlike an in-flight cancel this comes back as an error.
        if job.cancel.is_cancelled() {
            let _ = completion.send(Err(InferenceError::Cancelled));
            continue;
        }

        // Borrow the decoder first, since that's what loads the model if needed — so the
        // `tokenize`/`detokenize` calls below can count on the tokenizer being ready.
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

        // A request may ask for fewer tokens than the server's cap, but never more — the cap is an
        // operator-set limit, so anything larger is clamped down to it.
        let max_gen = match job.params.max_tokens {
            Some(requested) => requested.min(server.max_gen_tokens()),
            None => server.max_gen_tokens(),
        };

        // Give the new sequence the lowest-numbered free slot. Slots are just `0..max_slots`, and a
        // slot is free when no active sequence is using it; the admission check above guarantees at
        // least one is.
        let slot = (0..server.batch_capacity().max_slots)
            .find(|candidate| active.iter().all(|seq| seq.slot != *candidate))
            .expect("admission only runs while a slot is free");

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
            extra: JobMeta {
                emitter: job.emitter,
                detok: Utf8Buffer::new(),
                // Built once, here, from the server's config-driven sampler, so this sequence's RNG
                // lives for its whole generation and two concurrent requests sample off independent
                // streams. The sampler (the shared config) is grabbed fresh each round; only this
                // per-sequence state persists.
                sample_state: Some(server.sampler().fresh_state()),
                completion: Some(completion),
                cancel: job.cancel,
                queue_wait,
                cancelled: false,
            },
        });
    }
}

/// Run one round and deal with its results. The shared `step_round` core does the actual work — one
/// fused decode over every decoding sequence, plus at most one prefill — and this function handles
/// the serving-specific part around it: stream each new token out to its caller, and retire the
/// sequences that just finished.
fn step<S: BatchedInferenceServer>(server: &mut S, active: &mut Vec<JobSeq>) {
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

    // Borrow the decoder for the whole round. If the model isn't loaded we can't run anything, so
    // rather than panic the worker we retire every active sequence with that error.
    let outcomes = match server.decoder() {
        Ok(decoder) => {
            // One prefill budget covers the whole round and is computed across the full active set,
            // so a single long prompt can't hold up the in-flight decoders for more than one round.
            let mut budget = PrefillBudget::for_round(active);
            // Each request has its own per-sequence sampling state (a seeded RNG that persists across
            // its tokens). Move them out of `extra` into a parallel vec so the closure can reach them
            // by index while `step_round` holds `active` borrowed, then put them back after the round.
            let mut states: Vec<SamplingState> = active
                .iter_mut()
                .map(|seq| {
                    seq.extra
                        .sample_state
                        .take()
                        .expect("sample_state is only taken for the duration of a round")
                })
                .collect();
            let outcomes = step_round(
                decoder,
                active,
                &stop_ids,
                &mut budget,
                |logits, indices| {
                    // Gather the sampled rows' per-sequence states into a contiguous slice in row order,
                    // sample the whole batch through the one shared sampler, then return each state to
                    // its slot. `swap` with a throwaway lets us move a state out of `states[index]` and
                    // back without cloning — the index list has no duplicates within a round, so this
                    // round can't read a slot it just emptied.
                    let placeholder = || SamplingState {
                        rng: rand::SeedableRng::seed_from_u64(0),
                    };
                    let mut gathered: Vec<SamplingState> = indices
                        .iter()
                        .map(|&index| std::mem::replace(&mut states[index], placeholder()))
                        .collect();
                    let result = sampler.sample(logits, &mut gathered);
                    for (&index, state) in indices.iter().zip(gathered) {
                        states[index] = state;
                    }
                    result
                },
            );
            for (seq, state) in active.iter_mut().zip(states) {
                seq.extra.sample_state = Some(state);
            }
            outcomes
        }
        Err(err) => {
            // Couldn't borrow the decoder, so retire every sequence with this error. There's nothing
            // to release slots into (the caches live inside the decoder, and a fresh load starts
            // empty), so clearing `active` is what frees them.
            for seq in active.iter_mut() {
                retire(&mut seq.extra, Err(err.clone()));
            }
            active.clear();
            return;
        }
    };

    // Stream the round's output. For each sequence that advanced, push its new token's bytes through
    // the detok cursor and emit whatever text is now complete — a stop token isn't streamed, and a
    // mid-character byte is held back for next round. A sequence whose forward failed is retired here.
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
                // Flush the held-back bytes before completion fires, just like the decoder-error
                // path above, so that trailing text always reaches the caller before we tell them
                // the request is done.
                retire(&mut seq.extra, Err(err));
            }
            StepOutcome::Skipped => {}
        }
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
            // A request cancelled mid-flight still completes with `Ok`: the caller got real tokens,
            // and this finish-reason stat says why the stream ended early. Only the cancelled path
            // sets one today, so normally-finished requests keep their existing stats output.
            if seq.extra.cancelled {
                stats.entries.insert(crate::StatEntry::Named(
                    FINISH_REASON_STAT_NAME.to_string(),
                    "Cancelled".to_string(),
                ));
            }
            retire(&mut seq.extra, Ok(stats));
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

/// Flush a retiring sequence's detok cursor out to its emitter. At the true end of a stream a
/// held-back partial character can never be completed, so this is the one place we allow the lossy
/// U+FFFD replacement rather than silently dropping those bytes. Every retire path calls it: the
/// retire sweep in `step` and the decoder-error mass-retire.
fn flush_detok(meta: &mut JobMeta) {
    if let Some(text) = meta.detok.finish() {
        meta.emitter.completed(GeneratedItem::Text(text));
    }
}

/// Retire a single sequence: flush its detok cursor to the emitter, then send its completion reply
/// exactly once. The flush comes first so trailing held-back bytes always reach the caller before
/// the completion does, and the `take` keeps the bounded one-shot send-once — a second send would
/// block the worker. Every retire path goes through here; on success the caller builds the `Stats`
/// first and passes `Ok(stats)`.
fn retire(meta: &mut JobMeta, reply: InferenceResult<Stats>) {
    flush_detok(meta);
    if let Some(completion) = meta.completion.take() {
        let _ = completion.send(reply);
    }
}

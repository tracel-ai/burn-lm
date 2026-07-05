//! The emission side of the serving worker: streaming tokens out to callers, off the decode
//! round's critical path.
//!
//! A round produces one token per live lane, but delivering those tokens — pushing bytes through
//! each request's UTF-8 cursor, waking each request's listener, formatting and writing each
//! stream — is per-lane host work. Done inline it lands on the worker thread between forwards, so
//! every lane lengthens every round and aggregate throughput caps at `1/marginal-cost` no matter
//! how many lanes run (measured: ~16 ms/lane on a T4 server, ~40 on an M-series laptop). This
//! module moves that work to its own thread: the worker hands over ONE message per round with the
//! whole round's tokens and immediately starts the next forward; delivery overlaps the GPU.
//!
//! Ownership moves with the work. When a job is admitted the worker transfers the request's
//! emitter, detokenizer cursor, and completion channel here, and from then on this thread is the
//! only one that touches them — token delivery, the flush-before-completion ordering, and the
//! exactly-once completion reply all live in one place. The worker keeps what the scheduler needs
//! (cancel signal, stop bookkeeping) and speaks to this thread through a FIFO channel, which is
//! what preserves the ordering contract: a request's tokens always precede its completion.
//!
//! If the worker dies mid-flight, its end of the channel drops; this thread then flushes every
//! outstanding request's held-back bytes and answers it with `WorkerDied`, mirroring what the
//! worker's own panic path does for still-queued jobs. If a caller is gone (its emitter latched
//! done), delivery quietly stops for that request — same contract as before.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};

use crate::{
    errors::InferenceError, GeneratedItem, GeneratedItemEmitter, InferenceResult, Stats,
    Utf8Buffer,
};

/// What the worker tells the emission thread. Events for one request arrive in order because the
/// channel is FIFO and the worker is single-threaded.
pub(super) enum EmissionEvent {
    /// A job was admitted: take ownership of its delivery state. `id` is the worker's monotonic
    /// per-request key (slots are reused; ids are not).
    Admitted {
        id: u64,
        emitter: GeneratedItemEmitter,
        detok: Utf8Buffer,
        completion: std::sync::mpsc::SyncSender<InferenceResult<Stats>>,
    },
    /// One round's streamable tokens, already detokenized to raw bytes (the tokenizer lives with
    /// the server on the worker thread; the byte lookup is cheap — delivery is what's expensive).
    /// One event per round regardless of width: that is the decoupling.
    Tokens(Vec<(u64, Vec<u8>)>),
    /// A request is done: flush its held-back bytes, reply exactly once, drop its state. Any later
    /// event for the same id is ignored — the double-retire that used to be absorbed by
    /// `Option::take` on the completion sender is now absorbed by the state being gone.
    Retire {
        id: u64,
        reply: InferenceResult<Stats>,
    },
}

/// One request's delivery state, owned by the emission thread from admission to retirement.
struct Delivery {
    emitter: GeneratedItemEmitter,
    detok: Utf8Buffer,
    completion: std::sync::mpsc::SyncSender<InferenceResult<Stats>>,
}

impl Delivery {
    /// Push one token's bytes through the UTF-8 cursor and deliver whatever text is now complete.
    fn deliver(&mut self, bytes: &[u8]) {
        if let Some(text) = self.detok.push(bytes) {
            self.emitter.completed(GeneratedItem::Text(text));
        }
    }

    /// End of stream: flush the held-back partial character (lossily, as at any true end of
    /// stream) BEFORE the completion fires, so trailing text always reaches the caller before
    /// they are told the request is done.
    fn finish(mut self, reply: InferenceResult<Stats>) {
        if let Some(text) = self.detok.finish() {
            self.emitter.completed(GeneratedItem::Text(text));
        }
        let _ = self.completion.send(reply);
    }
}

/// Spawn the emission thread. It lives exactly as long as the worker holds the sender: when the
/// worker exits (shutdown or panic), the channel disconnects and every outstanding request is
/// answered with `WorkerDied`.
pub(super) fn spawn() -> InferenceResult<(Sender<EmissionEvent>, std::thread::JoinHandle<()>)> {
    let (tx, rx) = std::sync::mpsc::channel::<EmissionEvent>();
    let handle = std::thread::Builder::new()
        .name("burn-lm-batching-emission".to_string())
        .spawn(move || run(rx))
        .map_err(|_| InferenceError::WorkerDied)?;
    Ok((tx, handle))
}

fn run(rx: Receiver<EmissionEvent>) {
    let mut live: HashMap<u64, Delivery> = HashMap::new();
    while let Ok(event) = rx.recv() {
        match event {
            EmissionEvent::Admitted {
                id,
                emitter,
                detok,
                completion,
            } => {
                live.insert(
                    id,
                    Delivery {
                        emitter,
                        detok,
                        completion,
                    },
                );
            }
            EmissionEvent::Tokens(batch) => {
                for (id, bytes) in batch {
                    if let Some(delivery) = live.get_mut(&id) {
                        delivery.deliver(&bytes);
                    }
                }
            }
            EmissionEvent::Retire { id, reply } => {
                if let Some(delivery) = live.remove(&id) {
                    delivery.finish(reply);
                }
            }
        }
    }
    // The worker is gone (shutdown after a drained queue, or a panic). Anyone still live never got
    // a retire event: flush and answer them, exactly as the worker's panic path answers its queue.
    for (_, delivery) in live.drain() {
        delivery.finish(Err(InferenceError::WorkerDied));
    }
}

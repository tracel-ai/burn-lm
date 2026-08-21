use super::super::*;
use super::fakes::*;
use crate::job::{GenerationParams, InferenceJob, InferenceJobListener, InferenceTask};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

/// A listener whose `on_text` blocks until the test releases a gate — a stand-in for a slow SSE
/// socket whose `write` backpressures. The blocking write is the one thing the HTTP serving path
/// has that the in-memory `Vec`-writer probes never do, which is why those probes were all clean.
struct BlockingListener {
    gate: mpsc::Receiver<()>,
}

impl InferenceJobListener for BlockingListener {
    type CompletedItem = ();
    fn on_text(&mut self, _text: String) {
        // Block like a backpressured socket. Returns once the test drops the gate sender.
        let _ = self.gate.recv();
    }
    fn on_finished(self) {}
}

/// A slow consumer on ONE job must not stall the OTHERS. Today it does, and this is the production
/// HTTP hang reproduced in-process with no socket: the worker streams every sequence's tokens
/// inline on its single thread through a `sync_channel(1)` whose `send` BLOCKS when full (see
/// `job.rs`). A listener stuck in `on_text` stops draining, its 1-slot channel fills, and the
/// worker's next `send` wedges — head-of-line blocking every other in-flight request (ttft=null).
#[test]
fn a_blocking_listener_must_not_stall_other_jobs() {
    let log: OrderLog = Arc::new(Mutex::new(Vec::new()));
    let mut server = FakeServer::new(2, log); // two slots, so A and B share each round
    server.decoder.emit = 8; // enough tokens that the worker tries to send into A's full channel
    let channel = BatchingChannel::<FakeServer>::with_server(server);

    // Job A's listener blocks in `on_text`; the test holds the gate open.
    let (gate_tx, gate_rx) = mpsc::channel::<()>();
    let (job_a, _ha) = InferenceJob::create(
        InferenceTask::Prompt("a".into()),
        GenerationParams::default(),
        BlockingListener { gate: gate_rx },
    );
    // Job B is a normal, fast consumer that should finish promptly.
    let (job_b, _hb) = InferenceJob::create(
        InferenceTask::Prompt("b".into()),
        GenerationParams::default(),
        NullListener,
    );

    // Submit A first so it takes the lower slot and is streamed first in each round.
    let _rx_a = channel.submit(job_a).unwrap();
    let rx_b = channel.submit(job_b).unwrap();

    // B should complete quickly no matter how slow A's consumer is. Capture the outcome BEFORE
    // releasing the gate, so that even on failure the blocked worker and listener threads are
    // unwound rather than left hanging the test process.
    let b_finished = rx_b.recv_timeout(Duration::from_secs(3)).is_ok();
    drop(gate_tx); // release A's listener so every thread can finish

    assert!(
        b_finished,
        "job B never completed: it was head-of-line blocked by job A's slow listener — a slow \
         consumer on one stream stalled the shared worker and starved every other request"
    );
}

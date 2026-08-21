use axum::{
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use burn_lm_inference::{
    CancelSignal, GenerationParams, InferenceJob, InferenceTask, StatEntry, TextGenerationListener,
    WriteListener,
};
use std::io::Write;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::{
    errors::ServerResult,
    schemas::chat_schemas::{
        ChatCompletionChunkSchema, ChatCompletionParamsSchema, ChatCompletionRequestSchema,
        ChatCompletionSchema, ChoiceMessageRoleSchema, ChoiceMessageSchema, ChoiceSchema,
        FinishReasonSchema, StreamingChunk, UsageSchema,
    },
    stores::chat_store::{ModelStoreExt, ModelStoreState},
    utils::id::ChatCompletionId,
};

pub const REPLY_MARKER: &str = "##### Model Reply";

/// An SSE keepalive: a comment line — clients ignore anything starting with `:` — sent on the stream
/// while it would otherwise be silent, so a reverse proxy (e.g. Modal's `@web_server`) doesn't time
/// the idle connection out. The trailing blank line terminates the SSE event.
const SSE_HEARTBEAT: &str = ": heartbeat\n\n";

/// How often to emit `SSE_HEARTBEAT` while a blocking step is in flight — the model load, or a
/// generation with a long gap between tokens. A few seconds stays well under a typical proxy idle
/// timeout without adding meaningful traffic.
const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// Await a blocking task while keeping the SSE stream warm. Until the task finishes, emit an
/// `SSE_HEARTBEAT` on `tx` every `interval`, so the connection is never silent long enough for a
/// proxy to drop it. This is what covers a slow model load and any long inter-token gap — both run
/// on a blocking thread that produces no stream output while it works. If the client has gone (the
/// heartbeat send fails because the receiver was dropped) we stop emitting but keep awaiting the
/// task, so the spawned blocking work is driven to completion rather than leaked.
async fn await_with_heartbeat<T>(
    mut task: tokio::task::JoinHandle<T>,
    tx: &mpsc::Sender<String>,
    interval: std::time::Duration,
) -> Result<T, tokio::task::JoinError> {
    loop {
        tokio::select! {
            result = &mut task => return result,
            _ = tokio::time::sleep(interval) => {
                if tx.send(SSE_HEARTBEAT.to_string()).await.is_err() {
                    return (&mut task).await;
                }
            }
        }
    }
}

/// Per-request generation parameters derived from the payload, carried on the job itself —
/// immune to the shared-config mutation race the old per-request `parse_json_config` had once
/// requests ran concurrently. The only per-request knob is the token cap; sampling (temperature,
/// top-p, seed) is config-driven on the server.
fn generation_params(params: &ChatCompletionParamsSchema) -> GenerationParams {
    GenerationParams {
        max_tokens: params.max_tokens.map(|v| v as usize),
    }
}

/// Wire client disconnection to job cancellation: when the SSE receiver side is dropped (client
/// went away), fire the job's cancel signal so the engine stops spending forwards on it.
///
/// Returns the watcher's `JoinHandle`; the caller MUST abort it once the job is finished, because
/// the watcher holds a sender clone that would otherwise keep the SSE channel open forever.
fn wire_disconnect_cancel(
    tx: &mpsc::Sender<String>,
    cancel: CancelSignal,
) -> tokio::task::JoinHandle<()> {
    let tx = tx.clone();
    tokio::spawn(async move {
        // `closed()` completes when the receiver is dropped, regardless of how many sender
        // clones are still alive.
        tx.closed().await;
        cancel.cancel();
    })
}

struct SseWriter {
    tx: mpsc::Sender<String>,
    id: String,
    model: String,
    created: i64,
}

impl Write for SseWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
            &self.id,
            &self.model,
            self.created,
            &text,
        ));

        self.tx
            .blocking_send(chunk.to_event_stream())
            .map_err(|_| std::io::ErrorKind::BrokenPipe)?;

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub async fn chat_completions(
    State(state): State<ModelStoreState>,
    Json(payload): Json<ChatCompletionRequestSchema>,
) -> ServerResult<Response> {
    tracing::debug!("Received JSON payload: {:?}", payload);
    if payload.stream {
        handle_streaming_response(state.clone(), payload).await
    } else {
        handle_non_streaming_response(state.clone(), payload).await
    }
}

async fn handle_non_streaming_response(
    state: ModelStoreState,
    payload: ChatCompletionRequestSchema,
) -> ServerResult<Response> {
    // Lock held only inside `acquire_plugin`; released before generation so requests don't serialize.
    let (plugin, _) = state.acquire_plugin(&payload.model).await?;
    let messages: Vec<burn_lm_inference::Message> =
        payload.messages.into_iter().map(Into::into).collect();
    let task = InferenceTask::Context(messages);
    let (job, handle) = InferenceJob::create(
        task,
        generation_params(&payload.params),
        TextGenerationListener::default(),
    );
    // Map inference failures to HTTP errors instead of unwrapping: a shed job (`Overloaded`)
    // must become a 429, not a panicking handler — panicking here would defeat backpressure.
    let _stats = plugin
        .run_job(job)
        .map_err(crate::errors::ServerError::from)?;
    let content = handle.join();

    tracing::debug!("Answer: {}", content);
    let response = ChatCompletionSchema {
        id: ChatCompletionId::new().to_string(),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: payload.model.clone(),
        choices: vec![ChoiceSchema {
            index: 0,
            message: ChoiceMessageSchema {
                role: ChoiceMessageRoleSchema::Assistant,
                content,
                refusal: None,
            },
            finish_reason: FinishReasonSchema::Stop,
            logprobs: None,
        }],
        usage: UsageSchema::default(),
        system_fingerprint: "".to_string(),
    };
    Ok(Json(response).into_response())
}

async fn handle_streaming_response(
    state: ModelStoreState,
    payload: ChatCompletionRequestSchema,
) -> ServerResult<Response> {
    // Resolve the model up front (lock held only inside `acquire_plugin`). Doing it before the 200
    // stream starts returns a clean 4xx for an unknown model instead of panicking a worker
    // mid-stream, and the lock is released before generation so concurrent requests interleave
    // through the batching channel.
    let (plugin, old_model_name) = state.acquire_plugin(&payload.model).await?;

    // PRE-FLIGHT BACKPRESSURE CHECK, before any SSE byte is committed: once the 200 + headers
    // below are sent, a job shed with `Overloaded` inside the stream task can no longer become a
    // real 429 (the status line is gone — it would arrive as in-stream error text at best). The
    // check is ADVISORY: the queue can fill between this probe and the job's submission, in which
    // case the shed still happens at submit and is reported as in-stream error text (see the
    // `Ok(Err(..))` arm below) — but the common overload case answers with an honest 429.
    if plugin.is_overloaded() {
        return Err(crate::errors::ServerError::Overloaded);
    }

    let (tx, rx) = mpsc::channel(10);
    tokio::spawn({
        async move {
            let id = ChatCompletionId::new().to_string();
            let now = chrono::Utc::now().timestamp();
            let model = plugin.model_name();

            // feedback is we unloaded a previously loaded model
            if let Some(name) = old_model_name {
                let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
                    &id,
                    model,
                    now,
                    &format!("```Burn LM\nUnloaded model '{name}'!\n```\n\n"),
                ));
                tx.send(chunk.to_event_stream())
                    .await
                    .expect("should send unloading model chunk");
            }

            // load model and gives feedback in real time in the client
            if !plugin.is_loaded() {
                // loading model chunks
                let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
                    &id,
                    model,
                    now,
                    &format!("```Burn LM\nloading model '{}'... ", plugin.model_name()),
                ));
                tx.send(chunk.to_event_stream())
                    .await
                    .expect("should send loading model chunk");
                tracing::debug!("Loading model '{}'", plugin.model_name());
                let load_task = tokio::task::spawn_blocking({
                    let plugin = plugin.clone();
                    move || {
                        plugin.load().unwrap_or_else(|_| {
                            panic!("model '{}' should load", plugin.model_name())
                        })
                    }
                });
                // Loading weights blocks for seconds with nothing to stream; heartbeat across it so
                // the proxy doesn't drop the connection before the first token arrives.
                let loading_stats = await_with_heartbeat(load_task, &tx, HEARTBEAT_INTERVAL)
                    .await
                    .expect("should complete model loading");
                tracing::debug!("Model loaded '{}'", plugin.model_name());
                let loading_duration = match loading_stats {
                    Some(stats) => {
                        let model_duration_stat = stats
                            .entries
                            .iter()
                            .find(|e| matches!(e, StatEntry::ModelLoadingDuration(_)));
                        if let Some(stat) = model_duration_stat {
                            let duration = stat.get_duration().unwrap().as_secs_f64();
                            format!(" ({duration:.2}s)")
                        } else {
                            "".to_string()
                        }
                    }
                    _ => "".to_string(),
                };
                let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
                    &id,
                    model,
                    now,
                    &format!("model loaded ! ✓{loading_duration}\n```\n\n"),
                ));
                tx.send(chunk.to_event_stream())
                    .await
                    .expect("should send end of loading model chunk");
            }

            // answer chunk
            let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
                &id,
                model,
                now,
                &format!("\n{REPLY_MARKER}\n"),
            ));
            tx.send(chunk.to_event_stream())
                .await
                .expect("should send reply section title chunk");
            let mut messages: Vec<burn_lm_inference::Message> =
                payload.messages.into_iter().map(Into::into).collect();
            messages
                .iter_mut()
                .for_each(|m| m.cleanup(REPLY_MARKER, burn_lm_inference::STATS_MARKER));
            tracing::debug!("Cleaned up messages: {:?}", messages);
            let task = InferenceTask::Context(messages);
            let listener = WriteListener::new(SseWriter {
                tx: tx.clone(),
                id: id.clone(),
                model: model.to_string(),
                created: now,
            });
            let (job, handle) =
                InferenceJob::create(task, generation_params(&payload.params), listener);
            // Client disconnect -> job cancellation (observed by the worker's cancel sweep).
            let disconnect_watcher = wire_disconnect_cancel(&tx, job.cancel.clone());
            let gen_task = tokio::task::spawn_blocking({
                let plugin = plugin.clone();
                move || {
                    let result = plugin.run_job(job);
                    // Join the listener AFTER the job: guarantees every streamed chunk has been
                    // forwarded before stats/[DONE].
                    handle.join();
                    result
                }
            });
            // Tokens stream out through the listener, but a slow forward can leave a long gap
            // between them; heartbeat across generation so an idle gap can't trip the proxy either.
            let join_result = await_with_heartbeat(gen_task, &tx, HEARTBEAT_INTERVAL).await;
            // The watcher holds a sender clone; abort it on EVERY path (including the panic
            // arm below) so the SSE channel can close — otherwise the stream never ends.
            disconnect_watcher.abort();

            // `spawn_blocking` yields two layers of Result; peel them one at a time.
            // Layer 1: did the blocking task complete at all? `Err` means it panicked — today
            // that means the listener thread died mid-stream (client disconnected -> the SSE
            // write failed -> `join()` panicked), so treat it as a disconnect: close the stream
            // instead of re-panicking the handler task.
            let task_result = match join_result {
                Err(join_err) => {
                    tracing::debug!(
                        "generation task panicked (likely client disconnect): {join_err}"
                    );
                    let _ = tx.send(StreamingChunk::Done.to_event_stream()).await;
                    return;
                }
                Ok(result) => result,
            };
            // Layer 2: the task ran — did inference itself succeed?
            let stats = match task_result {
                Err(err) => {
                    // Includes Err(Cancelled) for a job cancelled while still queued: the client
                    // is gone, so just close the stream instead of panicking the handler.
                    //
                    // A job shed with `Overloaded` HERE slipped past the pre-flight check above
                    // (the queue filled in between) and the 429 window is gone — log it loudly
                    // and tell the client in-stream rather than fake a normal-looking truncation.
                    if matches!(err, burn_lm_inference::InferenceError::Overloaded) {
                        tracing::warn!(
                            "streaming job shed after the pre-flight overload check: {err}"
                        );
                        let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
                            &id,
                            model,
                            now,
                            &format!("\n```Burn LM\nerror: {err}\n```\n"),
                        ));
                        let _ = tx.send(chunk.to_event_stream()).await;
                    } else {
                        tracing::debug!("Inference did not complete: {err}");
                    }
                    let _ = tx.send(StreamingChunk::Done.to_event_stream()).await;
                    return;
                }
                Ok(stats) => stats,
            };
            let stats = format!("\n\n{}", stats.display_stats());
            let chunk =
                StreamingChunk::Data(ChatCompletionChunkSchema::new(&id, model, now, &stats));
            // A send error just means the client disconnected; nothing left to do.
            if tx.send(chunk.to_event_stream()).await.is_err() {
                return;
            }

            // Done chunk
            let done_chunk = StreamingChunk::Done;
            let _ = tx.send(done_chunk.to_event_stream()).await;
        }
    });

    let stream = ReceiverStream::new(rx).map(Ok::<_, std::io::Error>);
    let headers = HeaderMap::from_iter(vec![
        (
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("text/event-stream"),
        ),
        (
            HeaderName::from_static("cache-control"),
            HeaderValue::from_static("no-cache"),
        ),
        (
            HeaderName::from_static("connection"),
            HeaderValue::from_static("keep-alive"),
        ),
    ]);

    Ok((
        StatusCode::OK,
        headers,
        axum::body::Body::from_stream(stream),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_lm_inference::{GeneratedItem, WriteListener};
    use std::io::Write;
    use std::time::Duration;

    struct ChannelWriter {
        tx: mpsc::Sender<String>,
    }

    impl Write for ChannelWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let text = String::from_utf8_lossy(buf).to_string();
            self.tx
                .blocking_send(text)
                .map_err(|_| std::io::ErrorKind::BrokenPipe)?;
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The SSE disconnect wiring: dropping the stream's receiver (the client going away) must
    /// fire the job's cancel signal, and the watcher itself must finish so it cannot hold the
    /// channel open.
    #[tokio::test]
    async fn disconnect_fires_cancel_signal() {
        let (tx, rx) = mpsc::channel::<String>(1);
        let cancel = CancelSignal::default();
        let watcher = wire_disconnect_cancel(&tx, cancel.clone());
        assert!(!cancel.is_cancelled());

        // Client goes away.
        drop(rx);

        // The watcher observes the closed channel and fires the signal.
        tokio::time::timeout(Duration::from_secs(5), watcher)
            .await
            .expect("watcher should finish after receiver drop")
            .expect("watcher should not panic");
        assert!(cancel.is_cancelled());
    }

    /// The watcher holds a sender clone, so while the job is running the SSE channel must NOT
    /// read as closed — and aborting the watcher (what the handler does after generation) must
    /// release that clone so the stream can end. Reintroducing a watcher that is never aborted
    /// deadlocks every streaming request on its final `[DONE]`.
    #[tokio::test]
    async fn aborting_the_watcher_releases_the_sse_channel() {
        let (tx, mut rx) = mpsc::channel::<String>(1);
        let watcher = wire_disconnect_cancel(&tx, CancelSignal::default());

        // Drop the handler's own sender; only the watcher's clone remains.
        drop(tx);
        watcher.abort();

        // With the watcher aborted, no sender survives: the stream must end (recv -> None)
        // instead of hanging forever.
        let next = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .expect("stream must close once the watcher is aborted");
        assert!(next.is_none());
    }

    #[tokio::test]
    async fn write_listener_streams_text_as_soon_as_it_is_emitted() {
        let (tx, mut rx) = mpsc::channel(1);
        let task = InferenceTask::Prompt("prompt".to_string());
        let listener = WriteListener::new(ChannelWriter { tx });
        let (job, handle) = InferenceJob::create(task, GenerationParams::default(), listener);

        std::thread::spawn(move || {
            // This simulates the model emitting a token while generation is still running.
            job.emitter
                .completed(GeneratedItem::Text("first".to_string()));
            std::thread::sleep(Duration::from_millis(200));
            handle.join();
        });

        let first = tokio::time::timeout(Duration::from_millis(50), rx.recv())
            .await
            .expect("write listener should stream emitted text without waiting to finish")
            .expect("stream should still be open");
        assert_eq!(first, "first");
    }

    /// A blocking step that takes longer than the heartbeat interval must keep the stream warm:
    /// while it runs, `: heartbeat\n\n` comments flow so a proxy never sees a silent connection.
    /// This is the load gap and the slow-inter-token gap reproduced without a model.
    #[tokio::test]
    async fn heartbeat_keeps_a_slow_blocking_step_from_going_silent() {
        let (tx, mut rx) = mpsc::channel::<String>(8);
        let task = tokio::task::spawn_blocking(|| {
            std::thread::sleep(Duration::from_millis(150));
            "done"
        });

        // Drain the stream concurrently, like the response body would, counting heartbeats — so the
        // channel never backs up and the heartbeat sends never block waiting on an undrained rx.
        let drain = tokio::spawn(async move {
            let mut heartbeats = 0;
            while let Some(msg) = rx.recv().await {
                if msg == SSE_HEARTBEAT {
                    heartbeats += 1;
                }
            }
            heartbeats
        });

        let result = await_with_heartbeat(task, &tx, Duration::from_millis(20))
            .await
            .expect("the blocking task should complete");
        assert_eq!(result, "done");
        drop(tx); // close the stream so the drain task finishes

        let heartbeats = drain.await.unwrap();
        assert!(
            heartbeats >= 3,
            "expected several heartbeats across a 150ms block at 20ms cadence, got {heartbeats}"
        );
    }

    /// If the client is already gone, the heartbeat send fails — but the helper must still drive the
    /// blocking task to completion rather than hang or leak it.
    #[tokio::test]
    async fn heartbeat_stops_on_disconnect_but_still_awaits_the_task() {
        let (tx, rx) = mpsc::channel::<String>(1);
        drop(rx); // client already disconnected: every send will fail
        let task = tokio::task::spawn_blocking(|| {
            std::thread::sleep(Duration::from_millis(60));
            "done"
        });

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            await_with_heartbeat(task, &tx, Duration::from_millis(10)),
        )
        .await
        .expect("helper must not hang after a disconnect")
        .expect("the blocking task should still complete");
        assert_eq!(result, "done");
    }

    #[tokio::test]
    async fn rest_generation_streams_text_as_soon_as_it_is_emitted() {
        let (tx, mut rx) = mpsc::channel(1);
        let task = InferenceTask::Prompt("prompt".to_string());
        let listener = WriteListener::new(SseWriter {
            tx,
            id: "chatcmpl-test".to_string(),
            model: "test-model".to_string(),
            created: 42,
        });
        let (job, handle) = InferenceJob::create(task, GenerationParams::default(), listener);

        std::thread::spawn(move || {
            // This simulates the model emitting a token while generation is still running.
            job.emitter
                .completed(GeneratedItem::Text("first".to_string()));
            std::thread::sleep(Duration::from_millis(200));
            handle.join();
        });

        let first = tokio::time::timeout(Duration::from_millis(50), rx.recv())
            .await
            .expect("REST should stream emitted text without waiting for generation to finish")
            .expect("stream should still be open");
        assert!(first.contains("\"content\":\"first\""));
    }
}

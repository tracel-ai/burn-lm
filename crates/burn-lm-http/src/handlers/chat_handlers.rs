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

/// Per-request generation parameters derived from the payload, carried on the job itself —
/// immune to the shared-config mutation race the old per-request `parse_json_config` had once
/// requests ran concurrently.
fn generation_params(params: &ChatCompletionParamsSchema) -> GenerationParams {
    GenerationParams {
        max_tokens: params.max_tokens.map(|v| v as usize),
        temperature: params.temperature.map(f64::from),
        top_p: params.top_p.map(f64::from),
        seed: params.seed,
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
    let _stats = plugin.run_job(job).unwrap();
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
                let loading_stats = tokio::task::spawn_blocking({
                    let plugin = plugin.clone();
                    move || {
                        plugin.load().unwrap_or_else(|_| {
                            panic!("model '{}' should load", plugin.model_name())
                        })
                    }
                })
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
            let join_result = tokio::task::spawn_blocking({
                let plugin = plugin.clone();
                move || {
                    let result = plugin.run_job(job);
                    // Join the listener AFTER the job: guarantees every streamed chunk has been
                    // forwarded before stats/[DONE].
                    handle.join();
                    result
                }
            })
            .await;
            // The watcher holds a sender clone; abort it on EVERY path (including the panic
            // arm below) so the SSE channel can close — otherwise the stream never ends.
            disconnect_watcher.abort();

            let stats = match join_result {
                Err(join_err) => {
                    // The blocking task panicked. Today that means the listener thread died
                    // mid-stream (client disconnected -> the SSE write failed -> `join()`
                    // panicked), so treat it as a disconnect: close the stream instead of
                    // re-panicking the handler task.
                    tracing::debug!(
                        "generation task panicked (likely client disconnect): {join_err}"
                    );
                    let _ = tx.send(StreamingChunk::Done.to_event_stream()).await;
                    return;
                }
                Ok(Err(err)) => {
                    // Includes Err(Cancelled) for a job cancelled while still queued: the client
                    // is gone, so just close the stream instead of panicking the handler.
                    tracing::debug!("Inference did not complete: {err}");
                    let _ = tx.send(StreamingChunk::Done.to_event_stream()).await;
                    return;
                }
                Ok(Ok(stats)) => stats,
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

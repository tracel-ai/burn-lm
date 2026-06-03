use axum::{
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use burn_lm_inference::{
    InferenceJob, InferenceTask, StatEntry, TextGenerationListener, WriteListener,
};
use std::io::Write;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::{
    controllers::chat_controllers::ChatController,
    errors::ServerResult,
    schemas::chat_schemas::{
        ChatCompletionChunkSchema, ChatCompletionRequestSchema, ChatCompletionSchema,
        ChoiceMessageRoleSchema, ChoiceMessageSchema, ChoiceSchema, FinishReasonSchema,
        StreamingChunk, UsageSchema,
    },
    stores::chat_store::ModelStoreState,
    utils::id::ChatCompletionId,
};

pub const REPLY_MARKER: &str = "##### Model Reply";

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
    let mut store = state.lock().await;
    let (plugin, _) = store.get_plugin(&payload.model).await?;
    let messages: Vec<burn_lm_inference::Message> =
        payload.messages.into_iter().map(Into::into).collect();
    let json_params = serde_json::to_string(&payload.params)
        .expect("ChatCompletionParams should serialize to a JSON string");
    tracing::debug!("Json params from payload: {}", json_params);
    plugin.parse_json_config(&json_params);
    let task = InferenceTask::Context(messages);
    let (job, handle) = InferenceJob::create(task, TextGenerationListener::default());
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
    let (tx, rx) = mpsc::channel(10);
    tokio::spawn({
        async move {
            let mut store = state.lock().await;
            let id = ChatCompletionId::new().to_string();
            let (plugin, old_model_name) = store
                .get_plugin(&payload.model)
                .await
                .expect("should get model plugin");
            let json_params = serde_json::to_string(&payload.params)
                .expect("ChatCompletionParams should serialize to a JSON string");
            plugin.parse_json_config(&json_params);
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
            let (job, handle) = InferenceJob::create(task, listener);
            let stats = tokio::task::spawn_blocking({
                let plugin = plugin.clone();
                move || plugin.run_job(job).expect("should generate answer")
            })
            .await
            .expect("should complete answer generation");

            handle.join();
            let stats = format!("\n\n{}", stats.display_stats());
            let chunk =
                StreamingChunk::Data(ChatCompletionChunkSchema::new(&id, model, now, &stats));
            tx.send(chunk.to_event_stream())
                .await
                .expect("should send stats chunk");

            // Done chunk
            let done_chunk = StreamingChunk::Done;
            tx.send(done_chunk.to_event_stream())
                .await
                .expect("should send done chunk");
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

    #[tokio::test]
    async fn write_listener_streams_text_as_soon_as_it_is_emitted() {
        let (tx, mut rx) = mpsc::channel(1);
        let task = InferenceTask::Prompt("prompt".to_string());
        let listener = WriteListener::new(ChannelWriter { tx });
        let (job, handle) = InferenceJob::create(task, listener);

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
        let (job, handle) = InferenceJob::create(task, listener);

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

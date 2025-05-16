use axum::{
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use burnlm_inference::StatEntry;
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
    let messages: Vec<burnlm_inference::Message> = payload
        .messages
        .to_owned()
        .into_iter()
        .map(Into::into)
        .collect();
    let json_params = serde_json::to_string(&payload.params)
        .expect("ChatCompletionParams should serialize to a JSON string");
    tracing::debug!("Json params from payload: {}", json_params);
    plugin.parse_json_config(&json_params);
    let answer = plugin.run_completion(messages).unwrap();
    let content = answer.completion;
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
                    &format!("```BurnLM\nUnloaded model '{name}'!\n```\n\n"),
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
                    &format!("```BurnLM\nloading model '{}'... ", plugin.model_name()),
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
                    &format!("model loaded ! ✓{}\n```\n\n", loading_duration),
                ));
                tx.send(chunk.to_event_stream())
                    .await
                    .expect("should send end of loading model chunk");
            }

            // answer chunck
            let chunk = StreamingChunk::Data(ChatCompletionChunkSchema::new(
                &id,
                model,
                now,
                &format!("\n{REPLY_MARKER}\n"),
            ));
            tx.send(chunk.to_event_stream())
                .await
                .expect("should send reply section title chunk");
            let mut messages: Vec<burnlm_inference::Message> =
                payload.messages.into_iter().map(Into::into).collect();
            messages
                .iter_mut()
                .for_each(|m| m.cleanup(REPLY_MARKER, burnlm_inference::STATS_MARKER));
            tracing::debug!("Cleaned up messages: {:?}", messages);
            let answer = tokio::task::spawn_blocking({
                let plugin = plugin.clone();
                move || {
                    plugin
                        .run_completion(messages)
                        .expect("should generate answer")
                }
            })
            .await
            .expect("should complete answer generation");
            let content = format!("{}\n\n{}", answer.completion, answer.stats.display_stats());
            tracing::debug!("Answer: {}", answer.completion);
            let chunk =
                StreamingChunk::Data(ChatCompletionChunkSchema::new(&id, model, now, &content));
            tx.send(chunk.to_event_stream())
                .await
                .expect("should send answer chunk");

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

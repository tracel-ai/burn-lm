use axum::{
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tracing::{debug, info};

use crate::{
    controllers::model_controllers::ModelController,
    errors::ServerResult,
    schemas::chat_schemas::{
        ChatCompletionChunkSchema, ChatCompletionRequestSchema, ChatCompletionSchema,
        ChoiceMessageRoleSchema, ChoiceMessageSchema, ChoiceSchema, ChunkChoiceDeltaSchema,
        ChunkChoiceSchema, FinishReasonSchema, StreamingChunk, UsageSchema,
    },
    stores::model_store::ModelStoreState,
    utils::id::ChatCompletionId,
};

pub async fn chat_completions(
    State(state): State<ModelStoreState>,
    Json(payload): Json<ChatCompletionRequestSchema>,
) -> ServerResult<Response> {
    info!(
        "<<<<<<<<<<<< [CHAT COMPLETIONS] Received JSON payload: {:?} >>>>>>>>>>>>",
        payload
    );

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
    let store = state.lock().await;
    let plugin = store.get_model_plugin(&payload.model).await?;
    let messages: Vec<burnlm_inference::Message> = payload
        .messages
        .to_owned()
        .into_iter()
        .map(Into::into)
        .collect();
    let json_params = serde_json::to_string(&payload.params)
        .expect("ChatCompletionParams should serialize to a JSON string");
    info!("PARAMS JSON: {}", json_params);
    plugin.parse_json_config(&json_params);
    let answer = plugin.complete(messages).unwrap();
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
        let id = ChatCompletionId::new().to_string();
        async move {
            let store = state.lock().await;
            let plugin = store.get_model_plugin(&payload.model).await.unwrap();
            let json_params = serde_json::to_string(&payload.params)
                .expect("ChatCompletionParams should serialize to a JSON string");
            info!("PARAMS JSON: {}", json_params);
            plugin.parse_json_config(&json_params);
            let mut messages: Vec<burnlm_inference::Message> =
                payload.messages.iter().cloned().map(Into::into).collect();
            messages
                .iter_mut()
                .for_each(|m| m.remove_after(burnlm_inference::STATS_MARKER));
            debug!("MESSAGES CONTENT: {:?}", messages);
            let answer = plugin.complete(messages).unwrap();
            let content = format!("{}\n\n{}", answer.completion, answer.stats.display_stats());
            tracing::debug!("Answer: {}", answer.completion);
            let chunk = StreamingChunk::Data(ChatCompletionChunkSchema {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: plugin.model_name().to_owned(),
                choices: vec![ChunkChoiceSchema {
                    index: 0,
                    delta: Some(ChunkChoiceDeltaSchema {
                        role: None,
                        content: Some(content),
                    }),
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: "".to_string(),
                service_tier: None,
            });

            tx.send(chunk.to_event_stream())
                .await
                .expect("should send chunk_1");

            // Done chunk
            let done_chunk = StreamingChunk::Done;
            tx.send(done_chunk.to_event_stream())
                .await
                .expect("should send done_chunk");
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

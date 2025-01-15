use axum::{
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use rand::Rng;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tracing::info;

use crate::{
    errors::ServerResult,
    schemas::chat_schemas::{
        ChatCompletionChunkSchema, ChatCompletionRequestSchema, ChatCompletionSchema,
        ChoiceMessageRoleSchema, ChoiceMessageSchema, ChoiceSchema, ChunkChoiceDeltaSchema,
        ChunkChoiceSchema, FinishReasonSchema, StreamingChunk, UsageSchema,
    },
    utils::{id::ChatCompletionId, llm},
};

pub async fn chat_completions(
    Json(payload): Json<ChatCompletionRequestSchema>,
) -> ServerResult<Response> {
    info!(
        "<<<<<<<<<<<< [CHAT COMPLETIONS] Received JSON payload: {:?} >>>>>>>>>>>>",
        payload
    );

    if payload.stream {
        handle_streaming_response(&payload).await
    } else {
        handle_non_streaming_response(&payload).await
    }
}

async fn handle_non_streaming_response(
    payload: &ChatCompletionRequestSchema,
) -> ServerResult<Response> {
    let mut config = llm::Config::default();
    config.prompt = llm::forge_prompt(&payload.messages);
    config.seed = rand::thread_rng().gen::<u64>();
    config.temperature = 0.2;
    let content = llm::complete(&config);
    tracing::debug!("Answer: {content}");
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
    payload: &ChatCompletionRequestSchema,
) -> ServerResult<Response> {
    let (tx, rx) = mpsc::channel(10);
    tokio::spawn({
        let model = payload.model.clone();
        let messages = payload.messages.clone();
        let id = ChatCompletionId::new().to_string();
        async move {
            let mut config = llm::Config::default();
            config.prompt = llm::forge_prompt(&messages);
            config.seed = rand::thread_rng().gen::<u64>();
            let content = llm::complete(&config);
            tracing::debug!("Answer: {content}");
            let chunk = StreamingChunk::Data(ChatCompletionChunkSchema {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: model.clone(),
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

    let stream = ReceiverStream::new(rx).map(|item| Ok::<_, std::io::Error>(item));
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

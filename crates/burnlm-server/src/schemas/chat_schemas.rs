use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Deserialize, ToSchema)]
pub struct ChatCompletionRequestSchema {
    pub model: String,
    pub messages: Vec<ChoiceMessageSchema>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ChatCompletionSchema {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChoiceSchema>,
    pub usage: UsageSchema,
    pub system_fingerprint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct ChoiceMessageSchema {
    pub role: ChoiceMessageRoleSchema,
    pub content: String,
    pub refusal: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ChoiceSchema {
    pub index: u32,
    pub message: ChoiceMessageSchema,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: FinishReasonSchema,
}

#[derive(Clone, Debug, Serialize, Deserialize, strum::Display, ToSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceMessageRoleSchema {
    System,
    User,
    Assistant,
    Tool,
    Unknown(String),
}

#[derive(Default, Clone, Debug, Serialize, ToSchema)]
pub struct TokenDetailsSchema {
    pub cached_tokens: u32,
}

#[derive(Default, Clone, Debug, Serialize, ToSchema)]
pub struct CompletionTokenDetailsSchema {
    pub reasoning_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

#[derive(Default, Clone, Debug, Serialize, ToSchema)]
pub struct UsageSchema {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: TokenDetailsSchema,
    pub completion_tokens_details: CompletionTokenDetailsSchema,
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReasonSchema {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Unknown(String),
}

// Streaming -----------------------------------------------------------------

pub enum StreamingChunk {
    Data(ChatCompletionChunkSchema),
    Done,
}

impl StreamingChunk {
    pub fn to_event_stream(&self) -> String {
        match self {
            StreamingChunk::Data(data) => {
                format!(
                    "data: {}\n\n",
                    serde_json::to_string(data).expect("should serialize data")
                )
            }
            StreamingChunk::Done => "data: [DONE]\n\n".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ChatCompletionChunkSchema {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoiceSchema>,
    pub usage: Option<ChunkUsageSchema>,
    pub system_fingerprint: String,
    pub service_tier: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ChunkChoiceSchema {
    pub index: u32,
    pub delta: Option<ChunkChoiceDeltaSchema>,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<FinishReasonSchema>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ChunkChoiceDeltaSchema {
    pub role: Option<ChoiceMessageRoleSchema>,
    pub content: Option<String>,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ChunkUsageSchema {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

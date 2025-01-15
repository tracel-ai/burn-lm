use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, strum::Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
    Unknown(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub refusal: Option<String>,
}

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

impl Message {
    /// Removes from `self.content` everything after and including
    /// the first occurrence of `mark`.
    pub fn remove_after(&mut self, mark: &str) {
        if mark.is_empty() {
            return;
        }
        if let Some(pos) = self.content.find(mark) {
            self.content.truncate(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest(
        initial_content,
        mark,
        expected_content,
        case::marker_found("Hello, world! This is a test.", "world", "Hello, "),
        case::marker_not_found(
            "Hello, world! This is a test.",
            "foo",
            "Hello, world! This is a test."
        ),
        case::empty_marker("Hello, world!", "", "Hello, world!")
    )]
    fn test_remove_after(initial_content: &str, mark: &str, expected_content: &str) {
        let mut msg = Message {
            role: MessageRole::User,
            content: initial_content.to_string(),
            refusal: None,
        };
        msg.remove_after(mark);
        assert_eq!(
            msg.content, expected_content,
            "Content should be '{}' after removing marker '{}'",
            expected_content, mark
        );
    }
}

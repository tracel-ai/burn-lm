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
    /// Update 'content' to be the text between the first occurrence of 'start'
    /// and the last occurrence of 'end', excluding both markers.
    /// If either marker is empty, not found, or in the wrong order,
    /// the content remains unchanged.
    pub fn cleanup(&mut self, start: &str, end: &str) {
        if start.is_empty() || end.is_empty() {
            return;
        }
        if let Some(start_index) = self.content.find(start) {
            let content_start = start_index + start.len();
            if let Some(last_end_index) = self.content.rfind(end) {
                if last_end_index >= content_start {
                    // Update content to be the text between the markers
                    self.content = self.content[content_start..last_end_index].to_string();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest(
        initial_content,
        start,
        end,
        expected_content,
        case::markers_found(
            "Hello, [start]This is a test[end] Goodbye",
            "[start]",
            "[end]",
            "This is a test"
        ),
        case::start_marker_not_found(
            "Hello, This is a test[end] Goodbye",
            "[start]",
            "[end]",
            "Hello, This is a test[end] Goodbye"
        ),
        case::end_marker_not_found(
            "Hello, [start]This is a test. Goodbye",
            "[start]",
            "[end]",
            "Hello, [start]This is a test. Goodbye"
        ),
        case::both_markers_not_found(
            "Hello, world! This is a test.",
            "[start]",
            "[end]",
            "Hello, world! This is a test."
        ),
        case::empty_start_marker(
            "Hello, [start]This is a test[end] Goodbye",
            "",
            "[end]",
            "Hello, [start]This is a test[end] Goodbye"
        ),
        case::empty_end_marker(
            "Hello, [start]This is a test[end] Goodbye",
            "[start]",
            "",
            "Hello, [start]This is a test[end] Goodbye"
        ),
        case::multiple_occurrences(
            "Ignore [start]Keep this[end] and [start]not this[end] end part",
            "[start]",
            "[end]",
            "Keep this[end] and [start]not this"
        ),
        case::end_marker_before_start(
            "Hello [end] there [start] world",
            "[start]",
            "[end]",
            "Hello [end] there [start] world"
        ),
        case::same_marker("abcXdefXghi", "X", "X", "def")
    )]
    fn test_cleanup(initial_content: &str, start: &str, end: &str, expected_content: &str) {
        let mut msg = Message {
            role: MessageRole::User,
            content: initial_content.to_string(),
            refusal: None,
        };
        msg.cleanup(start, end);
        assert_eq!(
            msg.content, expected_content,
            "Content should be '{expected_content}' after cleaning up with start '{start}' and end '{end}'"
        );
    }
}

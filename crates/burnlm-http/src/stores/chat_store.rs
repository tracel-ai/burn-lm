use std::sync::Arc;

pub struct ChatStore {}

pub type ChatStoreState = Arc<tokio::sync::Mutex<ChatStore>>;

impl ChatStore {
    pub fn new() -> Self {
        Self {}
    }
}

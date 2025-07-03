use std::fmt;

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub struct ChatCompletionId {
    value: usize,
}

impl ChatCompletionId {
    pub fn new() -> Self {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        if value == usize::MAX {
            panic!("Memory ID overflowed");
        }
        Self { value }
    }
}

impl Default for ChatCompletionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ChatCompletionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chatcmpl-{}", self.value)
    }
}

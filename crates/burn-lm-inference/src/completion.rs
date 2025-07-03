use crate::Stats;

pub struct Completion {
    pub completion: String,
    pub stats: Stats,
}

impl Completion {
    /// Creates a new completion result.
    pub fn new(completion: &str) -> Self {
        Self {
            completion: completion.to_string(),
            stats: Stats::default(),
        }
    }
}

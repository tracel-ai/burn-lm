use std::any::Any;
use std::fmt::Debug;

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

pub trait InferenceChannel<Server: InferenceServer>: Send + Sync + Debug {
    /// Set configuration for inference.
    fn set_config(&self, config: Box<dyn Any>);

    /// Unload the model.
    fn unload(&self) -> InferenceResult<()>;

    /// Complete the passed prompt.
    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion>;
}

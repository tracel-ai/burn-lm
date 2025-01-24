use std::any::Any;

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

pub trait InferenceChannel<Server: InferenceServer>: Send + Sync {
    /// Set configuration for inference.
    fn set_config(&self, config: Box<dyn Any>);

    /// Return the selected version of the model.
    fn get_version(&self) -> String;

    /// Unload the model.
    fn unload(&self) -> InferenceResult<()>;

    /// Complete the passed prompt.
    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion>;
}

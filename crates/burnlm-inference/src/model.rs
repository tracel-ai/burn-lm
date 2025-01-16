use crate::{completion::Completion, errors::InferenceResult, message::Message, prompt::Prompt};

/// Marker trait for model configuration structure
pub trait InferenceModelConfig {}

/// Trait to implement so that a model can be used to make inferences
pub trait InferenceModel<C>: Default
where
    C: Default + clap::Parser,
{
    fn load(&mut self, config: C) -> InferenceResult<()>;
    fn unload(&mut self) -> InferenceResult<()>;
    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<Prompt>;
    fn complete(&self, prompt: Prompt, config: C) -> InferenceResult<Completion>;
}

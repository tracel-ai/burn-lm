use burn::prelude::Backend;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
// use burnlm_inference::plugin::*;
pub use crate::{
    completion::Completion, errors::InferenceResult, message::Message, prompt::Prompt,
};
pub use burnlm_macros::InferencePlugin;
pub use clap::{self, CommandFactory, Parser};
pub use inventory;
// ---------------------------------------------------------------------------

pub type ConfigFlagsFn = fn() -> clap::Command;
#[cfg(feature = "tch-cpu")]
pub type InferenceBackend = burn::backend::libtorch::LibTorch;
#[cfg(feature = "tch-gpu")]
pub type InferenceBackend = burn::backend::libtorch::LibTorch<burn::tensor::f16>;
#[cfg(feature = "wgpu")]
pub type InferenceBackend = burn::backend::wgpu::Wgpu;
#[cfg(feature = "cuda")]
pub type InferenceBackend = burn::backend::cuda_jit::CudaJit;

pub struct InferencePluginMetadata<B: Backend> {
    pub model_name: &'static str,
    pub model_name_lc: &'static str,
    pub config_flags_fn: ConfigFlagsFn,
    _phantom_b: PhantomData<B>,
}

impl<B: Backend> InferencePluginMetadata<B> {
    pub const fn new(
        model_name: &'static str,
        model_name_lc: &'static str,
        config_flags_fn: ConfigFlagsFn,
    ) -> Self {
        Self {
            model_name,
            model_name_lc,
            config_flags_fn,
            _phantom_b: PhantomData,
        }
    }
}

inventory::collect!(InferencePluginMetadata<InferenceBackend>);

/// Trait to enable to enabled inference on a plugin
pub trait InferencePlugin<C>: Default
where
    C: Default + clap::Parser,
{
    fn load(&mut self, config: C) -> InferenceResult<()>;
    fn unload(&mut self) -> InferenceResult<()>;
    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<Prompt>;
    fn complete(&mut self, prompt: Prompt, config: C) -> InferenceResult<Completion>;
}

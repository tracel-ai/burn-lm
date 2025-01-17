use burn::prelude::Backend;
use std::{any::Any, marker::PhantomData};

#[cfg(feature = "tch-cpu")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch;
    pub const INFERENCE_DEVICE: LibTorchDevice = LibTorchDevice::Cpu;
}
#[cfg(feature = "tch-gpu")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch<burn::tensor::f16>;
    #[cfg(not(target_os = "macos"))]
    pub const INFERENCE_DEVICE: LibTorchDevice = LibTorchDevice::Cuda(0);
    #[cfg(target_os = "macos")]
    pub const INFERENCE_DEVICE: LibTorchDevice = LibTorchDevice::Mps;
}
#[cfg(feature = "wgpu")]
pub mod burn_backend_types {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type InferenceBackend = Wgpu;
    pub const INFERENCE_DEVICE: WgpuDevice = WgpuDevice::default();
}
#[cfg(feature = "cuda")]
pub mod burn_backend_types {
    use burn::backend::{cuda_jit::CudaDevice, CudaJit};
    pub type InferenceBackend = CudaJit;
    pub const INFERENCE_DEVICE: CudaDevice = CudaDevice::default();
}

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
// use burnlm_inference::plugin::*;
pub use crate::{
    completion::Completion, errors::InferenceResult, message::Message, prompt::Prompt,
};
pub use burn_backend_types::*;
pub use burnlm_macros::InferencePlugin;
pub use clap::{self, CommandFactory, FromArgMatches, Parser};
pub use inventory;
// ---------------------------------------------------------------------------

pub type CreateCliFlagsFn = fn() -> clap::Command;
pub type ParseCliFlagsFn = fn(&clap::ArgMatches) -> Box<dyn Any>;
pub type CreatePluginFn = fn(Box<dyn Any>) -> Box<dyn InferencePlugin>;

pub struct InferencePluginMetadata<B: Backend> {
    pub model_name: &'static str,
    pub model_name_lc: &'static str,
    pub create_cli_flags_fn: CreateCliFlagsFn,
    pub parse_cli_flags_fn: ParseCliFlagsFn,
    pub create_plugin_fn: CreatePluginFn,
    _phantom_b: PhantomData<B>,
}

impl<B: Backend> InferencePluginMetadata<B> {
    pub const fn new(
        model_name: &'static str,
        model_name_lc: &'static str,
        create_cli_flags_fn: CreateCliFlagsFn,
        parse_cli_flags_fn: ParseCliFlagsFn,
        create_plugin_fn: CreatePluginFn,
    ) -> Self {
        Self {
            model_name,
            model_name_lc,
            create_cli_flags_fn,
            parse_cli_flags_fn,
            create_plugin_fn,
            _phantom_b: PhantomData,
        }
    }
}

inventory::collect!(InferencePluginMetadata<InferenceBackend>);

/// Plugin Inference associated functions
/// There are automatically derived when using the InferencePlugin derive.
pub trait InferencePluginAssociatedFn {
    /// Parse CLI flags from burnlm-cli and return a boxed plugin config object.
    fn parse_cli_config(args: &clap::ArgMatches) -> Box<dyn Any>
    where
        Self: Sized;
}

/// Plugin Inference interface
pub trait InferencePlugin {
    /// Create a new instance of plugin with the passed plugin configuration.
    fn new(config: Box<dyn Any>) -> Box<dyn InferencePlugin>
    where
        Self: Sized;

    /// Load the model.
    fn load(&mut self) -> InferenceResult<()>;

    /// Unload the model.
    fn unload(&mut self) -> InferenceResult<()>;

    /// Forge the prompt in the format expected by the model from the passed messages.
    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<Prompt>;

    /// Complete the passed prompt.
    fn complete(&mut self, prompt: Prompt) -> InferenceResult<Completion>;
}

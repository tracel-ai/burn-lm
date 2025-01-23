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
//                          model version    config
pub type CreatePluginFn = fn(Box<dyn Any>) -> Box<dyn InferencePlugin>;
pub type GetModelVersionsFn = fn() -> Vec<String>;

pub struct InferencePluginMetadata<B: Backend> {
    pub model_name: &'static str,
    pub model_name_lc: &'static str,
    pub model_creation_date: &'static str,
    pub owned_by: &'static str,
    pub create_cli_flags_fn: CreateCliFlagsFn,
    pub parse_cli_flags_fn: ParseCliFlagsFn,
    pub create_plugin_fn: CreatePluginFn,
    pub get_model_versions_fn: GetModelVersionsFn,
    _phantom_b: PhantomData<B>,
}

impl<B: Backend> InferencePluginMetadata<B> {
    pub const fn new(
        model_name: &'static str,
        model_name_lc: &'static str,
        model_creation_date: &'static str,
        owned_by: &'static str,
        create_cli_flags_fn: CreateCliFlagsFn,
        parse_cli_flags_fn: ParseCliFlagsFn,
        create_plugin_fn: CreatePluginFn,
        get_model_versions_fn: GetModelVersionsFn,
    ) -> Self {
        Self {
            model_name,
            model_name_lc,
            model_creation_date,
            owned_by,
            create_cli_flags_fn,
            parse_cli_flags_fn,
            create_plugin_fn,
            get_model_versions_fn,
            _phantom_b: PhantomData,
        }
    }
}

inventory::collect!(InferencePluginMetadata<InferenceBackend>);

/// Plugin Inference associated functions
/// There are automatically derived when using the InferencePlugin derive.
pub trait InferencePluginAssociatedFn {
    /// Returns supported versions for a given model.
    fn get_model_versions() -> Vec<String>
    where
        Self: Sized;

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

    /// Return the selected version of the model as a string.
    fn get_version(&self) -> String { "default".to_string() }

    /// Load the model.
    fn load(&mut self) -> InferenceResult<()>;

    /// Unload the model.
    fn unload(&mut self) -> InferenceResult<()>;

    /// Forge the prompt in the format expected by the model from the passed messages.
    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<Prompt>;

    /// Complete the passed prompt.
    fn complete(&mut self, prompt: Prompt) -> InferenceResult<Completion>;
}

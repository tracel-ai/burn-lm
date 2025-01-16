use burn::prelude::Backend;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
// use burnlm_inference::plugin::*;
pub use crate::{
    completion::Completion, errors::InferenceResult, message::Message, model::InferenceModel,
    prompt::Prompt,
};
pub use burnlm_macros::BurnLM;
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

pub struct InferenceModelPlugin<B: Backend> {
    pub name: &'static str,
    pub lc_name: &'static str,
    pub config_flags: ConfigFlagsFn,
    _phantom_b: PhantomData<B>,
}

impl<B: Backend> InferenceModelPlugin<B> {
    pub const fn new(
        name: &'static str,
        lc_name: &'static str,
        config_flags: ConfigFlagsFn,
    ) -> Self {
        Self {
            name,
            lc_name,
            config_flags,
            _phantom_b: PhantomData,
        }
    }
}

inventory::collect!(InferenceModelPlugin<InferenceBackend>);

pub mod channels;
pub mod client;
pub mod errors;
pub mod message;
pub mod plugin;
pub mod server;

pub type Completion = String;
pub type Prompt = String;

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
pub use crate::channels::mutex::MutexChannel;
pub use crate::channels::passthrough::SingleThreadedChannel;
pub use crate::client::InferenceClient;
pub use crate::errors::*;
pub use crate::message::{Message, MessageRole};
pub use crate::plugin::InferencePlugin;
pub use crate::server::{InferenceServer, InferenceServerConfig};
pub use burn::prelude::Backend;
pub use burn_backend_types::*;
pub use burnlm_macros::inference_server_config;
pub use burnlm_macros::InferenceServer;
// external re-export
pub use clap::{self, CommandFactory, FromArgMatches, Parser};
pub use serde::Deserialize;
pub use std::any::Any;
// ---------------------------------------------------------------------------

#[cfg(feature = "tch-cpu")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch;
    pub type InferenceDevice = LibTorchDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Cpu;
}
#[cfg(feature = "tch-gpu")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch<burn::tensor::f16>;
    pub type InferenceDevice = LibTorchDevice;
    #[cfg(not(target_os = "macos"))]
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Cuda(0);
    #[cfg(target_os = "macos")]
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Mps;
}
#[cfg(feature = "wgpu")]
pub mod burn_backend_types {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type InferenceBackend = Wgpu;
    pub type InferenceDevice = WgpuDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = WgpuDevice::default();
}
#[cfg(feature = "cuda")]
pub mod burn_backend_types {
    use burn::backend::{cuda_jit::CudaDevice, CudaJit};
    pub type InferenceBackend = CudaJit;
    pub type InferenceDevice = CudaDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = CudaDevice::default();
}

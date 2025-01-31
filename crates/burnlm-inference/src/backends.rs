// Candle --------------------------------------------------------------------

#[cfg(any(
    feature = "candle-accelerate",
    feature = "candle-cpu",
))]
pub mod burn_backend_types {
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = CandleDevice::Cpu;
}

#[cfg(feature = "candle-cuda")]
pub mod burn_backend_types {
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = CandleDevice::cuda(0);
}

#[cfg(feature = "candle-metal")]
pub mod burn_backend_types {
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = CandleDevice::metal(0);
}

// Cuda ----------------------------------------------------------------------

#[cfg(any(
    feature = "cuda",
    feature = "cuda-fusion",
))]
pub mod burn_backend_types {
    use burn::backend::cuda::{CudaDevice, Cuda};
    pub type InferenceBackend = Cuda;
    pub type InferenceDevice = CudaDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = CudaDevice::new(0);
}

// Hip -----------------------------------------------------------------------

#[cfg(feature = "hip")]
pub mod burn_backend_types {
    use burn::backend::hip::{Hip, HipDevice};
    pub type InferenceBackend = Hip;
    pub type InferenceDevice = HipDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = HipDevice::new(0);
}

// ndarray -------------------------------------------------------------------

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
pub mod burn_backend_types {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    pub type InferenceBackend = NdArray;
    pub type InferenceDevice = NdArrayDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = NdArrayDevice::Cpu;
}

// LibTorch ------------------------------------------------------------------

#[cfg(feature = "tch")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch<burn::tensor::f16>;
    pub type InferenceDevice = LibTorchDevice;
    #[cfg(not(target_os = "macos"))]
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Cuda(0);
    #[cfg(target_os = "macos")]
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Mps;
}

#[cfg(feature = "tch-cpu")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch;
    pub type InferenceDevice = LibTorchDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Cpu;
}

// WebGPU --------------------------------------------------------------------

#[cfg(any(
    feature = "wgpu",
    feature = "wgpu-fusion",
    feature = "wgpu-spirv",
    feature = "wgpu-spirv-fusion",
))]
pub mod burn_backend_types {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type InferenceBackend = Wgpu;
    pub type InferenceDevice = WgpuDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = WgpuDevice::DefaultDevice;
}

#[cfg(feature = "wgpu-cpu")]
pub mod burn_backend_types {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type InferenceBackend = Wgpu;
    pub type InferenceDevice = WgpuDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = WgpuDevice::Cpu;
}


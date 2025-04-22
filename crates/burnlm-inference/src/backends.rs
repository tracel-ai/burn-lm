// Candle --------------------------------------------------------------------

#[cfg(any(feature = "candle-accelerate", feature = "candle-cpu",))]
pub mod burn_backend_types {
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CandleDevice> =
        std::sync::LazyLock::new(|| CandleDevice::Cpu);
}

#[cfg(feature = "candle-cuda")]
pub mod burn_backend_types {
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CandleDevice> =
        std::sync::LazyLock::new(|| CandleDevice::cuda(0));
}

#[cfg(feature = "candle-metal")]
pub mod burn_backend_types {
    use burn::backend::candle::{Candle, CandleDevice};
    pub type InferenceBackend = Candle;
    pub type InferenceDevice = CandleDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CandleDevice> =
        std::sync::LazyLock::new(|| CandleDevice::metal(0));
}

// Cuda ----------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod burn_backend_types {
    use burn::backend::cuda::{Cuda, CudaDevice};
    pub type InferenceBackend = Cuda;
    pub type InferenceDevice = CudaDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<CudaDevice> =
        std::sync::LazyLock::new(|| CudaDevice::default());
}

// ROCm ----------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub mod burn_backend_types {
    use burn::backend::rocm::{Rocm, HipDevice};
    pub type InferenceBackend = Rocm;
    pub type InferenceDevice = HipDevice;
    pub const INFERENCE_DEVICE: std::sync::LazyLock<HipDevice> =
        std::sync::LazyLock::new(|| HipDevice::default());
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

#[cfg(feature = "libtorch")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch<burn::tensor::f16>;
    pub type InferenceDevice = LibTorchDevice;
    #[cfg(not(target_os = "macos"))]
    pub const INFERENCE_DEVICE: std::sync::LazyLock<LibTorchDevice> =
        std::sync::LazyLock::new(|| LibTorchDevice::Cuda(0));
    #[cfg(target_os = "macos")]
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Mps;
}

#[cfg(feature = "libtorch-cpu")]
pub mod burn_backend_types {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    pub type InferenceBackend = LibTorch;
    pub type InferenceDevice = LibTorchDevice;
    pub const INFERENCE_DEVICE: InferenceDevice = LibTorchDevice::Cpu;
}

// WebGPU (default) ----------------------------------------------------------

#[cfg(any(feature = "wgpu", feature = "vulkan",))]
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

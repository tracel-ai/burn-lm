mod elems {
    cfg_if::cfg_if! {
        // NOTE: f16/bf16 is not always supported on wgpu depending on the hardware
        // https://github.com/gfx-rs/wgpu/issues/7468
        if #[cfg(all(feature = "f16", any(feature = "cuda", feature = "wgpu", feature = "vulkan", feature = "metal", feature = "rocm", feature = "libtorch", feature = "candle-cuda")))]{
            pub type ElemType = burn::tensor::f16;
            pub const DTYPE_NAME: &str = "f16";
        }
        else if #[cfg(all(feature = "f16", any(feature = "cuda", feature = "wgpu", feature = "vulkan", feature = "metal", feature = "rocm", feature = "libtorch", feature = "candle-cuda")))]{
            pub type ElemType = burn::tensor::bf16;
            pub const DTYPE_NAME: &str = "bf16";
        } else {
            pub type ElemType = f32;
            pub const DTYPE_NAME: &str = "f32";
        }
    }
}

pub use elems::*;

use burn::tensor::{Device, DeviceIndex, DeviceKind};
use std::sync::LazyLock;

// Candle --------------------------------------------------------------------
// Candle is not part of the dispatch stack on Burn 0.22; use the default device.

#[cfg(any(feature = "candle-accelerate", feature = "candle-cpu"))]
pub mod burn_backend_types {
    use super::*;
    use burn_candle::{Candle, CandleDevice};

    pub type InferenceBackend = Candle;
    pub type InferenceBackendDevice = CandleDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<CandleDevice> =
        LazyLock::new(|| CandleDevice::Cpu);
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::default);
    pub const NAME: &str = "candle-cpu";
}

#[cfg(feature = "candle-cuda")]
pub mod burn_backend_types {
    use super::*;
    use burn_candle::{Candle, CandleDevice};

    pub type InferenceBackend = Candle;
    pub type InferenceBackendDevice = CandleDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<CandleDevice> =
        LazyLock::new(|| CandleDevice::cuda(0));
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::default);
    pub const NAME: &str = "candle-cuda";
}

#[cfg(feature = "candle-metal")]
pub mod burn_backend_types {
    use super::*;
    use burn_candle::{Candle, CandleDevice};

    pub type InferenceBackend = Candle;
    pub type InferenceBackendDevice = CandleDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<CandleDevice> =
        LazyLock::new(|| CandleDevice::metal(0));
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::default);
    pub const NAME: &str = "candle-metal";
}

// Cuda ----------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub type InferenceBackend = Cuda;
    pub type InferenceBackendDevice = CudaDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<CudaDevice> =
        LazyLock::new(CudaDevice::default);
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::cuda(DeviceIndex::Default));
    pub const NAME: &str = "cuda";
}

// ROCm ----------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::rocm::{Rocm, RocmDevice};

    pub type InferenceBackend = Rocm;
    pub type InferenceBackendDevice = RocmDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<RocmDevice> =
        LazyLock::new(RocmDevice::default);
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::rocm(DeviceIndex::Default));
    pub const NAME: &str = "rocm";
}

// ndarray -------------------------------------------------------------------
// This backend is used for testing and by default when no backend is selected.

#[cfg(any(feature = "ndarray", not(feature = "selected-backend")))]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    pub type InferenceBackend = NdArray;
    pub type InferenceBackendDevice = NdArrayDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<NdArrayDevice> =
        LazyLock::new(NdArrayDevice::default);
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::ndarray);
    pub const NAME: &str = "ndarray";
}

// LibTorch ------------------------------------------------------------------

#[cfg(feature = "libtorch")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub type InferenceBackend = LibTorch;
    pub type InferenceBackendDevice = LibTorchDevice;
    pub type InferenceDevice = Device;

    #[cfg(not(target_os = "macos"))]
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<LibTorchDevice> =
        LazyLock::new(|| LibTorchDevice::Cuda(0));
    #[cfg(target_os = "macos")]
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<LibTorchDevice> =
        LazyLock::new(|| LibTorchDevice::Mps);
    #[cfg(not(target_os = "macos"))]
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::libtorch_cuda(DeviceIndex::Default));
    #[cfg(target_os = "macos")]
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::libtorch_mps);
    pub const NAME: &str = "libtorch";
}

#[cfg(feature = "libtorch-cpu")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub type InferenceBackend = LibTorch;
    pub type InferenceBackendDevice = LibTorchDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<LibTorchDevice> =
        LazyLock::new(|| LibTorchDevice::Cpu);
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::libtorch);
    pub const NAME: &str = "libtorch-cpu";
}

// WebGPU --------------------------------------------------------------------

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub type InferenceBackend = Wgpu;
    pub type InferenceBackendDevice = WgpuDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<WgpuDevice> =
        LazyLock::new(|| WgpuDevice::DefaultDevice);
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::wgpu(DeviceKind::DefaultDevice));
    #[cfg(all(feature = "wgpu", not(feature = "vulkan"), not(feature = "metal")))]
    pub const NAME: &str = "wgpu";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "vulkan";
    #[cfg(feature = "metal")]
    pub const NAME: &str = "metal";
}

#[cfg(feature = "wgpu-cpu")]
pub mod burn_backend_types {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub type InferenceBackend = Wgpu;
    pub type InferenceBackendDevice = WgpuDevice;
    pub type InferenceDevice = Device;
    pub static INFERENCE_BACKEND_DEVICE: LazyLock<WgpuDevice> =
        LazyLock::new(|| WgpuDevice::Cpu);
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::wgpu(DeviceKind::Cpu));
    pub const NAME: &str = "wgpu-cpu";
}

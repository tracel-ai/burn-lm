mod elems {
    cfg_if::cfg_if! {
        // NOTE: f16/bf16 is not always supported on wgpu depending on the hardware
        // https://github.com/gfx-rs/wgpu/issues/7468
        if #[cfg(all(feature = "f16", any(feature = "cuda", feature = "wgpu", feature = "vulkan", feature = "metal", feature = "rocm", feature = "libtorch")))]{
            pub type ElemType = burn::tensor::f16;
            pub const DTYPE_NAME: &str = "f16";
        }
        else if #[cfg(all(feature = "f16", any(feature = "cuda", feature = "wgpu", feature = "vulkan", feature = "metal", feature = "rocm", feature = "libtorch")))]{
            pub type ElemType = burn::tensor::bf16;
            pub const DTYPE_NAME: &str = "bf16";
        } else {
            pub type ElemType = f32;
            pub const DTYPE_NAME: &str = "f32";
        }
    }
}

pub use elems::*;

use burn::tensor::Device;
use std::sync::LazyLock;

// Cuda ----------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod burn_backend_types {
    use super::*;

    pub type InferenceDevice = Device;
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::cuda(burn::tensor::DeviceIndex::Default));
    pub const NAME: &str = "cuda";
}

// ROCm ----------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub mod burn_backend_types {
    use super::*;

    pub type InferenceDevice = Device;
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::rocm(burn::tensor::DeviceIndex::Default));
    pub const NAME: &str = "rocm";
}

// ndarray -------------------------------------------------------------------
// This backend is used for testing and by default when no backend is selected.

#[cfg(any(feature = "ndarray", not(feature = "selected-backend")))]
pub mod burn_backend_types {
    use super::*;

    pub type InferenceDevice = Device;
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::ndarray);
    pub const NAME: &str = "ndarray";
}

// LibTorch ------------------------------------------------------------------

#[cfg(feature = "libtorch")]
pub mod burn_backend_types {
    use super::*;

    pub type InferenceDevice = Device;

    #[cfg(not(target_os = "macos"))]
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::libtorch_cuda(burn::tensor::DeviceIndex::Default));
    #[cfg(target_os = "macos")]
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::libtorch_mps);
    pub const NAME: &str = "libtorch";
}

#[cfg(feature = "libtorch-cpu")]
pub mod burn_backend_types {
    use super::*;

    pub type InferenceDevice = Device;
    pub static INFERENCE_DEVICE: LazyLock<Device> = LazyLock::new(Device::libtorch);
    pub const NAME: &str = "libtorch-cpu";
}

// WebGPU --------------------------------------------------------------------

#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
pub mod burn_backend_types {
    use super::*;

    pub type InferenceDevice = Device;
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::wgpu(burn::tensor::DeviceKind::DefaultDevice));
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

    pub type InferenceDevice = Device;
    pub static INFERENCE_DEVICE: LazyLock<Device> =
        LazyLock::new(|| Device::wgpu(burn::tensor::DeviceKind::Cpu));
    pub const NAME: &str = "wgpu-cpu";
}

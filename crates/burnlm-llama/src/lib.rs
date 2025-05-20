#![recursion_limit = "256"]

pub mod pretrained;
pub mod sampling;
pub mod tokenizer;

/// Neural network components.
pub mod nn;

mod base;

#[cfg(feature = "inference-server")]
pub mod server;

pub use base::*;

#[cfg(test)]
mod tests {
    #[cfg(any(not(feature = "test-non-default"), feature = "test-ndarray"))]
    pub type TestBackend = burn::backend::NdArray<f32, i32>;

    #[cfg(feature = "test-cuda")]
    pub type TestBackend = burn::backend::Cuda<f32, i32>;
    #[cfg(feature = "test-libtorch")]
    pub type TestBackend = burn::backend::LibTorch<f32>;
    #[cfg(feature = "test-wgpu")]
    pub type TestBackend = burn::backend::Wgpu<f32, i32>;

    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}

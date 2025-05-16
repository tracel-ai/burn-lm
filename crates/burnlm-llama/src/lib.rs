pub(crate) mod cache;
pub mod pretrained;
pub mod sampling;
pub mod tokenizer;

mod base;

/// Transformer model for Llama.
pub mod transformer;

pub use base::*;

#[cfg(test)]
mod tests {
    #[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "tch-gpu")))]
    pub type TestBackend = burn::backend::NdArray<f32, i32>;

    #[cfg(feature = "cuda")]
    pub type TestBackend = burn::backend::Cuda<f32, i32>;
    #[cfg(feature = "tch-gpu")]
    pub type TestBackend = burn::backend::LibTorch<f32>;

    pub type TestTensor<const D: usize> = burn::tensor::Tensor<TestBackend, D>;
}

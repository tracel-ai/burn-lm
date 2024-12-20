use std::time::Instant;

use burn::tensor::{backend::Backend, Device};
use llama_burn::{
    llama::{Llama, LlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};

use crate::schemas::chat_schemas::ChoiceMessageSchema;

const DEFAULT_PROMPT: &str = "";

#[derive(Debug)]
pub struct Config {
    /// Top-p probability threshold.
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    pub temperature: f64,
    /// Maximum sequence length for input text.
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    pub sample_len: usize,
    /// The seed to use when generating random samples.
    pub seed: u64,
    /// The input prompt.
    pub prompt: String,
    /// The Llama 3 model version.
    #[cfg(feature = "llama3")]
    pub model_version: Llama3,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            top_p: 0.9,
            temperature: 0.6,
            max_seq_len: 1024,
            sample_len: 1024,
            seed: 42,
            prompt: String::from(DEFAULT_PROMPT),
            #[cfg(feature = "llama3")]
            model_version: "llama-3.1-8b-instruct",
        }
    }
}

pub fn complete(config: &Config) -> String {
    #[cfg(feature = "tch-gpu")]
    return tch_gpu::run(config);
    #[cfg(feature = "tch-cpu")]
    return tch_cpu::run(config);
    #[cfg(feature = "wgpu")]
    return wgpu::run(config);
    #[cfg(feature = "cuda")]
    return cuda::run(config);
}

pub fn forge_prompt(messages: &Vec<ChoiceMessageSchema>) -> String {
    let mut prompt: Vec<String> =vec![];
    for message in messages {
        #[cfg(feature = "tinyllama")]
        prompt.push(format!("<|{}|>\n{}</s>\n", message.role.to_string(), message.content));
        #[cfg(feature = "llama3")]
        prompt.push(format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", message.role.to_string(), message.content));
    }
    let mut prompt = prompt.join("\n");
    #[cfg(feature = "tinyllama")]
    prompt.push_str("<|assistant|>\n");
    #[cfg(feature = "llama3")]
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt

}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) -> String {
    let now = Instant::now();
    let generated = llama.generate(prompt, sample_len, temperature, sampler);
    let elapsed = now.elapsed().as_secs();
    tracing::debug!(
        "{} tokens generated ({:.4} tokens/s)\n",
        generated.tokens,
        generated.tokens as f64 / generated.time
    );
    tracing::debug!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
    generated.text
}

pub fn chat<B: Backend>(config: &Config, device: Device<B>) -> String {
    // Sampling strategy
    let mut sampler = if config.temperature > 0.0 {
        Sampler::TopP(TopP::new(config.top_p, config.seed))
    } else {
        Sampler::Argmax
    };

    #[cfg(feature = "tinyllama")]
    let mut llama = LlamaConfig::tiny_llama_pretrained::<B>(config.max_seq_len, &device).unwrap();
    #[cfg(feature = "llama3")]
    let mut llama = LlamaConfig::llama3_8b_pretrained::<B>(config.max_seq_len, &device).unwrap();
    #[cfg(feature = "llama31")]
    let mut llama = LlamaConfig::llama3_1_8b_pretrained::<B>(config.max_seq_len, &device).unwrap();
    generate(
        &mut llama,
        &config.prompt,
        config.sample_len,
        config.temperature,
        &mut sampler,
    )
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use super::*;
    use burn::{
        backend::{libtorch::LibTorchDevice, LibTorch},
        tensor::f16,
    };

    pub fn run(config: &Config) -> String {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        chat::<LibTorch<f16>>(config, device)
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use super::*;
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};

    pub fn run(config: &Config) -> String {
        let device = LibTorchDevice::Cpu;

        chat::<LibTorch>(config, device)
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run(config: &Config) -> String {
        let device = WgpuDevice::default();

        chat::<Wgpu>(config, device)
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::{cuda_jit::CudaDevice, CudaJit};

    pub fn run(config: &Config) -> String {
        let device = CudaDevice::default();

        // NOTE: compilation errors in f16
        chat::<CudaJit<f32, i32>>(config, device)
    }
}


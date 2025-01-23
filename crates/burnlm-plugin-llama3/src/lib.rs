use clap::ValueEnum;
use rand::Rng;
use std::{any::Any, borrow::BorrowMut};

use burn::{prelude::Backend, tensor::Device};
use burnlm_plugin::*;
use llama_burn::{
    llama::{self, Llama},
    sampling::{Sampler, TopP},
    tokenizer::Tiktoken,
};

#[derive(Clone, Debug, clap::ValueEnum)]
/// Llama-3 model variants to load.
pub enum LlamaVersion {
    /// Llama-3-8B-Instruct.
    #[value(name = "llama-3-8b-instruct")]
    V3Instruct,
    /// Llama-3.1-8B-Instruct.
    #[value(name = "llama-3.1-8b-instruct")]
    V31Instruct,
}

impl Default for LlamaVersion {
    fn default() -> Self {
        LlamaVersion::V3Instruct
    }
}

#[derive(clap::Parser)]
pub struct Llama3PluginConfig {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = Llama3PluginConfig::default().top_p)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = Llama3PluginConfig::default().temperature)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = Llama3PluginConfig::default().max_seq_len)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, default_value_t = Llama3PluginConfig::default().sample_len)]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If u64::MAX then a random seed is used for each inference.
    #[arg(long, default_value_t = Llama3PluginConfig::default().seed)]
    pub seed: u64,
    /// The Llama 3 model version.
    #[arg(long)]
    pub model_version: LlamaVersion,
}

impl Default for Llama3PluginConfig {
    fn default() -> Self {
        Self {
            top_p: 0.9,
            temperature: 0.6,
            max_seq_len: 1024,
            sample_len: 1024,
            seed: u64::MAX,
            model_version: LlamaVersion::default(),
        }
    }
}

#[derive(InferencePlugin)]
#[inference_plugin(
    model_name = "Llama3",
    model_versions=LlamaVersion,
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc.",
)]
pub struct Llama3Plugin<B: Backend> {
    config: Llama3PluginConfig,
    device: Box<dyn Any>,
    model: Option<Llama<B, Tiktoken>>,
}

impl<B: Backend> InferencePlugin for Llama3Plugin<B> {
    fn new(config: Box<dyn Any>) -> Box<dyn InferencePlugin>
    where
        Self: Sized,
    {
        let config = config.downcast::<Llama3PluginConfig>().unwrap();
        Box::new(Self {
            config: *config,
            device: Box::new(INFERENCE_DEVICE),
            model: None,
        })
    }

    fn get_version(&self) -> String {
        self.config.model_version.to_possible_value().unwrap().get_name().to_string()
    }

    fn load(&mut self) -> InferenceResult<()> {
        let device = self.device.downcast_mut::<Device<B>>().unwrap();
        println!("Inference device used: {device:?}");
        if self.model.is_none() {
            self.model = match self.config.model_version {
                LlamaVersion::V3Instruct => Some(
                    llama::LlamaConfig::llama3_8b_pretrained::<B>(self.config.max_seq_len, &device)
                        .unwrap(),
                ),
                LlamaVersion::V31Instruct => Some(
                    llama::LlamaConfig::llama3_1_8b_pretrained::<B>(
                        self.config.max_seq_len,
                        &device,
                    )
                    .unwrap(),
                ),
            };
        }
        Ok(())
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.model = None;
        Ok(())
    }

    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<Prompt> {
        let mut prompt: Vec<String> = vec![];
        for message in messages {
            prompt.push(format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                message.role.to_string(),
                message.content
            ));
        }
        let mut prompt = prompt.join("\n");
        prompt.push_str("<|assistant|>\n");
        Ok(prompt)
    }

    fn complete(&mut self, prompt: Prompt) -> InferenceResult<Completion> {
        let model = match self.model.borrow_mut() {
            Some(m) => m,
            _ => return Err(burnlm_plugin::errors::InferenceError::ModelNotLoaded),
        };
        let seed = match self.config.seed {
            u64::MAX => rand::thread_rng().gen::<u64>(),
            s => s,
        };
        let mut sampler = if self.config.temperature > 0.0 {
            Sampler::TopP(TopP::new(self.config.top_p, seed))
        } else {
            Sampler::Argmax
        };
        println!("Generating...");
        let generated = model.generate(
            &prompt,
            self.config.sample_len,
            self.config.temperature,
            &mut sampler,
        );
        Ok(generated.text)
    }
}

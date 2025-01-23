use rand::Rng;
use std::{any::Any, borrow::BorrowMut};

use burn::{prelude::Backend, tensor::Device};
use burnlm_plugin::*;
use llama_burn::{
    llama::{self, Llama},
    sampling::{Sampler, TopP},
    tokenizer::SentiencePieceTokenizer,
};

#[derive(Parser)]
pub struct TinyLlamaPluginConfig {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = TinyLlamaPluginConfig::default().top_p)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = TinyLlamaPluginConfig::default().temperature)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = TinyLlamaPluginConfig::default().max_seq_len)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, default_value_t = TinyLlamaPluginConfig::default().sample_len)]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If u64::MAX then a random seed is used for each inference.
    #[arg(long, default_value_t = TinyLlamaPluginConfig::default().seed)]
    pub seed: u64,
}

impl Default for TinyLlamaPluginConfig {
    fn default() -> Self {
        Self {
            top_p: 0.9,
            temperature: 0.6,
            max_seq_len: 1024,
            sample_len: 1024,
            seed: u64::MAX,
        }
    }
}

#[derive(InferencePlugin)]
#[inference_plugin(
    model_name = "TinyLlama",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc.",
)]
pub struct TinyLlamaPlugin<B: Backend> {
    config: TinyLlamaPluginConfig,
    device: Box<dyn Any>,
    model: Option<Llama<B, SentiencePieceTokenizer>>,
}

impl<B: Backend> InferencePlugin for TinyLlamaPlugin<B> {
    fn new(config: Box<dyn Any>) -> Box<dyn InferencePlugin>
    where
        Self: Sized,
    {
        let config = config.downcast::<TinyLlamaPluginConfig>().unwrap();
        Box::new(Self {
            config: *config,
            device: Box::new(INFERENCE_DEVICE),
            model: None,
        })
    }

    fn load(&mut self) -> InferenceResult<()> {
        let device = self.device.downcast_mut::<Device<B>>().unwrap();
        println!("Inference device used: {device:?}");
        if self.model.is_none() {
            self.model = Some(
                llama::LlamaConfig::tiny_llama_pretrained::<B>(self.config.max_seq_len, &device)
                    .unwrap(),
            );
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
                "<|{}|>\n{}</s>\n",
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

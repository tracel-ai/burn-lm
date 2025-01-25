use rand::Rng;
use serde::Deserialize;
use std::{any::Any, borrow::BorrowMut};

use burn::prelude::Backend;
use burnlm_inference::*;
use llama_burn::{
    llama::{self, Llama},
    sampling::{Sampler, TopP},
    tokenizer::SentiencePieceTokenizer,
};

// #[derive(InferenceConfig, Parser, Deserialize, Debug)]
#[derive(Parser, Deserialize, Debug)]
pub struct TinyLlamaServerConfig {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default_top_p())]
    #[serde(default = "TinyLlamaServerConfig::default_top_p")]
    // #[config(default = 0.9)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default_temperature())]
    #[serde(default = "TinyLlamaServerConfig::default_temperature")]
    // #[config(default = 0.6)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default_max_seq_len())]
    #[serde(default = "TinyLlamaServerConfig::default_max_seq_len")]
    // #[config(default = 1024)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, default_value_t = TinyLlamaServerConfig::default_sample_len())]
    #[serde(default = "TinyLlamaServerConfig::default_sample_len")]
    // #[config(default = 1024)]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If it is 0 then a random seed is used for each inference.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default_seed())]
    #[serde(default = "TinyLlamaServerConfig::default_seed")]
    // #[config(default = 0)]
    pub seed: u64,
}

impl InferenceServerConfig for TinyLlamaServerConfig {}

impl TinyLlamaServerConfig {
    fn default_top_p() -> f64 { 0.9 }
    fn default_temperature() -> f64 { 0.6 }
    fn default_max_seq_len() -> usize { 1024 }
    fn default_sample_len() -> usize { 1024 }
    fn default_seed() -> u64 { 0 }
}

impl Default for TinyLlamaServerConfig {
    fn default() -> Self {
        Self {
            top_p: TinyLlamaServerConfig::default_top_p(),
            temperature: TinyLlamaServerConfig::default_temperature(),
            max_seq_len: TinyLlamaServerConfig::default_max_seq_len(),
            sample_len: TinyLlamaServerConfig::default_sample_len(),
            seed: TinyLlamaServerConfig::default_seed(),
        }
    }
}

#[derive(InferenceServer, Default, Debug)]
#[inference_server(
    model_name = "TinyLlama",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc.",
)]
pub struct TinyLlamaServer<B: Backend> {
    config: TinyLlamaServerConfig,
    model: Option<Llama<B, SentiencePieceTokenizer>>,
}

unsafe impl<B: Backend> Sync for TinyLlamaServer<B> {}

impl InferenceServer for TinyLlamaServer<InferenceBackend> {
    type Config = TinyLlamaServerConfig;

    fn set_config(&mut self, config: Box<dyn Any>) {
        self.config = *config.downcast::<TinyLlamaServerConfig>().unwrap();
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.model = None;
        Ok(())
    }

    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        println!("TinyLlama3Config: {:?}", self.config);
        self.load()?;
        let prompt = self.prompt(messages)?;
        let seed = match self.config.seed {
            0 => rand::thread_rng().gen::<u64>(),
            s => s,
        };
        let mut sampler = if self.config.temperature > 0.0 {
            Sampler::TopP(TopP::new(self.config.top_p, seed))
        } else {
            Sampler::Argmax
        };
        println!("Generating...");
        let generated = match self.model.borrow_mut() {
            Some(model) => model.generate(
                &prompt,
                self.config.sample_len,
                self.config.temperature,
                &mut sampler,
            ),
            _ => return Err(InferenceError::ModelNotLoaded),
        };
        Ok(generated.text)
    }
}

impl TinyLlamaServer<InferenceBackend> {
    fn load(&mut self) -> InferenceResult<()> {
        if self.model.is_none() {
            self.model = Some(
                llama::LlamaConfig::tiny_llama_pretrained::<InferenceBackend>(self.config.max_seq_len, &INFERENCE_DEVICE)
                    .unwrap(),
            );
        }
        Ok(())
    }

    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<burnlm_inference::Prompt> {
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
}

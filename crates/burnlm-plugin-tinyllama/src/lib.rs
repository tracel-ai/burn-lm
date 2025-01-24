use rand::Rng;
use std::{any::Any, borrow::BorrowMut};

use burn::prelude::Backend;
use burnlm_inference::*;
use llama_burn::{
    llama::{self, Llama},
    sampling::{Sampler, TopP},
    tokenizer::SentiencePieceTokenizer,
};

#[derive(Parser)]
pub struct TinyLlamaServerConfig {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default().top_p)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default().temperature)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default().max_seq_len)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, default_value_t = TinyLlamaServerConfig::default().sample_len)]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If u64::MAX then a random seed is used for each inference.
    #[arg(long, default_value_t = TinyLlamaServerConfig::default().seed)]
    pub seed: u64,
}

impl InferenceServerConfig for TinyLlamaServerConfig {}

impl Default for TinyLlamaServerConfig {
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

#[derive(InferenceServer, Default)]
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
        self.load()?;
        let prompt = self.prompt(messages)?;
        let model = match self.model.borrow_mut() {
            Some(m) => m,
            _ => return Err(InferenceError::ModelNotLoaded),
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

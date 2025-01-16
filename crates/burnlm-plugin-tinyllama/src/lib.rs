use burn::prelude::Backend;
use burnlm_inference::plugin::*;
use llama_burn::{
    llama::{self, Llama},
    tokenizer::SentiencePieceTokenizer,
};

#[derive(Parser)]
pub struct TinyLlamaConfig {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = TinyLlamaConfig::default().top_p)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = TinyLlamaConfig::default().temperature)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = TinyLlamaConfig::default().max_seq_len)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, default_value_t = TinyLlamaConfig::default().sample_len)]
    pub sample_len: usize,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = TinyLlamaConfig::default().seed)]
    pub seed: u64,
}

impl Default for TinyLlamaConfig {
    fn default() -> Self {
        Self {
            top_p: 0.9,
            temperature: 0.6,
            max_seq_len: 1024,
            sample_len: 1024,
            seed: 42,
        }
    }
}

#[derive(BurnLM)]
pub struct TinyLlama<B: Backend> {
    device: B::Device,
    model: Option<Llama<B, SentiencePieceTokenizer>>,
}

impl<B: Backend> Default for TinyLlama<B> {
    fn default() -> Self {
        Self {
            device: B::Device::default(),
            model: None,
        }
    }
}

impl<B: Backend> InferenceModel<TinyLlamaConfig> for TinyLlama<B> {
    fn load(&mut self, config: TinyLlamaConfig) -> InferenceResult<()> {
        if self.model.is_none() {
            self.model = Some(
                llama::LlamaConfig::tiny_llama_pretrained::<B>(config.max_seq_len, &self.device)
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

    fn complete(&self, _prompt: String, _config: TinyLlamaConfig) -> InferenceResult<Completion> {
        Ok("".to_string())
    }
}

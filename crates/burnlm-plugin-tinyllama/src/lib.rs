use burnlm_inference::{
    completion::Completion, errors::InferenceResult, message::Message, model::InferenceModel,
    plugin::InferenceModelPlugin, prompt::Prompt,
};
use burnlm_macros::BurnLM;
use clap::CommandFactory;

#[derive(clap::Parser)]
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
pub struct TinyLlama {}

impl InferenceModel<TinyLlamaConfig> for TinyLlama {
    fn load(&self, _config: TinyLlamaConfig) -> InferenceResult<()> {
        Ok(())
    }

    fn unload(&self) -> InferenceResult<()> {
        Ok(())
    }

    fn prompt(&self, _messages: Vec<Message>) -> InferenceResult<Prompt> {
        Ok("".to_string())
    }

    fn complete(&self, _prompt: String, _config: TinyLlamaConfig) -> InferenceResult<Completion> {
        Ok("".to_string())
    }
}

use burn::prelude::Backend;
use burnlm_inference::plugin::*;
use llama_burn::{
    llama::{self, Llama},
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
pub struct Llama3Config {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = Llama3Config::default().top_p)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = Llama3Config::default().temperature)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = Llama3Config::default().max_seq_len)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, default_value_t = Llama3Config::default().sample_len)]
    pub sample_len: usize,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = Llama3Config::default().seed)]
    pub seed: u64,
    /// The Llama 3 model version.
    #[arg(long)]
    pub model_version: LlamaVersion,
}

impl Default for Llama3Config {
    fn default() -> Self {
        Self {
            top_p: 0.9,
            temperature: 0.6,
            max_seq_len: 1024,
            sample_len: 1024,
            seed: 42,
            model_version: LlamaVersion::default(),
        }
    }
}

#[derive(BurnLM)]
pub struct Llama3<B: Backend> {
    device: B::Device,
    model: Option<Llama<B, Tiktoken>>,
}

impl<B: Backend> Default for Llama3<B> {
    fn default() -> Self {
        Self {
            device: B::Device::default(),
            model: None,
        }
    }
}

impl<B: Backend> InferenceModel<Llama3Config> for Llama3<B> {
    fn load(&mut self, config: Llama3Config) -> InferenceResult<()> {
        self.model = match config.model_version {
            LlamaVersion::V3Instruct => Some(
                llama::LlamaConfig::llama3_8b_pretrained::<B>(config.max_seq_len, &self.device)
                    .unwrap(),
            ),
            LlamaVersion::V31Instruct => Some(
                llama::LlamaConfig::llama3_1_8b_pretrained::<B>(config.max_seq_len, &self.device)
                    .unwrap(),
            ),
        };
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

    fn complete(&self, _prompt: String, _config: Llama3Config) -> InferenceResult<Completion> {
        Ok("".to_string())
    }
}

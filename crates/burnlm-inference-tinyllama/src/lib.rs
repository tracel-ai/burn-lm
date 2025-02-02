use rand::Rng;
use serde::Deserialize;
use std::borrow::BorrowMut;

use burn::prelude::Backend;
use burnlm_inference::*;
use llama_burn::{
    llama::{self, Llama},
    pretrained::{self, ModelMeta},
    sampling::{Sampler, TopP},
    tokenizer::SentiencePieceTokenizer,
};

#[inference_server_config]
pub struct TinyLlamaServerConfig {
    /// Top-p probability threshold.
    #[config(default = 0.9)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[config(default = 0.6)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[config(default = 1024)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[config(default = 128, openwebui_param = "max_tokens")]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If it is 0 then a random seed is used for each inference.
    #[config(default = 0)]
    pub seed: u64,
}

#[derive(InferenceServer, Default, Debug)]
#[inference_server(
    model_name = "TinyLlama",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc."
)]
pub struct TinyLlamaServer<B: Backend> {
    config: TinyLlamaServerConfig,
    model: Option<Llama<B, SentiencePieceTokenizer>>,
}

impl InferenceServer for TinyLlamaServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<()>> {
        Some(|| {
            let model = pretrained::Llama::TinyLlama.pretrained();
            model.download_weights().map_err(|err| {
                InferenceError::DownloadWeightError(Self::model_name().to_string(), err.to_string())
            })?;
            model.download_tokenizer().map_err(|err| {
                InferenceError::DownloadTokenizerError(
                    Self::model_name().to_string(),
                    err.to_string(),
                )
            })?;
            Ok(())
        })
    }

    fn is_downloaded(&mut self) -> bool {
        let model = pretrained::Llama::TinyLlama.pretrained();
        model.is_downloaded()
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.model = None;
        Ok(())
    }

    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        println!("{:?}", self.config);
        println!("Burn device: {:?}", INFERENCE_DEVICE);
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
                llama::LlamaConfig::tiny_llama_pretrained::<InferenceBackend>(
                    self.config.max_seq_len,
                    &INFERENCE_DEVICE,
                )
                .unwrap(),
            );
        }
        Ok(())
    }

    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<burnlm_inference::Prompt> {
        let mut prompt: Vec<String> = vec![];
        for message in messages {
            prompt.push(format!("<|{}|>\n{}</s>\n", message.role, message.content));
        }
        let mut prompt = prompt.join("\n");
        prompt.push_str("<|assistant|>\n");
        Ok(prompt)
    }
}

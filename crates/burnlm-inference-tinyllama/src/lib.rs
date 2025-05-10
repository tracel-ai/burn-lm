#![recursion_limit = "256"]

use rand::Rng;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

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
    #[config(default = 0.1)]
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

#[derive(InferenceServer, Clone, Default, Debug)]
#[inference_server(
    model_name = "TinyLlama",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc."
)]
pub struct TinyLlamaServer<B: Backend> {
    config: TinyLlamaServerConfig,
    model: Option<Arc<Mutex<Llama<B, SentiencePieceTokenizer>>>>,
}

impl InferenceServer for TinyLlamaServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        Some(|| {
            let now = std::time::Instant::now();
            let model = pretrained::Llama::TinyLlama.pretrained();
            model.download_weights().map_err(|err| {
                InferenceError::DownloadError(Self::model_name().to_string(), err.to_string())
            })?;
            model.download_tokenizer().map_err(|err| {
                InferenceError::DownloadError(Self::model_name().to_string(), err.to_string())
            })?;
            let mut stats = Stats::new();
            stats
                .entries
                .insert(StatEntry::ModelDownloadingDuration(now.elapsed()));
            Ok(Some(stats))
        })
    }

    fn is_downloaded(&mut self) -> bool {
        let model = pretrained::Llama::TinyLlama.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        Some(|| {
            let model = pretrained::Llama::TinyLlama.pretrained();
            model.delete_weights().map_err(|err| {
                InferenceError::DeleteError(Self::model_name().to_string(), err.to_string())
            })?;
            model.delete_tokenizer().map_err(|err| {
                InferenceError::DeleteError(Self::model_name().to_string(), err.to_string())
            })?;
            Ok(None)
        })
    }

    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        if !self.is_loaded() {
            let now = std::time::Instant::now();
            let model = llama::LlamaConfig::tiny_llama_pretrained::<InferenceBackend>(
                self.config.max_seq_len,
                &INFERENCE_DEVICE,
            )
            .unwrap();
            self.model = Some(Arc::new(Mutex::new(model)));
            let mut stats = Stats::new();
            stats
                .entries
                .insert(StatEntry::ModelLoadingDuration(now.elapsed()));
            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }

    fn is_loaded(&mut self) -> bool {
        self.model.is_some()
    }

    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        if let Some(arc_model) = self.model.take() {
            match Arc::try_unwrap(arc_model) {
                Ok(mutex) => {
                    let model = mutex
                        .into_inner()
                        .expect("should be able to extract model from mutex");
                    drop(model);
                }
                Err(_) => {
                    return Err(InferenceError::UnloadError(
                        Self::model_name().to_string(),
                        "Multiple references exist".to_string(),
                    ))
                }
            }
        }
        Ok(None)
    }

    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        let load_stats = self.load()?;
        let prompt = self.prompt(messages)?;
        let seed = match self.config.seed {
            0 => rand::rng().random::<u64>(),
            s => s,
        };
        let mut sampler = if self.config.temperature > 0.0 {
            Sampler::TopP(TopP::new(self.config.top_p, seed))
        } else {
            Sampler::Argmax
        };
        let generated = match &self.model {
            Some(arc_model) => arc_model
                .lock()
                .expect("should be able to lock the model for inference")
                .generate(
                    &prompt,
                    self.config.sample_len,
                    self.config.temperature,
                    &mut sampler,
                ),
            _ => return Err(InferenceError::ModelNotLoaded),
        };
        let mut completion = Completion::new(&generated.text);
        let mut total_duration = generated.time;
        completion.stats.entries.extend(vec![
            StatEntry::InferenceDuration(generated.time),
            StatEntry::TokensCount(generated.tokens),
            StatEntry::TokensPerSecond(generated.tokens, generated.time),
        ]);
        if let Some(stats) = load_stats {
            let model_loading = stats
                .entries
                .iter()
                .find(|e| matches!(e, StatEntry::ModelLoadingDuration(_)));
            if let Some(stat) = model_loading {
                total_duration += stat
                    .get_duration()
                    .expect("should be a ModelLoadingDuration stat")
            }
            completion.stats.entries.extend(stats.entries);
        }
        completion
            .stats
            .entries
            .insert(StatEntry::TotalDuration(total_duration));
        Ok(completion)
    }
}

impl TinyLlamaServer<InferenceBackend> {
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

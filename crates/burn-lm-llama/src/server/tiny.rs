use rand::Rng;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

use crate::{
    generation::{GenerationError, Sampler, TopP},
    pretrained::ModelMeta,
    tokenizer::SentiencePieceTokenizer,
    Llama, LlamaConfig, TinyLlamaVersion,
};
use burn::prelude::Backend;
use burn_lm_inference::{InferenceJob, *};

#[inference_server_config]
pub struct TinyLlamaServerConfig {
    /// Top-p probability threshold.
    #[config(default = 0.9)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[config(default = 0.0)]
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
    model_creation_date = "30/12/2023",
    owned_by = "Tracel Technologies Inc."
)]
pub struct TinyLlamaServer<B: Backend> {
    config: TinyLlamaServerConfig,
    model: Option<Arc<Mutex<Llama<B, SentiencePieceTokenizer>>>>,
}

impl TinyLlamaServer<InferenceBackend> {
    fn run_prompt(
        &mut self,
        prompt: Prompt,
        emitter: GeneratedItemEmitter,
    ) -> InferenceResult<Stats> {
        let load_stats = self.load()?;
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
            Some(arc_model) => match arc_model
                .lock()
                .expect("should be able to lock the model for inference")
                .generate(
                    &prompt,
                    self.config.sample_len,
                    self.config.temperature,
                    &mut sampler,
                    emitter,
                ) {
                Ok(result) => result,
                Err(GenerationError::MaxSequenceLengthExceeded { actual, max }) => {
                    return Err(InferenceError::ContextLengthExceeded(actual, max));
                }
            },
            _ => return Err(InferenceError::ModelNotLoaded),
        };
        let mut stats = Stats::default();
        let mut total_duration = generated.time;
        stats.entries.extend(vec![
            StatEntry::InferenceDuration(generated.time),
            StatEntry::TokensCount(generated.tokens),
            StatEntry::TokensPerSecond(generated.tokens, generated.time),
        ]);
        if let Some(load_stats) = load_stats {
            let model_loading = load_stats
                .entries
                .iter()
                .find(|e| matches!(e, StatEntry::ModelLoadingDuration(_)));
            if let Some(model_stats) = model_loading {
                total_duration += model_stats
                    .get_duration()
                    .expect("should be a ModelLoadingDuration stat")
            }
            stats.entries.extend(load_stats.entries);
        }
        stats
            .entries
            .insert(StatEntry::TotalDuration(total_duration));
        Ok(stats)
    }
}
impl InferenceServer for TinyLlamaServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        Some(|| {
            let now = std::time::Instant::now();
            let model = TinyLlamaVersion::V1.pretrained();
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
        let model = TinyLlamaVersion::V1.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        Some(|| {
            let model = TinyLlamaVersion::V1.pretrained();
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
            let model = LlamaConfig::tiny_llama_pretrained::<InferenceBackend>(
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

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        let prompt = match job.task {
            InferenceTask::Message(message) => self.prompt(vec![message])?,
            InferenceTask::Context(messages) => self.prompt(messages)?,
            InferenceTask::Prompt(prompt) => prompt,
        };
        self.run_prompt(prompt, job.emitter)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        match &self.model {
            Some(arc_model) => {
                let mut model = arc_model
                    .lock()
                    .expect("should lock the model for inference");
                model.reset();
                Ok(())
            }
            None => Err(InferenceError::ModelNotLoaded),
        }
    }
}

impl TinyLlamaServer<InferenceBackend> {
    fn prompt(&self, messages: Vec<Message>) -> InferenceResult<burn_lm_inference::Prompt> {
        let mut prompt: Vec<String> = vec![];
        for message in messages {
            prompt.push(format!("<|{}|>\n{}</s>\n", message.role, message.content));
        }
        let mut prompt = prompt.join("\n");
        prompt.push_str("<|assistant|>\n");
        Ok(prompt)
    }
}

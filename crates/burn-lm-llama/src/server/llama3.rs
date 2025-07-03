use rand::Rng;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

use crate::{
    generation::{GenerationError, Sampler, TopP},
    pretrained::ModelMeta,
    tokenizer::Tiktoken,
    Llama, LlamaConfig, LlamaVersion,
};
use burn::prelude::Backend;
use burn_lm_inference::*;

#[inference_server_config]
pub struct Llama3ServerConfig {
    /// Top-p probability threshold.
    #[config(default = 0.9)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[config(default = 0.0)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[config(default = 4096)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[config(default = 512, openwebui_param = "max_tokens")]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If it is 0 then a random seed is used for each inference.
    #[config(default = 0)]
    pub seed: u64,
}

#[derive(InferenceServer, Clone, Debug)]
#[inference_server(
    model_name = "Llama 3 (8B Instruct)",
    model_cli_param_name = "llama3",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc."
)]
pub struct Llama3InstructServer<B: Backend> {
    config: Llama3ServerConfig,
    server: Llama3BaseServer<B>,
}

impl<B: Backend> Default for Llama3InstructServer<B> {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::<B>::new(LlamaVersion::Llama3Instruct),
        }
    }
}

fn llama_downloader(version: LlamaVersion, name: &'static str) -> InferenceResult<Option<Stats>> {
    let now = std::time::Instant::now();
    let model = LlamaVersion::pretrained(&version);
    model
        .download_weights()
        .map_err(|err| InferenceError::DownloadError(name.to_string(), err.to_string()))?;
    model
        .download_tokenizer()
        .map_err(|err| InferenceError::DownloadError(name.to_string(), err.to_string()))?;
    let mut stats = Stats::new();
    stats
        .entries
        .insert(StatEntry::ModelDownloadingDuration(now.elapsed()));
    Ok(Some(stats))
}

fn llama_deleter(version: LlamaVersion, name: &'static str) -> InferenceResult<Option<Stats>> {
    let model = LlamaVersion::pretrained(&version);
    model
        .delete_weights()
        .map_err(|err| InferenceError::DeleteError(name.to_string(), err.to_string()))?;
    model
        .delete_tokenizer()
        .map_err(|err| InferenceError::DeleteError(name.to_string(), err.to_string()))?;
    Ok(None)
}

impl InferenceServer for Llama3InstructServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama3Instruct,
                Llama3InstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = LlamaVersion::Llama3Instruct.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn deleter() -> InferenceResult<Option<Stats>> {
            llama_deleter(
                LlamaVersion::Llama3Instruct,
                Llama3InstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(deleter)
    }

    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.load(&self.config)
    }

    fn is_loaded(&mut self) -> bool {
        self.server.is_loaded()
    }

    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.unload(Self::model_name())
    }

    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(InferenceServer, Clone, Debug)]
#[inference_server(
    model_name = "Llama 3.1 (8B Instruct)",
    model_cli_param_name = "llama31",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc."
)]
pub struct Llama31InstructServer<B: Backend> {
    config: Llama3ServerConfig,
    server: Llama3BaseServer<B>,
}

impl<B: Backend> Default for Llama31InstructServer<B> {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::<B>::new(LlamaVersion::Llama31Instruct),
        }
    }
}

impl InferenceServer for Llama31InstructServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama31Instruct,
                Llama31InstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = LlamaVersion::Llama31Instruct.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn deleter() -> InferenceResult<Option<Stats>> {
            llama_deleter(
                LlamaVersion::Llama31Instruct,
                Llama31InstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(deleter)
    }

    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.load(&self.config)
    }

    fn is_loaded(&mut self) -> bool {
        self.server.is_loaded()
    }

    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.unload(Self::model_name())
    }

    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(InferenceServer, Clone, Debug)]
#[inference_server(
    model_name = "Llama 3.2 (1B Instruct)",
    model_cli_param_name = "llama32",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc."
)]
pub struct Llama321bInstructServer<B: Backend> {
    config: Llama3ServerConfig,
    server: Llama3BaseServer<B>,
}

impl<B: Backend> Default for Llama321bInstructServer<B> {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::<B>::new(LlamaVersion::Llama321bInstruct),
        }
    }
}

impl InferenceServer for Llama321bInstructServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama321bInstruct,
                Llama321bInstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = LlamaVersion::Llama321bInstruct.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn deleter() -> InferenceResult<Option<Stats>> {
            llama_deleter(
                LlamaVersion::Llama321bInstruct,
                Llama321bInstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(deleter)
    }

    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.load(&self.config)
    }

    fn is_loaded(&mut self) -> bool {
        self.server.is_loaded()
    }

    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.unload(Self::model_name())
    }

    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(InferenceServer, Clone, Debug)]
#[inference_server(
    model_name = "Llama 3.2 (3B Instruct)",
    model_cli_param_name = "llama32-3b",
    model_creation_date = "05/01/2024",
    owned_by = "Tracel Technologies Inc."
)]
pub struct Llama323bInstructServer<B: Backend> {
    config: Llama3ServerConfig,
    server: Llama3BaseServer<B>,
}

impl<B: Backend> Default for Llama323bInstructServer<B> {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::<B>::new(LlamaVersion::Llama323bInstruct),
        }
    }
}

impl InferenceServer for Llama323bInstructServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama323bInstruct,
                Llama323bInstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = LlamaVersion::Llama323bInstruct.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn deleter() -> InferenceResult<Option<Stats>> {
            llama_deleter(
                LlamaVersion::Llama323bInstruct,
                Llama323bInstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(deleter)
    }

    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.load(&self.config)
    }

    fn is_loaded(&mut self) -> bool {
        self.server.is_loaded()
    }

    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.server.unload(Self::model_name())
    }

    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Llama3BaseServer<B: Backend> {
    model: Option<Arc<Mutex<Llama<B, Tiktoken>>>>,
    version: LlamaVersion,
}

unsafe impl<B: Backend> Sync for Llama3BaseServer<B> {}

impl<B: Backend> Llama3BaseServer<B> {
    pub fn new(version: LlamaVersion) -> Self {
        Self {
            model: None,
            version,
        }
    }
}

impl Llama3BaseServer<InferenceBackend> {
    fn unload(&mut self, model_name: &str) -> InferenceResult<Option<Stats>> {
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
                        model_name.to_string(),
                        "Multiple references exist".to_string(),
                    ))
                }
            }
        }
        Ok(None)
    }

    fn complete(
        &mut self,
        messages: Vec<Message>,
        config: &Llama3ServerConfig,
    ) -> InferenceResult<Completion> {
        let load_stats = self.load(config)?;
        let prompt = self.prompt(messages)?;
        let seed = match config.seed {
            0 => rand::rng().random::<u64>(),
            s => s,
        };
        let mut sampler = if config.temperature > 0.0 {
            Sampler::TopP(TopP::new(config.top_p, seed))
        } else {
            Sampler::Argmax
        };
        let generated = match &self.model {
            Some(arc_model) => {
                let mut model = arc_model
                    .lock()
                    .expect("should lock the model for inference");
                match model.generate(&prompt, config.sample_len, config.temperature, &mut sampler) {
                    Ok(result) => result,
                    Err(GenerationError::MaxSequenceLengthExceeded { actual, max }) => {
                        return Err(InferenceError::ContextLengthExceeded(actual, max));
                    }
                }
            }
            None => return Err(InferenceError::ModelNotLoaded),
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

    fn load(&mut self, config: &Llama3ServerConfig) -> InferenceResult<Option<Stats>> {
        if !self.is_loaded() {
            let now = std::time::Instant::now();
            let model = match self.version {
                LlamaVersion::Llama3Instruct => {
                    LlamaConfig::llama3_8b_pretrained::<InferenceBackend>(
                        config.max_seq_len,
                        &INFERENCE_DEVICE,
                    )
                    .unwrap()
                }
                LlamaVersion::Llama31Instruct => LlamaConfig::llama3_1_8b_pretrained::<
                    InferenceBackend,
                >(
                    config.max_seq_len, &INFERENCE_DEVICE
                )
                .unwrap(),
                LlamaVersion::Llama323bInstruct => LlamaConfig::llama3_2_3b_pretrained::<
                    InferenceBackend,
                >(
                    config.max_seq_len, &INFERENCE_DEVICE
                )
                .unwrap(),
                LlamaVersion::Llama321bInstruct => LlamaConfig::llama3_2_1b_pretrained::<
                    InferenceBackend,
                >(
                    config.max_seq_len, &INFERENCE_DEVICE
                )
                .unwrap(),
            };
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

    fn prompt(
        &self,
        messages: Vec<burn_lm_inference::message::Message>,
    ) -> InferenceResult<burn_lm_inference::Prompt> {
        let mut prompt: Vec<String> = vec![];
        for message in messages {
            prompt.push(format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                message.role.to_string().to_lowercase(),
                message.content
            ));
        }
        let mut prompt = prompt.join("");
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        Ok(prompt)
    }
}

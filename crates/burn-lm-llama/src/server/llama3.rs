use serde::Deserialize;

use crate::{
    generation::TemperatureSampler,
    inference::LlamaDecoder,
    pretrained::ModelMeta,
    tokenizer::{Tiktoken, Tokenizer},
    LlamaConfig, LlamaVersion,
};
use burn_lm_inference::{InferenceJob, *};

use super::{loaded_model::LoadedModel, params::SamplingSettings};

#[inference_server_config]
pub struct Llama3ServerConfig {
    /// Top-p probability threshold.
    #[config(default = 0.9)]
    pub top_p: f64,
    /// Temperature value for controlling randomness in sampling.
    #[config(default = 0.0)]
    pub temperature: f64,
    /// Maximum sequence length for input text.
    #[config(default = 8192)]
    pub max_seq_len: usize,
    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[config(default = 4096, openwebui_param = "max_tokens")]
    pub sample_len: usize,
    /// The seed to use when generating random samples. If it is 0 then a random seed is used for each inference.
    #[config(default = 0)]
    pub seed: u64,
    /// Maximum number of sequences the batched decoder runs concurrently. For a batched
    /// `BatchedInferenceServer` (the 1b, 3b, and both real 8b Llamas) this single knob both sizes the
    /// shared KV slab (one lane per slot) and is what the engine reads as
    /// `batch_capacity().max_slots`, so the two can never disagree. The Q4 1b ignores it and keeps a
    /// one-lane slab. Operators set the right value per deployment; the slab is sized at load, so a
    /// raised value takes effect on the next reload.
    #[config(default = 4)]
    pub max_slots: usize,
}

impl Llama3ServerConfig {
    /// This config's sampling fields as the per-job merge defaults (see `SamplingSettings`).
    fn sampling_defaults(&self) -> SamplingSettings {
        SamplingSettings {
            top_p: self.top_p,
            temperature: self.temperature,
            sample_len: self.sample_len,
            seed: self.seed,
        }
    }
}

#[derive(InferenceServer, Debug)]
#[inference_server(
    model_name = "Llama 3 (8B Instruct)",
    model_cli_param_name = "llama3",
    model_creation_date = "2024/04/18",
    created_by = "Meta"
)]
pub struct Llama3InstructServer {
    config: Llama3ServerConfig,
    server: Llama3BaseServer,
}

impl Default for Llama3InstructServer {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::new(LlamaVersion::Llama3Instruct),
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

impl InferenceServer for Llama3InstructServer {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama3Instruct,
                Llama3InstructServer::model_name(),
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
                Llama3InstructServer::model_name(),
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

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        self.server.run_job(job, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(InferenceServer, Debug)]
#[inference_server(
    model_name = "Llama 3.1 (8B Instruct)",
    model_cli_param_name = "llama31",
    model_creation_date = "2024/07/23",
    created_by = "Meta"
)]
pub struct Llama31InstructServer {
    config: Llama3ServerConfig,
    server: Llama3BaseServer,
}

impl Default for Llama31InstructServer {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::new(LlamaVersion::Llama31Instruct),
        }
    }
}

impl InferenceServer for Llama31InstructServer {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama31Instruct,
                Llama31InstructServer::model_name(),
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
                Llama31InstructServer::model_name(),
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

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        self.server.run_job(job, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(InferenceServer, Debug)]
#[inference_server(
    model_name = "Llama 3.2 (1B Instruct)",
    model_cli_param_name = "llama32",
    model_creation_date = "2024/09/25",
    created_by = "Meta"
)]
pub struct Llama321bInstructServer {
    config: Llama3ServerConfig,
    server: Llama3BaseServer,
}

impl Default for Llama321bInstructServer {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::new(LlamaVersion::Llama321bInstruct),
        }
    }
}

impl InferenceServer for Llama321bInstructServer {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama321bInstruct,
                Llama321bInstructServer::model_name(),
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
                Llama321bInstructServer::model_name(),
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

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        self.server.run_job(job, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

/// Generates the `BatchedInferenceServer` impl for a Llama server. The impl is identical pure
/// delegation for all four real Llama servers, since they are the same `{ config, server }` pair.
/// It is a macro rather than a blanket `impl<T> BatchedInferenceServer for T` because the trait is
/// foreign — it lives in `burn-lm-inference` — so Rust's orphan rule forbids the blanket impl. That
/// leaves duplicating the impl per server or generating it once; this generates it once.
macro_rules! impl_batched_llama_server {
    ($server:ty) => {
        impl BatchedInferenceServer for $server {
            type Decoder = LlamaDecoder;

            fn decoder(&mut self) -> InferenceResult<&mut LlamaDecoder> {
                self.server.decoder(&self.config)
            }

            fn batch_capacity(&self) -> BatchCapacity {
                // Cap the reported budget at the loaded lane count so a `max_slots` raised between
                // rounds cannot hand admission a slot past the fixed-size slab, which would index a
                // lane vector out of bounds and panic the worker. The slab only grows on reload.
                let max_slots = match self.server.loaded_lane_count() {
                    Some(lanes) => self.config.max_slots.min(lanes),
                    None => self.config.max_slots,
                };
                BatchCapacity { max_slots }
            }

            fn tokenize(&self, task: &InferenceTask) -> InferenceResult<Vec<u32>> {
                let prompt = match task {
                    InferenceTask::Message(message) => self.server.prompt(vec![message.clone()])?,
                    InferenceTask::Context(messages) => self.server.prompt(messages.clone())?,
                    InferenceTask::Prompt(prompt) => prompt.clone(),
                };
                self.server.encode(&prompt)
            }

            fn detokenize(&self, tokens: &[u32]) -> String {
                self.server.decode(tokens)
            }

            fn detokenize_bytes(&self, tokens: &[u32]) -> Vec<u8> {
                self.server.decode_bytes(tokens)
            }

            fn stop_ids(&self) -> Vec<u32> {
                self.server.stop_ids()
            }

            fn max_gen_tokens(&self) -> usize {
                self.config.sample_len
            }

            fn next_token_sampler(
                &self,
                params: &GenerationParams,
            ) -> Box<dyn NextTokenSampler + Send> {
                let settings = SamplingSettings::resolve(self.config.sampling_defaults(), params);
                Box::new(TemperatureSampler {
                    sampler: settings.sampler(),
                    temperature: settings.temperature,
                })
            }
        }
    };
}

impl_batched_llama_server!(Llama321bInstructServer);
impl_batched_llama_server!(Llama323bInstructServer);
impl_batched_llama_server!(Llama3InstructServer);
impl_batched_llama_server!(Llama31InstructServer);

#[derive(InferenceServer, Debug)]
#[inference_server(
    model_name = "Llama 3.2 (3B Instruct)",
    model_cli_param_name = "llama32-3b",
    model_creation_date = "2024/09/25",
    created_by = "Meta"
)]
pub struct Llama323bInstructServer {
    config: Llama3ServerConfig,
    server: Llama3BaseServer,
}

impl Default for Llama323bInstructServer {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::new(LlamaVersion::Llama323bInstruct),
        }
    }
}

impl InferenceServer for Llama323bInstructServer {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama323bInstruct,
                Llama323bInstructServer::model_name(),
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
                Llama323bInstructServer::model_name(),
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

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        self.server.run_job(job, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(InferenceServer, Debug)]
#[inference_server(
    model_name = "Llama 3.2 (1BQ4 Instruct)",
    model_cli_param_name = "llama32-q4",
    model_creation_date = "2024/09/25",
    created_by = "Meta"
)]
pub struct Llama321bInstructQ4Server {
    config: Llama3ServerConfig,
    server: Llama3BaseServer,
}

impl Default for Llama321bInstructQ4Server {
    fn default() -> Self {
        Self {
            config: Llama3ServerConfig::default(),
            server: Llama3BaseServer::new(LlamaVersion::Llama321bInstructQ4FB32),
        }
    }
}

impl InferenceServer for Llama321bInstructQ4Server {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn downloader() -> InferenceResult<Option<Stats>> {
            llama_downloader(
                LlamaVersion::Llama321bInstructQ4FB32,
                Llama321bInstructServer::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = LlamaVersion::Llama321bInstructQ4FB32.pretrained();
        model.is_downloaded()
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        fn deleter() -> InferenceResult<Option<Stats>> {
            llama_deleter(
                LlamaVersion::Llama321bInstructQ4FB32,
                Llama321bInstructServer::model_name(),
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

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        self.server.run_job(job, &self.config)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.server.clear_state()
    }
}

#[derive(Debug, Default)]
pub struct Llama3BaseServer {
    model: LoadedModel<Tiktoken>,
    version: LlamaVersion,
}

impl Llama3BaseServer {
    pub fn new(version: LlamaVersion) -> Self {
        Self {
            model: LoadedModel::default(),
            version,
        }
    }

    fn unload(&mut self, _model_name: &str) -> InferenceResult<Option<Stats>> {
        self.model.unload()
    }

    fn run_job(
        &mut self,
        job: InferenceJob,
        config: &Llama3ServerConfig,
    ) -> InferenceResult<Stats> {
        let prompt = match job.task {
            InferenceTask::Message(message) => self.prompt(vec![message])?,
            InferenceTask::Context(messages) => self.prompt(messages)?,
            InferenceTask::Prompt(prompt) => prompt,
        };
        self.complete(prompt, config, &job.params, job.emitter)
    }

    fn complete(
        &mut self,
        prompt: Prompt,
        config: &Llama3ServerConfig,
        params: &GenerationParams,
        emitter: GeneratedItemEmitter,
    ) -> InferenceResult<Stats> {
        let load_stats = self.load(config)?;
        // Request params merged over config — the same `SamplingSettings` resolution as the
        // batching path, so a request means the same thing on either channel. Default params
        // resolve to exactly the config, keeping config-driven callers (CLI) unchanged.
        let settings = SamplingSettings::resolve(config.sampling_defaults(), params);
        let mut sampler = settings.sampler();
        // Drive the single request through the batched path (a lone request is the batch-of-one
        // degenerate case of the same plumbing).
        let generated = {
            let model = self.model.get_mut()?;
            let mut outputs = model.generate_batch(
                vec![&prompt],
                settings.sample_len,
                settings.temperature,
                &mut sampler,
                vec![emitter],
            )?;
            outputs.pop().expect("one sequence in yields one output")
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

    fn clear_state(&mut self) -> InferenceResult<()> {
        self.model.get_mut()?.reset();
        Ok(())
    }

    /// The loaded decoder's KV-slab lane count, or `None` if the model is not loaded yet. The slab
    /// is sized once at load, so this is the real upper bound on admittable slots regardless of any
    /// later `config.max_slots` change.
    fn loaded_lane_count(&self) -> Option<usize> {
        self.model
            .get()
            .ok()
            .map(|model| model.decoder.cache.lane_count())
    }

    /// Mutably borrow the loaded decoder, loading the model first if needed.
    /// Used by `BatchedInferenceServer::decoder`.
    fn decoder(&mut self, config: &Llama3ServerConfig) -> InferenceResult<&mut LlamaDecoder> {
        self.load(config)?;
        Ok(&mut self.model.get_mut()?.decoder)
    }

    /// Encode a prompt into token ids using the loaded model's tokenizer.
    ///
    /// Thin wrapper over the existing Tiktoken tokenizer, exposed so the framework continuous loop
    /// can tokenize without owning the tokenizer. Requires the model to be loaded (the engine
    /// borrows the decoder, which loads the model, before tokenizing).
    fn encode(&self, prompt: &str) -> InferenceResult<Vec<u32>> {
        Ok(self.model.get()?.tokenizer.encode(prompt, false, false))
    }

    /// Decode token ids back to text using the loaded model's tokenizer.
    fn decode(&self, tokens: &[u32]) -> String {
        self.model
            .get()
            .map(|model| model.tokenizer.decode(tokens))
            .unwrap_or_default()
    }

    /// Decode token ids to raw bytes using the loaded model's tokenizer. Unlike `decode`, this is
    /// total per token: Tiktoken's byte-level decode cannot fail on a multi-byte character split
    /// across tokens.
    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        self.model
            .get()
            .map(|model| model.tokenizer.decode_bytes(tokens))
            .unwrap_or_default()
    }

    /// Stop token ids from the loaded model's tokenizer (EOS/EOT/EOM).
    fn stop_ids(&self) -> Vec<u32> {
        self.model
            .get()
            .map(|model| model.tokenizer.stop_ids())
            .unwrap_or_default()
    }

    fn load(&mut self, config: &Llama3ServerConfig) -> InferenceResult<Option<Stats>> {
        if !self.is_loaded() {
            let now = std::time::Instant::now();
            let model = match self.version {
                LlamaVersion::Llama3Instruct => LlamaConfig::llama3_8b_pretrained(
                    config.max_seq_len,
                    config.max_slots,
                    &*INFERENCE_DEVICE,
                )
                .unwrap(),
                LlamaVersion::Llama31Instruct => LlamaConfig::llama3_1_8b_pretrained(
                    config.max_seq_len,
                    config.max_slots,
                    &*INFERENCE_DEVICE,
                )
                .unwrap(),
                LlamaVersion::Llama323bInstruct => LlamaConfig::llama3_2_3b_pretrained(
                    config.max_seq_len,
                    config.max_slots,
                    &*INFERENCE_DEVICE,
                )
                .unwrap(),
                LlamaVersion::Llama321bInstruct => LlamaConfig::llama3_2_1b_pretrained(
                    config.max_seq_len,
                    config.max_slots,
                    &*INFERENCE_DEVICE,
                )
                .unwrap(),
                LlamaVersion::Llama321bInstructQ4FB32 => {
                    LlamaConfig::llama3_2_1b_pretrained_q4(config.max_seq_len, &*INFERENCE_DEVICE)
                        .unwrap()
                }
            };
            self.model.store(model);
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
        self.model.is_loaded()
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

use rand::Rng;
use serde::Deserialize;
use std::borrow::BorrowMut;

use burn::prelude::Backend;
use burnlm_inference::*;
use llama_burn::{
    llama::{self, Llama},
    pretrained::{self, ModelMeta},
    sampling::{Sampler, TopP},
    tokenizer::Tiktoken,
};

#[derive(Clone, Debug, Default)]
/// Llama-3 model variants to load.
pub enum LlamaVersion {
    /// Llama-3-8B-Instruct.
    Llama3Instruct,
    /// Llama-3.1-8B-Instruct.
    Llama31Instruct,
    /// Llama-3.2-3B-Instruct.
    Llama323bInstruct,
    #[default]
    /// Llama-3.2-1B-Instruct.
    Llama321bInstruct,
}

#[inference_server_config]
pub struct Llama3ServerConfig {
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

#[derive(InferenceServer, Debug)]
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

fn llama_downloader(version: pretrained::Llama, name: &'static str) -> InferenceResult<()> {
    let model = pretrained::Llama::pretrained(&version);
    model
        .download_weights()
        .map_err(|err| InferenceError::DownloadWeightError(name.to_string(), err.to_string()))?;
    model
        .download_tokenizer()
        .map_err(|err| InferenceError::DownloadTokenizerError(name.to_string(), err.to_string()))?;
    Ok(())
}

impl InferenceServer for Llama3InstructServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<()>> {
        fn downloader() -> InferenceResult<()> {
            llama_downloader(
                pretrained::Llama::Llama3,
                Llama3InstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = pretrained::Llama::Llama3.pretrained();
        model.is_downloaded()
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.server.unload()
    }

    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }
}

#[derive(InferenceServer, Debug)]
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
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<()>> {
        fn downloader() -> InferenceResult<()> {
            llama_downloader(
                pretrained::Llama::Llama31Instruct,
                Llama31InstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = pretrained::Llama::Llama31Instruct.pretrained();
        model.is_downloaded()
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.server.unload()
    }

    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }
}

#[derive(InferenceServer, Debug)]
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
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<()>> {
        fn downloader() -> InferenceResult<()> {
            llama_downloader(
                pretrained::Llama::Llama321bInstruct,
                Llama321bInstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = pretrained::Llama::Llama321bInstruct.pretrained();
        model.is_downloaded()
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.server.unload()
    }

    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }
}

#[derive(InferenceServer, Debug)]
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
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<()>> {
        fn downloader() -> InferenceResult<()> {
            llama_downloader(
                pretrained::Llama::Llama323bInstruct,
                Llama323bInstructServer::<InferenceBackend>::model_name(),
            )
        }
        Some(downloader)
    }

    fn is_downloaded(&mut self) -> bool {
        let model = pretrained::Llama::Llama323bInstruct.pretrained();
        model.is_downloaded()
    }

    fn unload(&mut self) -> InferenceResult<()> {
        self.server.unload()
    }

    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.server.complete(messages, &self.config)
    }
}

#[derive(Debug, Default)]
pub struct Llama3BaseServer<B: Backend> {
    model: Option<Llama<B, Tiktoken>>,
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
    fn unload(&mut self) -> InferenceResult<()> {
        self.model = None;
        Ok(())
    }

    fn complete(
        &mut self,
        messages: Vec<Message>,
        config: &Llama3ServerConfig,
    ) -> InferenceResult<Completion> {
        println!("{:?}", config);
        println!("Burn device: {:?}", INFERENCE_DEVICE);
        self.load(config)?;
        let prompt = self.prompt(messages)?;
        let seed = match config.seed {
            0 => rand::thread_rng().gen::<u64>(),
            s => s,
        };
        let mut sampler = if config.temperature > 0.0 {
            Sampler::TopP(TopP::new(config.top_p, seed))
        } else {
            Sampler::Argmax
        };
        println!("Generating...");
        let generated = match self.model.borrow_mut() {
            Some(model) => {
                model.generate(&prompt, config.sample_len, config.temperature, &mut sampler)
            }
            _ => return Err(InferenceError::ModelNotLoaded),
        };
        Ok(generated.text)
    }

    fn load(
        &mut self,
        config: &Llama3ServerConfig,
    ) -> burnlm_inference::errors::InferenceResult<()> {
        if self.model.is_none() {
            self.model = match self.version {
                LlamaVersion::Llama3Instruct => Some(
                    llama::LlamaConfig::llama3_8b_pretrained::<InferenceBackend>(
                        config.max_seq_len,
                        &INFERENCE_DEVICE,
                    )
                    .unwrap(),
                ),
                LlamaVersion::Llama31Instruct => Some(
                    llama::LlamaConfig::llama3_1_8b_pretrained::<InferenceBackend>(
                        config.max_seq_len,
                        &INFERENCE_DEVICE,
                    )
                    .unwrap(),
                ),
                LlamaVersion::Llama323bInstruct => Some(
                    llama::LlamaConfig::llama3_2_3b_pretrained::<InferenceBackend>(
                        config.max_seq_len,
                        &INFERENCE_DEVICE,
                    )
                    .unwrap(),
                ),
                LlamaVersion::Llama321bInstruct => Some(
                    llama::LlamaConfig::llama3_2_1b_pretrained::<InferenceBackend>(
                        config.max_seq_len,
                        &INFERENCE_DEVICE,
                    )
                    .unwrap(),
                ),
            };
        }
        Ok(())
    }

    fn prompt(
        &self,
        messages: Vec<burnlm_inference::message::Message>,
    ) -> burnlm_inference::errors::InferenceResult<burnlm_inference::Prompt> {
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

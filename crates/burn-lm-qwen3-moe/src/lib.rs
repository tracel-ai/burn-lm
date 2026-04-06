#![recursion_limit = "256"]

use burn_lm_inference::{InferenceJob, *};

use crate::hf_downloader::{delete, download_safetensors, exists};
use tracing::{error, info};

mod error_wrapper;
mod hf_downloader;
mod model;

const MODEL_NAME: &str = "Qwen/Qwen3-30B-A3B";

// This is where you can declare the configuration parameters for
// your model.
//
// On each field add the `config` attribute to define a default value for it.
//
// Those parameters will be available in `burn-lm` CLI.
//
// Parameters sent by Open WebUI will be automatically mapped to the configuration
// parameters with the same name. It also possible to map to an Open WebUI parameter
// with a different name with `#[config(openwebui_param = "param_name")]`.
// Examples of Open WebUI parameters are `temperature`, `seed` and `top_p`.
#[inference_server_config]
pub struct Qwen3MoeServerConfig {
    /// Temperature value for controlling randomness in sampling.
    #[config(default = 0.1)]
    pub temperature: f64,
}

// Declare the model server info using the `InferenceServer` derive
// and `inference_server` attribute. The structure must be generic over the
// Burn backends.
//
// Register the model by adding a dependency on this crate in the
// `burn-lm-registry` crate.
//
// Add an entry in the `inference_server_registry` attribute in `lib.rs`
// of `burn-lm-registry` crate. For instance for this dummy model:
//
//     server(
//         crate_namespace = "burnlm_inference_template",
//         server_type = "Qwen3MoeServer<InferenceBackend>",
//     ),
//
#[derive(InferenceServer, Clone, Default, Debug)]
#[inference_server(
    model_name = "Qwen/Qwen3-30B-A3B",
    model_creation_date = "2025/06/28",
    created_by = "Qwen team"
)]

pub struct Qwen3MoeServer<B: Backend> {
    config: Qwen3MoeServerConfig,
    // Remove the phantom data and add your model here, see TinyLLama example
    // in burn-lm-llama crate.
    // You'll likely need to wrap your model in an Arc Mutex because the server
    // needs to be clonable.
    _model: std::marker::PhantomData<B>,
}

impl InferenceServer for Qwen3MoeServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        Some(|| download_safetensors(MODEL_NAME))
    }

    fn is_downloaded(&mut self) -> bool {
        info!("Checking if model is downloaded...");
        match exists(MODEL_NAME) {
            Ok(exists) => exists,
            Err(e) => {
                error!("Error checking if model is downloaded: {}", e);
                false
            }
        }
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        Some(|| delete(MODEL_NAME))
    }

    fn load(&mut self) -> InferenceResult<Option<Stats>> {
        // Load the model here
        let now = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let mut stats = Stats::new();
        stats
            .entries
            .insert(StatEntry::ModelLoadingDuration(now.elapsed()));
        Ok(Some(stats))
    }

    fn is_loaded(&mut self) -> bool {
        // Return true when the model is loaded and ready for inference.
        true
    }

    fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        // Drop the model here.
        Ok(None)
    }

    fn run_job(&mut self, job: InferenceJob) -> InferenceResult<Stats> {
        match job.task {
            InferenceTask::Message(message) => {
                job.emitter.completed(GeneratedItem::Text(message.content));
            }
            InferenceTask::Context(mut messages) => {
                job.emitter.completed(GeneratedItem::Text(
                    messages
                        .pop()
                        .map(|msg| msg.content)
                        .unwrap_or_else(|| "...".to_string()),
                ));
            }
            InferenceTask::Prompt(text) => {
                job.emitter.completed(GeneratedItem::Text(text));
            }
        }

        // Example of returned statistics about the completion.
        let mut stats = Stats::default();
        stats
            .entries
            .insert(StatEntry::Named("Everything".to_string(), "42".to_string()));
        Ok(stats)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        // No state
        Ok(())
    }
}

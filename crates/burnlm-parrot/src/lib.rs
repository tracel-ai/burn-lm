#![recursion_limit = "256"]

use burnlm_inference::*;

// This is where you can declare the configuration parameters for
// your model.
//
// On each field add the `config` attribute to define a default value for it.
//
// Those parameters will be available in `burnlm` CLI.
//
// Parameters sent by Open WebUI will be automatically mapped to the configuration
// parameters with the same name. It also possible to map to an Open WebUI parameter
// with a different name with `#[config(openwebui_param = "param_name")]`.
// Examples of Open WebUI parameters are `temperature`, `seed` and `top_p`.
#[inference_server_config]
pub struct ParrotServerConfig {
    /// Temperature value for controlling randomness in sampling.
    #[config(default = 0.1)]
    pub temperature: f64,
}

// Declare the model server info using the `InferenceServer` derive
// and `inference_server` attribute. The structure must be generic over the
// Burn backends.
//
// Register the model by adding a dependency on this crate in the
// `burnlm-registry` crate.
//
// Add an entry in the `inference_server_registry` attribute in `lib.rs`
// of `burnlm-registry` crate. For instance for this dummy model:
//
//     server(
//         crate_namespace = "burnlm_inference_template",
//         server_type = "ParrotServer<InferenceBackend>",
//     ),
//
#[derive(InferenceServer, Clone, Default, Debug)]
#[inference_server(
    model_name = "Parrot",
    model_creation_date = "01/28/2025",
    owned_by = "Tracel Technologies Inc."
)]
pub struct ParrotServer<B: Backend> {
    config: ParrotServerConfig,
    // Remove the phantom data and add your model here, see TinyLLama example
    // in burnlm-llama crate.
    // You'll likely need to wrap your model in an Arc Mutex because the server
    // needs to be clonable.
    _model: std::marker::PhantomData<B>,
}

// Implement the `InferenceServer` trait for the server.
impl InferenceServer for ParrotServer<InferenceBackend> {
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        // Return a closure with code to download the model if available.
        // Return none if there is no possibility to download the model or if
        // this model does not need to be downloaded.
        None
    }

    fn is_downloaded(&mut self) -> bool {
        // This server example does not require downloading
        // thus is can be considered always installed.
        // Update accordingly.
        true
    }

    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        // Return a closure with code to delete the model if applicable.
        // Return none if there is no possibility to delete the model or if
        // this model does not need to be deleted.
        None
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

    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion> {
        // This is where the inference actually happens.
        let mut completion = match messages.last() {
            Some(msg) => Completion::new(&msg.content),
            _ => Completion::new("..."),
        };
        // Example of returned statistics about the completion.
        completion
            .stats
            .entries
            .insert(StatEntry::Named("Everything".to_string(), "42".to_string()));
        Ok(completion)
    }

    fn clear_state(&mut self) -> InferenceResult<()> {
        // No state
        Ok(())
    }
}

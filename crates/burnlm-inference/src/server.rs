use crate::{completion::Completion, errors::InferenceResult, message::Message, Stats};
use std::fmt::Debug;

/// Marker trait for server configurations.
pub trait InferenceServerConfig:
    clap::FromArgMatches + serde::de::DeserializeOwned + 'static + Debug
{
}

/// Trait to add parsing capability of server config from clap and serde
pub trait ServerConfigParsing {
    /// The configuration type to parse
    type Config: InferenceServerConfig;

    fn parse_cli_config(&mut self, args: &clap::ArgMatches);
    fn parse_json_config(&mut self, json: &str);
}

/// Inference server interface aimed to be implemented to be able to register a
/// model in Burn LM registry.
pub trait InferenceServer: ServerConfigParsing + Clone + Default + Send + Sync + Debug {
    /// Return closure of a function to download the model
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        None
    }

    /// Return true is the model has been downloaded.
    /// Return false if the model is not downloaded or there is no downloader.
    fn is_downloaded(&mut self) -> bool {
        false
    }

    /// Return closure of a function to delete the model
    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        None
    }

    /// Load the model.
    fn load(&mut self) -> InferenceResult<Option<Stats>>;

    /// Return true is the model is already loaded.
    fn is_loaded(&mut self) -> bool;

    /// Unload the model.
    fn unload(&mut self) -> InferenceResult<Option<Stats>>;

    /// Run inference to complete messages
    fn run_completion(&mut self, messages: Vec<Message>) -> InferenceResult<Completion>;

    /// Clear the model state
    fn clear_state(&mut self) -> InferenceResult<()>;
}

use std::any::Any;
use std::fmt::Debug;

use clap::FromArgMatches;
use serde::de::DeserializeOwned;

use crate::{errors::InferenceResult, message::Message, Completion};

/// Marker trait for server configurations.
pub trait InferenceServerConfig: clap::FromArgMatches + DeserializeOwned + 'static + Debug {}

/// Inference server interface aimed to be implemented to be able to register a
/// model in Burn LM registry.
pub trait InferenceServer: Default + Send + Sync + Debug {
    /// The configuration holding all the inference time parameter for a model
    type Config: InferenceServerConfig;

    /// Set configuration for inference.
    fn set_config(&mut self, config: Box<dyn Any>);

    /// Return closure of a function to download the model
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<()>> {
        None
    }

    /// Return true is the model has been downloaded.
    /// Return false if the model is not downloaded or there is no downloader.
    fn is_downloaded(&mut self) -> bool {
        false
    }

    /// Unload the model.
    fn unload(&mut self) -> InferenceResult<()>;

    /// Complete the prompt composed of formatted messages
    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion>;

    /// Parse CLI flags from burnlm-cli and return a config object.
    fn parse_cli_config(args: &clap::ArgMatches) -> Box<dyn Any> {
        let config = Self::Config::from_arg_matches(args)
            .expect("Should be able to parse arguments from CLI");
        Box::new(config)
    }

    /// Parse passed JSON and return a config oject
    fn parse_json_config(json: &str) -> Box<dyn Any> {
        let config: Self::Config =
            serde_json::from_str(json).expect("Should be able to parse JSON");
        Box::new(config)
    }
}

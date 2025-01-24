use std::any::Any;

use clap::FromArgMatches;

use crate::{errors::InferenceResult, message::Message, Completion};

/// Marker trait for server configurations.
pub trait InferenceServerConfig: clap::FromArgMatches + 'static {}

/// Inference server interface aimed to be implemented to be able to register a
/// model in Burn LM registry.
pub trait InferenceServer: Default + Sync {
    /// The configuration holding all the inference time parameter for a model
    type Config: InferenceServerConfig;

    /// Set configuration for inference.
    fn set_config(&mut self, config: Box<dyn Any>);

    /// Unload the model.
    fn unload(&mut self) -> InferenceResult<()>;

    /// Complete the prompt composed of formatted messages
    fn complete(&mut self, messages: Vec<Message>) -> InferenceResult<Completion>;

    /// Parse CLI flags from burnlm-cli and return a config object.
    fn parse_cli_config(args: &clap::ArgMatches) -> Box<dyn Any> {
        let config = Self::Config::from_arg_matches(args).expect("Should be able to parse arguments from CLI");
        Box::new(config)
    }

    /// Return the selected version of the model as a string.
    fn get_version(&self) -> String { "default".to_string() }

}

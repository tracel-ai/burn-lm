use std::fmt::Debug;

use crate::{Completion, InferenceResult, Message};

pub type CreateCliFlagsFn = fn() -> clap::Command;

pub trait InferencePlugin: Send + Sync + Debug {
    fn model_name(&self) -> &'static str;
    fn model_cli_param_name(&self) -> &'static str;
    fn model_creation_date(&self) -> &'static str;
    fn owned_by(&self) -> &'static str;
    fn create_cli_flags_fn(&self) -> CreateCliFlagsFn;
    fn downloader(&self) -> Option<fn() -> InferenceResult<()>>;
    fn parse_cli_config(&self, args: &clap::ArgMatches);
    fn parse_json_config(&self, json: &str);
    fn is_downloaded(&self) -> bool;
    fn unload(&self) -> InferenceResult<()>;
    fn complete(&self, messages: Vec<Message>) -> InferenceResult<Completion>;
}

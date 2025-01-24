use std::any::Any;

use crate::{Completion, InferenceResult, Message};

pub type CreateCliFlagsFn = fn() -> clap::Command;
pub type ParseCliFlagsFn = fn(&clap::ArgMatches) -> Box<dyn Any>;
pub type GetModelVersionsFn = fn() -> Vec<String>;

pub trait InferencePlugin: Sync {
    fn model_name(&self) -> &'static str;
    fn model_name_lc(&self) -> &'static str;
    fn model_creation_date(&self) -> &'static str;
    fn owned_by(&self) -> &'static str;
    fn create_cli_flags_fn(&self) -> CreateCliFlagsFn;
    fn parse_cli_flags_fn(&self) -> ParseCliFlagsFn;
    fn get_model_versions_fn(&self) -> GetModelVersionsFn;
    fn set_config(&self, config: Box<dyn Any>);
    fn get_version(&self) -> String;
    fn unload(&self) -> InferenceResult<()>;
    fn complete(&self, messages: Vec<Message>) -> InferenceResult<Completion>;
}

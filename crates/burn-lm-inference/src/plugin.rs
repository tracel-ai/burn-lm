use std::fmt::Debug;

use crate::{InferenceJob, InferenceResult, Stats};

pub type CreateCliFlagsFn = fn() -> clap::Command;

pub trait InferencePlugin: Send + Sync + Debug {
    fn clone_box(&self) -> Box<dyn InferencePlugin>;
    fn model_name(&self) -> &'static str;
    fn model_cli_param_name(&self) -> &'static str;
    fn model_creation_date(&self) -> &'static str;
    fn created_by(&self) -> &'static str;
    fn create_cli_flags_fn(&self) -> CreateCliFlagsFn;
    fn downloader(&self) -> Option<fn() -> InferenceResult<Option<Stats>>>;
    fn is_downloaded(&self) -> bool;
    fn deleter(&self) -> Option<fn() -> InferenceResult<Option<Stats>>>;
    fn parse_cli_config(&self, args: &clap::ArgMatches);
    fn parse_json_config(&self, json: &str);
    fn load(&self) -> InferenceResult<Option<Stats>>;
    fn is_loaded(&self) -> bool;
    fn unload(&self) -> InferenceResult<Option<Stats>>;
    fn run_job(&self, job: InferenceJob) -> InferenceResult<Stats>;
    fn clear_state(&self) -> InferenceResult<()>;
}

impl Clone for Box<dyn InferencePlugin> {
    fn clone(&self) -> Box<dyn InferencePlugin> {
        self.clone_box()
    }
}

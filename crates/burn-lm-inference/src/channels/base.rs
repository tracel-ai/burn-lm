use std::fmt::Debug;

use crate::{errors::InferenceResult, server::InferenceServer, InferenceJob, Stats};

pub trait InferenceChannel<Server: InferenceServer>: Clone + Send + Sync + Debug {
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

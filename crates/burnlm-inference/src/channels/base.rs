use std::fmt::Debug;

use crate::{
    completion::Completion, errors::InferenceResult, message::Message, server::InferenceServer,
    Stats,
};

pub trait InferenceChannel<Server: InferenceServer>: Clone + Send + Sync + Debug {
    fn downloader(&self) -> Option<fn() -> InferenceResult<Option<Stats>>>;
    fn is_downloaded(&self) -> bool;
    fn deleter(&self) -> Option<fn() -> InferenceResult<Option<Stats>>>;
    fn parse_cli_config(&self, args: &clap::ArgMatches);
    fn parse_json_config(&self, json: &str);
    fn load(&self) -> InferenceResult<Option<Stats>>;
    fn is_loaded(&self) -> bool;
    fn unload(&self) -> InferenceResult<Option<Stats>>;
    fn run_completion(&self, message: Vec<Message>) -> InferenceResult<Completion>;
}

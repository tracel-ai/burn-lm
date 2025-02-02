use std::fmt::Debug;

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

pub trait InferenceChannel<Server: InferenceServer>: Send + Sync + Debug {
    fn downloader(&self) -> Option<fn() -> InferenceResult<()>>;
    fn is_downloaded(&self) -> bool;
    fn parse_cli_config(&self, args: &clap::ArgMatches);
    fn parse_json_config(&self, json: &str);
    fn unload(&self) -> InferenceResult<()>;
    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion>;
}

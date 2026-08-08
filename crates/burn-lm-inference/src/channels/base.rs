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
    /// Advisory backpressure probe: whether a job submitted right now would be shed with
    /// `InferenceError::Overloaded`. Channels without a bounded queue never shed, hence the
    /// `false` default. "Advisory" because the answer can change between the probe and a
    /// subsequent submit; the submit itself is the authoritative shed point.
    fn is_overloaded(&self) -> bool {
        false
    }
}

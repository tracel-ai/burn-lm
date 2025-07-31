use std::{cell::RefCell, sync::Arc};

use crate::{errors::InferenceResult, server::InferenceServer, InferenceJob, Stats};

use super::InferenceChannel;

/// Simple passthrough channel that just provides interior mutability.
/// Not meant to be used in multithreaded environment.
#[derive(Debug, Clone)]
pub struct SingleThreadedChannel<Server: InferenceServer> {
    server: Arc<RefCell<Server>>,
}
unsafe impl<Server: InferenceServer> Send for SingleThreadedChannel<Server> {}
unsafe impl<Server: InferenceServer> Sync for SingleThreadedChannel<Server> {}

impl<Server: InferenceServer> Default for SingleThreadedChannel<Server> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Server: InferenceServer> SingleThreadedChannel<Server> {
    pub fn new() -> Self {
        Self {
            server: Arc::new(RefCell::new(Server::default())),
        }
    }
}

impl<Server: InferenceServer> InferenceChannel<Server> for SingleThreadedChannel<Server> {
    fn downloader(&self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        self.server.borrow_mut().downloader()
    }

    fn is_downloaded(&self) -> bool {
        self.server.borrow_mut().is_downloaded()
    }

    fn deleter(&self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        self.server.borrow_mut().deleter()
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        self.server.borrow_mut().parse_cli_config(args);
    }

    fn parse_json_config(&self, json: &str) {
        self.server.borrow_mut().parse_json_config(json);
    }

    fn load(&self) -> InferenceResult<Option<Stats>> {
        self.server.borrow_mut().load()
    }

    fn is_loaded(&self) -> bool {
        self.server.borrow_mut().is_loaded()
    }

    fn unload(&self) -> InferenceResult<Option<Stats>> {
        self.server.borrow_mut().unload()
    }

    fn run_job(&self, job: InferenceJob) -> InferenceResult<Stats> {
        self.server.borrow_mut().run_job(job)
    }

    fn clear_state(&self) -> InferenceResult<()> {
        self.server.borrow_mut().clear_state()
    }
}

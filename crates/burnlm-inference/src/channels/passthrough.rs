use std::{cell::RefCell, sync::Arc};

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

use super::InferenceChannel;

/// Simple passthrough channel that just provides interior mutability.
/// Not meant to be used in multithreaded environment.
#[derive(Debug)]
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
    fn downloader(&self) -> Option<fn() -> InferenceResult<()>> {
        self.server.borrow_mut().downloader()
    }

    fn is_downloaded(&self) -> bool {
        self.server.borrow_mut().is_downloaded()
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        self.server.borrow_mut().parse_cli_config(args);
    }

    fn parse_json_config(&self, json: &str) {
        self.server.borrow_mut().parse_json_config(json);
    }

    fn load(&self) -> InferenceResult<()> {
        self.server.borrow_mut().load()
    }

    fn unload(&self) -> InferenceResult<()> {
        self.server.borrow_mut().unload()
    }

    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion> {
        self.server.borrow_mut().complete(message)
    }
}

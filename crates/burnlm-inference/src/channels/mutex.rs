use std::sync::{Arc, Mutex};

use crate::{
    completion::Completion, errors::InferenceResult, message::Message, server::InferenceServer,
    Stats,
};

use super::InferenceChannel;

/// ARC Mutex channel that lock the server each time the client reaches to it.
#[derive(Debug, Clone)]
pub struct MutexChannel<Server: InferenceServer> {
    server: Arc<Mutex<Server>>,
}

impl<Server: InferenceServer> Default for MutexChannel<Server> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Server: InferenceServer> MutexChannel<Server> {
    pub fn new() -> Self {
        Self {
            server: Arc::new(Mutex::new(Server::default())),
        }
    }
}

impl<Server: InferenceServer> InferenceChannel<Server> for MutexChannel<Server> {
    fn downloader(&self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        let mut server = self.server.lock().unwrap();
        server.downloader()
    }

    fn is_downloaded(&self) -> bool {
        let mut server = self.server.lock().unwrap();
        server.is_downloaded()
    }

    fn deleter(&self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        let mut server = self.server.lock().unwrap();
        server.deleter()
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        let mut server = self.server.lock().unwrap();
        server.parse_cli_config(args);
    }

    fn parse_json_config(&self, json: &str) {
        let mut server = self.server.lock().unwrap();
        server.parse_json_config(json);
    }

    fn load(&self) -> InferenceResult<Option<Stats>> {
        let mut server = self.server.lock().unwrap();
        server.load()
    }

    fn is_loaded(&self) -> bool {
        let mut server = self.server.lock().unwrap();
        server.is_loaded()
    }

    fn unload(&self) -> InferenceResult<Option<Stats>> {
        let mut server = self.server.lock().unwrap();
        server.unload()
    }

    fn run_completion(&self, message: Vec<Message>) -> InferenceResult<Completion> {
        let mut server = self.server.lock().unwrap();
        server.run_completion(message)
    }
}

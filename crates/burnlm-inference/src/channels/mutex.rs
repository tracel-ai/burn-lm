use std::sync::{Arc, Mutex};

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

use super::InferenceChannel;

/// ARC Mutex channel that lock the server each time the client reaches to it.
#[derive(Debug)]
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
    fn downloader(&self) -> Option<fn() -> InferenceResult<()>> {
        let mut server = self.server.lock().unwrap();
        server.downloader()
    }

    fn is_downloaded(&self) -> bool {
        let mut server = self.server.lock().unwrap();
        server.is_downloaded()
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        let mut server = self.server.lock().unwrap();
        server.parse_cli_config(args);
    }

    fn parse_json_config(&self, json: &str) {
        let mut server = self.server.lock().unwrap();
        server.parse_json_config(json);
    }

    fn load(&self) -> InferenceResult<()> {
        let mut server = self.server.lock().unwrap();
        server.load()
    }

    fn unload(&self) -> InferenceResult<()> {
        let mut server = self.server.lock().unwrap();
        server.unload()
    }

    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion> {
        let mut server = self.server.lock().unwrap();
        server.complete(message)
    }
}

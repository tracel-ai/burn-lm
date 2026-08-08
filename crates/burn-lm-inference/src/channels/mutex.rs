use std::sync::{Arc, Mutex};

use crate::{errors::InferenceResult, server::InferenceServer, InferenceJob, Stats};

use super::InferenceChannel;

/// ARC Mutex channel that lock the server each time the client reaches to it.
#[derive(Debug)]
pub struct MutexChannel<Server: InferenceServer> {
    server: Arc<Mutex<Server>>,
}

// Manual `Clone` (clones the shared `Arc`) so we don't require `Server: Clone` — the derived impl
// would, even though the server is never deep-cloned.
impl<Server: InferenceServer> Clone for MutexChannel<Server> {
    fn clone(&self) -> Self {
        Self {
            server: Arc::clone(&self.server),
        }
    }
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

    fn run_job(&self, job: InferenceJob) -> InferenceResult<Stats> {
        let mut server = self.server.lock().unwrap();
        server.run_job(job)
    }

    fn clear_state(&self) -> InferenceResult<()> {
        let mut server = self.server.lock().unwrap();
        server.clear_state()
    }
}

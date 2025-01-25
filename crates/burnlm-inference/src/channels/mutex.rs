use std::sync::{Arc, Mutex};

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

use super::InferenceChannel;

/// ARC Mutex channel that lock the server each time the client reaches to it.
#[derive(Debug)]
pub struct MutexChannel<Server: InferenceServer> {
    server: Arc<Mutex<Server>>,
}

impl<Server: InferenceServer> MutexChannel<Server> {
    pub fn new() -> Self {
        Self {
            server: Arc::new(Mutex::new(Server::default()))
        }
    }
}

impl<Server: InferenceServer> InferenceChannel<Server> for MutexChannel<Server> {
    fn set_config(&self, config: Box<dyn std::any::Any>) {
        let mut server = self.server.lock().unwrap();
        server.set_config(config);
    }

    fn get_version(&self) -> String {
        let server = self.server.lock().unwrap();
        server.get_version()
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

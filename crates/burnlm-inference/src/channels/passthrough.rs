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

impl<Server: InferenceServer> SingleThreadedChannel<Server> {
    pub fn new() -> Self {
        Self {
            server: Arc::new(RefCell::new(Server::default()))
        }
    }
}

impl<Server: InferenceServer> InferenceChannel<Server> for SingleThreadedChannel<Server> {
    fn set_config(&self, config: Box<dyn std::any::Any>) {
        self.server.borrow_mut().set_config(config);
    }

    fn get_version(&self) -> String {
        self.server.borrow().get_version()
    }

    fn unload(&self) -> InferenceResult<()> {
        self.server.borrow_mut().unload()
    }

    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion> {
        self.server.borrow_mut().complete(message)
    }
}

use std::any::Any;
use std::fmt::Debug;

use crate::{errors::InferenceResult, message::Message, server::InferenceServer, Completion};

pub trait InferenceChannel<Server: InferenceServer>: Send + Sync + Debug {
    fn set_config(&self, config: Box<dyn Any>);
    fn downloader(&self) -> Option<fn() -> InferenceResult<()>>;
    fn is_downloaded(&self) -> bool;
    fn unload(&self) -> InferenceResult<()>;
    fn complete(&self, message: Vec<Message>) -> InferenceResult<Completion>;
}

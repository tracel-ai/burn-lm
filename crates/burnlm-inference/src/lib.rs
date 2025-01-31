pub mod backends;
pub mod channels;
pub mod client;
pub mod errors;
pub mod message;
pub mod plugin;
pub mod server;

pub type Completion = String;
pub type Prompt = String;

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
pub use crate::channels::mutex::MutexChannel;
pub use crate::channels::passthrough::SingleThreadedChannel;
pub use crate::client::InferenceClient;
pub use crate::errors::*;
pub use crate::message::{Message, MessageRole};
pub use crate::plugin::InferencePlugin;
pub use crate::server::{InferenceServer, InferenceServerConfig};
pub use burn::prelude::Backend;
pub use backends::burn_backend_types::*;
pub use burnlm_macros::inference_server_config;
pub use burnlm_macros::InferenceServer;
// external re-export
pub use clap::{self, CommandFactory, FromArgMatches, Parser};
pub use serde::Deserialize;
pub use std::any::Any;
// ---------------------------------------------------------------------------


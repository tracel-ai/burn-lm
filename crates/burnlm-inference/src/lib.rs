pub mod backends;
pub mod channels;
pub mod client;
pub mod completion;
pub mod errors;
pub mod message;
pub mod plugin;
pub mod server;
pub mod stats;
pub mod utils;

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
pub use crate::channels::mutex::MutexChannel;
pub use crate::channels::passthrough::SingleThreadedChannel;
pub use crate::client::InferenceClient;
pub use crate::completion::Completion;
pub use crate::errors::*;
pub use crate::message::{Message, MessageRole};
pub use crate::plugin::InferencePlugin;
pub use crate::server::{InferenceServer, InferenceServerConfig, ServerConfigParsing};
pub use crate::stats::{StatEntry, Stats, STATS_MARKER};
pub use backends::burn_backend_types::*;
pub use backends::DTYPE_NAME;
pub use burn::prelude::Backend;
pub use burnlm_macros::inference_server_config;
pub use burnlm_macros::InferenceServer;
// external re-export
pub use clap::{self, CommandFactory, FromArgMatches, Parser};
pub use serde::Deserialize;
pub use serde_json;
pub use std::any::Any;
// ---------------------------------------------------------------------------

pub type Prompt = String;

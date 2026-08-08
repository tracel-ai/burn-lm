mod job;

pub use job::*;

pub mod backends;
pub mod batching;
pub mod channels;
pub mod client;
pub mod errors;
pub mod message;
pub mod plugin;
pub mod sampler;
pub mod server;
pub mod stats;
pub mod utf8;
pub mod utils;

// ---------------------------------------------------------------------------
// Re-exports for convenience so plugins implementors can just do:
pub use crate::batching::{
    step_round, ActiveSeq, BatchCapacity, BatchedDecoder, BatchedInferenceServer, DecodeRow,
    PrefillBudget, StepOutcome,
};
pub use crate::channels::batching::BatchingChannel;
pub use crate::channels::mutex::MutexChannel;
pub use crate::channels::passthrough::SingleThreadedChannel;
pub use crate::client::InferenceClient;
pub use crate::errors::*;
pub use crate::message::{Message, MessageRole};
pub use crate::plugin::InferencePlugin;
pub use crate::sampler::{ids_to_host, Argmax, Sampler};
pub use crate::server::{InferenceServer, InferenceServerConfig, ServerConfigParsing};
pub use crate::stats::{StatEntry, Stats, STATS_MARKER};
pub use crate::utf8::Utf8Buffer;
pub use backends::burn_backend_types::*;
pub use backends::DTYPE_NAME;
pub use burn_lm_macros::inference_server_config;
pub use burn_lm_macros::InferenceServer;
// external re-export
pub use clap::{self, CommandFactory, FromArgMatches, Parser};
pub use serde::Deserialize;
pub use serde_json;
pub use std::any::Any;
// ---------------------------------------------------------------------------

pub type Prompt = String;

use burn_lm_inference::*;
use burn_lm_macros::inference_server_registry;
use std::{collections::HashMap, sync::Arc};

// `MutexChannel` is the default channel for every registered model: it locks the server for the
// duration of each call and works for any `InferenceServer`. Only models that implement
// `BatchedInferenceServer` (and therefore can run through the continuous-batching loop) opt into
// `BatchingChannel` via a per-server `channel_type` override below.
pub type Channel<B> = MutexChannel<B>;

pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

// Register model crates
#[inference_server_registry(
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama3InstructServer",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama31InstructServer",
    ),
    // The only model wired through the continuous-batching channel. It implements
    // `BatchedInferenceServer`; every other model stays on the default `MutexChannel`.
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama321bInstructServer",
        channel_type = "BatchingChannel",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama323bInstructServer",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama321bInstructQ4Server",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::tiny",
        server_type = "TinyLlamaServer",
    ),
    server(
        crate_namespace = "burn_lm_parrot",
        server_type = "ParrotServer",
    )
)]
#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
}

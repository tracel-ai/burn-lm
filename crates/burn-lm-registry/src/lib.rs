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
    // Every real (non-quantized) Llama is wired through the continuous-batching channel: the 1b, 3b,
    // and both 8b servers implement `BatchedInferenceServer`. The Q4 1b and the non-Llama models
    // stay on the default `MutexChannel`.
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama3InstructServer",
        channel_type = "BatchingChannel",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama31InstructServer",
        channel_type = "BatchingChannel",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama321bInstructServer",
        channel_type = "BatchingChannel",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama323bInstructServer",
        channel_type = "BatchingChannel",
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

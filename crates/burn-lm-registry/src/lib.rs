use burn_lm_inference::*;
use burn_lm_macros::inference_server_registry;
use std::{collections::HashMap, sync::Arc};

pub type Channel<B> = MutexChannel<B>;

pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

// Register model crates
#[inference_server_registry(
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama3InstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama31InstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama321bInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama323bInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burn_lm_llama::server::tiny",
        server_type = "TinyLlamaServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burn_lm_parrot",
        server_type = "ParrotServer<InferenceBackend>",
    )
)]
#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
}

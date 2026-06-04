use burn_lm_inference::*;
use burn_lm_macros::inference_server_registry;
use std::{collections::HashMap, sync::Arc};

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
    server(
        crate_namespace = "burn_lm_llama::server::llama3",
        server_type = "Llama321bInstructServer",
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

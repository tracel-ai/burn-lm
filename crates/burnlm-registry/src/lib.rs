use burnlm_inference::*;
use burnlm_macros::inference_server_registry;
use std::{collections::HashMap, sync::Arc};

pub type Channel<B> = MutexChannel<B>;

pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

// Register model crates
#[inference_server_registry(
    server(
        crate_namespace = "burnlm_inference_llama3",
        server_type = "LlamaV3Params8BInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_llama3",
        server_type = "LlamaV31Params8BInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_llama3",
        server_type = "LlamaV32Params1BInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_llama3",
        server_type = "LlamaV32Params3BInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_template",
        server_type = "ParrotServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_tinyllama",
        server_type = "TinyLlamaServer<InferenceBackend>",
    ),
)]
#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
}

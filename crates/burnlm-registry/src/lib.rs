use burnlm_inference::*;
use burnlm_macros::inference_server_registry;
use std::{collections::HashMap, sync::Arc};

pub type Channel<B> = MutexChannel<B>;

pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

// Register model crates
pub use burnlm_inference_llama3::*;
pub use burnlm_inference_tinyllama::*;
#[inference_server_registry(
    server(
        server_name = "TinyLlama",
        server_type = "TinyLlamaServer<InferenceBackend>"
    ),
    server(
        server_name = "Llama3",
        server_type = "LlamaV3Params8BInstructServer<InferenceBackend>"
    ),
    server(
        server_name = "Llama31",
        server_type = "LlamaV31Params8BInstructServer<InferenceBackend>"
    )
)]
#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
}

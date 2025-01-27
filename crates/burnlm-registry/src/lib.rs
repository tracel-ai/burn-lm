use burnlm_inference::*;
use std::{collections::HashMap, sync::Arc};

// Register model crates
pub use burnlm_inference_llama3::*;
pub use burnlm_inference_tinyllama::*;

pub type Channel<B> = MutexChannel<B>;

// TinyLlama
pub type TinyLlamaS = TinyLlamaServer<InferenceBackend>;
pub type TinyLlamaC = InferenceClient<TinyLlamaS, Channel<TinyLlamaS>>;
// Llama 3
pub type Llama3S = LlamaV3Params8BInstructServer<InferenceBackend>;
pub type Llama3C = InferenceClient<Llama3S, Channel<Llama3S>>;
// Llama 3.1
pub type Llama31S = LlamaV31Params8BInstructServer<InferenceBackend>;
pub type Llama31C = InferenceClient<Llama31S, Channel<Llama31S>>;

pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        let mut map: DynClients = HashMap::new();
        {
            type S = TinyLlamaS;
            map.insert(
                S::model_name(),
                Box::new(TinyLlamaC::new(
                    S::model_name(),
                    S::model_cli_param_name(),
                    S::model_creation_date(),
                    S::owned_by(),
                    <S as InferenceServer>::Config::command,
                    S::parse_cli_config,
                    S::parse_json_config,
                    Channel::<S>::new(),
                )),
            );
        }
        {
            type S = Llama3S;
            map.insert(
                S::model_name(),
                Box::new(Llama3C::new(
                    S::model_name(),
                    S::model_cli_param_name(),
                    S::model_creation_date(),
                    S::owned_by(),
                    <S as InferenceServer>::Config::command,
                    S::parse_cli_config,
                    S::parse_json_config,
                    Channel::<S>::new(),
                )),
            );
        }
        {
            type S = Llama31S;
            map.insert(
                S::model_name(),
                Box::new(Llama31C::new(
                    S::model_name(),
                    S::model_cli_param_name(),
                    S::model_creation_date(),
                    S::owned_by(),
                    <S as InferenceServer>::Config::command,
                    S::parse_cli_config,
                    S::parse_json_config,
                    Channel::<S>::new(),
                )),
            );
        }
        Self {
            clients: Arc::new(map),
        }
    }

    pub fn get(&self) -> &DynClients {
        &self.clients
    }
}

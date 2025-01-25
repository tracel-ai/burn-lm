use std::{collections::HashMap, sync::Arc};
use burnlm_inference::*;

// Register model crates
pub use burnlm_plugin_llama3::*;
pub use burnlm_plugin_tinyllama::*;

pub type Channel<B> = MutexChannel<B>;

// TinyLlama
pub type TinyLlamaS = TinyLlamaServer<InferenceBackend>;
pub type TinyLlamaC = InferenceClient<TinyLlamaS, Channel<TinyLlamaS>>;
// Llama3
pub type Llama3S = Llama3Server<InferenceBackend>;
pub type Llama3C = InferenceClient<Llama3S, Channel<Llama3S>>;

pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
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
                    S::model_name_lc(),
                    S::model_creation_date(),
                    S::owned_by(),
                    <S as InferenceServer>::Config::command,
                    S::parse_cli_config,
                    S::parse_json_config,
                    S::get_model_versions,
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
                    S::model_name_lc(),
                    S::model_creation_date(),
                    S::owned_by(),
                    <S as InferenceServer>::Config::command,
                    S::parse_cli_config,
                    S::parse_json_config,
                    S::get_model_versions,
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


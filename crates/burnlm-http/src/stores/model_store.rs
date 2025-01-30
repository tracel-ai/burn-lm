use std::sync::Arc;

use axum::async_trait;
use burnlm_inference::InferencePlugin;
use burnlm_registry::Registry;

use crate::{
    controllers::model_controllers::ModelController, errors::ServerResult,
    schemas::model_schemas::ModelSchema,
};

#[derive(Debug)]
pub struct ModelStore {
    registry: Registry,
}

pub type ModelStoreState = Arc<tokio::sync::Mutex<ModelStore>>;

impl ModelStore {
    fn new() -> Self {
        Self {
            registry: Registry::new(),
        }
    }

    pub fn create_state() -> ModelStoreState {
        Arc::new(tokio::sync::Mutex::new(ModelStore::new()))
    }
}

#[async_trait]
impl ModelController for ModelStore {
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>> {
        let mut models = vec![];
        let mut installed: Vec<_> = self.registry.get().iter().filter(|(_name, plugin)| plugin.is_downloaded()).collect();
        installed.sort_by_key(|(key, ..)| *key);
        for (_name, plugin) in installed {
            models.push(ModelSchema::from(plugin));
        }
        Ok(models)
    }

    async fn get_model_plugin(&self, name: &str) -> ServerResult<&Box<dyn InferencePlugin>> {
        let plugin = self
            .registry
            .get()
            .iter()
            .find(|(pname, _)| (**pname).to_lowercase() == name.to_lowercase())
            .map(|(_, plugin)| plugin);
        let plugin = plugin.unwrap_or_else(|| panic!("Model plugin should be registered: {name}"));
        Ok(plugin)
    }
}

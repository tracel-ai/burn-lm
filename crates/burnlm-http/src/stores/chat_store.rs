use std::sync::Arc;

use async_trait::async_trait;
use burnlm_inference::InferencePlugin;
use burnlm_registry::Registry;

use crate::{
    controllers::chat_controllers::ChatController, errors::ServerResult,
    schemas::model_schemas::ModelSchema,
};

#[derive(Debug)]
pub struct ChatStore {
    registry: Registry,
    // the current active plugin name
    current_plugin_name: Option<String>,
}

pub type ModelStoreState = Arc<tokio::sync::Mutex<ChatStore>>;

impl ChatStore {
    fn new() -> Self {
        Self {
            registry: Registry::new(),
            current_plugin_name: None,
        }
    }

    pub fn create_state() -> ModelStoreState {
        Arc::new(tokio::sync::Mutex::new(ChatStore::new()))
    }
}

#[async_trait]
impl ChatController for ChatStore {
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>> {
        let mut models = vec![];
        let mut installed: Vec<_> = self
            .registry
            .get()
            .iter()
            .filter(|(_name, plugin)| plugin.is_downloaded())
            .collect();
        installed.sort_by_key(|(key, ..)| *key);
        for (_name, plugin) in installed {
            models.push(ModelSchema::from(plugin));
        }
        Ok(models)
    }

    async fn get_plugin(
        &mut self,
        name: &str,
    ) -> ServerResult<(Box<dyn InferencePlugin>, Option<String>)> {
        let requested_plugin = self
            .registry
            .get()
            .iter()
            .find(|(pname, _)| (**pname).to_lowercase() == name.to_lowercase())
            .map(|(_, plugin)| plugin);
        let requested_plugin =
            requested_plugin.unwrap_or_else(|| panic!("plugin '{name}' should be registered"));

        // unload plugin if we request a different plugin than the current one
        let mut old_model_name = None;
        if self.current_plugin_name.is_some()
            && *self.current_plugin_name.as_ref().unwrap()
                != requested_plugin.model_name().to_string()
        {
            let current_plugin_name = self.current_plugin_name.as_ref().unwrap();
            let current_plugin = self
                .registry
                .get()
                .iter()
                .find(|(pname, _)| **pname == current_plugin_name)
                .map(|(_, plugin)| plugin);
            let current_plugin = current_plugin
                .unwrap_or_else(|| panic!("previous plugin '{name}' should be registered"));
            tracing::debug!("Unloading model '{current_plugin_name}'");
            if let Err(error) = current_plugin.unload() {
                tracing::error!(
                    "Cannot unload plugin '{name}' (reason: {})",
                    error.to_string()
                );
            }
            old_model_name = Some(current_plugin.model_name().to_string());
        }

        self.current_plugin_name = Some(requested_plugin.model_name().to_string());
        Ok((requested_plugin.clone(), old_model_name))
    }
}

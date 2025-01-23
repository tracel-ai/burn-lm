use std::sync::Arc;

use axum::async_trait;

use crate::{
    controllers::model_controllers::ModelController, errors::ServerResult,
    schemas::model_schemas::ModelSchema,
};

#[derive(Debug, Clone)]
pub struct ModelStore {
    models: Vec<Arc<ModelSchema>>,
    current: Option<Arc<ModelSchema>>,
}

pub type ModelStoreState = Arc<tokio::sync::Mutex<ModelStore>>;

impl ModelStore {
    fn new() -> Self {
        let mut models = vec![];
        for plugin in burnlm_registry::get_inference_plugins() {
            models.push(Arc::new(ModelSchema::from(plugin)));
        }
        Self {
            models,
            current: None,
        }
    }

    pub fn create_state() -> ModelStoreState {
        Arc::new(tokio::sync::Mutex::new(ModelStore::new()))
    }
}

#[async_trait]
impl ModelController for ModelStore {
    async fn list_models(&self) -> ServerResult<Vec<Arc<ModelSchema>>> {
        Ok(self.models.clone())
    }

    async fn set_current(&self, name: &str) -> ServerResult<()> {
        Ok(())
    }
}

use axum::async_trait;
use chrono::Utc;

use crate::{
    controllers::model_controllers::ModelController, errors::ServerResult,
    schemas::model_schemas::ModelSchema,
};

#[derive(Debug, Clone)]
pub struct ModelStore {
    models: Vec<ModelSchema>,
}

impl ModelStore {
    pub fn new() -> Self {
        let now = Utc::now().timestamp() as u32;
        let object = "model".to_string();
        let owned_by = "Tracel Technologies Inc.".to_string();
        Self {
            models: vec![
                #[cfg(feature = "tinyllama")]
                ModelSchema {
                    id: "TinyLlama".to_string(),
                    created: now,
                    object: object.clone(),
                    owned_by: owned_by.clone(),
                },
                #[cfg(feature = "llama3")]
                ModelSchema {
                    id: "llama 3".to_string(),
                    created: now,
                    object: object.clone(),
                    owned_by: owned_by.clone(),
                },
                #[cfg(feature = "llama31")]
                ModelSchema {
                    id: "llama 3.1".to_string(),
                    created: now,
                    object: object.clone(),
                    owned_by: owned_by.clone(),
                },
            ],
        }
    }
}

#[async_trait]
impl ModelController for ModelStore {
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>> {
        Ok(self.models.clone())
    }
}

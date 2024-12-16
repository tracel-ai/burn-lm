use axum::async_trait;
use chrono::Utc;

use crate::{controllers::model_controllers::ModelController, errors::ServerResult, schemas::model_schemas::ModelSchema};

#[derive(Debug, Clone)]
pub struct ModelStore {
    models: Vec<ModelSchema>,
}

impl ModelStore {
    pub fn new() -> Self {
        let now = Utc::now().timestamp() as u32;
        let object = "model".to_string();
        let owned_by = "Tracel Technologies Inc.".to_string();
        Self { models: vec![
            ModelSchema {
                id: "llama-31".to_string(),
                created: now,
                object: object.clone(),
                owned_by: owned_by.clone(),
            },
            ModelSchema {
                id: "llama-32".to_string(),
                created: now,
                object: object.clone(),
                owned_by: owned_by.clone(),
            },
        ]}
    }
}

#[async_trait]
impl ModelController for ModelStore {
    async fn list_models (&self) -> ServerResult<Vec<ModelSchema>> {
        Ok(self.models.clone())
    }
}

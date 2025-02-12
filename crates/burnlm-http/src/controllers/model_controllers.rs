use crate::{errors::ServerResult, schemas::model_schemas::ModelSchema};
use axum::async_trait;
use burnlm_inference::InferencePlugin;

#[async_trait]
pub trait ModelController {
    async fn get_model_plugin(&self, name: &str) -> ServerResult<Box<dyn InferencePlugin>>;
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>>;
}

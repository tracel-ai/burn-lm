use crate::{errors::ServerResult, schemas::model_schemas::ModelSchema};
use axum::async_trait;
use burnlm_inference::InferencePlugin;

#[async_trait]
pub trait ChatController {
    async fn get_plugin(
        &mut self,
        name: &str,
    ) -> ServerResult<(Box<dyn InferencePlugin>, Option<String>)>;
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>>;
}

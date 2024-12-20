use crate::{errors::ServerResult, schemas::model_schemas::ModelSchema};
use axum::async_trait;

#[async_trait]
pub trait ModelController {
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>>;
}

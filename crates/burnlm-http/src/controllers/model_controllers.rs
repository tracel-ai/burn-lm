use std::sync::Arc;

use crate::{errors::ServerResult, schemas::model_schemas::ModelSchema};
use axum::async_trait;

#[async_trait]
pub trait ModelController {
    async fn set_current(&self, name: &str) -> ServerResult<()>;
    async fn list_models(&self) -> ServerResult<Vec<Arc<ModelSchema>>>;
}

use axum::async_trait;
use crate::{errors::ServerResult, schemas::model_schemas::ModelSchema};

#[async_trait]
pub trait ModelController {

    async fn list_models (&self) -> ServerResult<Vec<ModelSchema>>;

}

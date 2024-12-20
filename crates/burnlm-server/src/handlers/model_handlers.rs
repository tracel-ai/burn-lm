use axum::extract::Path;
use axum::{extract::State, Json};

use crate::errors::ServerError;
use crate::{
    controllers::model_controllers::ModelController, errors::ServerResult,
    schemas::model_schemas::ModelSchema, stores::model_store::ModelStore,
};

use crate::constants::API_VERSION;

#[utoipa::path(
    get,
    path = format!("/{}/models", API_VERSION),
    responses(
        (status = 200, description = "Gets all models.", body = Vec<ModelSchema>),
    )
)]
pub async fn list_models(State(store): State<ModelStore>) -> ServerResult<Json<Vec<ModelSchema>>> {
    let models = store.list_models().await?;
    Ok(Json(models.into()))
}

#[utoipa::path(
    get,
    path = format!("/{}/models/{{model}}", API_VERSION),
    responses(
        (status = 200, description = "Get model information.", body = ModelSchema),
    )
)]
pub async fn get_model(
    State(store): State<ModelStore>,
    Path(model): Path<String>,
) -> ServerResult<Json<ModelSchema>> {
    let models = store.list_models().await?;
    models
        .iter()
        .find(|m| model == m.id)
        .map_or_else(|| Err(ServerError::NotFound), |info| Ok(Json(info.clone())))
}

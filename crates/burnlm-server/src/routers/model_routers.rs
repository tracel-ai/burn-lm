use crate::{handlers::model_handlers::*, stores::model_store::ModelStore};

use axum::{routing::get, Router};

pub fn public_router(state: ModelStore) -> Router {
    Router::new()
        .route("/models", get(list_models))
        .route("/models/:model", get(get_model))
        .with_state(state)
}

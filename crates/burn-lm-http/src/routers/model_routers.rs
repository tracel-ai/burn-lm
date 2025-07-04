use crate::{handlers::model_handlers::*, stores::chat_store::ModelStoreState};

use axum::{routing::get, Router};

pub fn public_router(state: ModelStoreState) -> Router {
    Router::new()
        .route("/models", get(list_models))
        .route("/models/{model}", get(get_model))
        .with_state(state)
}

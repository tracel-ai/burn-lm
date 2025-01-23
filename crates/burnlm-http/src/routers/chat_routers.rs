use crate::{handlers::chat_handlers::*, stores::model_store::ModelStoreState};

use axum::{routing::post, Router};

pub fn public_router(state: ModelStoreState) -> Router {
    Router::new().route("/chat/completions", post(chat_completions)).with_state(state)
}

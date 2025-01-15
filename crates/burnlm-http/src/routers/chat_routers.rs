use crate::handlers::chat_handlers::*;

use axum::{routing::post, Router};

pub fn public_router() -> Router {
    Router::new().route("/chat/completions", post(chat_completions))
}

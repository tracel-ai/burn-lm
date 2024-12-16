use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

pub type ServerResult<T> = core::result::Result<T, ServerError>;
pub type ServerOptionalResult<T> = core::result::Result<Option<T>, ServerError>;

#[derive(thiserror::Error, Debug)]
pub enum ServerError {
    #[error("Resource not found")]
    NotFound,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        match self {
            ServerError::NotFound => handle_not_found_error(),
        }
    }
}

// IntoResponse error handlers

fn handle_not_found_error() -> Response {
    let msg = "Resource not found";
    tracing::error!("{msg}");
    let status = StatusCode::NOT_FOUND;
    (status, msg).into_response()
}

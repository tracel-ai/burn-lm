use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::schemas::chat_schemas::ChoiceMessageRoleSchema;

pub type ServerResult<T> = core::result::Result<T, ServerError>;
pub type ServerOptionalResult<T> = core::result::Result<Option<T>, ServerError>;

#[derive(thiserror::Error, Debug)]
pub enum ServerError {
    #[error("Resource not found")]
    NotFound,
    #[error("Error loading model (reason: {0})")]
    LoadingError(String),
    #[error("")]
    UserRoleExpected(ChoiceMessageRoleSchema),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        match self {
            ServerError::NotFound => handle_not_found_error(),
            ServerError::UserRoleExpected(role) => handle_user_role_expected_error(role),
            ServerError::LoadingError(reason) => handle_loading_model_error(reason),
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

fn handle_loading_model_error(reason: String) -> Response {
    let msg = format!("Error loading model (reason: {reason}).");
    tracing::error!("{msg}");
    let status = StatusCode::INTERNAL_SERVER_ERROR;
    (status, msg).into_response()
}

fn handle_user_role_expected_error(role: ChoiceMessageRoleSchema) -> Response {
    let msg = format!("Role should be 'user' and not '{role}'.");
    tracing::error!("{msg}");
    let status = StatusCode::BAD_REQUEST;
    (status, msg).into_response()
}

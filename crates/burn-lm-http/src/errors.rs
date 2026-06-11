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
    #[error("Model '{0}' is not registered")]
    ModelNotFound(String),
    #[error("Error loading model (reason: {0})")]
    LoadingError(String),
    #[error("")]
    UserRoleExpected(ChoiceMessageRoleSchema),
    #[error("Server is overloaded; retry later")]
    Overloaded,
    #[error("Inference failed (reason: {0})")]
    Inference(String),
}

/// Map inference-layer errors onto HTTP semantics: a shed job (`Overloaded`) is the retryable
/// 429 case; everything else surfaces as a 500 with the error's own message.
impl From<burn_lm_inference::InferenceError> for ServerError {
    fn from(error: burn_lm_inference::InferenceError) -> Self {
        match error {
            burn_lm_inference::InferenceError::Overloaded => ServerError::Overloaded,
            other => ServerError::Inference(other.to_string()),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        match self {
            ServerError::NotFound => handle_not_found_error(),
            ServerError::ModelNotFound(name) => handle_model_not_found_error(name),
            ServerError::UserRoleExpected(role) => handle_user_role_expected_error(role),
            ServerError::LoadingError(reason) => handle_loading_model_error(reason),
            ServerError::Overloaded => handle_overloaded_error(),
            ServerError::Inference(reason) => handle_inference_error(reason),
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

fn handle_model_not_found_error(name: String) -> Response {
    let msg = format!("Model '{name}' is not registered.");
    tracing::error!("{msg}");
    (StatusCode::NOT_FOUND, msg).into_response()
}

fn handle_loading_model_error(reason: String) -> Response {
    let msg = format!("Error loading model (reason: {reason}).");
    tracing::error!("{msg}");
    let status = StatusCode::INTERNAL_SERVER_ERROR;
    (status, msg).into_response()
}

fn handle_overloaded_error() -> Response {
    let msg = "Server is overloaded; retry later.";
    tracing::warn!("{msg}");
    (StatusCode::TOO_MANY_REQUESTS, msg).into_response()
}

fn handle_inference_error(reason: String) -> Response {
    let msg = format!("Inference failed (reason: {reason}).");
    tracing::error!("{msg}");
    (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
}

fn handle_user_role_expected_error(role: ChoiceMessageRoleSchema) -> Response {
    let msg = format!("Role should be 'user' and not '{role}'.");
    tracing::error!("{msg}");
    let status = StatusCode::BAD_REQUEST;
    (status, msg).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_lm_inference::InferenceError;

    /// Backpressure contract with clients: a shed job (`Overloaded`) must surface as a retryable
    /// 429, not a generic 500.
    #[test]
    fn overloaded_maps_to_429() {
        let err: ServerError = InferenceError::Overloaded.into();
        assert!(matches!(err, ServerError::Overloaded));
        assert_eq!(err.into_response().status(), StatusCode::TOO_MANY_REQUESTS);
    }

    /// Every other inference failure (e.g. a dead worker) is a server-side 500 carrying the
    /// inference error's own message.
    #[test]
    fn other_inference_errors_map_to_500() {
        let err: ServerError = InferenceError::WorkerDied.into();
        assert_eq!(
            err.into_response().status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}

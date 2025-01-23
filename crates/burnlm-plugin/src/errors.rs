pub type InferenceResult<T> = Result<T, InferenceError>;
pub type InferenceOptionalResult<T> = Result<Option<T>, InferenceError>;

#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
    #[error("Error loading model: {0}")]
    LoadError(String),
    #[error("Model has not been loaded.")]
    ModelNotLoaded,
}

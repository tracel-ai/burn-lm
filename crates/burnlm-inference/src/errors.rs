pub type InferenceResult<T> = Result<T, InferenceError>;
pub type InferenceOptionalResult<T> = Result<Option<T>, InferenceError>;

#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
    #[error("Error deleting model: {0} (reason: {1})")]
    DeleteError(String, String),
    #[error("Error downloading model: {0} (reason: {1})")]
    DownloadError(String, String),
    #[error("Error loading model: {0}")]
    LoadError(String),
    #[error("Model has not been loaded.")]
    ModelNotLoaded,
    #[error("The plugin '{0}' does not support downloading.")]
    PluginDownloadUnsupportedError(String),
    #[error("Error unloading model: {0} (reason: {1})")]
    UnloadError(String, String),
}

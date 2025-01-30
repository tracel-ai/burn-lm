pub type InferenceResult<T> = Result<T, InferenceError>;
pub type InferenceOptionalResult<T> = Result<Option<T>, InferenceError>;

#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
    #[error("Error downloading model: {0} (reason: {1})")]
    DownloadError(String, String),
    #[error("Error downloading tokenizer for model: {0} (reason: {1})")]
    DownloadTokenizerError(String, String),
    #[error("Error downloading weights for model: {0} (reason: {1})")]
    DownloadWeightError(String, String),
    #[error("Error loading model: {0}")]
    LoadError(String),
    #[error("The plugin '{0}' does not support downloading.")]
    PluginDownloadUnsupportedError(String),
    #[error("Model has not been loaded.")]
    ModelNotLoaded,
}

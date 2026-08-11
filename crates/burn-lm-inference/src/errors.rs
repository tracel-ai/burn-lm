pub type InferenceResult<T> = Result<T, InferenceError>;
pub type InferenceOptionalResult<T> = Result<Option<T>, InferenceError>;

#[derive(thiserror::Error, Debug, Clone)]
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
    #[error("Input sequence length ({0} tokens) exceeds maximum context window ({1} tokens). Please shorten your input or increase the maximum context window.")]
    ContextLengthExceeded(usize, usize),
    #[error("The request needs {0} more KV block(s) than the pool holds. Lower max_tokens or shorten the prompt.")]
    KvPoolExhausted(usize),
    #[error("Decoder forward violated the batch contract: {0}")]
    BatchContractViolation(String),
    #[error("Model is busy ({0} active sequence(s), {1} queued job(s)); retry once in-flight generation completes.")]
    Busy(usize, usize),
    #[error("The job was cancelled before it produced a result.")]
    Cancelled,
    #[error("The server is overloaded: the job queue is full. Retry later.")]
    Overloaded,
    #[error("The inference worker died while the job was in flight. Retry the request.")]
    WorkerDied,
}

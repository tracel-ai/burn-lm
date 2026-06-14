use burn_lm_inference::{InferenceError, InferenceResult, Stats};

use crate::{inference::Llama, tokenizer::Tokenizer};

/// A model slot that is either empty or holds a loaded model.
///
/// Centralizes the "is it loaded? hand me a reference, else `ModelNotLoaded`" handshake so the
/// servers never repeat `match &self.model { Some(m) => ..., None => Err(...) }`. The model is owned
/// directly: the channel (mutex / passthrough / the batching worker thread) already provides
/// exclusive access, so no inner lock is needed.
pub(crate) struct LoadedModel<T: Tokenizer> {
    inner: Option<Llama<T>>,
}

impl<T: Tokenizer> Default for LoadedModel<T> {
    fn default() -> Self {
        Self { inner: None }
    }
}

impl<T: Tokenizer> std::fmt::Debug for LoadedModel<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("loaded", &self.inner.is_some())
            .finish()
    }
}

impl<T: Tokenizer + 'static> LoadedModel<T> {
    /// Whether a model is currently loaded.
    pub(crate) fn is_loaded(&self) -> bool {
        self.inner.is_some()
    }

    /// Install a freshly built model.
    pub(crate) fn store(&mut self, model: Llama<T>) {
        self.inner = Some(model);
    }

    /// Borrow the loaded model, or return `InferenceError::ModelNotLoaded`.
    pub(crate) fn get(&self) -> InferenceResult<&Llama<T>> {
        self.inner.as_ref().ok_or(InferenceError::ModelNotLoaded)
    }

    /// Mutably borrow the loaded model, or return `InferenceError::ModelNotLoaded`.
    pub(crate) fn get_mut(&mut self) -> InferenceResult<&mut Llama<T>> {
        self.inner.as_mut().ok_or(InferenceError::ModelNotLoaded)
    }

    /// Drop the loaded model.
    pub(crate) fn unload(&mut self) -> InferenceResult<Option<Stats>> {
        self.inner = None;
        Ok(None)
    }
}

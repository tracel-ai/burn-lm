// ---------------------------------------------------------------------------
// Register model crates
pub use burnlm_plugin_llama3::*;
pub use burnlm_plugin_tinyllama::*;
// ---------------------------------------------------------------------------
use burnlm_inference::plugin::*;

pub fn get_models() -> Vec<&'static InferenceModelPlugin<InferenceBackend>> {
    inventory::iter::<InferenceModelPlugin<InferenceBackend>>
        .into_iter()
        .collect()
}

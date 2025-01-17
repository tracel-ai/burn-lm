// ---------------------------------------------------------------------------
// Register model crates
pub use burnlm_plugin_llama3::*;
pub use burnlm_plugin_tinyllama::*;
// ---------------------------------------------------------------------------
use burnlm_inference::plugin::*;

pub fn get_inference_plugins() -> Vec<&'static InferencePluginMetadata<InferenceBackend>> {
    inventory::iter::<InferencePluginMetadata<InferenceBackend>>
        .into_iter()
        .collect()
}

pub fn get_inference_plugin(
    name: &str,
) -> Option<&'static InferencePluginMetadata<InferenceBackend>> {
    inventory::iter::<InferencePluginMetadata<InferenceBackend>>
        .into_iter()
        .find(|p| p.model_name_lc == name.to_lowercase())
}

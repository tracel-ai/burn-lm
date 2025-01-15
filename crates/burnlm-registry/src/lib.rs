// Register model crates
pub use burnlm_plugin_llama3::*;
pub use burnlm_plugin_tinyllama::*;

use burnlm_inference::plugin::InferenceModelPlugin;

pub fn get_models() -> Vec<&'static InferenceModelPlugin> {
    inventory::iter::<InferenceModelPlugin>
        .into_iter()
        .collect()
}

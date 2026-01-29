use std::path::PathBuf;

use burn::tensor;
use burn_lm_inference::{serde_json, InferenceResult};
use burn_store::{ModuleStore, SafetensorsStore};

use crate::error_wrapper::Wrap;
use crate::hf_downloader::get_file;

mod attention;
mod config;
mod decoder_layer;
pub fn load_model(name: &str, safe_tensors: &[PathBuf]) -> InferenceResult<()> {
    let config_path = get_file(name, "config.json")?;

    let reader = std::fs::File::open(config_path).w()?;
    let config: config::Qwen3MoeConfig = serde_json::from_reader(reader).w()?;

    let mut tensor_store = SafetensorsStore::default();
    let mut tensor_store = tensor_store
        .with_full_paths(safe_tensors.iter().map(|p| p.to_string_lossy()))
        .match_all();

    let tensors = tensor_store.get_all_snapshots().w()?;
    Ok(())
}

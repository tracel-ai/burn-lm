use std::path::PathBuf;

use burn_lm_inference::{serde_json, Backend, InferenceBackend, InferenceResult};
use burn_std::device::{Device, DeviceId};
use burn_store::{ModuleStore, SafetensorsStore};

use crate::error_wrapper::Wrap;
use crate::hf_downloader::get_file;
use crate::model::model::Qwen3MoeModel;

mod attention;
mod basic_device_mapper;
mod config;
mod decoder_layer;
mod model;
mod moe;

pub fn load_model<B: Backend>(name: &str, safe_tensors: &[PathBuf], b: &B) -> InferenceResult<()> {
    let config_path = get_file(name, "config.json")?;

    let reader = std::fs::File::open(config_path).w()?;
    let config: config::Qwen3MoeConfig = serde_json::from_reader(reader).w()?;

    let mut tensor_store = SafetensorsStore::default();
    let mut tensor_store = tensor_store
        .with_full_paths(safe_tensors.iter().map(|p| p.to_string_lossy()))
        .match_all();

    let tensors = tensor_store.get_all_snapshots().w()?;
    let device_mappings = basic_device_mapper::map_layers_to_devices::<B::Device>(
        config.base_config.num_hidden_layers,
    );

    Qwen3MoeModel::<B>::new(&tensors, &config, device_mappings).w()?;

    Ok(())
}

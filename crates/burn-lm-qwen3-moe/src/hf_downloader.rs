use burn_lm_inference::{InferenceResult, StatEntry, Stats};
use hf_hub::api::sync::Api;
use tracing::info;

use crate::error_wrapper::Wrap;

pub fn get_file(model_name: &str, file_name: &str) -> InferenceResult<std::path::PathBuf> {
    let api = Api::new().w()?;

    let model = api.model(model_name.into());
    let downloaded = model.get(file_name).w()?;

    Ok(downloaded)
}

pub fn download_safetensors(model_name: &str) -> InferenceResult<Option<Stats>> {
    let api = Api::new().w()?;

    let model = api.model(model_name.into());
    let repo_info = api.model(model_name.into()).info().w()?;

    let mut stats = Stats::new();

    let now = std::time::Instant::now();
    for file in repo_info.siblings {
        let downloaded = model.get(&file.rfilename).w()?;
        let metadata = std::fs::metadata(downloaded).w()?;

        stats.entries.insert(StatEntry::Named(
            file.rfilename.into(),
            metadata.len().to_string(),
        ));
    }

    stats
        .entries
        .insert(StatEntry::ModelDownloadingDuration(now.elapsed()));
    Ok(Some(stats))
}

pub fn exists(model_name: &str) -> InferenceResult<bool> {
    let cache = hf_hub::Cache::from_env();
    let api = Api::new().w()?;

    let repo_info = api.model(model_name.into()).info().w()?;
    let cached_repo = cache.model(model_name.into());
    let all_files_cached = repo_info
        .siblings
        .iter()
        .map(|s| cached_repo.get(&s.rfilename))
        .all(|e| e.is_some());
    info!("all files cached {}", all_files_cached);
    Ok(all_files_cached)
}

pub fn delete(model_name: &str) -> InferenceResult<Option<Stats>> {
    let cache = hf_hub::Cache::from_env();
    let api = Api::new().w()?;

    let repo_info = api.model(model_name.into()).info().w()?;
    let cached_repo = cache.model(model_name.into());
    let mut stats = Stats::new();

    for file in repo_info.siblings {
        if let Some(cached_file) = cached_repo.get(&file.rfilename) {
            std::fs::remove_file(&cached_file).w()?;
            stats.entries.insert(burn_lm_inference::StatEntry::Named(
                "Delete".into(),
                cached_file.to_string_lossy().into(),
            ));
        }
    }
    Ok(Some(stats))
}

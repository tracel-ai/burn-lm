use burnlm_inference::{InferenceError, InferenceResult};
use burnlm_registry::Registry;

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("download").about("Download models");
    let registry = Registry::new();
    // Download all
    let subcommand = clap::Command::new("all").about("Download all downloadable models");
    root = root.subcommand(subcommand);
    // Create a a subcommand for each registered model
    for (_name, plugin) in registry.get().iter() {
        let about = if plugin.downloader().is_some() {
            if  plugin.is_downloaded() {
                "✅ Downloaded"
            } else {
                "👍 Downloadable"
            }
        } else {
            "🚫 Not downloadable"
        };
        let subcommand = clap::Command::new(plugin.model_cli_param_name())
            .about(about);
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> anyhow::Result<()> {
    let registry = Registry::new();
    let downloaders: Vec<(String, fn() -> InferenceResult<()>)> = match args.subcommand_name().unwrap() {
        "all" => {
            registry.get().iter()
                .filter(|(_, plugin)| plugin.downloader().is_some())
                .map(|(name, plugin)| (name.to_string(), plugin.downloader().unwrap()))
                .collect()
        },
        model => {
            let (name, plugin) = registry.get().iter()
                .find(|(_, p)| p.model_cli_param_name() == model)
                .expect("Plugin should be registered");
            let downloader = plugin.downloader();
            match downloader {
                Some(dl) => vec![(name.to_string(), dl)],
                None => anyhow::bail!(InferenceError::PluginDownloadUnsupportedError(model.to_string())),
            }
        },
    };

    // let's download each model sequentially. Doing it concurrently, while working might
    // be awful if a large number of models are registered. Moerover currently the llama
    // models use the Burn downloader. The Burn downloader uses a progress bar that does
    // not support MultiProgress of indicatif crate.
    for (i, (name, dl)) in downloaders.iter().enumerate() {
        println!("[{}/{}] Downloading model: {name}\nPlease wait...", i + 1, downloaders.len());
        if let Err(err) = dl() {
            eprintln!("Download error: {}", err);
        }
        println!("✅ Download complete!");
    }
    Ok(())
}

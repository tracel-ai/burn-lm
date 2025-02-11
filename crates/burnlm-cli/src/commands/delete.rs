use burnlm_inference::{InferenceError, InferencePlugin, InferenceResult, Stats};
use burnlm_registry::Registry;

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("delete").about("Delete downloaded models");
    let registry = Registry::new();
    // delete all
    let subcommand = clap::Command::new("all").about("Delete all downloaded models");
    root = root.subcommand(subcommand);
    // Create a a subcommand for each downloaded model
    let mut reg_entries: Vec<_> = registry
        .get()
        .iter()
        .filter(|(_, p)| deletable(*p))
        .collect();
    reg_entries.sort_by_key(|(key, ..)| *key);
    for (_name, plugin) in reg_entries {
        root = root.subcommand(
            clap::Command::new(plugin.model_cli_param_name())
                .about(format!("Delete model '{}'", plugin.model_name())),
        );
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> super::HandleCommandResult {
    let registry = Registry::new();
    let deleters = match args.subcommand_name() {
        Some("all") => {
            let mut candidates: Vec<(String, fn() -> InferenceResult<Option<Stats>>)> = registry
                .get()
                .iter()
                .filter(|(_, p)| deletable(*p))
                .map(|(n, p)| (n.to_string(), p.deleter().unwrap()))
                .collect();
            candidates.sort_by(|(n1, ..), (n2, ..)| n1.cmp(n2));
            candidates
        }
        Some(model) => {
            let (name, plugin) = registry
                .get()
                .iter()
                .find(|(_, p)| p.model_cli_param_name() == model)
                .expect("Plugin should be registered");
            let deleter = plugin.deleter();
            match deleter {
                Some(rm) => vec![(name.to_string(), rm)],
                None => anyhow::bail!(InferenceError::PluginDownloadUnsupportedError(
                    model.to_string()
                )),
            }
        }
        None => {
            create().print_help().unwrap();
            return Ok(None);
        }
    };

    // let's delete each model sequentially to mimic the download command behabior.
    for (i, (name, rm)) in deleters.iter().enumerate() {
        println!(
            "[{}/{}] Deleting model: {name}\nPlease wait...",
            i + 1,
            deleters.len()
        );
        if let Err(err) = rm() {
            eprintln!("Deletion error: {}", err);
        }
        println!("✅ Delete complete!");
    }
    Ok(Some(super::ShellMetaAction::RefreshParser))
}

fn deletable(plugin: &Box<dyn InferencePlugin>) -> bool {
    plugin.downloader().is_some() && plugin.is_downloaded() && plugin.deleter().is_some()
}

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("run").about("Run inference on chosen model");
    // Create a a subcommand for each registered model with its associated config flags
    for plugin in burnlm_registry::get_inference_plugins() {
        let mut subcommand = clap::Command::new(plugin.model_name_lc)
            .about(format!("Use {} model", plugin.model_name));
        subcommand = subcommand.args((plugin.config_flags_fn)().get_arguments());
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle() -> anyhow::Result<()> {
    println!("Run inference");
    Ok(())
}

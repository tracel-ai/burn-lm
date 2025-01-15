pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("run").about("Run inference on chosen model");
    // Create a a subcommand for each registered model with its associated config flags
    for model in burnlm_registry::get_models() {
        let mut subcommand =
            clap::Command::new(model.lc_name).about(format!("Use {} model", model.name));
        subcommand = subcommand.args((model.config_flags)().get_arguments());
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle() -> anyhow::Result<()> {
    println!("Run inference");
    Ok(())
}

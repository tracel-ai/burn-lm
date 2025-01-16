pub(crate) fn create() -> clap::Command {
    clap::Command::new("models").about("List all available models")
}

pub(crate) fn handle() -> anyhow::Result<()> {
    println!("Available Models:");
    for plugin in burnlm_registry::get_inference_plugins() {
        println!("- {}", plugin.model_name);
    }
    Ok(())
}

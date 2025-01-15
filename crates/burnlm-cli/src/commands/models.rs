pub(crate) fn create() -> clap::Command {
    clap::Command::new("models").about("List all available models")
}

pub(crate) fn handle() -> anyhow::Result<()> {
    println!("Available Models:");
    for model in burnlm_registry::get_models() {
        println!("- {}", model.name);
    }
    Ok(())
}

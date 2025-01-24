use burnlm_registry::Registry;

pub(crate) fn create() -> clap::Command {
    clap::Command::new("models").about("List all available models")
}

pub(crate) fn handle() -> anyhow::Result<()> {
    println!("Available Models:");
    let mut registry = Registry::default();
    for (name, ..) in registry.get().iter() {
        println!("- {}", name);
    }
    Ok(())
}

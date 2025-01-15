use crate::commands;

pub fn run() -> anyhow::Result<()> {
    // Define CLI
    let cli = clap::command!()
        .subcommand(commands::models::create())
        .subcommand(commands::run::create());

    // Execute commands
    let matches = cli.get_matches();
    if let Some(_) = matches.subcommand_matches("models") {
        commands::models::handle()
    } else if let Some(_) = matches.subcommand_matches("run") {
        commands::run::handle()
    } else {
        Ok(())
    }
}

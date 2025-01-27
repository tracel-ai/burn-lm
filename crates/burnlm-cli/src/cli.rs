use crate::commands;

pub fn run() -> anyhow::Result<()> {
    // Define CLI
    let cli = clap::command!()
        .subcommand(commands::models::create())
        .subcommand(commands::run::create());

    // Execute commands
    let matches = cli.get_matches();
    if matches.subcommand_matches("models").is_some() {
        commands::models::handle()
    } else if let Some(args) = matches.subcommand_matches("run") {
        commands::run::handle(args)
    } else {
        Ok(())
    }
}

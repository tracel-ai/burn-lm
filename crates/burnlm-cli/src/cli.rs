use crate::commands;

pub fn run() -> anyhow::Result<()> {
    // Define CLI
    let cli = clap::command!()
        .subcommand(commands::download::create())
        .subcommand(commands::models::create())
        .subcommand(commands::run::create())
        .subcommand(commands::new::create())
        .subcommand(commands::web::create());

    // Execute commands
    let matches = cli.get_matches();
    if let Some(args) = matches.subcommand_matches("download") {
        commands::download::handle(args)
    } else if matches.subcommand_matches("models").is_some() {
        commands::models::handle()
    } else if let Some(args) = matches.subcommand_matches("new") {
        commands::new::handle(args)
    } else if let Some(args) = matches.subcommand_matches("run") {
        commands::run::handle(args)
    } else if let Some(args) = matches.subcommand_matches("web") {
        commands::web::handle(args)
    } else {
        Ok(())
    }
}

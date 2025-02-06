use crate::commands;

pub fn run() -> anyhow::Result<()> {
    println!();
    // Define CLI
    let mut cli = clap::command!()
        .subcommand(commands::backends::create())
        .subcommand(commands::chat::create())
        .subcommand(commands::download::create())
        .subcommand(commands::models::create())
        .subcommand(commands::new::create())
        .subcommand(commands::run::create())
        .subcommand(commands::server::create())
        .subcommand(commands::shell::create())
        .subcommand(commands::web::create());

    // Execute commands
    let matches = cli.clone().get_matches();

    if matches.subcommand_matches("backends").is_some() {
        commands::backends::handle().map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("chat") {
        commands::chat::handle(args, None).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("download") {
        commands::download::handle(args).map(|_| ())
    } else if matches.subcommand_matches("models").is_some() {
        commands::models::handle().map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("new") {
        commands::new::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("run") {
        commands::run::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("server") {
        commands::server::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("shell") {
        commands::shell::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("web") {
        commands::web::handle(args).map(|_| ())
    } else {
        cli.print_help().unwrap();
        Ok(())
    }
}

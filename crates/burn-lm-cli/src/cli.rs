use clap::Arg;

use crate::commands;

pub fn run(backend: &str, dtype: &str) -> anyhow::Result<()> {
    // Define CLI
    let cli = clap::command!()
        // Those values are going to be parsed by the burn-lm crate.
        //
        // This ensures that this CLI is correctly compiled with the proper backend and dtype.
        // The goal is also that all values are parsed using a unified CLI with seamless
        // recompilation if the backend or dtype changes.
        .args([
            Arg::new("backend")
                .short('b')
                .long("backend")
                .help("The backend selected."),
            Arg::new("dtype")
                .short('d')
                .long("dtype")
                .help("The element type used."),
        ])
        .subcommand(commands::backends::create())
        .subcommand(commands::chat::create())
        .subcommand(commands::delete::create())
        .subcommand(commands::download::create())
        .subcommand(commands::models::create())
        .subcommand(commands::new::create())
        .subcommand(commands::run::create())
        .subcommand(commands::server::create())
        .subcommand(commands::shell::create())
        .subcommand(commands::web::create());

    // Execute commands
    let matches = cli.clone().get_matches();

    if let Some(b) = matches.get_one::<String>("backend") {
        assert_eq!(b, backend);
    }

    if let Some(d) = matches.get_one::<String>("dtype") {
        assert_eq!(d, dtype);
    }

    if matches.subcommand_matches("backends").is_some() {
        commands::backends::handle().map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("chat") {
        commands::chat::handle(args, backend, dtype).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("delete") {
        commands::delete::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("download") {
        commands::download::handle(args).map(|_| ())
    } else if matches.subcommand_matches("models").is_some() {
        commands::models::handle(false).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("new") {
        commands::new::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("run") {
        commands::run::handle(args).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("server") {
        commands::server::handle(args, backend, dtype).map(|_| ())
    } else if matches.subcommand_matches("shell").is_some() {
        commands::shell::handle(backend, dtype).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("web") {
        commands::web::handle(args, backend, dtype).map(|_| ())
    } else {
        // default action is to start a shell
        commands::shell::handle(backend, dtype).map(|_| ())
    }
}

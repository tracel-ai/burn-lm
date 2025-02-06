use clap::ValueEnum;
use yansi::Paint;

use crate::{backends::BackendValues, commands};

const BURNLM_DEFAULT_BACKEND_ENVVAR: &str = "BURNLM_DEFAULT_BACKEND";

pub fn run() -> anyhow::Result<()> {
    println!();
    // Define CLI
    let cli = clap::command!()
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
        commands::shell::handle(Some(args), None).map(|_| ())
    } else if let Some(args) = matches.subcommand_matches("web") {
        commands::web::handle(args).map(|_| ())
    } else {
        // default command is to launch a shell
        let backend = match std::env::var(BURNLM_DEFAULT_BACKEND_ENVVAR) {
            Ok(backend) => BackendValues::from_str(&backend, true).unwrap(),
            Err(_) => {
                let mut table = comfy_table::Table::new();
                table.load_preset(comfy_table::presets::UTF8_FULL)
                    .apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS)
                    .set_content_arrangement(comfy_table::ContentArrangement::Dynamic)
                    .set_width(80)
                    .add_row(
                        vec![comfy_table::Cell::new(format!("💡 Hint: No environment variable '{BURNLM_DEFAULT_BACKEND_ENVVAR}' defined. Using default Burn backend which is 'wgpu'."))]);
                println!("{}\n", table.bright_yellow().bold().italic());
                BackendValues::Wgpu
            }
        };
        commands::shell::handle(None, Some(&backend)).map(|_| ())
    }
}

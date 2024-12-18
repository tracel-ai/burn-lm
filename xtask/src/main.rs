mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
use tracel_xtask::prelude::*;

#[macros::base_commands(Build, Bump, Check, Compile, Fix, Test)]
enum Command {
    /// Start Chat interface using open-webui.
    Chat(commands::chat::ChatCmdArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>()?;
    match args.command {
        Command::Chat(args) => commands::chat::handle_command(args).await?,
        _ => dispatch_base_commands(args)?,
    };
    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );
    Ok(())
}

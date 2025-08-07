mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
use tracel_xtask::prelude::*;

#[macros::base_commands(Build, Bump, Check, Compile, Fix, Test, Publish)]
enum Command {
    /// Run commands to manage the book.
    Book(commands::book::BookArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>(parse_args::<Command>()?)?;

    println!("HER");
    match args.command {
        Command::Book(book_args) => book_args.parse(),
        _ => {
            dispatch_base_commands(args)
        },
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );
    Ok(())
}

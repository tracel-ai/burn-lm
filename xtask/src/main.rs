#[macro_use]
extern crate log;

use std::time::Instant;
use tracel_xtask::prelude::*;

#[macros::base_commands(Build, Bump, Check, Compile, Fix, Test)]
enum Command {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>()?;
    // keep this match with only a catch-all arm in case we need to add a new
    // custom command in Command enum above.
    dispatch_base_commands(args)?;
    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );
    Ok(())
}

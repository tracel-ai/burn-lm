use std::process::Command as StdCommand;

use crate::utils;

const DOCKER_COMPOSE_CONFIG: &str = "./crates/burnlm-cli/config/docker-compose.web.yml";
const DOCKER_COMPOSE_PROJECT: &str = "burn-lm-web";
const MPROC_WEB: &str = "./crates/burnlm-cli/config/mprocs_web.yml";

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("web").about("Run inference in an Open WebUI client");
    let start = clap::Command::new("start").about("Start web client");
    root = root.subcommand(start);
    let stop = clap::Command::new("stop").about("Stop web client");
    root = root.subcommand(stop);
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> anyhow::Result<()> {
    let action = match args.subcommand_name() {
        Some(cmd) => cmd,
        None => {
            create().print_help().unwrap();
            return Ok(());
        }
    };
    match action {
        "start" => start_web(),
        "stop" => stop_web(),
        _ => Err(anyhow::format_err!("Error: command unknown {action}")),
    }
}

pub(crate) fn start_web() -> anyhow::Result<()> {
    println!("Starting containerized services...",);
    up_docker_compose()?;
    println!("Launching web stack...",);
    StdCommand::new("mprocs")
        .args(["--config", MPROC_WEB])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to start web stack: {e}"))?;
    println!("Web stack shutdown!",);
    Ok(())
}

pub(crate) fn stop_web() -> anyhow::Result<()> {
    println!("Stopping containerized services...",);
    down_docker_compose()?;
    Ok(())
}

pub fn up_docker_compose() -> anyhow::Result<()> {
    let args = vec![
        "compose",
        "-f",
        DOCKER_COMPOSE_CONFIG,
        "-p",
        DOCKER_COMPOSE_PROJECT,
        "up",
        "-d",
    ];
    utils::run_process(
        "docker",
        &args,
        None,
        None,
        "Failed to execute 'docker compose' to start the container!",
    )
}

pub fn down_docker_compose() -> anyhow::Result<()> {
    let args = vec![
        "compose",
        "-f",
        DOCKER_COMPOSE_CONFIG,
        "-p",
        DOCKER_COMPOSE_PROJECT,
        "down",
    ];
    utils::run_process(
        "docker",
        &args,
        None,
        None,
        "Failed to execute 'docker compose' to stop the container!",
    )
}

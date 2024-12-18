use std::process::Command as StdCommand;

use tracel_xtask::prelude::*;

const DOCKER_COMPOSE_CONFIG: &str = "xtask/config/docker-compose.chat.yml";
const DOCKER_COMPOSE_PROJECT: &str = "burn-lm-chat";

#[derive(clap::Args)]
pub struct ChatCmdArgs {
    #[command(subcommand)]
    command: ChatSubCommand,
}

#[derive(clap::Subcommand)]
enum ChatSubCommand {
    /// Start chat stack.
    Start,
    /// Ensure that all containerized services are down.
    Stop,
}

pub(crate) async fn handle_command(args: ChatCmdArgs) -> anyhow::Result<()> {
    match args.command {
        ChatSubCommand::Start => start_chat().await,
        ChatSubCommand::Stop => stop_chat().await,
    }
}

pub(crate) async fn start_chat() -> anyhow::Result<()> {
    let mproc_config = "xtask/config/mprocs_chat.yml";
    info!("Starting containerized services...",);
    up_docker_compose()?;
    info!("Launching chat stack...",);
    StdCommand::new("mprocs")
        .args(["--config", &mproc_config])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to start chat: {e}"))?;
    info!("Chat shutdown!",);
    Ok(())
}

pub(crate) async fn stop_chat() -> anyhow::Result<()> {
    info!("Stopping containerized services...",);
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
    run_process(
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
    run_process(
        "docker",
        &args,
        None,
        None,
        "Failed to execute 'docker compose' to stop the container!",
    )
}

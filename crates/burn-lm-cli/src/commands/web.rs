use std::io::Write;
use std::process::Command as StdCommand;

use crate::utils::{self, ensure_cargo_crate_is_installed, find_executable};

const DOCKER_COMPOSE_CONFIG: &str = "./crates/burn-lm-cli/config/docker-compose.web.yml";
const DOCKER_COMPOSE_PROJECT: &str = "burn-lm-web";
const MPROC_WEB_TEMPLATE: &str = "./crates/burn-lm-cli/config/mprocs_web.yml";
const MPROC_WEB_CONFIG: &str = "./tmp/mprocs_web.yml";

pub(crate) fn create() -> clap::Command {
    clap::Command::new("web")
        .about("Run inference in an Open WebUI client")
        .subcommand(clap::Command::new("start").about("Start web client"))
        .subcommand(clap::Command::new("stop").about("Stop web client"))
}

pub(crate) fn handle(
    args: &clap::ArgMatches,
    backend: &str,
    dtype: &str,
) -> super::HandleCommandResult {
    let action = match args.subcommand_name() {
        Some(cmd) => cmd,
        None => {
            create().print_help().unwrap();
            return Ok(None);
        }
    };
    match action {
        "start" => start_web(backend, dtype),
        "stop" => stop_web(),
        _ => Err(anyhow::format_err!("Error: command unknown {action}")),
    }
}

fn install_dependencies() -> super::HandleCommandResult {
    ensure_cargo_crate_is_installed("mprocs", None, None, false)?;
    ensure_cargo_crate_is_installed("cargo-watch", None, None, true)
}

fn start_web(backend: &str, dtype: &str) -> super::HandleCommandResult {
    println!("Starting containerized services...",);
    install_dependencies()?;
    ensure_docker_executable_is_installed();
    up_docker_compose()?;
    // write mprocs file from template
    let template = std::fs::read_to_string(MPROC_WEB_TEMPLATE).unwrap();
    let script = template.replace("{{BACKEND}}", backend);
    let script = script.replace("{{DTYPE}}", dtype);
    std::fs::create_dir_all("tmp").expect("directory should be created");
    let mut file = std::fs::File::create(MPROC_WEB_CONFIG).unwrap();
    file.write_all(script.as_bytes()).unwrap();
    println!("Launching web stack...",);
    StdCommand::new("mprocs")
        .args(["--config", MPROC_WEB_CONFIG])
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to start web stack: {e}"))?;
    println!("Web stack shutdown!",);
    Ok(None)
}

fn stop_web() -> super::HandleCommandResult {
    println!("Stopping containerized services...",);
    down_docker_compose()?;
    Ok(None)
}

pub fn ensure_docker_executable_is_installed() {
    if !find_executable("docker") {
        println!("error: 'docker' executable not found on the system.");
        println!("Make sure you have docker installed (https://docs.docker.com)");
        std::process::exit(1);
    }
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

const BURNLM_BACKEND_ENVVAR: &str = "BURNLM_BACKEND";

fn main() -> anyhow::Result<()> {
    let backend = match std::env::var(BURNLM_BACKEND_ENVVAR) {
        Ok(backend) => backend,
        Err(_) => anyhow::bail!("Environment variable '{BURNLM_BACKEND_ENVVAR}' must be set."),
    };
    burnlm_cli::cli::run(&backend)
}

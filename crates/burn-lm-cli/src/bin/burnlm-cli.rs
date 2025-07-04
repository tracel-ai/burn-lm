fn main() -> anyhow::Result<()> {
    burn_lm_cli::cli::run(burn_lm_inference::NAME, burn_lm_inference::DTYPE_NAME)
}

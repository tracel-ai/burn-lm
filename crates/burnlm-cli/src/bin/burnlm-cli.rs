fn main() -> anyhow::Result<()> {
    burnlm_cli::cli::run(burnlm_inference::NAME, burnlm_inference::DTYPE_NAME)
}

pub(crate) mod backends;
pub(crate) mod chat;
pub(crate) mod download;
pub(crate) mod models;
pub(crate) mod new;
pub(crate) mod run;
pub(crate) mod server;
pub(crate) mod shell;
pub(crate) mod web;

const INNER_BURNLM_CLI_TARGET_DIR: &str = "target/inner";
const INNER_BURNLM_CLI_ENVVAR: &str = "__INNER_BURNLM_CLI";
const BURNLM_SHELL_ENVVAR: &str = "__BURNLM_SHELL";
const ANSI_CODE_DELETE_COMPILING_MESSAGES: &str = "\r\x1b[K\x1b[F\x1b[K\x1b[F";
const RESTART_SHELL_EXIT_CODE: i32 = 8;

/// Meta action used in shell mode.
/// It is returned by the handle function of each command.
pub(crate) enum ShellMetaAction {
    Initialize,
    RefreshParser,
    RestartShell,
}

type HandleCommandResult = anyhow::Result<Option<ShellMetaAction>>;

use yansi::Paint;
/// Rustyline custom line editor helper
/// Principal aim for this is to provide a way to stylize the prompt.
#[derive(
    Default, rustyline::Completer, rustyline::Helper, rustyline::Hinter, rustyline::Validator,
)]
struct BurnLMPromptHelper {
    style: yansi::Style,
}

impl BurnLMPromptHelper {
    pub fn new(style: yansi::Style) -> Self {
        Self { style }
    }
}

impl rustyline::highlight::Highlighter for BurnLMPromptHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> std::borrow::Cow<'b, str> {
        if default {
            std::borrow::Cow::Owned(format!("{}", prompt.paint(self.style)))
        } else {
            std::borrow::Cow::Borrowed(prompt)
        }
    }
}

use std::io::{stdout, Write};
/// Build burnlm for the specified backend and run the burnlm command
/// via its run arguments.
/// It returns the exist status of the run process.
fn build_and_run_burnlm(
    initial_message: &str,
    backend: &str,
    burnlm_run_args: &[impl AsRef<str>],
    extra_env_vars: &[(&str, &str)],
) -> std::process::ExitStatus {
    // Print the initial message.
    println!("{}", initial_message);

    // Start a spinner with a compilation message.
    let comp_msg = format!("Compiling for requested Burn backend {}...", backend);
    let mut spinner = spinners::Spinner::new(
        spinners::Spinners::Bounce,
        comp_msg.bright_black().rapid_blink().to_string().into(),
    );

    // Compute the common build/run parameters.
    let inference_feature = format!("burnlm-inference/{}", backend);
    let target_dir = format!("{}/{backend}", INNER_BURNLM_CLI_TARGET_DIR);

    // Define the base environment variable(s) needed for all commands.
    let mut env_vars = vec![(INNER_BURNLM_CLI_ENVVAR, "1")];
    // Append any extra environment variables provided.
    env_vars.extend_from_slice(extra_env_vars);

    // Common arguments for the build command.
    let build_args = [
        "build",
        "--release",
        "--bin",
        "burnlm",
        "--no-default-features",
        "--features",
        &inference_feature,
        "--target-dir",
        &target_dir,
        "--quiet",
        "--color",
        "always",
    ];

    // Execute the build command.
    let mut build_cmd = std::process::Command::new("cargo");
    for (key, value) in &env_vars {
        build_cmd.env(key, value);
    }
    let build_output = build_cmd
        .args(&build_args)
        .output()
        .expect("cargo build should compile burnlm successfully");

    // Stop the spinner and clear the temporary compiling message.
    spinner.stop();
    print!("{}", ANSI_CODE_DELETE_COMPILING_MESSAGES);
    stdout().flush().unwrap();

    // Print any stderr output from the build.
    let stderr_text = String::from_utf8_lossy(&build_output.stderr);
    if !stderr_text.is_empty() {
        println!("{stderr_text}");
    }
    if !build_output.status.success() {
        std::process::exit(build_output.status.code().unwrap_or(1));
    }

    // Base arguments for the run command.
    let mut run_args = vec![
        "run",
        "--release",
        "--bin",
        "burnlm",
        "--no-default-features",
        "--features",
        &inference_feature,
        "--target-dir",
        &target_dir,
        "--quiet",
        "--color",
        "always",
        "--",
    ];
    // Append any extra run arguments provided by the caller.
    run_args.extend(burnlm_run_args.iter().map(|arg| arg.as_ref()));

    // Execute the run command.
    let mut run_cmd = std::process::Command::new("cargo");
    for (key, value) in &env_vars {
        run_cmd.env(key, value);
    }
    run_cmd.args(&run_args)
        .status()
        .expect("cargo run should execute burnlm successfully")
}

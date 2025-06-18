use spinners::{Spinner, Spinners};
use std::{
    path::PathBuf,
    process::{exit, Command},
};

use yansi::Paint;

const BURNLM_SUPERVISOR_RESTART_EXIT_CODE: i32 = 8;

struct CargoCommand {
    subcommand: String,
    features: String,
    extra_args: Vec<String>,
}

impl CargoCommand {
    /// New build command.
    fn build(backend: &str, dtype: &str) -> Self {
        Self {
            subcommand: "build".to_string(),
            features: Self::features(backend, dtype),
            extra_args: Vec::new(),
        }
    }

    /// New run command.
    fn run(backend: &str, dtype: &str, args: Vec<String>) -> Self {
        Self {
            subcommand: "run".to_string(),
            features: Self::features(backend, dtype),
            extra_args: args,
        }
    }

    fn features(backend: &str, dtype: &str) -> String {
        format!("{backend},{dtype}")
    }

    fn set_features(&mut self, backend: &str, dtype: &str) {
        self.features = Self::features(backend, dtype)
    }

    fn args(&self) -> Vec<&str> {
        let mut args = vec![
            self.subcommand.as_str(),
            "--release",
            "--features",
            self.features.as_str(),
            "--bin",
            "burnlm-cli",
            "--quiet",
            "--color",
            "always",
        ];

        if !self.extra_args.is_empty() {
            args.push("--");
            args.extend(self.extra_args.iter().map(|a| a.as_str()));
        }

        args
    }
}

struct CliConfig {
    backend: String,
    dtype: String,
}

impl Default for CliConfig {
    fn default() -> Self {
        let (backend, dtype) = Self::load();
        Self { backend, dtype }
    }
}

impl CliConfig {
    fn config_path() -> PathBuf {
        let mut path = std::env::var("BURNLM_CONFIG_DIR")
            .map(|dir| PathBuf::from(dir))
            .unwrap_or(std::env::current_dir().expect("should get valid directory"));
        path.push("burnlm.config");
        path
    }

    fn load() -> (String, String) {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let mut lines = content.lines();
                if let (Some(backend), Some(dtype)) = (lines.next(), lines.next()) {
                    return (backend.to_string(), dtype.to_string());
                }
            }
        }
        // TODO: platform specific defaults
        ("ndarray".to_string(), "f32".to_string())
    }

    fn reload(&mut self) {
        let (backend, dtype) = Self::load();
        self.backend = backend;
        self.dtype = dtype;
    }
}

fn main() {
    println!();
    // retrieve backend
    let mut exit_code = BURNLM_SUPERVISOR_RESTART_EXIT_CODE;
    // build and run arguments
    let mut args = std::env::args();

    let mut backend = None;
    let mut dtype = None;

    // We manually parse the settings in the command line that might change the features flags to
    // be activated.
    while let Some(arg) = args.next() {
        if arg.contains("--backend") || arg.contains("-b") {
            backend = Some(args.next().expect("A backend must be set"));
        }
        if arg.contains("--dtype") || arg.contains("-d") {
            dtype = Some(args.next().expect("A dtype must be set"));
        }

        if backend.is_some() && dtype.is_some() {
            break;
        }
    }

    let mut config = CliConfig::default();
    let passed_args = std::env::args().skip(1).collect();

    let mut build_cmd = CargoCommand::build(&config.backend, &config.dtype);
    let mut run_cmd = CargoCommand::run(&config.backend, &config.dtype, passed_args);

    // Rebuild and restart burnlm while its exit code is SUPERVISOR_RESTART_EXIT_CODE
    while exit_code == BURNLM_SUPERVISOR_RESTART_EXIT_CODE {
        config.reload();
        build_cmd.set_features(&config.backend, &config.dtype);
        run_cmd.set_features(&config.backend, &config.dtype);

        let compile_msg = "compiling burnlm CLI, please wait...";
        let mut sp = Spinner::new(Spinners::Bounce, compile_msg.bright_black().to_string());
        // build burnlm cli
        let build_output = Command::new("cargo")
            .args(&build_cmd.args())
            .output()
            .expect("build command should compile burnlm successfully");
        // build step results
        let stderr_text = String::from_utf8_lossy(&build_output.stderr);
        if !stderr_text.is_empty() {
            println!("{stderr_text}");
        }
        if !build_output.status.success() {
            exit(build_output.status.code().unwrap_or(1));
        }
        // stop the spinner
        let completion_msg = format!(
            "{} {}",
            "✓".bright_green().bold(),
            "burnlm CLI ready!".bright_black().bold(),
        );
        sp.stop_with_message(completion_msg);
        // execute burnlm
        let run_status = Command::new("cargo")
            .args(&run_cmd.args())
            .status()
            .expect("burnlm command should execute successfully");
        exit_code = run_status.code().unwrap_or(1);
    }
    exit(exit_code);
}

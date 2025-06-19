use spinners::{Spinner, Spinners};
use std::{
    path::PathBuf,
    process::{exit, Command},
};

use yansi::Paint;

const BURNLM_SUPERVISOR_RESTART_EXIT_CODE: i32 = 8;
const BURNLM_CONFIG_FILE: &str = "burnlm.config";

fn cargo_args<'a>(
    subcommand: &'a str,
    features: &'a str,
    extra_args: &'a [String],
) -> Vec<&'a str> {
    let mut args = vec![
        subcommand,
        "--release",
        "--features",
        features,
        "--bin",
        "burnlm-cli",
        "--quiet",
        "--color",
        "always",
    ];

    if !extra_args.is_empty() {
        args.push("--");
        args.extend(extra_args.iter().map(|s| s.as_str()));
    }

    args
}

struct BurnLmConfig {
    backend: String,
    dtype: String,
}

impl Default for BurnLmConfig {
    fn default() -> Self {
        let (backend, dtype) = Self::load();
        Self { backend, dtype }
    }
}

impl BurnLmConfig {
    fn config_path() -> PathBuf {
        let mut path = std::env::var("BURNLM_CONFIG_DIR")
            .map(|dir| PathBuf::from(dir))
            .unwrap_or(std::env::current_dir().expect("should get valid directory"));
        path.push(BURNLM_CONFIG_FILE);
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

    let mut config = BurnLmConfig::default();
    let passed_args: Vec<String> = std::env::args().skip(1).collect();

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

    // Rebuild and restart burnlm while its exit code is SUPERVISOR_RESTART_EXIT_CODE
    while exit_code == BURNLM_SUPERVISOR_RESTART_EXIT_CODE {
        config.reload();

        // Force feature flags over config
        let feat_backend = backend.clone().unwrap_or(config.backend.clone());
        let feat_dtype = dtype.clone().unwrap_or(config.dtype.clone());
        let features = format!("{},{}", feat_backend, feat_dtype);

        let build_args = cargo_args("build", &features, &[]);
        let run_args = cargo_args("run", &features, &passed_args);

        let compile_msg = "compiling burnlm CLI, please wait...";
        let mut sp = Spinner::new(Spinners::Bounce, compile_msg.bright_black().to_string());
        // build burnlm cli
        let build_output = Command::new("cargo")
            .args(build_args)
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
            .args(run_args)
            .status()
            .expect("burnlm command should execute successfully");
        exit_code = run_status.code().unwrap_or(1);
    }
    exit(exit_code);
}

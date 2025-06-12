use spinners::{Spinner, Spinners};
use std::process::{exit, Command};

use yansi::Paint;

const BURNLM_SUPERVISOR_RESTART_EXIT_CODE: i32 = 8;

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

    let backend = backend.unwrap_or("ndarray".to_string());
    let dtype = dtype.unwrap_or("f32".to_string());

    let features = format!("{backend},{dtype}");

    let common_args = vec![
        "--release",
        "--features",
        features.as_str(),
        "--bin",
        "burnlm-cli",
        "--quiet",
        "--color",
        "always",
    ];
    let mut build_args = vec!["build"];
    build_args.extend(common_args.clone());
    let mut run_args = vec!["run"];

    run_args.extend(common_args);
    run_args.push("--");
    let passed_args: Vec<String> = std::env::args().skip(1).collect();
    run_args.extend(passed_args.iter().map(|s| s.as_str()));

    // Rebuild and restart burnlm while its exit code is SUPERVISOR_RESTART_EXIT_CODE
    while exit_code == BURNLM_SUPERVISOR_RESTART_EXIT_CODE {
        let compile_msg = "compiling burnlm CLI, please wait...";
        let mut sp = Spinner::new(Spinners::Bounce, compile_msg.bright_black().to_string());
        // build burnlm cli
        let build_output = Command::new("cargo")
            .args(&build_args)
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
            .args(&run_args)
            .status()
            .expect("burnlm command should execute successfully");
        exit_code = run_status.code().unwrap_or(1);
    }
    exit(exit_code);
}

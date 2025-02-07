use spinners::{Spinner, Spinners};
use std::{
    io::{stdout, Write},
    process::{exit, Command},
};

use yansi::Paint;

const SUPERVISER_RESTART_EXIT_CODE: i32 = 8;

fn main() {
    let mut exit_code = SUPERVISER_RESTART_EXIT_CODE;
    let build_args = vec![
        "build",
        "--release",
        "--bin",
        "burnlm",
        "--quiet",
        "--color",
        "always",
    ];
    let mut run_args = vec!["run", "--release", "--bin", "burnlm", "--quiet", "--"];
    let passed_args: Vec<String> = std::env::args().skip(1).collect();
    run_args.extend(passed_args.iter().map(|s| s.as_str()));
    // Rebuild and restart burnlm while its exit code is SUPERVISER_RESTART_EXIT_CODE
    while exit_code == SUPERVISER_RESTART_EXIT_CODE {
        let mut sp = Spinner::new(
            Spinners::Bounce,
            "Compiling burnlm CLI, please wait..."
                .bright_black()
                .rapid_blink()
                .to_string()
                .into(),
        );
        // build burnlm cli
        let build_output = Command::new("cargo")
            .args(&build_args)
            .output()
            .expect("build command should compile burnlm successfully");
        // stop the spinner and remove the line
        sp.stop();
        print!("\r\x1b[K");
        stdout().flush().unwrap();
        // build step results
        let stderr_text = String::from_utf8_lossy(&build_output.stderr);
        if !stderr_text.is_empty() {
            println!("{stderr_text}");
        }
        if !build_output.status.success() {
            exit(build_output.status.code().unwrap_or(1));
        }
        // execute burnlm
        let run_status = Command::new("cargo")
            .args(&run_args)
            .status()
            .expect("burnlm command should execute successfully");
        exit_code = run_status.code().unwrap_or(1);
    }
    exit(exit_code);
}

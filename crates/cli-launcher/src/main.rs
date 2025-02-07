use spinners::{Spinner, Spinners};
use yansi::Paint;
use std::{io::{stdout, Write}, process::{exit, Command}};

fn main() {
    let mut sp = Spinner::new(
        Spinners::Bounce,
        "Compiling burnlm CLI, please wait...".bright_black().rapid_blink().to_string().into()
    );
    // build burnlm cli
    let build_output = Command::new("cargo")
        .args(["build", "--release", "--bin", "burnlm", "--quiet", "--color", "always"])
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
    let mut args = vec!["run", "--release", "--bin", "burnlm", "--quiet", "--"];
    let passed_args: Vec<String> = std::env::args().skip(1).collect();
    args.extend(passed_args.iter().map(|s| s.as_str()));
    let run_status = Command::new("cargo")
        .args(&args)
        .status()
        .expect("burnlm command should execute successfully");
    exit(run_status.code().unwrap_or(1));
}

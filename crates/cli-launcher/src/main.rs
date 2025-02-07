use spinners::{Spinner, Spinners};
use yansi::Paint;
use std::{io::{stdout, Write}, process::{exit, Command}};

fn main() {
    let mut sp = Spinner::new(
        Spinners::Bounce,
        "Compiling burnlm CLI, please wait...".bright_black().rapid_blink().to_string().into()
    );
    // build burnlm cli
    let build_status = Command::new("cargo")
        .args(["build", "--release", "--bin", "burnlm", "--quiet"])
        .status()
        .expect("build command should compile burnlm successfully");
    // stop the spinner and remove the line
    sp.stop();
    print!("\r\x1b[K");
    stdout().flush().unwrap();

    // launch burnlm
    if !build_status.success() {
        exit(build_status.code().unwrap_or(1));
    }
    let mut args = vec!["run", "--release", "--bin", "burnlm", "--quiet", "--"];
    let passed_args: Vec<String> = std::env::args().skip(1).collect();
    args.extend(passed_args.iter().map(|s| s.as_str()));
    let run_status = Command::new("cargo")
        .args(&args)
        .status()
        .expect("burnlm command should execute successfully");
    exit(run_status.code().unwrap_or(1));
}

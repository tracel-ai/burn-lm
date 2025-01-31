/// Simple wrapper that print a message and then execute burnlm
fn main() {
    println!("Compiling burnlm CLI, please wait...");
    let mut args = vec!["run", "--release", "--bin", "burnlm", "--quiet", "--"];
    let passed_args: Vec<String> = std::env::args().skip(1).collect();
    args.extend(passed_args.iter().map(|s| s.as_str()));
    let status = std::process::Command::new("cargo")
        .args(&args)
        .status()
        .expect("burnlm command should execute successfully");
    std::process::exit(status.code().unwrap_or(1));
}

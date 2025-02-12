pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("server").about("Run a simple OpenAI API compatible server");
    let run = clap::Command::new("run")
        .about("Run an OpenAI API compatible server")
        .arg(
            clap::Arg::new("port")
                .long("port")
                .value_parser(clap::value_parser!(u16))
                .default_value("3001")
                .required(false)
                .help("The listening port for the server"),
        );
    root = root.subcommand(run);
    root
}

pub(crate) fn handle(args: &clap::ArgMatches, backend: &str) -> super::HandleCommandResult {
    match args.subcommand_name() {
        Some("run") => {
            let run_args = args.subcommand_matches("run").unwrap();
            let port = run_args.get_one::<u16>("port").unwrap();
            let port_string = port.to_string();
            let inference_feature = format!("burnlm-inference/{}", backend);
            let common_args = vec![
                "--release",
                "--package",
                "burnlm-http",
                "--bin",
                "burnlm-http",
                "--quiet",
                "--features",
                &inference_feature,
            ];
            let mut build_args = vec!["build"];
            build_args.extend(common_args.clone());
            let mut run_args = vec!["run"];
            run_args.extend(common_args);
            run_args.extend(vec!["--", "run", "--port", &port_string]);
            let mut spin_msg = super::SpinningMessage::new(
                &format!("compiling {backend} server..."),
                "server ready!",
            );
            // build server
            let build_output = std::process::Command::new("cargo")
                .args(&build_args)
                .output()
                .expect("build command should compile burnlm-http successfully");
            // build step results
            let stderr_text = String::from_utf8_lossy(&build_output.stderr);
            if !stderr_text.is_empty() {
                println!("{stderr_text}");
            }
            if !build_output.status.success() {
                std::process::exit(build_output.status.code().unwrap_or(1));
            }
            // stop the spinner
            spin_msg.end(false);
            // run server
            let run_status = std::process::Command::new("cargo")
                .args(&run_args)
                .status()
                .expect("burnlm-http should execute successfully");
            std::process::exit(run_status.code().unwrap_or(1));
        }
        _ => {
            create().print_help().unwrap();
            Ok(None)
        }
    }
}

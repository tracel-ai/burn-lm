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
            println!("Starting server listening on port {port_string}...");
            println!("Compiling burnlm server with Burn backend {backend}, please wait...");
            let inference_feature = format!("burnlm-inference/{}", backend);
            let args = vec![
                "run",
                "--release",
                "--package",
                "burnlm-http",
                "--bin",
                "server",
                "--quiet",
                "--features",
                &inference_feature,
                "--",
                "run",
                "--port",
                &port_string,
            ];
            let status = std::process::Command::new("cargo")
                .args(&args)
                .status()
                .expect("burnlm command should execute successfully");
            std::process::exit(status.code().unwrap_or(1));
        }
        _ => {
            create().print_help().unwrap();
            Ok(None)
        }
    }
}

use crate::backends::{BackendValues, DEFAULT_BURN_BACKEND};

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
        )
        .arg(
            clap::Arg::new("backend")
                .long("backend")
                .value_parser(clap::value_parser!(BackendValues))
                .default_value(DEFAULT_BURN_BACKEND)
                .required(false)
                .help("The Burn backend used for inference"),
        );
    root = root.subcommand(run);
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> anyhow::Result<()> {
    match args.subcommand_name() {
        Some("run") => {
            let run_args = args.subcommand_matches("run").unwrap();
            run(run_args)
        }
        _ => {
            create().print_help().unwrap();
            Ok(())
        }
    }
}

fn run(args: &clap::ArgMatches) -> anyhow::Result<()> {
    let backend = args.get_one::<BackendValues>("backend").unwrap();
    let port = args.get_one::<u16>("port").unwrap();
    let port_string = port.to_string();
    println!("Starting server listening on port {port_string}...");
    println!("Compiling burnlm server with Burn backend {backend}, please wait...");
    let inference_feature = format!("burnlm-inference/{}", backend.to_string());
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

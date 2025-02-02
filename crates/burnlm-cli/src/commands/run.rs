use burnlm_inference::{message::MessageRole, Message};
use burnlm_registry::Registry;

use crate::backends::{BackendValues, DEFAULT_BURN_BACKEND};

const INNER_BURNLM_CLI: &'static str = "__INNER_BURNLM_CLI";

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("run").about("Run inference on chosen model in the terminal");
    let registry = Registry::new();
    // Create a a subcommand for each registered model with its associated  flags
    let mut installed: Vec<_> = registry
        .get()
        .iter()
        .filter(|(_name, plugin)| plugin.is_downloaded())
        .collect();
    installed.sort_by_key(|(key, ..)| *key);
    for (_name, plugin) in installed {
        let mut subcommand = clap::Command::new(plugin.model_cli_param_name())
            .about(format!("Use {} model", plugin.model_name()));
        subcommand = subcommand
            .args((plugin.create_cli_flags_fn())().get_arguments())
            .arg(
                clap::Arg::new("prompt")
                    .help("The prompt to send to the model.")
                    .required(true)
                    .index(1),
            )
            .arg(
                clap::Arg::new("backend")
                    .long("backend")
                    .value_parser(clap::value_parser!(BackendValues))
                    .default_value(DEFAULT_BURN_BACKEND) // we pass as litteral as enum default does not work here
                    .required(false)
                    .help("The Burn backend for the inference"),
            );
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> anyhow::Result<()> {
    let plugin_name = match args.subcommand_name() {
        Some(cmd) => cmd,
        None => {
            create().print_help().unwrap();
            return Ok(());
        }
    };
    let run_args = args.subcommand_matches(plugin_name).unwrap();
    if std::env::var(INNER_BURNLM_CLI).is_ok() {
        run(plugin_name, run_args)
    } else {
        let backend = run_args.get_one::<BackendValues>("backend").unwrap();
        println!("Running inference...");
        println!("Compiling for requested Burn backend {backend}...");
        let inference_feature = format!("burnlm-inference/{}", backend.to_string());
        let mut args = vec![
            "run",
            "--release",
            "--bin",
            "burnlm",
            "--no-default-features",
            "--features",
            &inference_feature,
            "--quiet",
            "--",
        ];
        let passed_args: Vec<String> = std::env::args().skip(1).collect();
        args.extend(passed_args.iter().map(|s| s.as_str()));
        std::process::Command::new("cargo")
            .env(INNER_BURNLM_CLI, "1")
            .args(&args)
            .status()
            .expect("burnlm command should execute successfully");
        Ok(())
    }
}

fn run(plugin_name: &str, run_args: &clap::ArgMatches) -> anyhow::Result<()> {
    let registry = Registry::new();
    let plugin = registry
        .get()
        .iter()
        .find(|(_, p)| p.model_cli_param_name() == plugin_name.to_lowercase())
        .map(|(_, plugin)| plugin);
    let plugin = plugin.unwrap_or_else(|| panic!("Plugin should be registered: {plugin_name}"));
    plugin.parse_cli_config(run_args);
    let prompt = run_args
        .get_one::<String>("prompt")
        .expect("The prompt argument should be set.");
    let message = Message {
        role: MessageRole::User,
        content: prompt.clone(),
        refusal: None,
    };
    let result = plugin.complete(vec![message]);
    match result {
        Ok(answer) => {
            let bold_orange = "\x1b[1;38;5;214m";
            let reset = "\x1b[0m";
            println!("\n{bold_orange}{answer}{reset}");
            Ok(())
        }
        Err(err) => anyhow::bail!("An error occured: {err}"),
    }
}

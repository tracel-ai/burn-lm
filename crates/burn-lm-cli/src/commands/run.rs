use burn_lm_inference::{message::MessageRole, Message};
use burn_lm_registry::Registry;
use yansi::Paint;

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
        let subcommand = clap::Command::new(plugin.model_cli_param_name())
            .about(format!("Use {} model", plugin.model_name()))
            .args((plugin.create_cli_flags_fn())().get_arguments())
            .arg(
                clap::Arg::new("no-stats")
                    .help("Disable display of statistics at the end of the inference")
                    .long("no-stats")
                    .action(clap::ArgAction::SetTrue)
                    .required(false),
            )
            .arg(
                clap::Arg::new("prompt")
                    .help("The prompt to send to the model")
                    .required(true)
                    .index(1),
            );
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> super::HandleCommandResult {
    let plugin_name = match args.subcommand_name() {
        Some(cmd) => cmd,
        None => {
            create().print_help().unwrap();
            return Ok(None);
        }
    };
    let run_args = args.subcommand_matches(plugin_name).unwrap();
    run(plugin_name, run_args)
}

fn run(plugin_name: &str, run_args: &clap::ArgMatches) -> super::HandleCommandResult {
    let registry = Registry::new();
    let plugin = registry
        .get()
        .iter()
        .find(|(_, p)| p.model_cli_param_name() == plugin_name.to_lowercase())
        .map(|(_, plugin)| plugin);
    let plugin = plugin.unwrap_or_else(|| panic!("Plugin should be registered: {plugin_name}"));
    plugin.parse_cli_config(run_args);

    // load the model
    let mut spin_msg = super::SpinningMessage::new(
        &format!("loading model '{}'...", plugin.model_name()),
        "model loaded!",
    );
    plugin.load()?;
    spin_msg.end(false);

    // generation
    let prompt = run_args
        .get_one::<String>("prompt")
        .expect("The prompt argument should be set.");
    let message = Message {
        role: MessageRole::User,
        content: prompt.clone(),
        refusal: None,
    };
    let mut spin_msg = super::SpinningMessage::new("generating answer...", "answer generated!");
    let result = plugin.run_completion(vec![message]);
    match result {
        Ok(answer) => {
            spin_msg.end(false);
            let fmt_answer = answer.completion.bright_black();
            println!("\n{fmt_answer}");
            if !run_args.get_flag("no-stats") {
                crate::utils::display_stats(&answer);
            }
            Ok(None)
        }
        Err(err) => anyhow::bail!("An error occurred: {err}"),
    }
}

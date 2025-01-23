use burnlm_registry::burnlm_plugin::{message::MessageRole, Message};

pub(crate) fn create() -> clap::Command where {
    let mut root = clap::Command::new("run").about("Run inference on chosen model");
    // Create a a subcommand for each registered model with its associated  flags
    for plugin in burnlm_registry::get_inference_plugins() {
        let mut subcommand = clap::Command::new(plugin.model_name_lc)
            .about(format!("Use {} model", plugin.model_name));
        subcommand = subcommand
            .args((plugin.create_cli_flags_fn)().get_arguments())
            .arg(
                clap::Arg::new("prompt")
                    .help("The prompt to send to the model.")
                    .required(true)
                    .index(1),
            );
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> anyhow::Result<()> {
    let plugin_name = args.subcommand_name().unwrap();
    let plugin_metadata = burnlm_registry::get_inference_plugin(&plugin_name)
        .expect("The plugin should be registered.");
    let config_flags = args
        .subcommand_matches(args.subcommand_name().unwrap())
        .unwrap();
    let config = (plugin_metadata.parse_cli_flags_fn)(config_flags);
    let versions = (plugin_metadata.get_model_versions_fn)();
    println!("Available model versions: {versions:?}");
    let mut plugin = (plugin_metadata.create_plugin_fn)(config);
    println!("Selected model versions: {}", plugin.get_version());
    let prompt = config_flags
        .get_one::<String>("prompt")
        .expect("The prompt argument should be set.");
    println!("Prompt: {prompt}");
    println!("Running inference...");
    plugin.load()?;
    let message = Message {
        role: MessageRole::User,
        content: prompt.clone(),
        refusal: None,
    };
    let prompt = plugin.prompt(vec![message]).unwrap();
    let result = plugin.complete(prompt);
    println!("Result: {result:?}");
    Ok(())
}

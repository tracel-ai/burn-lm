use burnlm_inference::{message::MessageRole, Message};
use burnlm_registry::Registry;

pub(crate) fn create() -> clap::Command where {
    let mut root = clap::Command::new("run").about("Run inference on chosen model");
    let registry = Registry::new();
    // Create a a subcommand for each registered model with its associated  flags
    for (_name, plugin) in registry.get().iter() {
        let mut subcommand = clap::Command::new(plugin.model_cli_param_name())
            .about(format!("Use {} model", plugin.model_name()));
        subcommand = subcommand
            .args((plugin.create_cli_flags_fn())().get_arguments())
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
    let registry = Registry::new();
    let plugin_name = args.subcommand_name().unwrap();
    let plugin = registry
        .get()
        .iter()
        .find(|(_, p)| p.model_cli_param_name() == plugin_name.to_lowercase())
        .map(|(_, plugin)| plugin);
    let plugin = plugin.unwrap_or_else(|| panic!("Plugin should be registered: {plugin_name}"));
    let config_flags = args
        .subcommand_matches(args.subcommand_name().unwrap())
        .unwrap();
    let config = (plugin.parse_cli_flags_fn())(config_flags);
    plugin.set_config(config);
    let prompt = config_flags
        .get_one::<String>("prompt")
        .expect("The prompt argument should be set.");
    println!("Prompt: {prompt}");
    println!("Running inference...");
    let message = Message {
        role: MessageRole::User,
        content: prompt.clone(),
        refusal: None,
    };
    let result = plugin.complete(vec![message]);
    println!("Result: {result:?}");
    Ok(())
}

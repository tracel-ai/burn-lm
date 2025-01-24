use burnlm_inference::{message::MessageRole, Message};
use burnlm_registry::Registry;

pub(crate) fn create() -> clap::Command where {
    let mut root = clap::Command::new("run").about("Run inference on chosen model");
    let mut registry = Registry::default();
    // Create a a subcommand for each registered model with its associated  flags
    for (_name, plugin) in registry.get().iter() {
        let mut subcommand = clap::Command::new(plugin.model_name_lc())
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
    let mut registry = Registry::default();
    let client_name = args.subcommand_name().unwrap();
    let plugin = registry.get().iter().find(|(name, _)| (**name).to_lowercase() == client_name.to_lowercase() ).map(|(_, client)| client);
    let plugin = plugin.expect(&format!("Client should be registered: {client_name}"));
    let versions = (plugin.get_model_versions_fn())();
    println!("Available model versions: {versions:?}");
    let config_flags = args
        .subcommand_matches(args.subcommand_name().unwrap())
        .unwrap();
    let config = (plugin.parse_cli_flags_fn())(config_flags);
    plugin.set_config(config);
    let prompt = config_flags
        .get_one::<String>("prompt")
        .expect("The prompt argument should be set.");
    println!("Selected version: {}", plugin.get_version());
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

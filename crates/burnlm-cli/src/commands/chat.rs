use burnlm_inference::{Message, MessageRole};
use burnlm_registry::Registry;
use rustyline::{history::DefaultHistory, Editor};
use yansi::Paint;

use super::BurnLMPromptHelper;
use crate::{backends::BackendValues, utils};

#[derive(clap::Subcommand)]
pub enum MessageCommand {
    Msg { message: String },
    // slash commands
    Exit,
}

// custom rustyline editor to automatically insert the 'msg' command
// in front of the message and parse slash commands (for instance /exit).
struct ChatEditor<H: rustyline::Helper> {
    editor: Editor<H, DefaultHistory>,
}

impl ChatEditor<BurnLMPromptHelper> {
    fn new() -> Self {
        let mut editor = Editor::<BurnLMPromptHelper, DefaultHistory>::new().unwrap();
        let helper = BurnLMPromptHelper::new(yansi::Color::Yellow.bold());
        editor.set_helper(Some(helper));
        Self { editor }
    }
}

impl cloop::InputReader for ChatEditor<BurnLMPromptHelper> {
    fn read(&mut self, prompt: &str) -> std::io::Result<cloop::InputResult> {
        match self.editor.read(prompt) {
            Ok(cloop::InputResult::Input(s)) => {
                if let (Some(cmd), rest) = utils::parse_command(&s) {
                    Ok(cloop::InputResult::Input(format!("{cmd} {rest}")))
                } else {
                    // consider any freefrom input a message
                    Ok(cloop::InputResult::Input(format!("msg \"{s}\"")))
                }
            }
            other => other,
        }
    }
}

#[derive(Default)]
struct ChatContext {
    messages: Vec<Message>,
}

impl ChatContext {
    pub fn new() -> Self {
        Self::default()
    }
}

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("chat").about("Start a chat session with the choosen model");
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
            .about(format!("Chat with {} model", plugin.model_name()))
            .args((plugin.create_cli_flags_fn())().get_arguments());
        if std::env::var(super::BURNLM_SHELL_ENVVAR).is_err() {
            subcommand = subcommand.arg(
                clap::Arg::new("backend")
                    .long("backend")
                    .value_parser(clap::value_parser!(BackendValues))
                    .required(true)
                    .help("The Burn backend used for chat inference"),
            );
        }
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(
    args: &clap::ArgMatches,
    backend: Option<&BackendValues>,
) -> super::HandleCommandResult {
    let plugin_name = match args.subcommand_name() {
        Some(cmd) => cmd,
        None => {
            create().print_help().unwrap();
            return Ok(None);
        }
    };
    let plugin_args = args.subcommand_matches(plugin_name).unwrap();
    let backend = match backend {
        Some(b) => b,
        None => plugin_args.get_one::<BackendValues>("backend").unwrap(),
    };
    if std::env::var(super::INNER_BURNLM_CLI_ENVVAR).is_ok() {
        // retrieve registered plugin
        let registry = Registry::new();
        let plugin = registry
            .get()
            .iter()
            .find(|(_, p)| p.model_cli_param_name() == plugin_name.to_lowercase())
            .map(|(_, plugin)| plugin);
        let plugin = plugin.unwrap_or_else(|| panic!("Plugin should be registered: {plugin_name}"));
        plugin.parse_cli_config(plugin_args);

        // create chat shell
        let app_name = format!("({backend}) chat|{}", plugin.model_name());
        let delim = "> ";
        let handler = |args: MessageCommand, ctx: &mut ChatContext| -> cloop::ShellResult {
            match args {
                MessageCommand::Msg { message } => {
                    let formatted_msg = Message {
                        role: MessageRole::User,
                        content: message,
                        refusal: None,
                    };
                    ctx.messages.push(formatted_msg);
                    let result = plugin.complete(ctx.messages.clone());
                    match result {
                        Ok(answer) => {
                            let formatted_ans = Message {
                                role: MessageRole::Assistant,
                                content: answer.clone(),
                                refusal: None,
                            };
                            ctx.messages.push(formatted_ans);
                            let fmt_answer = answer.bright_black().bold();
                            println!("{fmt_answer}");
                        }
                        Err(err) => anyhow::bail!("An error occured: {err}"),
                    }
                    Ok(cloop::ShellAction::Continue)
                }
                MessageCommand::Exit => Ok(cloop::ShellAction::Exit),
            }
        };

        let mut shell = cloop::Shell::new(
            format!("{app_name}{delim}"),
            ChatContext::new(),
            ChatEditor::new(),
            cloop::ClapSubcommandParser::default(),
            handler,
        );

        println!("Chat session started! (press CTRL+D or type /exit to close session)");
        shell.run().unwrap();
        println!("Chat session closed!");
    } else {
        println!("Starting burnlm chat session...");
        println!("Compiling for requested Burn backend {backend}...");
        let inference_feature = format!("burnlm-inference/{backend}");
        let target_dir = format!("{}/{backend}", super::INNER_BURNLM_CLI_TARGET_DIR);
        let mut chat_args = vec![
            "run",
            "--release",
            "--bin",
            "burnlm",
            "--no-default-features",
            "--features",
            &inference_feature,
            "--target-dir",
            &target_dir,
            "--quiet",
            "--",
        ];
        let cli_args: Vec<String> = std::env::args().skip(1).collect();
        chat_args.extend(cli_args.iter().map(|s| s.as_str()));
        std::process::Command::new("cargo")
            .env(super::INNER_BURNLM_CLI_ENVVAR, "1")
            .args(&chat_args)
            .status()
            .expect("burnlm command should execute successfully");
    }

    Ok(None)
}

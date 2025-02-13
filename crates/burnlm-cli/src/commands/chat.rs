use burnlm_inference::{Message, MessageRole};
use burnlm_registry::Registry;
use clap::CommandFactory as _;
use rustyline::{history::DefaultHistory, Editor};
use yansi::Paint;

use super::BurnLMPromptHelper;
use crate::utils;

#[derive(clap::Subcommand)]
pub enum MessageCommand {
    /// Exit chat session
    Exit,
    /// Display slash commands help
    Help,
    /// Message (prompt) for inference
    Msg {
        message: String,
    },
    /// Toggle stats
    Stats,
}

// Dummy wrapper to get CommandFactory implemented
#[derive(clap::Parser)]
#[command(name = "chat",
          about = "Burn LM Chat",
          disable_help_subcommand = true)]
struct MessageCli {
    #[command(subcommand)]
    command: MessageCommand,
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
    stats: bool,
}

impl ChatContext {
    pub fn new() -> Self {
        Self {
            messages: vec![],
            stats: true,
        }
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
        let subcommand = clap::Command::new(plugin.model_cli_param_name())
            .about(format!("Chat with {} model", plugin.model_name()))
            .args((plugin.create_cli_flags_fn())().get_arguments());
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches, backend: &str) -> super::HandleCommandResult {
    let plugin_name = match args.subcommand_name() {
        Some(cmd) => cmd,
        None => {
            create().print_help().unwrap();
            return Ok(None);
        }
    };

    // retrieve registered plugin
    let registry = Registry::new();
    let plugin = registry
        .get()
        .iter()
        .find(|(_, p)| p.model_cli_param_name() == plugin_name.to_lowercase())
        .map(|(_, plugin)| plugin);
    let plugin = plugin.unwrap_or_else(|| panic!("Plugin should be registered: {plugin_name}"));
    let plugin_args = args.subcommand_matches(plugin_name).unwrap();
    plugin.parse_cli_config(plugin_args);

    // load the model
    let mut spin_msg = super::SpinningMessage::new(
        &format!("loading model '{}'...", plugin.model_name()),
        "model loaded!",
    );
    plugin.load()?;
    spin_msg.end(false);

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
                let mut spin_msg =
                    super::SpinningMessage::new("generating answer...", "generation complete!");
                let result = plugin.run_completion(ctx.messages.clone());
                match result {
                    Ok(answer) => {
                        let formatted_ans = Message {
                            role: MessageRole::Assistant,
                            content: answer.completion.to_owned(),
                            refusal: None,
                        };
                        ctx.messages.push(formatted_ans);
                        spin_msg.end(true);
                        let fmt_answer = answer.completion.bright_black().bold();
                        println!("{fmt_answer}");
                        if ctx.stats {
                            crate::utils::display_stats(&answer);
                        }
                    }
                    Err(err) => anyhow::bail!("An error occured: {err}"),
                }
                Ok(cloop::ShellAction::Continue)
            }
            MessageCommand::Help => {
                MessageCli::command()
                    .print_help()
                    .expect("help output should be printed");
                Ok(cloop::ShellAction::Continue)
            },
            MessageCommand::Stats => {
                ctx.stats = !ctx.stats;
                let msg = format!("Stats toggled {}!", if ctx.stats { "on" } else { "off" });
                println!("{}", msg.bright_black().bold());
                Ok(cloop::ShellAction::Continue)
            }
            MessageCommand::Exit => Ok(cloop::ShellAction::Exit),
        }
    };

    println!("");
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
    Ok(None)
}

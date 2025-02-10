use std::{cell::RefCell, process::exit, rc::Rc};

use rustyline::{history::DefaultHistory, Editor};

use super::{BurnLMPromptHelper, ShellMetaAction};

const RESTART_SHELL_EXIT_CODE: i32 = 8;

// custom rustyline editor to stylize the prompt
struct ShellEditor<H: rustyline::Helper> {
    editor: Rc<RefCell<Editor<H, DefaultHistory>>>,
}

impl ShellEditor<BurnLMPromptHelper> {
    fn new(editor: Rc<RefCell<Editor<BurnLMPromptHelper, DefaultHistory>>>) -> Self {
        Self { editor }
    }
}

impl cloop::InputReader for ShellEditor<BurnLMPromptHelper> {
    fn read(&mut self, prompt: &str) -> std::io::Result<cloop::InputResult> {
        self.editor.borrow_mut().read(prompt)
    }
}

pub(crate) fn create() -> clap::Command {
    clap::Command::new("shell").about("Start a burnlm shell session")
}

type ShellContext = ();

fn create_parser() -> clap::Command {
    clap::Command::default()
        .subcommand(super::backends::create())
        .subcommand(super::chat::create())
        .subcommand(super::download::create())
        .subcommand(super::models::create())
        .subcommand(super::new::create())
        .subcommand(super::run::create())
        .subcommand(super::server::create())
        .subcommand(super::web::create())
        .multicall(true)
}

pub(crate) fn handle(backend: &str) -> anyhow::Result<()> {
    // meta action used to control the outer loop
    // we need interior mutability here because the shell handler
    // is bound to the Fn trait
    let meta_action = RefCell::new(Some(ShellMetaAction::Initialize));

    // allow to refresh the parser given the meta_action set by
    // the executed command
    let mut parser = create_parser();

    // define the editor outside the loop to be able to persist
    // the history between parser refresh.
    let editor = Rc::new(RefCell::new(
        Editor::<BurnLMPromptHelper, DefaultHistory>::new().unwrap(),
    ));
    let helper = BurnLMPromptHelper::new(yansi::Color::Green.bold());
    editor.borrow_mut().set_helper(Some(helper));

    println!("\nWelcome to Burn LM shell 🔥 (press CTRL+D to exit)");
    let app_name = format!("({backend}) burnlm");
    let delim = "> ";

    while meta_action.borrow().is_some() {
        match meta_action.borrow().as_ref().unwrap() {
            ShellMetaAction::Initialize => (),
            ShellMetaAction::RefreshParser => {
                println!("Refreshing shell...");
                parser = create_parser()
            }
            ShellMetaAction::RestartShell => {
                println!("Restarting shell...");
                exit(RESTART_SHELL_EXIT_CODE);
            }
        }
        *meta_action.borrow_mut() = None;

        let handler = |args: clap::ArgMatches, _: &mut ShellContext| -> cloop::ShellResult {
            *meta_action.borrow_mut() = if args.subcommand_matches("backends").is_some() {
                super::backends::handle()?
            } else if let Some(args) = args.subcommand_matches("chat") {
                super::chat::handle(args, backend)?
            } else if let Some(args) = args.subcommand_matches("download") {
                super::download::handle(args)?
            } else if args.subcommand_matches("models").is_some() {
                super::models::handle(true)?
            } else if let Some(args) = args.subcommand_matches("new") {
                super::new::handle(args)?
            } else if let Some(args) = args.subcommand_matches("run") {
                super::run::handle(args)?
            } else if let Some(args) = args.subcommand_matches("server") {
                super::server::handle(args, backend)?
            } else if let Some(args) = args.subcommand_matches("web") {
                super::web::handle(args, backend)?
            } else {
                None
            };
            if meta_action.borrow().is_some() {
                Ok(cloop::ShellAction::Exit)
            } else {
                Ok(cloop::ShellAction::Continue)
            }
        };

        let mut shell = cloop::Shell::new(
            format!("{app_name}{delim}"),
            ShellContext::default(),
            ShellEditor::new(editor.clone()),
            parser.clone(),
            handler,
        );

        shell.run().unwrap();
    }
    println!("Bye!");
    Ok(())
}

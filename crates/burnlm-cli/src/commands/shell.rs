use std::{cell::RefCell, io::{stdout, Write}, process::exit, rc::Rc};

use rustyline::{history::DefaultHistory, Editor};
use yansi::Paint;

use super::{BurnLMPromptHelper, ShellMetaAction};
use crate::backends::{BackendValues, DEFAULT_BURN_BACKEND};

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
    clap::Command::new("shell")
        .about("Start a burnlm shell session")
        .arg(
            clap::Arg::new("backend")
                .long("backend")
                .value_parser(clap::value_parser!(BackendValues))
                .default_value(DEFAULT_BURN_BACKEND)
                .required(false)
                .help("The Burn backend used for inference"),
        )
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

pub(crate) fn handle(
    args: Option<&clap::ArgMatches>,
    backend: Option<&BackendValues>,
) -> anyhow::Result<()> {
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

    // Burn backend for this shell session
    let backend = match backend {
        Some(b) => b,
        None => args
            .expect("should have parsed args when no backend function argument has been provided")
            .get_one::<BackendValues>("backend")
            .unwrap(),
    };

    if std::env::var(super::INNER_BURNLM_CLI_ENVVAR).is_ok() {
        println!("Welcome to Burn LM shell 🔥 (press CTRL+D to exit)");
        let app_name = format!("({backend}) burnlm");
        let delim = "> ";

        // toto
        while meta_action.borrow().is_some() {
            match meta_action.borrow().as_ref().unwrap() {
                ShellMetaAction::RefreshParser => {
                    println!("Refreshing shell...");
                    parser = create_parser()
                }
                _ => (),
            }
            *meta_action.borrow_mut() = None;

            let handler = |args: clap::ArgMatches, _: &mut ShellContext| -> cloop::ShellResult {
                *meta_action.borrow_mut() = if args.subcommand_matches("backends").is_some() {
                    super::backends::handle()?
                } else if let Some(args) = args.subcommand_matches("chat") {
                    super::chat::handle(args, Some(backend))?
                } else if let Some(args) = args.subcommand_matches("download") {
                    super::download::handle(args)?
                } else if args.subcommand_matches("models").is_some() {
                    super::models::handle()?
                } else if let Some(args) = args.subcommand_matches("new") {
                    super::new::handle(args)?
                } else if let Some(args) = args.subcommand_matches("run") {
                    super::run::handle(args)?
                } else if let Some(args) = args.subcommand_matches("server") {
                    super::server::handle(args)?
                } else if let Some(args) = args.subcommand_matches("web") {
                    super::web::handle(args)?
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
    } else {
        println!("Running burnlm shell...");
        let comp_msg = format!("Compiling for requested Burn backend {backend}...");
        let mut sp = spinners::Spinner::new(
            spinners::Spinners::Bounce,
            comp_msg.bright_black().rapid_blink().to_string().into()
        );
        let inference_feature = format!("burnlm-inference/{}", backend);
        let target_dir = format!("{}/{backend}", super::INNER_BURNLM_CLI_TARGET_DIR);
        let args = vec![
            "build",
            "--release",
            "--bin",
            "burnlm",
            "--no-default-features",
            "--features",
            &inference_feature,
            "--target-dir",
            &target_dir,
            "--quiet",
            "--color",
            "always",
        ];
        let build_output = std::process::Command::new("cargo")
            .env(super::INNER_BURNLM_CLI_ENVVAR, "1")
            .env(super::BURNLM_SHELL_ENVVAR, "1")
            .args(&args)
            .output()
            .expect("burnlm command should build successfully");
        // Stop the spinner and clear the temporary message
        sp.stop();
        print!("\r\x1b[K");
        stdout().flush().unwrap();
        // Build step results
        let stderr_text = String::from_utf8_lossy(&build_output.stderr);
        if !stderr_text.is_empty() {
            println!("{stderr_text}");
        }
        if !build_output.status.success() {
            exit(build_output.status.code().unwrap_or(1));
        }
        // launch shell
        let backend_str = &backend.to_string();
        let args = vec![
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
            "shell",
            "--backend",
            &backend_str,
        ];
        std::process::Command::new("cargo")
            .env(super::INNER_BURNLM_CLI_ENVVAR, "1")
            .env(super::BURNLM_SHELL_ENVVAR, "1")
            .args(&args)
            .status()
            .expect("burnlm command should execute successfully");
    }

    Ok(())
}

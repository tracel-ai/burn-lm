use crate::backends::{BackendValues, DEFAULT_BURN_BACKEND};

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

// Define our own Rustyline because we need all commands to start with 'burnlm'
// so that the passed cli can parse the line just fine.
pub struct ShellEditor {
    editor: rustyline::DefaultEditor,
}

impl ShellEditor {
    fn new() -> Self {
        Self {
            editor: rustyline::DefaultEditor::new().unwrap(),
        }
    }
}

impl cloop::InputReader for ShellEditor {
    fn read(&mut self, prompt: &str) -> std::io::Result<cloop::InputResult> {
        match self.editor.read(prompt) {
            Ok(cloop::InputResult::Input(s)) => {
                Ok(cloop::InputResult::Input(format!("burnlm {s}")))
            }
            other => other,
        }
    }
}

pub(crate) fn handle(cli: clap::Command, args: &clap::ArgMatches) -> anyhow::Result<()> {
    let backend = args.get_one::<BackendValues>("backend").unwrap();
    if std::env::var(super::INNER_BURNLM_CLI).is_ok() {
        let app_name = format!("({backend}) burnlm");
        let delim = "> ";

        let handler = |args: clap::ArgMatches, _: &mut ()| -> cloop::ShellResult {
            if args.subcommand_matches("backends").is_some() {
                super::backends::handle()?;
                Ok(cloop::ShellAction::Continue)
            } else if let Some(args) = args.subcommand_matches("chat") {
                super::chat::handle(args, Some(backend))?;
                Ok(cloop::ShellAction::Continue)
            } else if let Some(args) = args.subcommand_matches("download") {
                super::download::handle(args)?;
                Ok(cloop::ShellAction::Continue)
            } else if args.subcommand_matches("models").is_some() {
                super::models::handle()?;
                Ok(cloop::ShellAction::Continue)
            } else if let Some(args) = args.subcommand_matches("new") {
                super::new::handle(args)?;
                Ok(cloop::ShellAction::Continue)
            } else if let Some(args) = args.subcommand_matches("run") {
                super::run::handle(args)?;
                Ok(cloop::ShellAction::Continue)
            } else if let Some(args) = args.subcommand_matches("server") {
                super::server::handle(args)?;
                Ok(cloop::ShellAction::Continue)
            } else if let Some(args) = args.subcommand_matches("web") {
                super::web::handle(args)?;
                Ok(cloop::ShellAction::Continue)
            } else {
                Ok(cloop::ShellAction::Continue)
            }
        };

        let bold_green = "\x1b[1;32m";
        let reset = "\x1b[0m";
        let mut shell = cloop::Shell::new(
            format!("{bold_green}{app_name}{delim}{reset}"),
            (),
            ShellEditor::new(),
            cli,
            handler,
        );

        shell.run().unwrap();
    } else {
        println!("Running burnlm shell...");
        println!("Compiling for requested Burn backend {backend}...");
        let inference_feature = format!("burnlm-inference/{}", backend);
        let backend_str = &backend.to_string();
        let args = vec![
            "run",
            "--release",
            "--bin",
            "burnlm",
            "--no-default-features",
            "--features",
            &inference_feature,
            "--quiet",
            "--",
            "shell",
            "--backend",
            &backend_str,
        ];
        std::process::Command::new("cargo")
            .env(super::INNER_BURNLM_CLI, "1")
            .args(&args)
            .status()
            .expect("burnlm command should execute successfully");
    }

    Ok(())
}

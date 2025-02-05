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

pub(crate) fn handle(cli: &clap::Command, args: &clap::ArgMatches) -> anyhow::Result<()> {
    let backend = args.get_one::<BackendValues>("backend").unwrap();
    if std::env::var(super::INNER_BURNLM_CLI).is_ok() {
        println!("Welcome to Burn LM shell! (press CTRL+D to exit)");
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
            rustyline::DefaultEditor::new().unwrap(),
            cli.clone().multicall(true),
            handler,
        );

        shell.run().unwrap();
        println!("Bye!");
    } else {
        println!("Running burnlm shell...");
        println!("Compiling for requested Burn backend {backend}...");
        let inference_feature = format!("burnlm-inference/{}", backend);
        let backend_str = &backend.to_string();
        let target_dir = format!("{}/shell/{backend}", super::INNER_BURNLM_CLI_TARGET_DIR);
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
            .env(super::INNER_BURNLM_CLI, "1")
            .env(super::BURNLM_SHELL, "1")
            .args(&args)
            .status()
            .expect("burnlm command should execute successfully");
    }

    Ok(())
}

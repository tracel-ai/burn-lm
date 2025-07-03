use strum::IntoEnumIterator;

use crate::backends::BackendValues;

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("backend").about("Change the Burn backend");

    // Create a a subcommand for each backend entry
    for backend in BackendValues::iter() {
        let backend = backend.to_string();
        let subcommand = clap::Command::new(&backend)
            .about(format!("{backend} backend"))
            .arg(
                clap::Arg::new("dtype")
                    .help("The data type to use with the backend")
                    .index(1)
                    .required(false),
            );
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> super::HandleCommandResult {
    match args.subcommand() {
        Some((cmd, sub_args)) if is_valid_backend(cmd) => {
            let dtype = sub_args
                .get_one::<String>("dtype")
                .filter(|x| super::dtype::is_valid_dtype(x))
                .cloned();
            Ok(Some(super::ShellMetaAction::ChangeBackend(
                cmd.to_string(),
                dtype,
            )))
        }
        _ => {
            create().print_help().unwrap();
            Ok(None)
        }
    }
}

pub fn is_valid_backend(input: &str) -> bool {
    BackendValues::iter().any(|variant| variant.to_string() == input)
}

use strum::IntoEnumIterator;

use crate::backends::BackendValues;

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("backend").about("Change the Burn backend");

    // Create a a subcommand for each backend entry
    for backend in BackendValues::iter() {
        let backend = backend.to_string();
        let subcommand = clap::Command::new(&backend).about(format!("{backend} backend"));
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> super::HandleCommandResult {
    match args.subcommand_name() {
        Some(cmd) if is_valid_backend(cmd) => {
            Ok(Some(super::ShellMetaAction::ChangeBackend(cmd.to_string())))
        }
        _ => {
            create().print_help().unwrap();
            return Ok(None);
        }
    }
}

pub fn is_valid_backend(input: &str) -> bool {
    BackendValues::iter().any(|variant| variant.to_string() == input)
}

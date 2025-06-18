const VALID_DTYPES: [&str; 3] = ["f32", "f16", "bf16"];

pub(crate) fn create() -> clap::Command {
    let mut root = clap::Command::new("dtype").about("Change the Burn backend data type");

    // Create a a subcommand for each dtype entry
    for dtype in VALID_DTYPES.iter() {
        let subcommand = clap::Command::new(dtype).about(format!("{dtype} data type"));
        root = root.subcommand(subcommand);
    }
    root
}

pub(crate) fn handle(args: &clap::ArgMatches) -> super::HandleCommandResult {
    match args.subcommand_name() {
        Some(cmd) if is_valid_dtype(cmd) => {
            Ok(Some(super::ShellMetaAction::ChangeDtype(cmd.to_string())))
        }
        _ => {
            create().print_help().unwrap();
            return Ok(None);
        }
    }
}

// TODO: check that dtype is valid for current backend (otherwise program will panic)
pub fn is_valid_dtype(input: &str) -> bool {
    VALID_DTYPES.iter().any(|dtype| *dtype == input)
}

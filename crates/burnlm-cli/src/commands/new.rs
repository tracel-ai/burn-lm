use crate::utils::{self, copy_directory};
use std::io::Write;

const CRATE_DIR: &str = "./crates";
const TEMPLATE_DIR_NAME: &str = "burnlm-inference-template";
const REGISTRY_DIR_NAME: &str = "burnlm-registry";

pub(crate) fn create() -> clap::Command {
    clap::Command::new("new")
        .about("Create a new inference server crate")
        .arg(
            clap::Arg::new("name")
                .help("The name of the crate without the prefx 'burnlm-inference-'")
                .required(true),
        )
}

pub(crate) fn handle(args: &clap::ArgMatches) -> super::HandleCommandResult {
    // sanitize the desired crate name
    let crate_name = args
        .get_one::<String>("name")
        .expect("The name argument should be set.");
    let crate_name = utils::sanitize_crate_name(crate_name);
    let crate_fullname = format!("burnlm-inference-{crate_name}");

    println!("Check if crate directory already exist...");
    let crate_path = std::path::Path::new(CRATE_DIR).join(&crate_fullname);
    if std::fs::metadata(&crate_path)
        .map(|m| m.is_dir())
        .unwrap_or(false)
    {
        anyhow::bail!("Crate with name {crate_fullname} already exists.");
    }

    // copy template contents to new crate directory
    println!("Copy infererence server template...");
    let src = std::path::Path::new(CRATE_DIR).join(TEMPLATE_DIR_NAME);
    copy_directory(&src, &crate_path)?;

    println!("Update new crate files...");
    // Cargo.toml
    let mut replacements = std::collections::HashMap::new();
    replacements.insert(
        "burnlm-inference-template".to_string(),
        crate_fullname.clone(),
    );
    utils::replace_in_file(crate_path.join("Cargo.toml"), &replacements);
    // lib.rs
    let crate_namespace = crate_fullname.replace("-", "_");
    let ty_prefix = utils::remove_and_capitalize_dashes(&crate_name);
    let mut replacements = std::collections::HashMap::new();
    replacements.insert(
        "burnlm_inference_template".to_string(),
        crate_namespace.clone(),
    );
    replacements.insert("Parrot".to_string(), ty_prefix.clone());
    utils::replace_in_file(crate_path.join("src").join("lib.rs"), &replacements);

    println!("Register new server in registry...");
    // update the lib.rs file of the registry crate
    let registry_lib = std::path::Path::new(CRATE_DIR)
        .join(REGISTRY_DIR_NAME)
        .join("src")
        .join("lib.rs");
    let server_type = format!("{ty_prefix}Server<InferenceBackend>");
    add_inference_server_registration(registry_lib, &crate_namespace, &server_type)?;
    // append the server crate to the registry Cargo.toml
    let registry_cargo_path = std::path::Path::new(CRATE_DIR)
        .join(REGISTRY_DIR_NAME)
        .join("Cargo.toml");
    let dependency = format!("{crate_fullname} = {{ path = \"../{crate_fullname}\" }}");
    let mut registry_cargo_file = std::fs::OpenOptions::new()
        .append(true)
        .open(registry_cargo_path)?;
    writeln!(registry_cargo_file, "{dependency}")?;

    Ok(Some(super::ShellMetaAction::RefreshParser))
}

/// Inserts a new server(...) entry into the #[inference_server_registry(...)]
/// attribute in `lib.rs` of the `burnlm-registyr`.
/// Returns an error if the exact same server entry (crate_namespace + server_type)
/// is already present.
pub fn add_inference_server_registration<P: AsRef<std::path::Path>>(
    path_to_lib_rs: P,
    crate_namespace: &str,
    server_type: &str,
) -> anyhow::Result<()> {
    let file_path = path_to_lib_rs.as_ref();
    let original = std::fs::read_to_string(file_path).unwrap_or_else(|_| {
        panic!(
            "should be able to read file '{}'",
            file_path.to_str().unwrap()
        )
    });
    let new_server_line = format!(
        "    server(\n        crate_namespace = \"{crate_namespace}\",\n        server_type = \"{server_type}\",\n    ),"
    );
    // Check if it is already in the file
    let already_present = original.contains(&new_server_line);
    if already_present {
        return Err(anyhow::format_err!(
            "Server entry already exists: {crate_namespace}, {server_type}"
        ));
    }
    // Find the inference_server_registry attribute start
    let start_marker = "#[inference_server_registry(";
    let start_index = original.find(start_marker).ok_or_else(|| {
        anyhow::format_err!("registry file should contain the inference_server_registry attribute")
    })?;
    // Find the end of that attribute
    let end_marker = ")]";
    let end_index = match original[start_index..].find(end_marker) {
        Some(rel_idx) => start_index + rel_idx,
        None => {
            return Err(anyhow::format_err!(
                "Could not find the closing )] for the inference_server_registry attribute."
            ));
        }
    };
    // Insert new server(...) entry
    let before = &original[..end_index];
    let after = &original[end_index..];
    let updated = format!("{before}{new_server_line}\n{after}");
    // Update the file
    std::fs::write(file_path, updated)
        .map_err(|err| anyhow::format_err!("Failed to write file: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Dummy registry lib.rs
    const BASE_LIB_RS: &str = r#"
use burnlm_inference::*;
use burnlm_macros::inference_server_registry;
use std::{collections::HashMap, sync::Arc};

pub type Channel<B> = MutexChannel<B>;
pub type DynClients = HashMap<&'static str, Box<dyn InferencePlugin>>;

// Register model crates
#[inference_server_registry(
    server(
        crate_namespace = "burnlm_inference_llama3",
        server_type = "LlamaV3Params8BInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_llama3",
        server_type = "LlamaV31Params8BInstructServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_template",
        server_type = "ParrotServer<InferenceBackend>",
    ),
    server(
        crate_namespace = "burnlm_inference_tinyllama",
        server_type = "TinyLlamaServer<InferenceBackend>",
    ),
)]
#[derive(Debug)]
pub struct Registry {
    clients: Arc<DynClients>,
}
"#;
    // A minimal variant that is missing the attribute altogether
    const NO_MACRO_LIB_RS: &str = r#"
use burnlm_inference::*;
use burnlm_macros::inference_server_registry;
"#;

    #[rstest]
    #[case::success_new_entry(
        BASE_LIB_RS,
        "burnlm_inference_extra",
        "ExtraServer<InferenceBackend>",
        true
    )]
    #[case::duplicate_entry(
        BASE_LIB_RS,
        "burnlm_inference_llama3",
        "LlamaV3Params8BInstructServer<InferenceBackend>",
        false
    )]
    #[case::missing_macro(NO_MACRO_LIB_RS, "anything", "anything", false)]
    fn test_add_inference_server_registration(
        #[case] initial_lib_content: &str,
        #[case] crate_namespace: &str,
        #[case] server_type: &str,
        #[case] expect_success: bool,
    ) {
        let mut tmp = NamedTempFile::new().expect("should create temp file");
        write!(tmp, "{}", initial_lib_content).expect("should write initial content");
        tmp.flush().expect("should flush");

        let path = tmp.path();
        let result = add_inference_server_registration(path, crate_namespace, server_type);
        if expect_success {
            assert!(result.is_ok(), "Expected successful insertion");
            let final_contents =
                std::fs::read_to_string(path).expect("should read updated lib content");
            let inserted_text = format!(
                "server(\n        crate_namespace = \"{crate_namespace}\",\n        server_type = \"{server_type}\",\n    ),"
            );
            assert!(
                final_contents.contains(&inserted_text),
                "Updated file should contain the new server(...) entry."
            );
        } else {
            assert!(result.is_err(), "Expected an error to occur");
        }
    }
}

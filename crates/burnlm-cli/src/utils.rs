use anyhow::Context;
use burnlm_inference::Completion;
use yansi::Paint;
use which::which;

use crate::commands::HandleCommandResult;

/// Sanitizes a given crate name by replacing invalid characters, merging consecutive
/// hyphens, and ensuring it adheres to common crate naming conventions.
pub(crate) fn sanitize_crate_name(input: &str) -> String {
    // Replace any disallowed character with '-'
    let replaced: String = input
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    // Cleanup consecutinve '-'
    let merged = replaced
        .split('-')
        .filter(|chunk| !chunk.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    // Trim any special characters
    let trimmed = merged.trim_matches(|c: char| !c.is_alphanumeric());
    trimmed.to_string().to_lowercase()
}

pub(crate) fn remove_and_capitalize_dashes(input: &str) -> String {
    input
        .split('-')
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                // `s.chars()` is an iterator. We take the first character, uppercase it,
                // then append the rest of the substring.
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

/// Recursively copies a directory and its contents from a source path to a destination path.
pub(crate) fn copy_directory(src: &std::path::Path, dst: &std::path::Path) -> anyhow::Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)
            .with_context(|| format!("Failed to create directory: {}", dst.display()))?;
    }
    for entry in std::fs::read_dir(src)
        .with_context(|| format!("Failed to read directory: {}", src.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_directory(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            std::fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "Failed to copy file from {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        } else {
            return Err(anyhow::anyhow!(
                "Unsupported file type at: {}",
                src_path.display()
            ));
        }
    }
    Ok(())
}

/// In-place replace of all occurrences of each key in `replacements`
/// with its corresponding value.
pub(crate) fn replace_in_file<P: AsRef<std::path::Path>>(
    path: P,
    replacements: &std::collections::HashMap<String, String>,
) {
    let path_ref = path.as_ref();
    let content = std::fs::read_to_string(path_ref)
        .expect("should read the entire file content successfully");
    let mut updated_content = content;
    for (target, replacement) in replacements {
        updated_content = updated_content.replace(target, replacement);
    }
    // Write the modified content back to the file
    std::fs::write(path_ref, updated_content)
        .expect("should write the updated content successfully");
}

/// Run a process
pub(crate) fn run_process(
    name: &str,
    args: &[&str],
    envs: Option<std::collections::HashMap<&str, &str>>,
    path: Option<&std::path::Path>,
    error_msg: &str,
) -> anyhow::Result<()> {
    let mut command = std::process::Command::new(name);
    if let Some(path) = path {
        command.current_dir(path);
    }
    if let Some(envs) = envs {
        command.envs(&envs);
    }
    let status = command.args(args).status().map_err(|e| {
        anyhow::anyhow!(
            "Failed to execute {} {}: {}",
            name,
            args.first().unwrap(),
            e
        )
    })?;
    if !status.success() {
        return Err(anyhow::anyhow!("{}", error_msg));
    }
    anyhow::Ok(())
}

/// Display stats from a completion
pub fn display_stats(completion: &Completion) {
    let stats = completion.stats.display_stats().to_string();
    let fmt_stats = stats.italic();
    println!("{fmt_stats}");
}

/// Ensure that a cargo crate is installed
pub fn ensure_cargo_crate_is_installed(
    crate_name: &str,
    features: Option<&str>,
    version: Option<&str>,
    locked: bool,
) -> HandleCommandResult {
    if !is_cargo_crate_installed(crate_name) {
        println!("Installing cargo crate '{}'", crate_name);
        let mut args = vec!["install", crate_name];
        if locked {
            args.push("--locked");
        }
        if let Some(features) = features {
            if !features.is_empty() {
                args.extend(vec!["--features", features]);
            }
        }
        if let Some(version) = version {
            args.extend(vec!["--version", version]);
        }
        run_process(
            "cargo",
            &args,
            None,
            None,
            &format!("crate '{}' should be installed", crate_name),
        )?;
    }
    Ok(None)
}

/// Returns true if the passed cargo crate is installed locally
pub fn is_cargo_crate_installed(crate_name: &str) -> bool {
    let output = std::process::Command::new("cargo")
        .arg("install")
        .arg("--list")
        .output()
        .expect("Should get the list of installed cargo commands");
    let output_str = String::from_utf8_lossy(&output.stdout);
    output_str.lines().any(|line| line.contains(crate_name))
}

pub fn find_executable(name: &str) -> bool {
    which(name).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[rstest]
    #[case::empty("", "")]
    #[case::basic_already_valid("abc", "abc")]
    #[case::double_dash_in_middle("abc--def", "abc-def")]
    #[case::leading_and_trailing_dashes("--abc--", "abc")]
    #[case::special_chars("hello world!", "hello-world")]
    #[case::all_special_chars("!@#$%^", "")]
    #[case::leading_trailing_underscores("___rust___", "rust")]
    #[case::underscores_in_middle("crate_name", "crate_name")]
    #[case::weird_chars("some?weird?chars", "some-weird-chars")]
    fn to_valid_crate_name_cases(#[case] input: &str, #[case] expected: &str) {
        let actual = sanitize_crate_name(input);
        assert_eq!(
            actual, expected,
            "Should transform `{}` into `{}`, but got `{}`",
            input, expected, actual
        );
    }

    #[test]
    fn test_replace_in_file() {
        let mut tmp_file = NamedTempFile::new().expect("should create a temporary file");
        writeln!(tmp_file, "Hello, BURN!").expect("should write to the temp file");
        let mut replacements = std::collections::HashMap::new();
        replacements.insert("BURN".to_string(), "Ember".to_string());
        replacements.insert("Hello".to_string(), "Hi".to_string());

        replace_in_file(tmp_file.path(), &replacements);
        let updated_content = std::fs::read_to_string(tmp_file.path())
            .expect("should read updated temp file content");
        assert_eq!(updated_content.trim(), "Hi, Ember!");
    }

    #[test]
    fn test_copy_empty_directory() {
        let temp_src = TempDir::new().expect("should create temp src directory");
        let temp_dst = TempDir::new().expect("should create temp dst directory");
        let src_dir = temp_src.path().join("empty_src");
        std::fs::create_dir_all(&src_dir).expect("should create empty source directory");
        let dst_dir = temp_dst.path().join("copied_empty_src");

        copy_directory(&src_dir, &dst_dir).unwrap();
        assert!(dst_dir.exists(), "copied directory should exist");
        assert!(dst_dir.is_dir(), "copied path should be a directory");
    }

    #[test]
    fn test_copy_directory_with_files() {
        let temp_src = TempDir::new().expect("should create temp src directory");
        let temp_dst = TempDir::new().expect("should create temp dst directory");
        let src_dir = temp_src.path().join("src_with_files");
        std::fs::create_dir_all(&src_dir).expect("should create source directory");
        let file_paths = ["file_a.txt", "file_b.txt"];
        for file_name in &file_paths {
            let file_path = src_dir.join(file_name);
            let mut file = std::fs::File::create(&file_path).expect("should create test file");
            writeln!(file, "Hello from {}", file_name).expect("should write to file");
        }

        let dst_dir = temp_dst.path().join("copied_src_with_files");
        copy_directory(&src_dir, &dst_dir).unwrap();
        for file_name in &file_paths {
            let copied_file = dst_dir.join(file_name);
            assert!(copied_file.exists(), "copied file should exist");
            assert!(copied_file.is_file(), "copied file path should be a file");
        }
    }

    #[test]
    fn test_copy_directory_with_subdirectories() {
        let temp_src = TempDir::new().expect("should create temp src directory");
        let temp_dst = TempDir::new().expect("should create temp dst directory");
        let src_dir = temp_src.path().join("nested_src");
        let nested_subdir = src_dir.join("subdir");
        std::fs::create_dir_all(&nested_subdir).expect("should create nested source directory");
        let nested_file_path = nested_subdir.join("nested_file.txt");
        let mut file =
            std::fs::File::create(&nested_file_path).expect("should create nested test file");
        writeln!(file, "Hello from nested file").expect("should write to file");

        let dst_dir = temp_dst.path().join("copied_nested_src");
        copy_directory(&src_dir, &dst_dir).unwrap();
        let copied_subdir = dst_dir.join("subdir");
        assert!(copied_subdir.exists(), "copied subdirectory should exist");
        assert!(
            copied_subdir.is_dir(),
            "copied subdirectory should be a directory"
        );
        let copied_nested_file = copied_subdir.join("nested_file.txt");
        assert!(
            copied_nested_file.exists(),
            "copied nested file should exist"
        );
        assert!(
            copied_nested_file.is_file(),
            "copied nested file should be a file"
        );
    }

    #[rstest]
    #[case::empty("", "")]
    #[case::no_dash("hello", "Hello")]
    #[case::single_dash("hello-world", "HelloWorld")]
    #[case::multiple_dashes("multiple-dashes-in-a-row", "MultipleDashesInARow")]
    #[case::leading_dash("-hello", "Hello")]
    #[case::consecutive_dashes("hello--world", "HelloWorld")]
    fn test_remove_and_capitalize_dashes(#[case] input: &str, #[case] expected: &str) {
        let result = remove_and_capitalize_dashes(input);
        assert_eq!(result, expected);
    }
}

/// Parses a command from an input string.
///
/// If the input starts with a '/', it splits the string into:
///  - the command name (without the leading '/') normalized to lowercase,
///  - the remainder after the command (with the separating space removed).
pub fn parse_command(input: &str) -> (Option<String>, &str) {
    input
        .strip_prefix('/')
        .map(|s| {
            let mut parts = s.splitn(2, ' ');
            let command = parts
                .next()
                .filter(|cmd| !cmd.is_empty())
                .map(str::to_lowercase);
            let args = parts.next().unwrap_or("").trim();
            (command, args)
        })
        .unwrap_or((None, ""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::non_command("hello world", (None, ""))]
    #[case::only_slash("/", (None, ""))]
    #[case::command_only("/command", (Some("command".to_string()), ""))]
    #[case::command_with_args("/command argument", (Some("command".to_string()), "argument"))]
    #[case::command_with_multiple_words("/cmd arg1 arg2", (Some("cmd".to_string()), "arg1 arg2"))]
    #[case::slash_followed_by_space("/ greet", (None, "greet"))]
    #[case::command_with_trailing_space("/cmd ", (Some("cmd".to_string()), ""))]
    #[case::command_with_multiple_spaces("/cmd   args", (Some("cmd".to_string()), "args"))]
    #[case::command_with_multiple_spaces_2("/cmd   args  args", (Some("cmd".to_string()), "args  args"))]
    #[case::uppercase_command("/CoMmAnD someArg", (Some("command".to_string()), "someArg"))]
    fn test_parse_command(#[case] input: &str, #[case] expected: (Option<String>, &str)) {
        let result = parse_command(input);
        assert_eq!(result, expected);
    }
}

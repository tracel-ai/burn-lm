pub(crate) mod backends;
pub(crate) mod chat;
pub(crate) mod download;
pub(crate) mod models;
pub(crate) mod new;
pub(crate) mod run;
pub(crate) mod server;
pub(crate) mod shell;
pub(crate) mod web;

const INNER_BURNLM_CLI_TARGET_DIR: &str = "target/inner";
const INNER_BURNLM_CLI_ENVVAR: &str = "__INNER_BURNLM_CLI";
const BURNLM_SHELL_ENVVAR: &str = "__BURNLM_SHELL";

/// Meta action used in shell mode.
/// It is returned by the handle function of each command.
pub(crate) enum ShellMetaAction {
    Initialize,
    RefreshParser,
}

type HandleCommandResult = anyhow::Result<Option<ShellMetaAction>>;

use yansi::Paint;
/// Rustyline custom line editor helper
/// Principal aim for this is to provide a way to stylize the prompt.
#[derive(
    Default, rustyline::Completer, rustyline::Helper, rustyline::Hinter, rustyline::Validator,
)]
struct BurnLMPromptHelper {
    style: yansi::Style,
}

impl BurnLMPromptHelper {
    pub fn new(style: yansi::Style) -> Self {
        Self { style }
    }
}

impl rustyline::highlight::Highlighter for BurnLMPromptHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> std::borrow::Cow<'b, str> {
        if default {
            std::borrow::Cow::Owned(format!("{}", prompt.paint(self.style)))
        } else {
            std::borrow::Cow::Borrowed(prompt)
        }
    }
}

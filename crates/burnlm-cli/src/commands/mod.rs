pub(crate) mod backends;
pub(crate) mod chat;
pub(crate) mod delete;
pub(crate) mod download;
pub(crate) mod models;
pub(crate) mod new;
pub(crate) mod run;
pub(crate) mod server;
pub(crate) mod shell;
pub(crate) mod web;

const ANSI_CODE_DELETE_LINE: &str = "\r\x1b[K";

/// Meta action used in shell mode.
/// It is returned by the handle function of each command.
pub(crate) enum ShellMetaAction {
    Initialize,
    RefreshParser,
    RestartShell,
}

pub(crate) type HandleCommandResult = anyhow::Result<Option<ShellMetaAction>>;

use std::io::{stdout, Write};

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

/// A message with a spinner that can display the elpased time once finished.
pub(crate) struct SpinningMessage {
    end_message: String,
    spinner: spinners::Spinner,
    start_time: std::time::Instant,
}

impl SpinningMessage {
    pub fn new(start_msg: &str, end_msg: &str) -> Self {
        let now = std::time::Instant::now();
        let spinner = spinners::Spinner::new(
            spinners::Spinners::Bounce,
            start_msg.bright_black().to_string(),
        );
        Self {
            end_message: end_msg.to_owned(),
            spinner,
            start_time: now,
        }
    }

    /// Stop the spinner and replace the line with the end message.
    /// If delete is true then delete the spinner line altogether.
    pub fn end(&mut self, delete: bool) {
        if delete {
            self.spinner.stop();
            print!("{}", ANSI_CODE_DELETE_LINE);
            stdout().flush().unwrap();
        } else {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let elapsed_msg = format!("({:.3}s)", elapsed);
            let completion_msg = format!(
                "{} {} {}",
                "✓".bright_green().bold(),
                self.end_message.bright_black().bold(),
                elapsed_msg.bright_black().italic(),
            );
            self.spinner.stop_with_message(completion_msg);
        }
    }
}

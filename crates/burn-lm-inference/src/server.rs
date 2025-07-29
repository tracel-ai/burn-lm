use crate::{errors::InferenceResult, message::Message, Stats};
use std::{
    any::Any,
    fmt::Debug,
    io::Write,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
        Arc,
    },
};

/// Marker trait for server configurations.
pub trait InferenceServerConfig:
    clap::FromArgMatches + serde::de::DeserializeOwned + 'static + Debug
{
}

/// Trait to add parsing capability of server config from clap and serde
pub trait ServerConfigParsing {
    /// The configuration type to parse
    type Config: InferenceServerConfig;

    fn parse_cli_config(&mut self, args: &clap::ArgMatches);
    fn parse_json_config(&mut self, json: &str);
}

enum CompletionMessage {
    Text(String),
    Finished(SyncSender<Box<dyn Any + Send>>),
}

#[derive(Clone)]
pub struct Completion {
    sender: SyncSender<CompletionMessage>,
    done: Arc<AtomicBool>,
}

pub trait CompletionCallback: Send + 'static {
    type Result: Send;

    fn on_text(&mut self, text: String);
    fn on_finished(self) -> Self::Result;
}

#[derive(Default)]
pub struct StringCallback {
    pub value: String,
}

#[derive(Default)]
pub struct StdOutCallback {}

impl CompletionCallback for StdOutCallback {
    type Result = ();

    fn on_text(&mut self, text: String) {
        let mut io = std::io::stdout();

        write!(io, "{text}").unwrap();
        io.flush().unwrap();
    }

    fn on_finished(self) -> Self::Result {
        ()
    }
}

impl CompletionCallback for StringCallback {
    type Result = String;

    fn on_text(&mut self, text: String) {
        self.value += &text;
    }

    fn on_finished(self) -> Self::Result {
        self.value
    }
}

pub struct CompletionHandle<C: CompletionCallback> {
    sender: SyncSender<CompletionMessage>,
    _c: PhantomData<C>,
}

impl<C: CompletionCallback> CompletionHandle<C> {
    pub fn finished(&self) -> C::Result {
        let (sender, rec) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(CompletionMessage::Finished(sender))
            .unwrap();

        if let Ok(any) = rec.recv() {
            return *any.downcast().unwrap();
        } else {
            panic!()
        }
    }
}

impl Completion {
    /// Creates a new completion and process it on another thread.
    pub fn start<C: CompletionCallback>(mut callback: C) -> (Self, CompletionHandle<C>) {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<CompletionMessage>(1);

        let handle = CompletionHandle {
            sender: sender.clone(),
            _c: PhantomData,
        };
        let done = Arc::new(AtomicBool::new(false));
        let this = Self {
            sender,
            done: done.clone(),
        };

        std::thread::spawn(move || {
            for msg in receiver {
                match msg {
                    CompletionMessage::Text(text) => callback.on_text(text),
                    CompletionMessage::Finished(c) => {
                        let result = callback.on_finished();
                        let result: Box<dyn Any + Send> = Box::new(result);
                        c.send(result).unwrap();
                        done.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        });

        (this, handle)
    }
    pub fn text(&self, text: String) {
        if !self.done.load(Ordering::Relaxed) {
            self.sender.send(CompletionMessage::Text(text)).unwrap();
        }
    }
}

/// Inference server interface aimed to be implemented to be able to register a
/// model in Burn LM registry.
pub trait InferenceServer: ServerConfigParsing + Clone + Default + Send + Sync + Debug {
    /// Return closure of a function to download the model
    fn downloader(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        None
    }

    /// Return true is the model has been downloaded.
    /// Return false if the model is not downloaded or there is no downloader.
    fn is_downloaded(&mut self) -> bool {
        false
    }

    /// Return closure of a function to delete the model
    fn deleter(&mut self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        None
    }

    /// Load the model.
    fn load(&mut self) -> InferenceResult<Option<Stats>>;

    /// Return true is the model is already loaded.
    fn is_loaded(&mut self) -> bool;

    /// Unload the model.
    fn unload(&mut self) -> InferenceResult<Option<Stats>>;

    /// Run inference to complete messages
    fn run_completion(
        &mut self,
        messages: Vec<Message>,
        completion: Completion,
    ) -> InferenceResult<Stats>;

    /// Clear the model state
    fn clear_state(&mut self) -> InferenceResult<()>;
}

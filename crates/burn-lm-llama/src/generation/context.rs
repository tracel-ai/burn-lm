use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::Sender,
        Arc,
    },
    thread::JoinHandle,
};

use burn_lm_inference::{GeneratedItem, GeneratedItemEmitter};

use crate::tokenizer::Tokenizer;

use super::StreamingDecoder;

/// The streaming side of a generating sequence: a background decoder thread that detokenizes and
/// emits each token as it arrives, plus a count of how many tokens were generated.
///
/// Stop detection happens synchronously in the generic decode core (`step_round`), so only
/// non-stop tokens ever reach this context.
///
/// Not `Clone`: it owns the [`JoinHandle`] of the background decoder thread and is finalized via
/// [`finish`](Self::finish), so cloning it would not make sense.
pub struct GenerationContext {
    num_generated: Arc<AtomicUsize>,
    sender: Sender<u32>,
    decoder_handle: JoinHandle<()>,
}

impl GenerationContext {
    /// Create a new generation context.
    pub fn new<T: Tokenizer + 'static>(emitter: GeneratedItemEmitter, tokenizer: T) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<u32>();
        let num_generated = Arc::new(AtomicUsize::new(0));

        let mut generation = TokenGeneration::new(emitter, tokenizer, num_generated.clone());

        let decoder_handle = std::thread::spawn(move || {
            for token in receiver.iter() {
                generation.process(token);
            }
        });

        Self {
            num_generated,
            sender,
            decoder_handle,
        }
    }

    /// Finish the generation, ensuring every generated token has been decoded and emitted.
    ///
    /// Drops the channel sender so the decoder thread's `receiver.iter()` loop terminates, joins
    /// that thread so all in-flight tokens are emitted before returning, and returns the final
    /// number of generated tokens.
    pub fn finish(self) -> usize {
        let Self {
            sender,
            decoder_handle,
            num_generated,
        } = self;

        // Dropping the sender closes the channel, ending the decoder thread's `receiver.iter()`.
        drop(sender);
        // Join so the final in-flight token is decoded and emitted before we return.
        decoder_handle.join().unwrap();

        num_generated.load(Ordering::Relaxed)
    }

    /// Update the state with a newly generated (non-stop) token, streaming it to the decoder.
    pub fn update(&mut self, token: u32) {
        self.sender.send(token).unwrap();
    }
}

struct TokenGeneration<T: Tokenizer> {
    emitter: GeneratedItemEmitter,
    decoder: StreamingDecoder<T>,
    num_tokens_generated: Arc<AtomicUsize>,
    num_generated: usize,
}

impl<T: Tokenizer> TokenGeneration<T> {
    fn new(
        emitter: GeneratedItemEmitter,
        tokenizer: T,
        num_tokens_generated: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            emitter,
            decoder: StreamingDecoder::new(tokenizer),
            num_tokens_generated,
            num_generated: 0,
        }
    }

    fn process(&mut self, token: u32) {
        self.num_generated += 1;

        if let Some(text) = self.decoder.push_tokens(&[token]) {
            self.emitter.completed(GeneratedItem::Text(text));
        }

        self.num_tokens_generated
            .store(self.num_generated, Ordering::Relaxed);
    }
}

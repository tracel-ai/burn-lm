use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::Sender,
        Arc,
    },
    thread::JoinHandle,
};

use burn::tensor::{Device, Int, Tensor};
use burn_lm_inference::{GeneratedItem, GeneratedItemEmitter};

use crate::tokenizer::Tokenizer;

use super::StreamingDecoder;

/// The text generation context, used to check when a stop token has been reached.
///
/// Not `Clone`: it owns the [`JoinHandle`] of the background decoder thread and is finalized via
/// [`finish`](Self::finish), so cloning it would not make sense.
pub struct GenerationContext {
    pub tokens: Tensor<1, Int>,
    num_tokens: usize,
    stop: Arc<AtomicBool>,
    num_generated: Arc<AtomicUsize>,
    sender: Sender<Tensor<1, Int>>,
    decoder_handle: JoinHandle<()>,
}

impl GenerationContext {
    /// Create a new generation context.
    pub fn new<T: Tokenizer + 'static>(
        max_sample_len: usize,
        emitter: GeneratedItemEmitter,
        tokenizer: T,
        device: &Device,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Tensor<1, Int>>();
        let stop = Arc::new(AtomicBool::new(false));
        let num_generated = Arc::new(AtomicUsize::new(0));

        let mut generation =
            TokenGeneration::new(emitter, tokenizer, stop.clone(), num_generated.clone());

        let decoder_handle = std::thread::spawn(move || {
            for tokens in receiver.iter() {
                let tokens = tokens
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .unwrap();

                generation.process(tokens);
            }
        });

        Self {
            tokens: Tensor::empty([max_sample_len], device),
            num_tokens: 0,
            stop,
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
            ..
        } = self;

        // Dropping the sender closes the channel, ending the decoder thread's `receiver.iter()`.
        drop(sender);
        // Join so the final in-flight token is decoded and emitted before we return.
        decoder_handle.join().unwrap();

        num_generated.load(Ordering::Relaxed)
    }

    /// Add generated tokens to the state (without checking for stop condition).
    pub fn append(&mut self, tokens: Tensor<1, Int>) {
        let num_tokens_prev = self.num_tokens;
        self.num_tokens += tokens.shape().num_elements();
        self.tokens
            .inplace(|toks| toks.slice_assign(num_tokens_prev..self.num_tokens, tokens));
    }

    /// Update the state with newly generated tokens.
    pub fn update(&mut self, tokens: Tensor<1, Int>) {
        self.append(tokens.clone());

        if !self.should_stop() {
            self.sender.send(tokens).unwrap();
        }
    }

    /// True if the state previously detected a stop token.
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Returns the number of tokens generated.
    pub fn num_tokens_generated(&self) -> usize {
        self.num_generated.load(Ordering::Relaxed)
    }
}

struct TokenGeneration<T: Tokenizer> {
    emitter: GeneratedItemEmitter,
    decoder: StreamingDecoder<T>,
    stop_tokens: Vec<u32>,
    stop: Arc<AtomicBool>,
    num_tokens_generated: Arc<AtomicUsize>,
    num_generated: usize,
}

impl<T: Tokenizer> TokenGeneration<T> {
    fn new(
        emitter: GeneratedItemEmitter,
        tokenizer: T,
        stop: Arc<AtomicBool>,
        num_tokens_generated: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            emitter,
            stop_tokens: tokenizer.stop_ids(),
            decoder: StreamingDecoder::new(tokenizer),
            stop,
            num_tokens_generated,
            num_generated: 0,
        }
    }

    fn process(&mut self, tokens: Vec<u32>) {
        let mut finished = false;
        let mut generated = Vec::new();

        self.num_generated += tokens.len();

        for token in tokens {
            if self.stop_tokens.contains(&token) {
                finished = true;
            }

            if !finished {
                generated.push(token);
            }
        }

        if !generated.is_empty() {
            if let Some(text) = self.decoder.push_tokens(&generated) {
                self.emitter.completed(GeneratedItem::Text(text));
            }
        }

        if finished {
            self.stop.store(true, Ordering::Relaxed);
        }

        self.num_tokens_generated
            .store(self.num_generated, Ordering::Relaxed);
    }
}

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::Sender,
    Arc,
};

use burn::tensor::{Int, Tensor};
use burn_lm_inference::{Backend, GeneratedItem, GeneratedItemEmitter};

use crate::tokenizer::Tokenizer;

#[derive(Clone)]
/// The text generation context, used to check when a stop token has been reached.
pub struct GenerationContext<B: Backend> {
    pub tokens: Tensor<B, 1, Int>,
    num_tokens: usize,
    stop: Arc<AtomicBool>,
    num_generated: Arc<AtomicUsize>,
    sender: Sender<Tensor<B, 1, Int>>,
}

impl<B: Backend> GenerationContext<B> {
    /// Create a new instance.
    pub fn new<T: Tokenizer + 'static>(
        max_sample_len: usize,
        emitter: GeneratedItemEmitter,
        tokenizer: T,
        device: &B::Device,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Tensor<B, 1, Int>>();
        let stop = Arc::new(AtomicBool::new(false));
        let num_generated = Arc::new(AtomicUsize::new(0));

        let mut generation =
            TokenGeneration::new(emitter, tokenizer, stop.clone(), num_generated.clone());

        std::thread::spawn(move || {
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
            tokens: Tensor::<B, 1, Int>::empty([max_sample_len], device),
            num_tokens: 0,
            stop,
            num_generated,
            sender,
        }
    }

    /// Add generated tokens to the state (without checking for stop condition).
    pub fn append(&mut self, tokens: Tensor<B, 1, Int>) {
        let num_tokens_prev = self.num_tokens;
        self.num_tokens += tokens.shape().num_elements();
        self.tokens
            .inplace(|toks| toks.slice_assign([num_tokens_prev..self.num_tokens], tokens));
    }

    /// Update the state with newly generated tokens.
    pub fn update(&mut self, tokens: Tensor<B, 1, Int>) {
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
    tokenizer: T,
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
            tokenizer,
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
            let text = self.tokenizer.decode(generated);
            self.emitter.completed(GeneratedItem::Text(text));
        }

        if finished {
            self.stop.store(true, Ordering::Relaxed);
        }

        self.num_tokens_generated
            .store(self.num_generated, Ordering::Relaxed);
    }
}

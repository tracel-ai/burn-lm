use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::Sender,
    Arc,
};

use burn::tensor::{Int, Tensor};
use burn_lm_inference::Backend;

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
    pub fn new(max_sample_len: usize, stop_tokens: Tensor<B, 1, Int>) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Tensor<B, 1, Int>>();
        let stop = Arc::new(AtomicBool::new(false));
        let num_generated = Arc::new(AtomicUsize::new(0));
        let num_generated_clone = Arc::clone(&num_generated);
        let state_clone = Arc::clone(&stop);
        let device = stop_tokens.device();

        std::thread::spawn(move || {
            for tokens in receiver.iter() {
                let num_tokens = tokens.shape().num_elements();
                let total_num_tokens = num_generated_clone.load(Ordering::Relaxed) + num_tokens;
                if stop_tokens
                    .clone()
                    .equal(tokens)
                    .any()
                    .into_data()
                    .convert::<u8>()
                    .as_slice::<u8>()
                    .unwrap()[0]
                    == 1
                {
                    state_clone.store(true, Ordering::Relaxed);
                }
                num_generated_clone.store(total_num_tokens, Ordering::Relaxed);
            }
        });

        Self {
            tokens: Tensor::<B, 1, Int>::empty([max_sample_len], &device),
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

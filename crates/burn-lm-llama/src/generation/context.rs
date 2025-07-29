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

pub struct StreamChat<T: Tokenizer> {
    pub(crate) emitter: GeneratedItemEmitter,
    pub(crate) tokenizer: T,
}

#[derive(Default)]
struct TokenStreamState {
    stop_tokens_match_count: usize,
    previous_tokens: Vec<u32>,
}

impl TokenStreamState {
    pub fn check_end_token(&mut self, stop_tokens: &[u32], tok: u32) -> bool {
        //  println!("{stop_tokens:?}[{}] == {tok}", self.stop_tokens_match_count);

        //  if self.stop_tokens_match_count >= stop_tokens.len() {
        //      return true;
        //  }

        //  if stop_tokens[self.stop_tokens_match_count] == tok {
        //      self.stop_tokens_match_count += 1;
        //  } else {
        //      self.stop_tokens_match_count = 0;
        //  };
        //
        // self.stop_tokens_match_count == stop_tokens.len()
        if stop_tokens.contains(&tok) {
            return true;
        }

        self.previous_tokens.push(tok);
        false
    }

    pub fn stream(&mut self) -> Option<Vec<u32>> {
        if self.stop_tokens_match_count == 0 {
            Some(core::mem::take(&mut self.previous_tokens))
        } else {
            None
        }
    }
}
impl<B: Backend> GenerationContext<B> {
    /// Create a new instance.
    pub fn new<T: Tokenizer + 'static>(
        max_sample_len: usize,
        stop_tokens: Tensor<B, 1, Int>,
        stream: StreamChat<T>,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Tensor<B, 1, Int>>();
        let stop = Arc::new(AtomicBool::new(false));
        let num_generated = Arc::new(AtomicUsize::new(0));
        let num_generated_clone = Arc::clone(&num_generated);
        let state_clone = Arc::clone(&stop);
        let device = stop_tokens.device();

        let stop_tokens = stop_tokens
            .into_data()
            .convert::<u32>()
            .into_vec::<u32>()
            .unwrap();

        std::thread::spawn(move || {
            let mut state = TokenStreamState::default();

            for tokens in receiver.iter() {
                let num_tokens = tokens.shape().num_elements();
                let total_num_tokens = num_generated_clone.load(Ordering::Relaxed) + num_tokens;
                let tokens = tokens
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .unwrap();

                let mut finished = false;
                for token in tokens {
                    finished = state.check_end_token(&stop_tokens, token);
                }

                if let Some(tokens) = state.stream() {
                    let text = stream.tokenizer.decode(tokens);
                    stream.emitter.completed(GeneratedItem::Text(text));
                }

                if finished {
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

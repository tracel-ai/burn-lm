use std::time::Instant;

use burn::prelude::*;
use burn_lm_inference::GeneratedItemEmitter;

use crate::{inference::LlamaDecoder, tokenizer::Tokenizer};

use super::{
    temperature_scaled_softmax, GenerationContext, GenerationError, GenerationOutput, Sampler,
};

/// Drives a single autoregressive request through a reusable Llama decoder.
pub struct SingleRequestEngine<'a, T: Tokenizer> {
    decoder: &'a mut LlamaDecoder,
    tokenizer: T,
}

impl<'a, T: Tokenizer + 'static> SingleRequestEngine<'a, T> {
    pub fn new(decoder: &'a mut LlamaDecoder, tokenizer: T) -> Self {
        Self { decoder, tokenizer }
    }

    pub fn generate(
        &mut self,
        input_tokens: Tensor<1, Int>,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
        emitter: GeneratedItemEmitter,
    ) -> Result<GenerationOutput, GenerationError> {
        let prompt_len = input_tokens.dims()[0];

        let mut state = GenerationContext::new(
            prompt_len + sample_len,
            emitter,
            self.tokenizer.clone(),
            &self.decoder.device,
        );
        state.append(input_tokens);

        let mut input_pos = Tensor::<1, Int>::arange(0..prompt_len as i64, &self.decoder.device);
        let now = Instant::now();

        for _ in 0..sample_len {
            if state.should_stop() {
                break;
            }

            let x = state
                .tokens
                .clone()
                .select(0, input_pos.clone())
                .reshape([1, -1]);

            let logits = self.decoder.forward(x)?;

            let [batch_size, seq_len, vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .reshape([batch_size, vocab_size]);

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let next_token = sampler.sample(next_token_logits).reshape([batch_size]);
            state.update(next_token);

            let t = input_pos.dims()[0];
            input_pos = input_pos.slice(t - 1..t) + 1;
        }

        Ok(GenerationOutput {
            tokens: state.num_tokens_generated(),
            time: now.elapsed(),
        })
    }
}

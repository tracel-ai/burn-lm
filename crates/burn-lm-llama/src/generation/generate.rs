use std::time::Instant;

use super::{GenerationContext, Sampler};
use crate::{tokenizer::Tokenizer, Llama};
use burn::{prelude::*, tensor::activation::softmax};
use burn_lm_inference::GeneratedItemEmitter;

pub(crate) fn temperature_scaled_softmax<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
) -> Tensor<B, 2> {
    softmax(logits / temperature, 1)
}

/// Generated text sample output.
pub struct GenerationOutput {
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: std::time::Duration,
}

#[derive(Debug)]
pub enum GenerationError {
    MaxSequenceLengthExceeded { actual: usize, max: usize },
}

impl<B: Backend, T: Tokenizer + 'static> Llama<B, T> {
    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate (i.e., the number of generation steps to take).
    /// - `temperature`: Temperature value for controlling randomness in sampling (scales logits by `1 / temperature`).
    ///                  High values result in more random sampling.
    /// - `sampler`: The sampling strategy to use when selecting the next token based on the predicted probabilities.
    ///
    /// # Returns
    /// The generated text along with some other metadata (see [GenerationOutput]).
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
        emitter: GeneratedItemEmitter,
    ) -> Result<GenerationOutput, GenerationError> {
        let input_tokens = self.tokenize(prompt);
        let prompt_len = input_tokens.dims()[0];

        let mut state = GenerationContext::new(
            prompt_len + sample_len,
            emitter,
            self.tokenizer.clone(),
            &self.device,
        );
        state.append(input_tokens);

        let mut input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &self.device);
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

            let [_, seq_len] = x.dims();

            // Prepare cache and RoPE for current sequence length and position
            let mask = self.cache.prepare(seq_len)?;
            self.pos_encoding.prepare(seq_len);

            let logits = self
                .model
                .forward(x, &mut self.cache, &self.pos_encoding, mask);

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1); // [batch_size=1, vocab_size]

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let next_token = sampler.sample(next_token_logits).squeeze(0);
            // Update with the new generated token
            state.update(next_token);

            // Advance
            let t = input_pos.dims()[0];
            input_pos = input_pos.slice([t - 1..t]) + 1;
        }

        let num_tokens = state.num_tokens_generated();

        Ok(GenerationOutput {
            tokens: num_tokens,
            time: now.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tests::*, tokenizer::byte::ByteTokenizer, LlamaConfig};

    use burn::{
        module::Reinitializer,
        tensor::{TensorData, Tolerance},
    };
    use burn_lm_inference::TextGenerationListener;

    #[test]
    fn test_temperature_softmax() {
        let tensor = TestTensor::<2>::from([[21.3125, 19.859375, 19.0625, 18.75, 18.171875]]);

        let output = temperature_scaled_softmax(tensor, 0.6);
        let expected = TensorData::from([[
            0.8691406,
            0.07836914,
            0.020767212,
            0.0124053955,
            0.0047035217,
        ]]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05)); // 5% tolerance
    }

    #[test]
    fn test_llama3_2_3b_test() {
        let device: Device<TestBackend> = Default::default();
        let config = LlamaConfig::llama3_2_1b_test();
        let mut llama = config.init::<TestBackend, ByteTokenizer>(&device).unwrap();

        llama.model = Reinitializer::new()
            .random_float(0, -1.0, 1.0)
            .apply(llama.model);

        let (emitter, handle) = GeneratedItemEmitter::init(TextGenerationListener::default());
        llama
            .generate("This is a test", 64, 0.0, &mut Sampler::Argmax, emitter)
            .unwrap();
        let result = handle.join();
        let expected = "[187, 114, 51, 146, 146, 250, 112, 224, 192, 99, 132, 0, 0, 180, 192, 99, 19, 114, 19, 174, 0, 180, 192, 131, 132, 19, 99, 114, 131, 132, 249, 146, 82, 28, 226, 226, 148, 84, 19, 192, 83, 99, 19, 249, 19, 251, 222, 19, 192, 180, 192, 180, 192, 0, 180, 192, 146, 20, 0, 180, 192, 180]";

        assert_eq!(result, expected);
    }
}

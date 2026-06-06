use super::{Sampler, SingleRequestEngine};
use crate::{inference::Llama, tokenizer::Tokenizer};
use burn::{prelude::*, tensor::activation::softmax};
use burn_lm_inference::GeneratedItemEmitter;

pub(crate) fn temperature_scaled_softmax(logits: Tensor<2>, temperature: f64) -> Tensor<2> {
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

impl<T: Tokenizer + 'static> Llama<T> {
    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate (i.e., the number of generation steps to take).
    /// - `temperature`: Temperature value for controlling randomness in sampling (scales logits by `1 / temperature`).
    ///   High values result in more random sampling.
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
        self.reset();

        let input_tokens = self.tokenize(prompt);
        let tokenizer = self.tokenizer.clone();
        let mut engine = SingleRequestEngine::new(&mut self.decoder, tokenizer);

        engine.generate(input_tokens, sample_len, temperature, sampler, emitter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tests::*, tokenizer::byte::ByteTokenizer, LlamaConfig};

    use crate::tests::Reinitializer;
    use burn::tensor::{TensorData, Tolerance};
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
        let device: Device = Default::default();
        let config = LlamaConfig::llama3_2_1b_test();
        let mut llama = config.init::<ByteTokenizer>(&device).unwrap();

        llama.decoder.model = Reinitializer::default()
            .random_float(0, -1.0, 1.0)
            .apply(llama.decoder.model);

        let (emitter, handle) = GeneratedItemEmitter::init(TextGenerationListener::default());
        llama
            .generate("This is a test", 64, 0.0, &mut Sampler::Argmax, emitter)
            .unwrap();

        let result = handle.join();
        let expected = "[187][114][51][146][146][250][112][224][192][99][132][0][0][180][192][99][19][114][19][174][0][180][192][131][132][19][99][114][131][132][249][146][82][28][226][226][148][84][19][192][83][99][19][249][19][251][222][19][192][180][192][180][192][0][180][192][146][20][0][180][192][180][20]";

        assert_eq!(result, expected);
    }

    #[test]
    fn llama_generate_leaks_autoregressive_kv_cache_across_independent_generations() {
        fn run_once(
            llama: &mut crate::inference::Llama<ByteTokenizer>,
            prompt: &str,
        ) -> String {
            let (emitter, handle) = GeneratedItemEmitter::init(TextGenerationListener::default());

            // This observes streamed text, which can race with the decoder thread.
            // Give the decoder a short grace period before finishing the listener;
            // the emitter lifecycle should be fixed separately from this cache test.
            llama
                .generate(prompt, 48, 0.0, &mut Sampler::Argmax, emitter)
                .unwrap();

            // Note: I hate this, but fixing it properly would require intrusive changes to generate or even deeper.
            // See comment above for more details.
            std::thread::sleep(std::time::Duration::from_millis(100));

            handle.join()
        }

        fn reset_generation_state(llama: &mut crate::inference::Llama<ByteTokenizer>) {
            llama.reset();
            // Keep the clean baseline explicit: a stateless request starts at position 0.
            llama.decoder.pos_encoding.next_position = 0;
            llama.decoder.pos_encoding.curr_seq_len = 0;
            llama.decoder.pos_encoding.start_offset = 0;
        }

        let device: Device = Default::default();
        let config = LlamaConfig::llama3_2_1b_test();
        let mut llama = config.init::<ByteTokenizer>(&device).unwrap();

        llama.decoder.model = Reinitializer::default()
            .random_float(0, -1.0, 1.0)
            .apply(llama.decoder.model);

        let prompt_1 = "This is a deterministic stateless generation test.";
        let prompt_2 = "A different request should not become hidden context.";

        // With fixed weights, argmax sampling, and clean generation state,
        // prompt_1 should have one deterministic answer.
        reset_generation_state(&mut llama);
        let expected_prompt_1 = run_once(&mut llama, prompt_1);

        // Clearing generation state before another prompt_1 run gives the same baseline.
        // This verifies that the test is deterministic when no stale state is present.
        reset_generation_state(&mut llama);
        let prompt_1_after_reset = run_once(&mut llama, prompt_1);
        assert_eq!(expected_prompt_1, prompt_1_after_reset);

        // Poison the model with an unrelated request, then run prompt_1 without clearing
        // state. For a stateless API this must still match the clean prompt_1 baseline.
        // If it differs, prompt_2's KV/position state leaked into the next request.
        reset_generation_state(&mut llama);
        let _ = run_once(&mut llama, prompt_2);
        let prompt_1_after_unrelated_prompt = run_once(&mut llama, prompt_1);

        assert_eq!(
            expected_prompt_1, prompt_1_after_unrelated_prompt,
            "Llama::generate leaks autoregressive KV cache across independent generations"
        );
    }
}

use std::time::Instant;

use super::Sampler;
use crate::{inference::Llama, tokenizer::Tokenizer};
use burn::{prelude::*, tensor::activation::softmax, tensor::TensorData};
use burn_lm_inference::{
    batching::{BatchCapacity, BatchedDecoder, ForwardBatch},
    GeneratedItemEmitter, InferenceResult,
};

use super::GenerationContext;

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
    ///
    /// Single-request generation is just batched generation with one sequence, so this delegates to
    /// [`generate_batch`](Self::generate_batch). Keeping one decode loop means there is a single
    /// place to maintain (and, later, to replace with a fused forward).
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
        emitter: GeneratedItemEmitter,
    ) -> InferenceResult<GenerationOutput> {
        let mut outputs =
            self.generate_batch(vec![prompt], sample_len, temperature, sampler, vec![emitter])?;
        Ok(outputs
            .pop()
            .expect("one prompt in yields exactly one output"))
    }

    /// Generate for a batch of prompts, advancing them round-robin one token at a time.
    ///
    /// This is a thin batch driver that exercises the [`BatchedDecoder`] seam exactly the way the
    /// framework's continuous engine does: each sequence owns its own engine-side cache
    /// ([`LlamaDecoder::Cache`](crate::inference::LlamaDecoder)), allocated up front and passed back
    /// into [`forward`](BatchedDecoder::forward) on every step. The round-robin sweep makes the
    /// sequences' output streams interleave. This is **not** fused batching — Phase 2 replaces the
    /// per-row forward with a single multi-row call.
    pub fn generate_batch(
        &mut self,
        prompts: Vec<&str>,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
        emitters: Vec<GeneratedItemEmitter>,
    ) -> InferenceResult<Vec<GenerationOutput>> {
        assert_eq!(
            prompts.len(),
            emitters.len(),
            "each prompt needs exactly one emitter"
        );

        self.reset();

        let capacity = BatchCapacity {
            free_slots: prompts.len(),
            free_kv_tokens: usize::MAX,
        };

        // Per-sequence state: engine-owned cache + streaming/stop context + position cursor.
        struct Seq {
            state: GenerationContext,
            cache: <crate::inference::LlamaDecoder as BatchedDecoder>::Cache,
            /// Full token buffer (prompt + generated). The next forward consumes the unprocessed
            /// tail of this buffer.
            tokens: Vec<u32>,
            /// Number of tokens already forwarded through the decoder (absolute position of the
            /// next input). Starts at 0; after prefill it equals the prompt length.
            processed: usize,
            steps_left: usize,
            finished: bool,
        }

        let device = self.decoder.device.clone();

        let mut active: Vec<Seq> = prompts
            .into_iter()
            .zip(emitters)
            .map(|(prompt, emitter)| {
                let input_tokens = self.tokenize(prompt);
                let prompt_len = input_tokens.dims()[0];
                let token_ids: Vec<u32> = input_tokens
                    .clone()
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .expect("prompt tokens should convert to u32");
                let mut state = GenerationContext::new(
                    prompt_len + sample_len,
                    emitter,
                    self.tokenizer.clone(),
                    &device,
                );
                state.append(input_tokens);
                Seq {
                    state,
                    cache: self.decoder.allocate_cache(capacity),
                    tokens: token_ids,
                    processed: 0,
                    steps_left: sample_len,
                    finished: false,
                }
            })
            .collect();

        let now = Instant::now();
        let mut remaining = active.len();
        while remaining > 0 {
            for seq in active.iter_mut() {
                if seq.finished {
                    continue;
                }
                if seq.steps_left == 0 || seq.state.should_stop() {
                    seq.finished = true;
                    remaining -= 1;
                    continue;
                }

                // Unprocessed tail: whole prompt on the first step (prefill), one token afterwards.
                let input_ids: Vec<i32> =
                    seq.tokens[seq.processed..].iter().map(|&t| t as i32).collect();
                let seq_len = input_ids.len();
                let position = seq.processed;
                let x = Tensor::<2, Int>::from_data(
                    TensorData::new(input_ids, [1, seq_len]),
                    &device,
                );

                let batch = ForwardBatch {
                    input_tokens: x,
                    positions: vec![position],
                    cache_slots: vec![0],
                };
                // Drive the batched seam explicitly (UFCS) so this exercises
                // `BatchedDecoder::forward` against the engine-owned per-seq cache, not the
                // inherent `LlamaDecoder::forward`.
                let output = BatchedDecoder::forward(&mut self.decoder, batch, &mut seq.cache)?;

                let [batch_size, out_seq_len, vocab_size] = output.logits.dims();
                let mut next_token_logits = output
                    .logits
                    .slice([0..batch_size, out_seq_len - 1..out_seq_len])
                    .reshape([batch_size, vocab_size]);
                if temperature > 0.0 {
                    next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
                }
                let next_token = sampler.sample(next_token_logits).reshape([batch_size]);
                let next_id = next_token
                    .clone()
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .expect("sampled token should convert to u32")[0];

                // Everything up to here is now processed; the next step consumes only the new token.
                seq.processed = seq.tokens.len();
                seq.tokens.push(next_id);
                seq.state.update(next_token);
                seq.steps_left -= 1;
            }
        }

        let elapsed = now.elapsed();
        Ok(active
            .into_iter()
            .map(|seq| GenerationOutput {
                // `finish` joins the decoder thread so all tokens are emitted before we return,
                // making the streamed output deterministic for the caller's `handle.join()`.
                tokens: seq.state.finish(),
                time: elapsed,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tests::*, tokenizer::byte::ByteTokenizer, LlamaConfig};

    use crate::tests::Reinitializer;
    use burn::tensor::{TensorData, Tolerance};
    use burn_lm_inference::{GeneratedItemEmitter, TextGenerationListener, WriteListener};
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// A writer that records, in global emission order, which sequence produced each chunk of text.
    struct TagWriter {
        id: usize,
        log: Arc<Mutex<Vec<usize>>>,
    }

    impl Write for TagWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.log.lock().unwrap().push(self.id);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Two requests driven through `generate_batch` must interleave their output streams, proving
    /// the engine advances them round-robin rather than one fully then the other.
    #[test]
    fn generate_batch_interleaves_two_sequences() {
        let device: Device = Default::default();
        let config = LlamaConfig::llama3_2_1b_test();
        let mut llama = config.init::<ByteTokenizer>(&device).unwrap();
        llama.decoder.model = Reinitializer::default()
            .random_float(0, -1.0, 1.0)
            .apply(llama.decoder.model);

        let log = Arc::new(Mutex::new(Vec::<usize>::new()));
        let (emitter_a, handle_a) = GeneratedItemEmitter::init(WriteListener::new(TagWriter {
            id: 0,
            log: log.clone(),
        }));
        let (emitter_b, handle_b) = GeneratedItemEmitter::init(WriteListener::new(TagWriter {
            id: 1,
            log: log.clone(),
        }));

        llama
            .generate_batch(
                vec!["First request", "Second request"],
                16,
                0.0,
                &mut Sampler::Argmax,
                vec![emitter_a, emitter_b],
            )
            .unwrap();

        // Flush both streaming pipelines.
        handle_a.join();
        handle_b.join();

        let log = log.lock().unwrap();
        assert!(
            log.contains(&0) && log.contains(&1),
            "both sequences should produce output: {log:?}"
        );

        // Interleaving ⇒ the two streams overlap in time: each starts before the other ends.
        // A serial implementation would produce all of one then all of the other (no overlap).
        let first0 = log.iter().position(|&x| x == 0).unwrap();
        let last0 = log.iter().rposition(|&x| x == 0).unwrap();
        let first1 = log.iter().position(|&x| x == 1).unwrap();
        let last1 = log.iter().rposition(|&x| x == 1).unwrap();
        assert!(
            first1 < last0 && first0 < last1,
            "sequences did not interleave (ran serially): {log:?}"
        );
    }

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
        // Full sample_len=64 output. (The previous value was one token short: the old generation
        // path didn't join the streaming/decoder thread before the caller read the result, so the
        // final token was dropped at record time. `GenerationContext::finish` now joins that thread,
        // so all 64 generated tokens are emitted deterministically.)
        let expected = "[187][114][51][146][146][250][112][224][192][99][132][0][0][180][192][99][19][114][19][174][0][180][192][131][132][19][99][114][131][132][249][146][82][28][226][226][148][84][19][192][83][99][19][249][19][251][222][19][192][180][192][180][192][0][180][192][146][20][0][180][192][180][20][0]";

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

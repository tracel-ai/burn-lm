use std::time::Instant;

use crate::{inference::Llama, tokenizer::Tokenizer};
use burn_lm_inference::{
    batching::{step_round, ActiveSeq, BatchedDecoder, PrefillBudget, StepOutcome},
    GeneratedItemEmitter, InferenceResult, Sampler,
};

use super::GenerationContext;

/// Generated text sample output.
pub struct GenerationOutput {
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: std::time::Duration,
}

impl<T: Tokenizer + 'static> Llama<T> {
    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate (i.e., the number of generation steps to take).
    /// - `sampler`: The sampling strategy to use when selecting the next token based on the predicted
    ///   logits. It carries its own config (temperature, top-p): temperature scaling now lives inside
    ///   the sampler, not as a separate argument.
    ///
    /// # Returns
    /// The generated text along with some other metadata (see [GenerationOutput]).
    ///
    /// Single-request generation is just batched generation with one sequence, so this delegates to
    /// `generate_batch`. Keeping one decode loop means there is a single place to maintain (and,
    /// later, to replace with a fused forward).
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        sampler: &dyn Sampler,
        emitter: GeneratedItemEmitter,
    ) -> InferenceResult<GenerationOutput> {
        let mut outputs = self.generate_batch(vec![prompt], sample_len, sampler, vec![emitter])?;
        Ok(outputs
            .pop()
            .expect("one prompt in yields exactly one output"))
    }

    /// Generate for a batch of prompts, advancing all of them one token per round.
    ///
    /// This is a thin batch driver that exercises the `BatchedDecoder` seam exactly the way the
    /// framework's continuous engine does: each sequence is assigned a decoder slot up front (this
    /// driver owns the slot list, `0..prompts.len()`), prompts prefill into their slots, and each
    /// round advances every decoding sequence through one fused `decode` call before releasing
    /// every slot on return. The sequences' output streams interleave round by round.
    pub fn generate_batch(
        &mut self,
        prompts: Vec<&str>,
        sample_len: usize,
        sampler: &dyn Sampler,
        emitters: Vec<GeneratedItemEmitter>,
    ) -> InferenceResult<Vec<GenerationOutput>> {
        assert_eq!(
            prompts.len(),
            emitters.len(),
            "each prompt needs exactly one emitter"
        );
        // The decoder owns a fixed KV slab with one lane per slot, so the batch cannot exceed its
        // lane count (`max_batch_size`). Fail fast with a clear message instead of a cryptic
        // out-of-bounds inside the cache. The framework serving path never trips this — admission
        // bounds in-flight sequences by `batch_capacity().max_slots`, wired to the same number.
        assert!(
            prompts.len() <= self.decoder.cache.lane_count(),
            "generate_batch got {} prompts but the decoder slab has only {} lane(s); \
             build the model with a larger max_batch_size",
            prompts.len(),
            self.decoder.cache.lane_count(),
        );

        self.reset();

        // Build the active set from the prompts. Each sequence carries its decoder slot and token
        // buffer (the generic decode state) plus its `GenerationContext` (the library-side
        // streaming/detok payload) in `extra`. `max_gen = sample_len` caps generation.
        let mut active: Vec<ActiveSeq<GenerationContext>> = prompts
            .into_iter()
            .zip(emitters)
            .enumerate()
            .map(|(slot, (prompt, emitter))| {
                let token_ids: Vec<u32> = self
                    .tokenize(prompt)
                    .into_data()
                    .convert::<u32>()
                    .into_vec::<u32>()
                    .expect("prompt tokens should convert to u32");
                let state = GenerationContext::new(emitter, self.tokenizer.clone());
                ActiveSeq {
                    slot,
                    tokens: token_ids,
                    processed: 0,
                    generated: 0,
                    max_gen: sample_len,
                    finished: false,
                    // No engine reservation on the library path: this driver bounds the batch by
                    // the decoder's lane count up front, and the pool matches that rectangle, so
                    // block accounting is the serving worker's concern, not this one's.
                    kv_reservation: 0,
                    extra: state,
                }
            })
            .collect();

        let stop_ids = self.tokenizer.stop_ids();

        let now = Instant::now();
        // Run the shared generic decode core to completion: one round advances every still-active
        // sequence by a token (one fused decode call, so their streams interleave). `step_round`
        // does the forward, the sample, and the synchronous stop-check; the driver-side work left is
        // streaming each new non-stop token through its `GenerationContext`. A stop id retires its
        // sequence in the same round it is produced, so no token is generated past it.
        let mut failure = None;
        'rounds: while active.iter().any(|seq| !seq.finished) {
            // A fresh prefill budget per round over the whole batch.
            let mut budget = PrefillBudget::for_round(&active);
            // The sampler carries no per-sequence state — greedy argmax draws nothing, and a
            // stochastic strategy draws from the backend RNG — so the whole round samples through the
            // one shared `sampler`, the same shape the serving worker uses.
            let outcomes = step_round(
                &mut self.decoder,
                &mut active,
                &stop_ids,
                &mut budget,
                // Unbounded prefill (the whole prompt in one round) for the local generation path; the
                // chunked-prefill width is server config, applied by the serving worker.
                0,
                |logits| sampler.sample(logits),
            );
            for (seq, outcome) in active.iter_mut().zip(outcomes) {
                match outcome {
                    StepOutcome::Stepped { token, is_stop, .. } => {
                        if !is_stop {
                            seq.extra.update(token);
                        }
                    }
                    // A failed forward (e.g. context length exceeded) aborts the whole batch,
                    // exactly as the pre-seam loop's `forward(..)?` did.
                    StepOutcome::Failed(err) => {
                        failure = Some(err);
                        break 'rounds;
                    }
                    StepOutcome::Skipped => {}
                    // Unreachable on this path (it prefills unbounded, so a prompt is never split into
                    // intermediate chunks), but the match is exhaustive: an intermediate chunk produces
                    // no token and leaves the sequence running.
                    StepOutcome::Prefilling => {}
                }
            }
        }

        // Free every slot before returning — on failure too. The decoder outlives this call, and
        // a slot left behind would leak this generation's KV history into the next one.
        for seq in active.iter() {
            self.decoder.release(seq.slot);
        }
        if let Some(err) = failure {
            return Err(err);
        }

        let elapsed = now.elapsed();
        Ok(active
            .into_iter()
            .map(|seq| {
                // `finish` joins the decoder thread so all tokens are emitted before we return,
                // making the streamed output deterministic for the caller's `handle.join()`. The
                // reported count is the engine's `generated` counter, which includes a terminating
                // stop token — the historical convention, and the one the serving driver reports.
                let tokens = seq.generated;
                seq.extra.finish();
                GenerationOutput {
                    tokens,
                    time: elapsed,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::sampling::temperature_scaled_softmax;
    use crate::{tests::*, tokenizer::byte::ByteTokenizer, LlamaConfig};

    use crate::tests::Reinitializer;
    use burn::prelude::*;
    use burn::tensor::{TensorData, Tolerance};
    use burn_lm_inference::{Argmax, GeneratedItemEmitter, TextGenerationListener, WriteListener};
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
    /// the engine advances them a token each per round rather than one fully then the other.
    #[test]
    fn generate_batch_interleaves_two_sequences() {
        let device: Device = Default::default();
        // Two sequences ⇒ the decoder slab needs two lanes (slot == lane).
        let config = LlamaConfig::llama3_2_1b_test().with_max_batch_size(2);
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
                &Argmax,
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

    /// Round-robin must be ISOLATED: a sequence's output is identical whether it runs alone or
    /// interleaved with another. This is the per-sequence KV-cache isolation that makes batch>1
    /// safe — a leak across the engine's cache swap would change a sequence's tokens when batched.
    #[test]
    fn generate_batch_isolates_each_sequence() {
        fn run(llama: &mut Llama<ByteTokenizer>, prompts: Vec<&str>) -> Vec<String> {
            llama.reset(); // each run starts from clean state (generation mutates shared KV)
            let (emitters, handles): (Vec<_>, Vec<_>) = prompts
                .iter()
                .map(|_| GeneratedItemEmitter::init(TextGenerationListener::default()))
                .unzip();
            llama
                .generate_batch(prompts, 16, &Argmax, emitters)
                .unwrap();
            handles.into_iter().map(|h| h.join()).collect()
        }

        let device: Device = Default::default();
        // Runs batches of up to two prompts ⇒ two lanes (solo runs use lane 0).
        let mut llama = LlamaConfig::llama3_2_1b_test()
            .with_max_batch_size(2)
            .init::<ByteTokenizer>(&device)
            .unwrap();
        llama.decoder.model = Reinitializer::default()
            .random_float(0, -1.0, 1.0)
            .apply(llama.decoder.model);

        // Each prompt run solo, then the two run interleaved (batch of 2).
        let a_alone = run(&mut llama, vec!["First request"]).remove(0);
        let b_alone = run(&mut llama, vec!["Second request"]).remove(0);
        let together = run(&mut llama, vec!["First request", "Second request"]);

        assert_eq!(
            together[0], a_alone,
            "sequence A's output changed when batched with B (KV cache not isolated)"
        );
        assert_eq!(
            together[1], b_alone,
            "sequence B's output changed when batched with A (KV cache not isolated)"
        );
        // Guard against a trivially-true test: the two prompts must actually diverge.
        assert_ne!(a_alone, b_alone, "test prompts produced identical output");
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
            .generate("This is a test", 64, &Argmax, emitter)
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
        fn run_once(llama: &mut crate::inference::Llama<ByteTokenizer>, prompt: &str) -> String {
            let (emitter, handle) = GeneratedItemEmitter::init(TextGenerationListener::default());

            // This observes streamed text, which can race with the decoder thread.
            // Give the decoder a short grace period before finishing the listener;
            // the emitter lifecycle should be fixed separately from this cache test.
            llama.generate(prompt, 48, &Argmax, emitter).unwrap();

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

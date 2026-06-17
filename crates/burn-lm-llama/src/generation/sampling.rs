// `Tensor` is needed by `temperature_scaled_softmax` (test-or-server gate); the rest are needed only
// by `top_p_sample_row` (server gate). Gate each `use` to where its items are actually consumed so a
// plain non-test build that compiles neither helper carries no unused imports.
#[cfg(all(
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
use burn::tensor::Int;
#[cfg(any(
    test,
    all(
        feature = "inference-server",
        any(feature = "llama3", feature = "tiny")
    )
))]
use burn::tensor::Tensor;
#[cfg(all(
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rngs::StdRng,
};

// `temperature_scaled_softmax` is exercised both by `LlamaSampler` (under the server-feature gate)
// and by this module's unit test. Gate it to those two so a plain non-test build that compiles
// neither doesn't carry it as dead code.
#[cfg(any(
    test,
    all(
        feature = "inference-server",
        any(feature = "llama3", feature = "tiny")
    )
))]
pub(crate) fn temperature_scaled_softmax(logits: Tensor<2>, temperature: f64) -> Tensor<2> {
    burn::tensor::activation::softmax(logits / temperature, 1)
}

/// Top-p (nucleus) sampling over a single `[1, vocab]` row of probabilities, drawing from the given
/// RNG. Selects the smallest set of tokens whose cumulative probability mass exceeds `p`, then
/// samples one token from that nucleus, returning a `[1, 1]` token id.
///
/// This is the exact nucleus math the old `TopP` sampler ran, lifted out so the config-driven
/// `LlamaSampler` can call it per row with that row's own RNG. The single-row `assert` is preserved:
/// `LlamaSampler::sample` slices the batch into rows and calls this once per row.
///
/// Gated to the `LlamaSampler` build (its only caller), so a build without the server features
/// doesn't carry it as dead code.
#[cfg(all(
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
pub(crate) fn top_p_sample_row(probs: Tensor<2>, p: f64, rng: &mut StdRng) -> Tensor<2, Int> {
    assert_eq!(
        probs.dims()[0],
        1,
        "Naive top-p sampling only supports single-batch tensors"
    );
    let (probs_sort, probs_idx) = probs.sort_descending_with_indices(1);

    // TODO: cumsum + Distribution::Multinomial support

    let mut probs_sort = probs_sort.to_data().iter::<f64>().collect::<Vec<_>>();

    let mut cumsum = 0.;
    probs_sort.iter_mut().for_each(|x| {
        if cumsum >= p {
            *x = 0.0;
        } else {
            cumsum += *x;
        }
    });

    let next_token_idx = WeightedIndex::new(probs_sort).unwrap().sample(rng);

    probs_idx.slice([0..1, next_token_idx..next_token_idx + 1])
}

// The config-driven sampler that plugs the Llama generation strategy (temperature scaling plus
// top-p, or greedy argmax) into the framework's object-safe `Sampler` seam. It is gated the same way
// as `SamplingSettings` (which it is built from): that type lives in `server::params`, behind
// `any(llama3, tiny)` inside `mod server`, which is behind `inference-server` — and the framework
// `Sampler` trait comes from `burn_lm_inference`, also behind `inference-server`. So this whole sampler
// only exists when all of those line up.
#[cfg(all(
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
pub use llama_sampler::LlamaSampler;

#[cfg(all(
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
mod llama_sampler {
    use super::{temperature_scaled_softmax, top_p_sample_row};
    use crate::server::params::SamplingSettings;
    use burn::tensor::Tensor;
    use burn_lm_inference::{ids_to_host, InferenceResult, Sampler, SamplingState};
    use rand::SeedableRng;

    /// The Llama models' config-driven next-token sampler.
    ///
    /// It holds the server's sampling-config defaults (temperature, top-p threshold, seed) and is
    /// shared by every in-flight sequence — the framework hands one `LlamaSampler` to a whole round.
    /// The per-sequence variation is only the RNG, which rides in each row's `SamplingState`, seeded
    /// in `fresh_state` from `effective_seed()` so two concurrent sequences draw off independent
    /// streams. At `temperature == 0.0` this is plain argmax, byte-identical to the old
    /// `Sampler::Argmax` path; above zero it temperature-scales the logits then draws from the
    /// preserved top-p nucleus using that row's RNG.
    pub struct LlamaSampler {
        settings: SamplingSettings,
    }

    impl LlamaSampler {
        /// Build the sampler from the server's resolved sampling settings (config defaults).
        pub fn new(settings: SamplingSettings) -> Self {
            Self { settings }
        }
    }

    impl Sampler for LlamaSampler {
        fn fresh_state(&self) -> SamplingState {
            // Each sequence seeds its own RNG from the config seed (a `0` seed draws a fresh random
            // one per sequence), so two concurrent requests never share a stream.
            SamplingState {
                rng: rand::rngs::StdRng::seed_from_u64(self.settings.effective_seed()),
            }
        }

        fn sample(
            &self,
            logits: Tensor<2>,
            states: &mut [SamplingState],
        ) -> InferenceResult<Vec<u32>> {
            // Greedy fast path: at temperature 0 there is nothing stochastic, so the whole batch is a
            // single argmax — byte-identical to the old argmax sampler, and the per-sequence RNGs are
            // untouched. This is the path the equivalence and worker tests exercise.
            if self.settings.temperature <= 0.0 {
                return ids_to_host(logits.argmax(1));
            }

            // Stochastic path: temperature-scale the whole batch once, then draw each row from its own
            // top-p nucleus using that row's RNG. We slice the `[n, vocab]` probabilities into single
            // rows because the moved top-p math is single-row (its preserved `assert`), and each row's
            // RNG must advance independently across the sequence's tokens.
            let [rows, vocab] = logits.dims();
            let probs = temperature_scaled_softmax(logits, self.settings.temperature);
            let mut ids = Vec::with_capacity(rows);
            for (row, state) in states.iter_mut().enumerate().take(rows) {
                let row_probs = probs.clone().slice([row..row + 1, 0..vocab]);
                let id = top_p_sample_row(row_probs, self.settings.top_p, &mut state.rng);
                ids.extend(ids_to_host(id)?);
            }
            Ok(ids)
        }
    }
}

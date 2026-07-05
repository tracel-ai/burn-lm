// `temperature_scaled_softmax` is needed in test builds and in the server build; `LlamaSampler` is
// needed only in the server build. Gate each item to where it is consumed so a plain build that
// compiles neither carries no dead code.
#[cfg(any(
    test,
    all(
        feature = "inference-server",
        any(feature = "llama3", feature = "tiny")
    )
))]
use burn::tensor::Tensor;

// `temperature_scaled_softmax` is exercised both by `LlamaSampler` (under the server-feature gate)
// and by a unit test in `generate`. Gate it to those two so a plain non-test build that compiles
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
    use super::temperature_scaled_softmax;
    use crate::server::params::SamplingSettings;
    use burn::tensor::{Distribution, Tensor};
    use burn_lm_inference::{ids_to_host, InferenceResult, Sampler};

    /// The lower bound on the uniform draws feeding the Gumbel noise. The Gumbel transform is
    /// `-log(-log(u))`, which is `NaN` at `u == 0` and `+inf` at `u == 1`. Burn's `Uniform(low, high)`
    /// is `[low, high)`, so the high end is already excluded; clamping the low end just above zero
    /// keeps `log(u)` finite. The value is far smaller than any probability that ever wins, so it does
    /// not bias the draw.
    const GUMBEL_LOW: f64 = 1e-7;

    /// The Llama models' config-driven next-token sampler.
    ///
    /// It holds the server's sampling-config defaults (temperature, top-p threshold) and is shared by
    /// every in-flight sequence — the framework hands one `LlamaSampler` to a whole round. It keeps no
    /// per-sequence state: at `temperature == 0.0` it is plain greedy argmax, and above zero it draws
    /// from the top-p nucleus entirely on the device, taking its randomness from the tensor backend's
    /// own RNG. Reproducibility, when a test or deployment wants it, comes from seeding that backend
    /// RNG (`Device::seed`).
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
        fn sample(&self, logits: Tensor<2>) -> InferenceResult<Vec<u32>> {
            // Greedy fast path: at temperature 0 there is nothing stochastic, so the whole batch is a
            // single argmax — byte-identical to the framework's argmax sampler. This is the path the
            // equivalence and worker tests exercise.
            // TEMP PROFILING: which sampling path runs, and why.
            tracing::debug!(
                target: "batching",
                temperature = self.settings.temperature,
                top_p = self.settings.top_p,
                greedy = self.settings.temperature <= 0.0,
                "sampler-path"
            );
            if self.settings.temperature <= 0.0 {
                return ids_to_host(logits.argmax(1));
            }

            // Stochastic top-p, batched on the device with a single readback. The old path sorted, read
            // back, and drew one row at a time on the host; this does the whole round on the device and
            // only reads back the final `[rows, 1]` token ids.
            let [rows, vocab] = logits.dims();
            let device = logits.device();
            let probs = temperature_scaled_softmax(logits, self.settings.temperature);

            // Sort every row's vocabulary by probability so the nucleus is a contiguous leading run,
            // and keep the original token ids alongside so the winner can be mapped back.
            let (sorted, idx) = probs.sort_descending_with_indices(1);

            // The nucleus is the smallest leading run whose mass first reaches `p`. A token belongs to
            // it exactly when the mass *before* it is still below `p`; that exclusive prefix mass is the
            // inclusive cumulative sum minus the token itself. Everything past the nucleus is dropped to
            // `-inf` log-probability, so it can never win the argmax below.
            let exclusive_mass = sorted.clone().cumsum(1) - sorted.clone();
            let dropped = exclusive_mass.greater_equal_elem(self.settings.top_p);
            let masked = sorted.log().mask_fill(dropped, f32::NEG_INFINITY);

            // Gumbel-max: the argmax of `log p + g`, with `g` a Gumbel(0,1) draw per token, is an exact
            // categorical sample from the (renormalised) nucleus. The `-inf` tokens stay `-inf` and
            // never win, so the draw is confined to the nucleus. `g = -log(-log(u))`, `u ~ Uniform`.
            let uniform = Tensor::random(
                [rows, vocab],
                Distribution::Uniform(GUMBEL_LOW, 1.0),
                &device,
            );
            let gumbel = uniform.log().neg().log().neg();
            let winner = (masked + gumbel).argmax(1);

            // `winner` indexes into the sorted order; map it back to the original token ids and read
            // back the one column per row — the single device-to-host transfer of the whole sampler.
            ids_to_host(idx.gather(1, winner))
        }
    }
}

#[cfg(all(
    test,
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
mod tests {
    use super::LlamaSampler;
    use crate::server::params::SamplingSettings;
    use crate::tests::TestTensor;
    use burn_lm_inference::Sampler;
    use std::collections::HashSet;

    fn settings(temperature: f64, top_p: f64) -> SamplingSettings {
        SamplingSettings {
            top_p,
            temperature,
            sample_len: 16,
            seed: 0,
        }
    }

    #[test]
    fn temperature_zero_is_argmax() {
        // The greedy fast path must pick each row's largest logit and draw no randomness.
        let logits = TestTensor::<2>::from([[0.1, 5.0, 0.2, 0.0], [3.0, 0.1, 0.2, 0.9]]);
        let sampler = LlamaSampler::new(settings(0.0, 0.9));
        assert_eq!(sampler.sample(logits).unwrap(), vec![1, 0]);
    }

    #[test]
    fn a_dominant_token_is_always_drawn() {
        // Token 0 carries more than `p` of the mass on its own, so the nucleus is just {0}: every draw
        // must return 0 whatever the Gumbel noise. Deterministic, so it needs no seed — it pins that
        // the nucleus mask is right and that `-inf` tokens never win.
        let logits = TestTensor::<2>::from([[10.0, 0.0, 0.0, 0.0]]);
        let sampler = LlamaSampler::new(settings(1.0, 0.9));
        for _ in 0..16 {
            assert_eq!(sampler.sample(logits.clone()).unwrap(), vec![0]);
        }
    }

    #[test]
    fn draws_stay_in_the_nucleus_and_actually_vary() {
        // Two tokens dominate (0 and 1, roughly 57/43 of the mass) and the rest are negligible, so the
        // nucleus is {0, 1}. Over many draws every token must be in {0, 1} — the mask is correct — AND
        // both must appear, which proves it genuinely samples rather than collapsing to argmax. The
        // draw is stochastic (the backend's RNG is not seeded here), but with a 57/43 split over 200
        // draws both tokens appearing is a certainty, so the test is not flaky.
        let logits = TestTensor::<2>::from([[1.0, 0.7, -20.0, -20.0]]);
        let sampler = LlamaSampler::new(settings(1.0, 0.95));

        let mut seen = HashSet::new();
        for _ in 0..200 {
            let id = sampler.sample(logits.clone()).unwrap()[0];
            assert!(id == 0 || id == 1, "drew {id} outside the nucleus {{0, 1}}");
            seen.insert(id);
        }
        assert_eq!(
            seen.len(),
            2,
            "top-p collapsed to a single token instead of sampling the nucleus"
        );
    }
}

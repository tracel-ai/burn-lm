//! Per-job sampling settings: the server's sampling config, with the one per-request knob
//! (`max_tokens`) applied on top.
//!
//! Both generation paths resolve through here — the batching worker (via `sampler`/admission) and
//! the single-request `run_job` path — so a request means the same thing whichever channel serves
//! it, and neither path mutates shared server config. Sampling itself is config-driven: temperature
//! and top-p come from the server config; the only thing a request carries is its token cap. (The
//! `seed` field is also config-driven but currently inert — see `SamplingSettings::seed`.)

use burn_lm_inference::GenerationParams;
use rand::RngExt;

/// The effective sampling settings for one job: the server's sampling config defaults with the
/// per-job `max_tokens` cap applied. Built fresh per job, never mutating shared server config, so
/// concurrent requests cannot clobber each other.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingSettings {
    pub top_p: f64,
    pub temperature: f64,
    pub sample_len: usize,
    /// The configured sampling seed (`0` historically meant "draw a fresh random seed per job").
    ///
    /// This is currently inert: the device-side sampler draws its randomness from the tensor
    /// backend's own RNG, not from this value, so nothing in production reads it. Reproducibility
    /// today comes from seeding the backend RNG (`Device::seed`). The field is kept because it is an
    /// existing server-config knob, and the planned follow-up is to wire it through to a backend seed
    /// at a generation entry point.
    pub seed: u64,
}

impl SamplingSettings {
    /// Apply per-job `params` over the server config `defaults`. Only `max_tokens` is per-request,
    /// and it can only lower the configured `sample_len`, never raise it, so the operator-set cap
    /// stays authoritative. Sampling fields (top-p, temperature, seed) always come from the config.
    pub fn resolve(defaults: Self, params: &GenerationParams) -> Self {
        Self {
            top_p: defaults.top_p,
            temperature: defaults.temperature,
            sample_len: params
                .max_tokens
                .map_or(defaults.sample_len, |m| m.min(defaults.sample_len)),
            seed: defaults.seed,
        }
    }

    /// The seed to actually use: `0` draws a fresh random seed per job.
    ///
    /// Not yet wired into sampling — the device-side sampler seeds nothing from here (see the `seed`
    /// field). This is exercised only by its unit test, kept ready for the follow-up that seeds the
    /// backend RNG from the config seed.
    pub fn effective_seed(&self) -> u64 {
        match self.seed {
            0 => rand::rng().random::<u64>(),
            s => s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> SamplingSettings {
        SamplingSettings {
            top_p: 0.9,
            temperature: 0.0,
            sample_len: 4096,
            seed: 0,
        }
    }

    #[test]
    fn unset_params_fall_back_to_config_defaults() {
        let resolved = SamplingSettings::resolve(defaults(), &GenerationParams::default());
        assert_eq!(resolved, defaults());
    }

    #[test]
    fn max_tokens_lowers_but_cannot_raise_the_configured_cap() {
        let lowered = SamplingSettings::resolve(
            defaults(),
            &GenerationParams {
                max_tokens: Some(8),
            },
        );
        assert_eq!(lowered.sample_len, 8);

        let raised = SamplingSettings::resolve(
            defaults(),
            &GenerationParams {
                max_tokens: Some(1_000_000),
            },
        );
        assert_eq!(
            raised.sample_len, 4096,
            "request cannot exceed the server cap"
        );
    }

    #[test]
    fn sampling_fields_always_come_from_config_not_the_request() {
        // The only per-request knob is `max_tokens`; top-p, temperature, and seed are config-driven,
        // so a request cannot move them off the configured values.
        let resolved = SamplingSettings::resolve(
            defaults(),
            &GenerationParams {
                max_tokens: Some(8),
            },
        );
        assert_eq!(resolved.top_p, 0.9);
        assert_eq!(resolved.temperature, 0.0);
        assert_eq!(resolved.seed, 0);
    }

    #[test]
    fn explicit_seed_is_reproducible_and_zero_seed_is_random() {
        let fixed = SamplingSettings {
            seed: 42,
            ..defaults()
        };
        assert_eq!(fixed.effective_seed(), 42);
        let random = SamplingSettings {
            seed: 0,
            ..defaults()
        };
        // Astronomically unlikely to collide twice; assert two draws differ to show 0 is not
        // used literally.
        let (a, b) = (random.effective_seed(), random.effective_seed());
        assert!(a != 0 || b != 0);
    }
}

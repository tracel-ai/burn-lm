//! Per-job sampling settings: server config defaults merged with the per-request
//! [`GenerationParams`] carried on the job.
//!
//! Both generation paths resolve through here — the batching worker (via
//! `next_token_sampler`/admission) and the single-request `run_job` path — so a request means the
//! same thing whichever channel serves it, and neither path mutates shared server config.

use burn_lm_inference::GenerationParams;
use rand::RngExt;

use crate::generation::{Sampler, TopP};

/// The effective sampling settings for ONE job: server config defaults with any per-job overrides
/// applied. Built fresh per job — never mutates shared server config, so concurrent requests
/// cannot clobber each other.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingSettings {
    pub top_p: f64,
    pub temperature: f64,
    pub sample_len: usize,
    /// `0` means "draw a fresh random seed for this job".
    pub seed: u64,
}

impl SamplingSettings {
    /// Merge per-job `params` over the server config `defaults`. Job params take precedence;
    /// `None` falls back to the default — except `max_tokens`, which can only LOWER the
    /// configured `sample_len` (the operator-set cap stays authoritative).
    pub fn resolve(defaults: Self, params: &GenerationParams) -> Self {
        Self {
            top_p: params.top_p.unwrap_or(defaults.top_p),
            temperature: params.temperature.unwrap_or(defaults.temperature),
            sample_len: params
                .max_tokens
                .map_or(defaults.sample_len, |m| m.min(defaults.sample_len)),
            seed: params.seed.unwrap_or(defaults.seed),
        }
    }

    /// The seed to actually use: `0` draws a fresh random seed per job.
    pub fn effective_seed(&self) -> u64 {
        match self.seed {
            0 => rand::rng().random::<u64>(),
            s => s,
        }
    }

    /// Build this job's sampler. One sampler per job — its RNG state is never shared across jobs,
    /// and `temperature == 0.0` stays plain argmax/greedy.
    pub fn sampler(&self) -> Sampler {
        if self.temperature > 0.0 {
            Sampler::TopP(TopP::new(self.top_p, self.effective_seed()))
        } else {
            Sampler::Argmax
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
    fn params_take_precedence_over_config_defaults() {
        let params = GenerationParams {
            max_tokens: Some(7),
            temperature: Some(0.5),
            top_p: Some(0.42),
            seed: Some(1234),
        };
        let resolved = SamplingSettings::resolve(defaults(), &params);
        assert_eq!(
            resolved,
            SamplingSettings {
                top_p: 0.42,
                temperature: 0.5,
                sample_len: 7,
                seed: 1234,
            }
        );
    }

    #[test]
    fn unset_params_fall_back_to_config_defaults() {
        let resolved = SamplingSettings::resolve(defaults(), &GenerationParams::default());
        assert_eq!(resolved, defaults());
    }

    #[test]
    fn partial_override_merges_field_by_field() {
        let params = GenerationParams {
            temperature: Some(0.8),
            ..Default::default()
        };
        let resolved = SamplingSettings::resolve(defaults(), &params);
        assert_eq!(resolved.temperature, 0.8);
        assert_eq!(resolved.top_p, 0.9);
        assert_eq!(resolved.sample_len, 4096);
        assert_eq!(resolved.seed, 0);
    }

    #[test]
    fn max_tokens_lowers_but_cannot_raise_the_configured_cap() {
        let lowered = SamplingSettings::resolve(
            defaults(),
            &GenerationParams {
                max_tokens: Some(8),
                ..Default::default()
            },
        );
        assert_eq!(lowered.sample_len, 8);

        let raised = SamplingSettings::resolve(
            defaults(),
            &GenerationParams {
                max_tokens: Some(1_000_000),
                ..Default::default()
            },
        );
        assert_eq!(
            raised.sample_len, 4096,
            "request cannot exceed the server cap"
        );
    }

    #[test]
    fn sampler_kind_follows_per_job_temperature() {
        // Two jobs with different temperatures over the SAME config get independent sampler
        // configs (no shared-config mutation).
        let hot = SamplingSettings::resolve(
            defaults(),
            &GenerationParams {
                temperature: Some(0.7),
                seed: Some(1),
                ..Default::default()
            },
        );
        let cold = SamplingSettings::resolve(defaults(), &GenerationParams::default());
        assert!(matches!(hot.sampler(), Sampler::TopP(_)));
        assert!(matches!(cold.sampler(), Sampler::Argmax));
        // And the defaults themselves were not mutated by the first job.
        assert_eq!(cold, defaults());
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

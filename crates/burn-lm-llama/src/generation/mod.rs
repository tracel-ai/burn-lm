#[cfg(test)]
mod batched_equivalence;
mod context;
mod generate;
pub(crate) mod sampling;
mod streaming;

pub use context::*;
pub use generate::*;
pub use streaming::*;

// The config-driven sampler is the only public item in `sampling`, and it only exists when the
// server features that build `SamplingSettings` (and pull in the framework `Sampler` trait) are on.
// Re-export it under the same gate so the glob doesn't pull in zero public items the rest of the
// time. Everything else in `sampling` is `pub(crate)` and reached as `sampling::...`.
#[cfg(all(
    feature = "inference-server",
    any(feature = "llama3", feature = "tiny")
))]
pub use sampling::LlamaSampler;

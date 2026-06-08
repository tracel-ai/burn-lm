#[cfg(any(feature = "llama3", feature = "tiny"))]
mod loaded_model;

#[cfg(feature = "llama3")]
pub mod llama3;

#[cfg(feature = "tiny")]
pub mod tiny;

use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
    },
    tensor::{Device, Int, Tensor},
};

use crate::nn::{
    attention::*,
    fftn::{FeedForward, FeedForwardConfig},
};

/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config, Debug)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
}

impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init(&self, device: &Device) -> Transformer {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}

impl TransformerConfig {
    /// This transformer's KV shape, for building its paged cache: the cache cares about layer
    /// count, KV-head geometry, and the context window — nothing else about the model.
    pub fn kv_layout(&self) -> KvLayout {
        KvLayout {
            n_layers: self.n_layers,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.d_model / self.n_heads,
            max_seq_len: self.max_seq_len,
        }
    }
}

/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer {
    pub tok_embeddings: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
    // TODO: tied weights, helps with reduced memory
    pub output: Linear,
}

impl Transformer {
    /// Advance the active lanes through every layer by `seq_len` tokens and return their logits,
    /// shaped `[n, seq_len, vocab]`. All lanes in one call share the same input length (one token for
    /// a decode round, the prompt length for a prefill); they differ only in where each one sits in
    /// the KV buffer, which the `plan` carries as the per-lane start position and mask. The plan is
    /// built by `PagedKvCache::prepare_lanes` and described on `LanePlan`.
    pub fn forward_lanes(
        &self,
        input: Tensor<2, Int>,
        cache: &mut PagedKvCache,
        rope: &RotaryEncoding,
        plan: &LanePlan,
    ) -> Tensor<3> {
        // TEMP PROFILING (BURN_LM_PROFILE_OPS=1): the top-level phases, force-synced — in
        // particular the LM HEAD, the [2048 x 128256] projection the per-layer profile never
        // covered.
        let profile = std::env::var_os("BURN_LM_PROFILE_OPS").is_some();
        // A slice-read can be NARROWED by the lazy engine (it computes only the sliced element,
        // as the head's impossible 0.0ms proved). A full-tensor sum cannot — every element must
        // exist — so this forces true materialization at the phase boundary.
        let sync3 = |t: &Tensor<3>| {
            let _: f32 = t.clone().sum().into_scalar();
        };
        let t0 = std::time::Instant::now();
        let mut h = self.tok_embeddings.forward(input);
        let embed_us = if profile { sync3(&h); t0.elapsed().as_micros() as u64 } else { 0 };

        let t1 = std::time::Instant::now();
        for (layer, c) in self.layers.iter().zip(cache.layers_mut()) {
            h = layer.forward_lanes(h, c, rope, plan);
        }
        let layers_us = if profile { sync3(&h); t1.elapsed().as_micros() as u64 } else { 0 };

        let t2 = std::time::Instant::now();
        let h = self.norm.forward(h);
        let norm_us = if profile { sync3(&h); t2.elapsed().as_micros() as u64 } else { 0 };

        let t3 = std::time::Instant::now();
        // EXPERIMENT (head-only flatten): hand the LM head a 2-D [n·seq, d] activation so the
        // vocab projection is ONE GEMM sharing a single weight-stream across lanes, instead of a
        // batched matmul with M=1 rows that re-streams the 1.05 GB weight per lane.
        let [n_, seq_, d_] = h.dims();
        let out = self.output.forward(h.reshape([n_ * seq_, d_]));
        let vocab = out.dims()[1];
        let out = out.reshape([n_, seq_, vocab]);
        if profile {
            sync3(&out);
            tracing::debug!(
                target: "batching",
                n = out.dims()[0],
                embed_us,
                layers_us,
                norm_us,
                head_us = t3.elapsed().as_micros() as u64,
                "phase-top"
            );
        }
        out
    }

    /// Forward with non-autoregressive and creates a mask for training.
    pub fn forward_train(&self, input: Tensor<2, Int>, rope: &RotaryEncoding) -> Tensor<3> {
        let mut h = self.tok_embeddings.forward(input);

        for layer in self.layers.iter() {
            h = layer.forward_train(h, rope);
        }

        let h = self.norm.forward(h);
        self.output.forward(h)
    }
}

/// Configuration to create a [decoder-only transformer block](TransformerBlock).
#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Initialize a new [decoder-only transformer block](TransformerBlock).
    pub fn init(&self, device: &Device) -> TransformerBlock {
        let attention =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_kv_heads).init(device);
        let feed_forward = FeedForwardConfig::new(self.d_model, self.hidden_size).init(device);
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }
}

/// Decoder-only transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock {
    // The fields are `pub(crate)` so the batched-equivalence test harness
    // (`generation/batched_equivalence.rs`) can hand-roll a per-lane forward pass with the production
    // weights and check the batched path against it.
    /// Self-attention.
    pub(crate) attention: MultiHeadAttention,
    /// Feed-forward transformation.
    pub(crate) feed_forward: FeedForward,
    /// Attention pre-normalization.
    pub(crate) attention_norm: RmsNorm,
    /// Feed-forward pre-normalization.
    pub(crate) ffn_norm: RmsNorm,
}

impl TransformerBlock {
    /// One block of the lane-aware forward: pre-norm, the cached per-lane attention, and the
    /// feed-forward, each wrapped in its residual. The `plan` carries the per-lane RoPE positions, KV
    /// offsets, and mask through to the attention layer.
    pub fn forward_lanes(
        &self,
        input: Tensor<3>,
        cache: &mut KeyValueCache,
        rope: &RotaryEncoding,
        plan: &LanePlan,
    ) -> Tensor<3> {
        let h = input.clone()
            + self.attention.forward_cache_lanes(
                self.attention_norm.forward(input),
                cache,
                rope,
                plan,
            );
        // TEMP PROFILING: the FFN half of the block, force-synced (see forward_cache_lanes).
        if std::env::var_os("BURN_LM_PROFILE_OPS").is_some() {
            let t = std::time::Instant::now();
            let out = h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h));
            let _ = out.clone().slice([0..1, 0..1, 0..1]).into_data();
            tracing::debug!(target: "batching", ffn_us = t.elapsed().as_micros() as u64, "phase-ffn");
            return out;
        }
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }

    /// Forward with non-autoregressive and a required mask for training.
    pub fn forward_train(&self, input: Tensor<3>, rope: &RotaryEncoding) -> Tensor<3> {
        let h = input.clone()
            + self
                .attention
                .forward_masked(self.attention_norm.forward(input), rope);
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    use burn::tensor::{TensorData, Tolerance};

    #[test]
    fn test_rms_norm() {
        let device = Default::default();

        let rms = RmsNormConfig::new(4).with_epsilon(1e-5).init(&device);
        let input = TestTensor::<3>::from([[
            [0.0025997162, 0.0030002594, -0.006000519, 0.006000519],
            [0.0010004044, 0.00080013275, 0.0015001297, -0.01600647],
        ]]);

        let output = rms.forward(input);
        let expected = TensorData::from([[
            [0.45996094, 0.5307617, -1.0615234, 1.0615234],
            [0.11553955, 0.09240723, 0.17321777, -1.8486328],
        ]]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }
}

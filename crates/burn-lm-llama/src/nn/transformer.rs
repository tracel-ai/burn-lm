use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
    },
    tensor::{Bool, Device, Int, Tensor},
};

use crate::{
    generation::GenerationError,
    nn::{
        attention::*,
        fftn::{FeedForward, FeedForwardConfig},
    },
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
    /// Lane-aware forward: advance the given lanes (uniform input length) through the shared slab,
    /// using each lane's own start position for RoPE and KV writes and the per-lane mask, then
    /// returns logits `[n, seq_len, vocab]`. See [`LanePlan`] / [`TransformerCache::prepare_lanes`].
    pub fn forward_lanes(
        &self,
        input: Tensor<2, Int>,
        cache: &mut TransformerCache,
        rope: &RotaryEncoding,
        plan: &LanePlan,
    ) -> Tensor<3> {
        let mut h = self.tok_embeddings.forward(input);

        for (layer, c) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward_lanes(h, c, rope, plan);
        }

        let h = self.norm.forward(h);
        self.output.forward(h)
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

/// One lane-aware forward, planned by [`TransformerCache::prepare_lanes`]:
/// which buffer lanes participate, each lane's start position (RoPE + KV
/// write offset), and the per-lane causal+padding mask over the shared
/// KV buffer.
#[derive(Debug)]
pub struct LanePlan {
    /// Active buffer lanes, one per batch row of the forward input.
    pub lanes: Vec<usize>,
    /// Each lane's sequence length BEFORE this forward (its write offset and
    /// the absolute RoPE position of its first new token).
    pub starts: Vec<usize>,
    /// `[n, 1, q, l_max]` bool mask, `true` = masked: row `r` of lane `j`
    /// attends to columns `0..=starts[j] + r`; everything past that (the
    /// lane's own future and the buffer tail up to the longest active lane)
    /// is masked. The attention op turns masked positions into negative
    /// infinity before the softmax. Unlike the broadcast tril of the
    /// single-sequence path, decode (`q == 1`) is masked too — ragged lanes
    /// make it mandatory.
    pub mask: Tensor<4, Bool>,
}

#[derive(Clone, Debug)]
pub struct TransformerCache {
    layers: Vec<KeyValueCache>,
    device: Device,
    max_seq_len: usize,
    /// Per-lane bookkeeping (`prepare_lanes`/`reset_lane`) — the production decode path. One entry
    /// per slab lane. Must stay in lockstep with each layer's KV `lane_len` (see `prepare_lanes`).
    lens: Vec<usize>,
}

impl TransformerCache {
    pub fn new(config: &TransformerConfig, max_batch_size: usize, device: &Device) -> Self {
        let cache = (0..config.n_layers)
            .map(|_| {
                KeyValueCache::new(
                    max_batch_size,
                    config.n_kv_heads,
                    config.max_seq_len,
                    config.d_model / config.n_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        Self {
            layers: cache,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            lens: vec![0; max_batch_size],
        }
    }

    /// Number of buffer lanes (the model's `max_batch_size`).
    pub fn lane_count(&self) -> usize {
        self.lens.len()
    }

    /// Sequence length of one lane.
    pub fn lane_len(&self, lane: usize) -> usize {
        self.lens[lane]
    }

    /// Debug self-check: this cache's per-lane lengths (which drive RoPE positions + the mask) agree
    /// with each layer's KV write offset for the given lanes. The two advance together on every
    /// lane forward and reset together, so they must match before a forward; this makes that
    /// otherwise convention-only invariant self-checking at the production entry point.
    pub fn lanes_in_lockstep(&self, lanes: &[usize]) -> bool {
        lanes.iter().all(|&lane| {
            self.layers
                .iter()
                .all(|layer| layer.lane_len(lane) == self.lens[lane])
        })
    }

    /// Plan one lane-aware forward of `seq_len` new tokens for `lanes`:
    /// validate per-lane capacity, snapshot each lane's start position, build
    /// the per-lane mask, and advance the lane lengths. This is the production
    /// decode path (prefill = one lane; fused decode = the active lanes).
    ///
    /// There is NO Shift eviction in lane mode: a lane that would exceed
    /// `max_seq_len` is an error — the runtime finishes lanes that hit their
    /// token budget before they get here.
    pub fn prepare_lanes(
        &mut self,
        lanes: &[usize],
        seq_len: usize,
    ) -> Result<LanePlan, GenerationError> {
        // Local self-checking invariants the equivalence rests on (debug-only, cheap):
        // - lanes are DISTINCT: a duplicate would double-advance the bookkeeper while the per-row
        //   RoPE/mask used a stale start, silently corrupting that lane's attention.
        // - lanes are in range: a slot past the slab would otherwise be a raw index panic.
        // (The cross-component lockstep check — this cache's lens vs each layer's KV write offset —
        // lives in `LlamaDecoder::forward_lanes`, the production entry point, because `prepare_lanes`
        // is also exercised standalone by unit tests that never run a layer forward.)
        debug_assert!(
            lanes.iter().collect::<std::collections::HashSet<_>>().len() == lanes.len(),
            "prepare_lanes got a duplicate lane: {lanes:?}"
        );
        debug_assert!(
            lanes.iter().all(|&lane| lane < self.lens.len()),
            "prepare_lanes got a lane >= lane_count ({}): {lanes:?}",
            self.lens.len()
        );

        for &lane in lanes {
            if self.lens[lane] + seq_len > self.max_seq_len {
                return Err(GenerationError::MaxSequenceLengthExceeded {
                    actual: self.lens[lane] + seq_len,
                    max: self.max_seq_len,
                });
            }
        }

        let starts: Vec<usize> = lanes.iter().map(|&lane| self.lens[lane]).collect();
        let n = lanes.len();
        let l_max = starts.iter().map(|s| s + seq_len).max().expect("n >= 1");

        // Host-built per-lane causal + padding mask, `true` = masked.
        let mut mask_data = Vec::with_capacity(n * seq_len * l_max);
        for s in starts.iter() {
            for r in 0..seq_len {
                for c in 0..l_max {
                    mask_data.push(c > s + r);
                }
            }
        }
        let mask = Tensor::<4, Bool>::from_data(
            burn::tensor::TensorData::new(mask_data, [n, 1, seq_len, l_max]),
            &self.device,
        );

        for &lane in lanes {
            self.lens[lane] += seq_len;
        }

        Ok(LanePlan {
            lanes: lanes.to_vec(),
            starts,
            mask,
        })
    }

    /// Reset every lane: zero all lane lengths in this bookkeeping AND in every layer's KV cache.
    /// Used between independent generations.
    pub fn reset(&mut self) {
        for lane in 0..self.lens.len() {
            self.reset_lane(lane);
        }
    }

    /// Free one lane: zero its length in this bookkeeping AND in every
    /// layer's KV cache. The buffer row is overwritten on the next use.
    ///
    /// Releasing a lane outside the slab is a no-op (there is nothing to free): a defensive guard so
    /// a slot the slab never had — e.g. if `config.max_slots` were raised above the loaded lane
    /// count — cannot index the fixed-length lane vector out of bounds and panic. Admission also
    /// caps slots at the slab's `lane_count` (see `batch_capacity`), so this is belt-and-suspenders.
    pub fn reset_lane(&mut self, lane: usize) {
        if lane >= self.lens.len() {
            return;
        }
        self.lens[lane] = 0;
        self.layers
            .iter_mut()
            .for_each(|cache| cache.reset_lane(lane));
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
    // NOTE: fields are `pub(crate)` so the batched-equivalence characterization
    // harness (`generation/batched_equivalence.rs`, phase-2 S4 gate) can
    // hand-roll a per-lane forward pass with the production weights.
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
    /// Lane-aware forward: per-lane RoPE positions, lane-sliced KV,
    /// and the per-lane mask from [`LanePlan`].
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

    use burn::{
        nn::RotaryEncodingConfig,
        tensor::{TensorData, Tolerance},
    };

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

    fn lane_test_config(max_seq_len: usize) -> TransformerConfig {
        TransformerConfig::new(8, 2, 4, 4, 2, 1).with_max_seq_len(max_seq_len)
    }

    fn mask_rows(mask: &Tensor<4, Bool>) -> Vec<bool> {
        mask.clone().into_data().iter::<bool>().collect()
    }

    /// Decode step for two lanes at divergent positions: each lane's mask row
    /// allows exactly its own history plus the new token, and masks the
    /// buffer tail up to the longest active lane.
    #[test]
    fn test_prepare_lanes_decode_mask_covers_exactly_the_dead_columns() {
        let config = lane_test_config(8);
        let mut cache = TransformerCache::new(&config, 2, &Default::default());

        // Lane 0 holds 3 positions, lane 1 holds 1.
        cache.prepare_lanes(&[0], 3).unwrap();
        cache.prepare_lanes(&[1], 1).unwrap();
        assert_eq!(cache.lane_len(0), 3);
        assert_eq!(cache.lane_len(1), 1);

        // Fused decode: one new token per lane; l_max = 4.
        let plan = cache.prepare_lanes(&[0, 1], 1).unwrap();
        assert_eq!(plan.lanes, vec![0, 1]);
        assert_eq!(plan.starts, vec![3, 1]);
        assert_eq!(plan.mask.dims(), [2, 1, 1, 4]);
        // Lane 0 attends to columns 0..=3 (nothing masked); lane 1 attends to
        // columns 0..=1 and columns 2..4 (stale tail) are masked.
        assert_eq!(
            mask_rows(&plan.mask),
            vec![false, false, false, false, false, false, true, true]
        );
        assert_eq!(cache.lane_len(0), 4);
        assert_eq!(cache.lane_len(1), 2);
    }

    /// Single-lane prefill: the per-lane mask reduces to the ordinary causal
    /// triangle over the lane's own (empty) history.
    #[test]
    fn test_prepare_lanes_prefill_mask_is_causal() {
        let config = lane_test_config(8);
        let mut cache = TransformerCache::new(&config, 2, &Default::default());

        let plan = cache.prepare_lanes(&[1], 3).unwrap();
        assert_eq!(plan.starts, vec![0]);
        assert_eq!(plan.mask.dims(), [1, 1, 3, 3]);
        assert_eq!(
            mask_rows(&plan.mask),
            vec![false, true, true, false, false, true, false, false, false]
        );
    }

    /// A lane that would exceed its capacity is an error; lane mode has no
    /// Shift eviction.
    #[test]
    fn test_prepare_lanes_exceeded_max_seq_len() {
        let config = lane_test_config(4);
        let mut cache = TransformerCache::new(&config, 2, &Default::default());

        cache.prepare_lanes(&[0], 3).unwrap();
        // Lane 1 is fine on its own...
        cache.prepare_lanes(&[1], 1).unwrap();
        // ...but lane 0 cannot take 2 more positions.
        assert!(matches!(
            cache.prepare_lanes(&[0, 1], 2),
            Err(GenerationError::MaxSequenceLengthExceeded { actual: 5, max: 4 })
        ));
        // A failed plan advances nothing.
        assert_eq!(cache.lane_len(0), 3);
        assert_eq!(cache.lane_len(1), 1);
    }

    /// `reset_lane` zeroes one lane's bookkeeping and every layer's KV length
    /// for that lane, leaving the other lane untouched.
    #[test]
    fn test_reset_lane_isolates_one_lane() {
        let device: Device = Default::default();
        let config = lane_test_config(8);
        let mut cache = TransformerCache::new(&config, 2, &device);
        let head_dim = config.d_model / config.n_heads;

        // Write real KV rows so the per-layer lane lengths advance too.
        let write = |cache: &mut TransformerCache, lanes: &[usize], seq_len: usize| {
            let x = Tensor::ones([lanes.len(), config.n_kv_heads, seq_len, head_dim], &device);
            cache.prepare_lanes(lanes, seq_len).unwrap();
            for layer in cache.layers.iter_mut() {
                layer.forward_lanes(lanes, x.clone(), x.clone());
            }
        };

        write(&mut cache, &[0], 3);
        write(&mut cache, &[1], 1);
        write(&mut cache, &[0, 1], 1);
        assert_eq!(cache.lane_len(0), 4);
        assert_eq!(cache.lane_len(1), 2);
        for layer in cache.layers.iter() {
            assert_eq!(layer.lane_len(0), 4);
            assert_eq!(layer.lane_len(1), 2);
        }

        cache.reset_lane(0);
        assert_eq!(cache.lane_len(0), 0);
        assert_eq!(cache.lane_len(1), 2);
        for layer in cache.layers.iter() {
            assert_eq!(layer.lane_len(0), 0);
            assert_eq!(layer.lane_len(1), 2);
        }
    }

    /// Releasing a lane outside the slab is a no-op, not a panic — defends against a
    /// `config.max_slots` raised above the loaded lane count handing admission an out-of-range slot.
    #[test]
    fn test_reset_lane_out_of_range_is_a_noop() {
        let config = lane_test_config(8);
        let mut cache = TransformerCache::new(&config, 2, &Default::default());
        cache.prepare_lanes(&[0], 3).unwrap();
        cache.reset_lane(5); // 5 >= lane_count 2 — must not panic
        assert_eq!(cache.lane_count(), 2);
        assert_eq!(cache.lane_len(0), 3, "an in-range lane is untouched");
    }
}

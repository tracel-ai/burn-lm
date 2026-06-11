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

use super::pos_encoding::PositionalEncodingState;

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
    pub fn forward(
        &self,
        input: Tensor<2, Int>,
        cache: &mut TransformerCache,
        pos_encoding: &PositionalEncodingState,
        mask: Option<Tensor<4, Bool>>,
    ) -> Tensor<3> {
        let mut h = self.tok_embeddings.forward(input);

        for (layer, c) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(h, c, pos_encoding, mask.clone());
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
    /// Single-sequence bookkeeping (`prepare`/`reset`).
    curr_seq_len: usize,
    /// Per-lane bookkeeping (`prepare_lanes`/`reset_lane`). A cache instance
    /// uses one mode or the other, never both; the decoder switch-over onto
    /// the lane-aware path lands in the next change.
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
            curr_seq_len: 0,
            lens: vec![0; max_batch_size],
        }
    }

    /// Number of buffer lanes (the model's `max_batch_size`).
    // The decoder switch-over onto the lane-aware path lands in the next change.
    #[allow(dead_code)]
    pub fn lane_count(&self) -> usize {
        self.lens.len()
    }

    /// Sequence length of one lane.
    // The decoder switch-over onto the lane-aware path lands in the next change.
    #[allow(dead_code)]
    pub fn lane_len(&self, lane: usize) -> usize {
        self.lens[lane]
    }

    /// Plan one lane-aware forward of `seq_len` new tokens for `lanes`:
    /// validate per-lane capacity, snapshot each lane's start position, build
    /// the per-lane mask, and advance the lane lengths.
    ///
    /// There is NO Shift eviction in lane mode: a lane that would exceed
    /// `max_seq_len` is an error — the runtime finishes lanes that hit their
    /// token budget before they get here.
    // The decoder switch-over onto the lane-aware path lands in the next change.
    #[allow(dead_code)]
    pub fn prepare_lanes(
        &mut self,
        lanes: &[usize],
        seq_len: usize,
    ) -> Result<LanePlan, GenerationError> {
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

    pub fn prepare(&mut self, seq_len: usize) -> Result<Option<Tensor<4, Bool>>, GenerationError> {
        if seq_len > self.max_seq_len {
            return Err(GenerationError::MaxSequenceLengthExceeded {
                actual: seq_len,
                max: self.max_seq_len,
            });
        }

        self.curr_seq_len += seq_len;
        if self.curr_seq_len > self.max_seq_len {
            let num_removed = self.curr_seq_len - self.max_seq_len;
            self.layers
                .iter_mut()
                .for_each(|cache| cache.prepare(num_removed));
            self.curr_seq_len -= num_removed;
        }

        Ok(self.mask_attn(seq_len))
    }

    fn mask_attn(&self, seq_len: usize) -> Option<Tensor<4, Bool>> {
        if seq_len <= 1 {
            return None;
        }

        let mask = Tensor::<2, Bool>::tril_mask(
            [seq_len, self.curr_seq_len],
            (self.curr_seq_len - seq_len) as i64, // offset
            &self.device,
        );

        Some(mask.unsqueeze::<4>())
    }

    pub fn reset(&mut self) {
        self.curr_seq_len = 0;
        self.lens.iter_mut().for_each(|len| *len = 0);
        self.layers.iter_mut().for_each(|cache| cache.reset());
    }

    /// Free one lane: zero its length in this bookkeeping AND in every
    /// layer's KV cache. The buffer row is overwritten on the next use.
    // The decoder switch-over onto the lane-aware path lands in the next change.
    #[allow(dead_code)]
    pub fn reset_lane(&mut self, lane: usize) {
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
    pub fn forward(
        &self,
        input: Tensor<3>,
        cache: &mut KeyValueCache,
        pos_encoding: &PositionalEncodingState,
        mask: Option<Tensor<4, Bool>>,
    ) -> Tensor<3> {
        let h = input.clone()
            + self.attention.forward_cache(
                self.attention_norm.forward(input),
                cache,
                pos_encoding,
                mask,
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

    #[test]
    fn test_transformer() {
        let device: Device = Default::default();
        let config = TransformerConfig::new(8, 2, 8, 16, 2, 1);
        let transformer: Transformer = config.init(&device);

        let batch_size = 2;
        let seq_length = 2;

        let mut cache = TransformerCache::new(&config, batch_size, &device);

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);
        let rope = PositionalEncodingState::new(rope);

        let input = Tensor::arange(0..(batch_size * seq_length) as i64, &device)
            .reshape([batch_size, seq_length]);

        let transformer = Reinitializer::default()
            .range_float(0.0, 5.0)
            .apply(transformer);

        let mask = cache.prepare(seq_length).unwrap();
        let output = transformer.forward(input, &mut cache, &rope, mask);

        let expected = TensorData::from([
            [
                [
                    56.37573, 57.77283, 59.169933, 60.567043, 61.964146, 63.361248, 64.758354,
                    66.15546,
                ],
                [
                    56.374626, 57.77171, 59.168793, 60.56588, 61.962963, 63.360046, 64.75713,
                    66.15422,
                ],
            ],
            [
                [
                    56.374252, 57.771328, 59.168407, 60.565487, 61.962566, 63.359642, 64.75672,
                    66.1538,
                ],
                [
                    56.37408, 57.771156, 59.168232, 60.565304, 61.96238, 63.359455, 64.75653,
                    66.15361,
                ],
            ],
        ]);
        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.001));
    }

    pub struct ForwardCacheTestCase {
        cache: TransformerCache,
        config: TransformerConfig,
        device: Device,
    }

    impl ForwardCacheTestCase {
        fn new(config: TransformerConfig, device: Device) -> Self {
            Self {
                cache: TransformerCache::new(&config, 1, &device),
                config,
                device,
            }
        }

        fn forward_seq(&mut self, seq_len: usize) {
            let x = Tensor::ones(
                [
                    1,
                    self.config.n_kv_heads,
                    seq_len,
                    self.config.d_model / self.config.n_heads,
                ],
                &self.device,
            );
            self.cache.prepare(seq_len).unwrap();
            self.forward(x);
        }

        fn forward(&mut self, x: Tensor<4>) {
            for cache in self.cache.layers.iter_mut() {
                // - input:  `[batch_size, num_heads, seq_len_input, d_model]`
                // - output: `[batch_size, num_heads, seq_len_previous + seq_len_input, d_model]`
                cache.forward(x.clone(), x.clone());
            }
        }

        fn assert_eq_cache_len(&self, len: usize) {
            for cache in self.cache.layers.iter() {
                assert_eq!(cache.len(), len);
            }
        }
    }

    #[test]
    fn test_transformer_cache_should_shrink() {
        let max_seq_len = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let d_model = 4;
        let config = TransformerConfig::new(8, 2, d_model, 4, num_heads, num_kv_heads)
            .with_max_seq_len(max_seq_len);

        let mut model = ForwardCacheTestCase::new(config, Default::default());
        assert_eq!(model.cache.max_seq_len, max_seq_len);
        assert_eq!(model.cache.curr_seq_len, 0);

        let seq_len = 4;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, seq_len);
        model.assert_eq_cache_len(seq_len);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, 5);
        model.assert_eq_cache_len(5);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, 6);
        model.assert_eq_cache_len(6);

        // Shrink: any subsequent calls will shift the cache and have `max_seq_len`
        let seq_len = 6;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, max_seq_len);
        model.assert_eq_cache_len(max_seq_len);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, max_seq_len);
        model.assert_eq_cache_len(max_seq_len);

        let seq_len = 1;
        model.forward_seq(seq_len);
        assert_eq!(model.cache.curr_seq_len, max_seq_len);
        model.assert_eq_cache_len(max_seq_len);
    }

    #[test]
    fn test_transformer_cache_exceeded_max_seq_len() {
        let max_seq_len = 8;
        let num_heads = 2;
        let num_kv_heads = 1;
        let d_model = 4;
        let config = TransformerConfig::new(8, 2, d_model, 4, num_heads, num_kv_heads)
            .with_max_seq_len(max_seq_len);
        let mut cache = TransformerCache::new(&config, 1, &Default::default());

        // When the previous inputs and generated tokens are accumulated and provided as context
        // with a new input, or the input sequence simply exceeds the max_seq_len, the cache should
        // return an error
        assert!(matches!(
            cache.prepare(16),
            Err(GenerationError::MaxSequenceLengthExceeded { actual: 16, max: 8 })
        ));
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
}

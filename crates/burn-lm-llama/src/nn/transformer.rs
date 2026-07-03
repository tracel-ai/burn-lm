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
    /// Advance the active lanes through every layer by `seq_len` tokens and return their logits,
    /// shaped `[n, seq_len, vocab]`. All lanes in one call share the same input length (one token for
    /// a decode round, the prompt length for a prefill); they differ only in where each one sits in
    /// the KV buffer, which the `plan` carries as the per-lane start position and mask. The plan is
    /// built by `TransformerCache::prepare_lanes` and described on `LanePlan`.
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

/// Everything one lane-aware forward needs to know about where its lanes sit, produced by
/// `TransformerCache::prepare_lanes` and consumed unchanged all the way down to the attention layer.
/// It names which buffer lanes take part, each lane's start position (used both as the RoPE position
/// and as the KV write offset), and the per-lane mask over the shared KV buffer.
#[derive(Debug)]
pub struct LanePlan {
    /// The active buffer lanes, one per row of the forward input.
    pub lanes: Vec<usize>,
    /// Each lane's sequence length before this forward. It serves two roles: the absolute RoPE
    /// position of the lane's first new token, and the offset its new tokens are written to in the KV
    /// buffer.
    pub starts: Vec<usize>,
    /// Per active lane, in `lanes` order: the KV block ids covering the lane's positions after this
    /// forward, entry `i` covering positions `[i·block_size, (i+1)·block_size)`. The allocator grew
    /// each table before this plan was built, so every block a layer's write will touch already
    /// exists and the layers allocate nothing. This snapshot is also, deliberately, the handoff
    /// artifact a future prefill/decode split would ship: "prefill output = block list" is a literal
    /// value here.
    pub tables: Vec<Vec<u32>>,
    /// The round's KV gather index, prebuilt from `tables`: `blocks_per_lane` block ids per lane in
    /// position order, short lanes padded with the zeroed sentinel. A pure function of the round, so
    /// it is uploaded once here and every layer's K and V gather through the same handle.
    pub gather_idx: Tensor<1, Int>,
    /// Blocks per lane in `gather_idx`: enough to cover `l_max`.
    pub blocks_per_lane: usize,
    /// The longest active lane's length after this forward — the width of the gathered KV and of
    /// the mask's last dimension.
    pub l_max: usize,
    /// The per-lane attention mask, shaped `[n, 1, q, l_max]`, where `true` means masked. Row `r` of
    /// lane `j` may attend to columns `0..=starts[j] + r`; everything past that is masked off — both
    /// the lane's own future and the stale buffer tail out to the longest active lane. The attention
    /// op turns masked positions into negative infinity before the softmax. Because the lanes are
    /// ragged, even a single-token decode round (`q == 1`) needs this mask to hide each lane's tail,
    /// which the old single-sequence path could skip.
    pub mask: Tensor<4, Bool>,
}

/// The model-owned KV cache for one batch of lanes: the per-layer key/value block stores plus the
/// only length-and-ownership bookkeeping in the system. Each lane is an independent sequence drawing
/// blocks from a shared pool; a lane is a logical index here, mapped to physical blocks by its
/// table. The whole batched decode loop runs against one of these.
#[derive(Clone, Debug)]
pub struct TransformerCache {
    layers: Vec<KeyValueCache>,
    device: Device,
    max_seq_len: usize,
    /// The per-lane ledger: each lane's length and block table, plus the shared free stack — the
    /// only length-and-ownership bookkeeping in the system, in one type so lengths and tables can
    /// never drift apart. One block-id space serves every layer's K and V stores — block `b` names
    /// the same row in all of them.
    pool: BlockPool,
}

/// Tokens per KV block. Small enough that a short sequence's footprint is a few blocks instead of a
/// whole `max_seq_len` stripe (the point of paging), large enough that per-block gather granularity
/// stays cheap — at most `block_size - 1` wasted columns per lane per read. A model with a context
/// window smaller than this just uses one block per lane (the block is clamped to the window).
pub(crate) const DEFAULT_BLOCK_SIZE: usize = 128;

impl TransformerCache {
    pub fn new(config: &TransformerConfig, max_batch_size: usize, device: &Device) -> Self {
        Self::new_with_block_size(
            config,
            max_batch_size,
            DEFAULT_BLOCK_SIZE.min(config.max_seq_len),
            device,
        )
    }

    /// Like [`Self::new`] with an explicit block size — the seam the block-size equivalence tests
    /// drive, and where a measured, per-platform tuning of the default would plug in.
    pub fn new_with_block_size(
        config: &TransformerConfig,
        max_batch_size: usize,
        block_size: usize,
        device: &Device,
    ) -> Self {
        assert!(
            block_size >= 1 && block_size <= config.max_seq_len,
            "block_size must be in 1..=max_seq_len"
        );
        // The pool holds the same tokens as the old rectangle — `max_batch_size × max_seq_len` —
        // just cut into blocks, plus the zeroed sentinel. Decoupling the pool's size from the
        // rectangle (so lanes can oversubscribe it) is a later, engine-side change; the layout
        // stops depending on it here.
        let num_blocks = (max_batch_size * config.max_seq_len).div_ceil(block_size) + 1;
        let cache = (0..config.n_layers)
            .map(|_| {
                KeyValueCache::new(
                    num_blocks,
                    config.n_kv_heads,
                    block_size,
                    config.d_model / config.n_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        Self {
            layers: cache,
            device: device.clone(),
            max_seq_len: config.max_seq_len,
            pool: BlockPool::new(block_size, num_blocks, max_batch_size),
        }
    }

    /// Number of buffer lanes (the model's `max_batch_size`).
    pub fn lane_count(&self) -> usize {
        self.pool.lane_count()
    }

    /// Sequence length of one lane.
    pub fn lane_len(&self, lane: usize) -> usize {
        self.pool.lane_len(lane)
    }

    /// The hard per-lane token capacity (the context window). Lane mode does not evict, so no lane
    /// can hold more than this; the engine reads it to retire a sequence before its next forward
    /// would overflow.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Plan one forward of `seq_len` new tokens over the given lanes, and commit it. It checks each
    /// lane has room, snapshots the lanes' start positions, builds the per-lane mask, advances the
    /// lane lengths, and returns the resulting `LanePlan`. A prefill calls this with a single lane;
    /// a decode round calls it with all the active lanes at once.
    ///
    /// Lane mode does not evict: a lane that would pass `max_seq_len` is an error rather than having
    /// its oldest tokens dropped. The engine retires a sequence with `ContextLengthExceeded` before
    /// its next forward would push a lane past `max_seq_len` (see `step_round`'s classification), so
    /// a live lane never reaches the check below — it stays here as the model-side guard.
    pub fn prepare_lanes(
        &mut self,
        lanes: &[usize],
        seq_len: usize,
    ) -> Result<LanePlan, GenerationError> {
        // The lanes must be in range, or the ledger lookups below would be raw indexing panics.
        // (Duplicate lanes are asserted inside `begin_round`, where a repeat would corrupt the
        // ledger by advancing a length twice.)
        debug_assert!(
            lanes.iter().all(|&lane| lane < self.pool.lane_count()),
            "prepare_lanes got a lane >= lane_count ({}): {lanes:?}",
            self.pool.lane_count()
        );

        for &lane in lanes {
            if self.pool.lane_len(lane) + seq_len > self.max_seq_len {
                return Err(GenerationError::MaxSequenceLengthExceeded {
                    actual: self.pool.lane_len(lane) + seq_len,
                    max: self.max_seq_len,
                });
            }
        }

        // Commit the round in the ledger: one transactional call grows every lane's table and
        // advances every lane's length together — all lanes or none — and hands back the round's
        // start positions. Allocation happens only here, before any layer runs, so the write path
        // downstream never allocates. (While the pool matches the old rectangle it cannot actually
        // run dry — the ceiling check above fires first — but the rollback is the contract an
        // oversubscribed pool inherits.)
        let starts = self
            .pool
            .begin_round(lanes, seq_len)
            .map_err(|exhausted| GenerationError::KvPoolExhausted {
                short_by: exhausted.short_by,
            })?;
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

        // Snapshot each lane's covering blocks into the plan, in `lanes` order — the same order the
        // mask and RoPE rows use — so the layers address KV purely through the plan.
        let tables: Vec<Vec<u32>> = lanes
            .iter()
            .map(|&lane| self.pool.lane_blocks(lane).to_vec())
            .collect();

        // Prebuild the round's gather index — like the mask, a pure per-round artifact: every
        // layer's K and V gather through this one uploaded tensor instead of rebuilding it.
        let blocks_per_lane = l_max.div_ceil(self.pool.block_size());
        let ids: Vec<i32> = tables
            .iter()
            .flat_map(|table| {
                (0..blocks_per_lane).map(|i| *table.get(i).unwrap_or(&SENTINEL_BLOCK) as i32)
            })
            .collect();
        let gather_idx = Tensor::<1, Int>::from_data(
            burn::tensor::TensorData::new(ids, [lanes.len() * blocks_per_lane]),
            &self.device,
        );

        Ok(LanePlan {
            lanes: lanes.to_vec(),
            starts,
            tables,
            gather_idx,
            blocks_per_lane,
            l_max,
            mask,
        })
    }

    /// Reset the whole cache by zeroing every lane's length, reused between independent generations.
    pub fn reset(&mut self) {
        for lane in 0..self.pool.lane_count() {
            self.reset_lane(lane);
        }
    }

    /// Free one lane by zeroing its length. Nothing clears the buffer row itself; the next use of the
    /// lane overwrites it, because the next `prepare_lanes` reads a zeroed start and the KV write
    /// lands at offset 0.
    ///
    /// A lane index past the buffer is silently ignored rather than panicking. This guards against a
    /// slot the buffer never had — for instance if `config.max_slots` were set above the loaded lane
    /// count — reaching here and indexing the fixed-length length vector out of bounds. Admission
    /// already caps slots at `lane_count` (see `batch_capacity`), so this is a second line of defense.
    pub fn reset_lane(&mut self, lane: usize) {
        if lane >= self.pool.lane_count() {
            return;
        }
        self.pool.free_lane(lane);
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

    /// `reset_lane` zeroes one lane's length, leaving the other lane untouched.
    /// With the pool's ledger the single source of length, the freed
    /// lane's next write lands at offset 0 (the KV caches carry no counter).
    #[test]
    fn test_reset_lane_isolates_one_lane() {
        let device: Device = Default::default();
        let config = lane_test_config(8);
        let mut cache = TransformerCache::new(&config, 2, &device);
        let head_dim = config.d_model / config.n_heads;

        // Write real KV rows; the plan's starts come from this cache's lengths.
        let write = |cache: &mut TransformerCache, lanes: &[usize], seq_len: usize| {
            let x = Tensor::ones([lanes.len(), config.n_kv_heads, seq_len, head_dim], &device);
            let plan = cache.prepare_lanes(lanes, seq_len).unwrap();
            for layer in cache.layers.iter_mut() {
                layer.write_lanes(&plan.tables, &plan.starts, x.clone(), x.clone());
            }
        };

        write(&mut cache, &[0], 3);
        write(&mut cache, &[1], 1);
        write(&mut cache, &[0, 1], 1);
        assert_eq!(cache.lane_len(0), 4);
        assert_eq!(cache.lane_len(1), 2);

        cache.reset_lane(0);
        assert_eq!(cache.lane_len(0), 0);
        assert_eq!(cache.lane_len(1), 2);

        // Lane 0 is recycled from position 0; lane 1 keeps growing from its own length.
        write(&mut cache, &[0], 2);
        write(&mut cache, &[1], 1);
        assert_eq!(cache.lane_len(0), 2);
        assert_eq!(cache.lane_len(1), 3);
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

    /// Pool exhaustion mid-round is all-or-nothing across the WHOLE round: when one lane of a
    /// multi-lane `prepare_lanes` cannot get its block, the lanes already grown this round are
    /// unwound too — no length advances, no table grows, and the free stack is exactly as it was.
    /// The production pool (one block per lane) can never run dry, so the test swaps in a smaller
    /// one; smaller block sizes inherit this contract, where exhaustion becomes reachable for real.
    #[test]
    fn test_pool_exhaustion_rolls_back_the_whole_round() {
        let config = lane_test_config(8);
        let mut cache = TransformerCache::new(&config, 3, &Default::default());
        // Three lanes, but a pool with only two usable blocks.
        cache.pool = BlockPool::new(8, 3, 3);

        let err = cache.prepare_lanes(&[0, 1, 2], 4).unwrap_err();
        assert!(
            matches!(err, GenerationError::KvPoolExhausted { short_by: 1 }),
            "expected a one-block shortfall: {err:?}"
        );
        for lane in 0..3 {
            assert_eq!(cache.lane_len(lane), 0, "lane {lane}: no length may survive the rollback");
            assert!(
                cache.pool.lane_blocks(lane).is_empty(),
                "lane {lane}: no block may survive the rollback"
            );
        }
        assert_eq!(cache.pool.free_blocks(), 2, "the free stack is exactly as it was");

        // The pool still serves what fits: two lanes prefill fine after the failed round.
        let plan = cache.prepare_lanes(&[0, 1], 4).unwrap();
        assert_eq!(plan.tables.len(), 2);
        assert!(plan.tables.iter().all(|t| t.len() == 1));
    }
}

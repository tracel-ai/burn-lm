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
    /// The per-lane attention mask, shaped `[n, 1, q, l_max]`, where `true` means masked. Row `r` of
    /// lane `j` may attend to columns `0..=starts[j] + r`; everything past that is masked off — both
    /// the lane's own future and the stale buffer tail out to the longest active lane. The attention
    /// op turns masked positions into negative infinity before the softmax. Because the lanes are
    /// ragged, even a single-token decode round (`q == 1`) needs this mask to hide each lane's tail,
    /// which the old single-sequence path could skip.
    pub mask: Tensor<4, Bool>,
}

/// The model-owned KV cache for one batch of lanes: the per-layer key/value buffers plus the only
/// length bookkeeping in the system. Each lane is an independent sequence sharing the fixed-size
/// buffers, and a lane is addressed by its row index throughout. The whole batched decode loop runs
/// against one of these.
#[derive(Clone, Debug)]
pub struct TransformerCache {
    layers: Vec<KeyValueCache>,
    device: Device,
    max_seq_len: usize,
    /// Each lane's current sequence length, one entry per buffer lane. This is the single source of
    /// truth for lane lengths: `prepare_lanes` reads these into `LanePlan.starts` and the model
    /// threads them down as the KV write offsets, so the underlying KV buffers need no counter of
    /// their own. `prepare_lanes` grows an entry; `reset_lane` zeroes one.
    lens: Vec<usize>,
    /// The block allocator: which KV blocks each lane owns, and the shared free stack. One id space
    /// serves every layer's K and V pools — block `b` names the same row in all of them — so the
    /// bookkeeping lives here, once, next to `lens`. Tables and lengths advance and roll back
    /// together: `prepare_lanes` grows both or neither, `reset_lane` clears both.
    pool: BlockPool,
}

impl TransformerCache {
    pub fn new(config: &TransformerConfig, max_batch_size: usize, device: &Device) -> Self {
        // One block spans a lane's whole sequence space at this stage, so the pool is one block per
        // lane plus the sentinel — the same bytes as the old per-lane slab plus one row. Shrinking
        // the block (many small blocks per lane) is the next stage; only these two numbers change.
        let block_size = config.max_seq_len;
        let num_blocks = max_batch_size + 1;
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
            lens: vec![0; max_batch_size],
            pool: BlockPool::new(block_size, num_blocks, max_batch_size),
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
        // Two invariants the per-lane correctness rests on, checked only in debug builds. The lanes
        // must be distinct: a repeated lane would advance its length twice while the RoPE and mask
        // for the second row used the stale start, silently corrupting that lane. And the lanes must
        // be in range, since an out-of-range lane would otherwise be a raw indexing panic below.
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

        // Grow every lane's block table to cover this forward, BEFORE any length advances — ensure,
        // then advance, so a failure leaves every lane exactly as it was. `ensure_capacity` is
        // all-or-nothing per lane; the loop below extends that to the whole round by unwinding the
        // lanes already grown when a later one cannot fit. Allocation happens only here, before any
        // layer runs, so the write path downstream never allocates. (While one block spans the whole
        // sequence space the pool cannot actually run dry — every lane needs at most one block, and
        // the pool holds one per lane — but the rollback is the contract the smaller block sizes of
        // the next stage inherit.)
        let mut grown: Vec<(usize, usize)> = Vec::with_capacity(lanes.len());
        for &lane in lanes {
            let before = self.pool.lane_blocks(lane).len();
            match self.pool.ensure_capacity(lane, self.lens[lane] + seq_len) {
                Ok(()) => grown.push((lane, before)),
                Err(exhausted) => {
                    for &(grown_lane, keep) in &grown {
                        self.pool.truncate_lane(grown_lane, keep);
                    }
                    return Err(GenerationError::KvPoolExhausted {
                        short_by: exhausted.short_by,
                    });
                }
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

        // Snapshot each lane's covering blocks into the plan, in `lanes` order — the same order the
        // mask and RoPE rows use — so the layers address KV purely through the plan.
        let tables: Vec<Vec<u32>> = lanes
            .iter()
            .map(|&lane| self.pool.lane_blocks(lane).to_vec())
            .collect();

        Ok(LanePlan {
            lanes: lanes.to_vec(),
            starts,
            tables,
            mask,
        })
    }

    /// Reset the whole cache by zeroing every lane's length, reused between independent generations.
    pub fn reset(&mut self) {
        for lane in 0..self.lens.len() {
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
        if lane >= self.lens.len() {
            return;
        }
        self.lens[lane] = 0;
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
    /// With `TransformerCache.lens` the single source of length, the freed
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
                layer.forward_lanes(&plan.tables, &plan.starts, x.clone(), x.clone());
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

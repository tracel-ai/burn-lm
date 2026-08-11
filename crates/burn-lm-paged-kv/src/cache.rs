use burn::tensor::{Bool, Device, Int, Tensor};

use crate::block_pool::{BlockPool, SENTINEL_BLOCK};
use crate::kv_cache::KeyValueCache;

/// Tokens per KV block, by default. Small enough that a short sequence's footprint is a few blocks
/// instead of a whole `max_seq_len` stripe (the point of paging), large enough that per-block
/// gather granularity stays cheap — at most `block_size - 1` wasted columns per lane per read. A
/// model with a context window smaller than this just uses one block per lane.
pub const DEFAULT_BLOCK_SIZE: usize = 128;

/// The KV shape of a decoder-only model, from the cache's point of view: how many layers write KV,
/// the per-layer key/value geometry, and the context window. Everything else about the model is
/// irrelevant here.
#[derive(Debug, Clone, Copy)]
pub struct KvLayout {
    /// Transformer layers (each owns one K and one V store).
    pub n_layers: usize,
    /// Key/value heads per layer (the grouped-query count, not the query-head count).
    pub n_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// The context window: the logical per-lane token ceiling (RoPE table size, mask width).
    pub max_seq_len: usize,
}

/// What can go wrong planning a round.
#[derive(Debug)]
pub enum PagedKvError {
    /// A lane's new length would pass the context window. Lane mode does not evict, so this is an
    /// error rather than dropping the oldest tokens; the serving engine retires the sequence before
    /// its next forward would trigger it, leaving this as the model-side guard.
    MaxSequenceLengthExceeded { actual: usize, max: usize },
    /// The block pool could not cover the round: it is `short_by` blocks short. An engine that
    /// reserves worst-case blocks per sequence at admission never lets a live sequence reach this,
    /// so it indicates an accounting bug there, not a runtime condition — callers retire the
    /// sequence with the error rather than crashing the batch.
    PoolExhausted { short_by: usize },
}

/// Everything one lane-aware forward needs to know about where its lanes sit, produced by
/// [`PagedKvCache::prepare_lanes`] and consumed unchanged all the way down to the attention layer.
/// It names which lanes take part, each lane's start position (used both as the RoPE position and
/// as the KV write offset), the physical addressing (block tables and the prebuilt gather index),
/// and the per-lane mask.
#[derive(Debug)]
pub struct LanePlan {
    /// The active lanes, one per row of the forward input.
    pub lanes: Vec<usize>,
    /// Each lane's sequence length before this forward. It serves two roles: the absolute RoPE
    /// position of the lane's first new token, and the logical offset its new tokens are written to.
    pub starts: Vec<usize>,
    /// Per active lane, in `lanes` order: the KV block ids covering the lane's positions after this
    /// forward, entry `i` covering positions `[i·block_size, (i+1)·block_size)`. The allocator grew
    /// each table before this plan was built, so every block a layer's write will touch already
    /// exists and the layers allocate nothing. This snapshot is also, deliberately, the handoff
    /// artifact a future prefill/decode split would ship: "prefill output = block list" is a
    /// literal value here.
    pub tables: Vec<Vec<u32>>,
    /// The round's KV gather index, prebuilt from `tables`: `blocks_per_lane` block ids per lane in
    /// position order, short lanes padded with the zeroed sentinel. A pure function of the round, so
    /// it is uploaded once here and every layer's K and V gather through the same handle.
    pub gather_idx: Tensor<1, Int>,
    /// Blocks per lane in `gather_idx`: enough to cover `l_max`.
    pub blocks_per_lane: usize,
    /// The round's KV write index, prebuilt like `gather_idx`: `(block, offset)` destination pairs
    /// for every new token, lane-major in `lanes` order — `[n·seq_len, 2]`. One scatter per store
    /// per layer writes the whole round through this; a token crossing a block boundary is just a
    /// different pair. These are the exact inputs a dedicated cache-write kernel takes.
    pub write_idx: Tensor<2, Int>,
    /// The longest active lane's length after this forward — the width of the gathered KV and of
    /// the mask's last dimension.
    pub l_max: usize,
    /// The per-lane attention mask, shaped `[n, 1, q, l_max]`, where `true` means masked. Row `r` of
    /// lane `j` may attend to columns `0..=starts[j] + r`; everything past that is masked off — both
    /// the lane's own future and the stale tail out to the longest active lane. The attention op
    /// turns masked positions into negative infinity before the softmax. Because the lanes are
    /// ragged, even a single-token decode round (`q == 1`) needs this mask to hide each lane's tail.
    pub mask: Tensor<4, Bool>,
}

/// The model-owned paged KV cache for one batch of lanes: the per-layer key/value block stores plus
/// the only length-and-ownership bookkeeping in the system. Each lane is an independent sequence
/// drawing blocks from a shared pool; a lane is a logical index here, mapped to physical blocks by
/// its table. The whole batched decode loop runs against one of these.
#[derive(Clone, Debug)]
pub struct PagedKvCache {
    layers: Vec<KeyValueCache>,
    device: Device,
    /// The KV shape this cache was built with — kept so the pool can be rebuilt (`resize_pool`).
    layout: KvLayout,
    /// The per-lane ledger: each lane's length and block table, plus the shared free stack — in one
    /// type so lengths and tables can never drift apart. One block-id space serves every layer's K
    /// and V stores — block `b` names the same row in all of them.
    pool: BlockPool,
}

impl PagedKvCache {
    /// A cache with the default block size and a window-per-lane pool (see
    /// [`Self::with_window_per_lane`]).
    pub fn with_default_blocks(layout: KvLayout, max_lanes: usize, device: &Device) -> Self {
        Self::with_window_per_lane(
            layout,
            max_lanes,
            DEFAULT_BLOCK_SIZE.min(layout.max_seq_len),
            device,
        )
    }

    /// The unpaged configuration: one block spans each lane's whole context window, so nothing is
    /// ever split or padded — exactly the old per-lane slab layout (plus the sentinel block). A
    /// model that wants no paging chooses this; it is a configuration of the same type, not a
    /// separate implementation, and the equivalence gates cover it as the degenerate block size.
    pub fn unpaged(layout: KvLayout, max_lanes: usize, device: &Device) -> Self {
        Self::with_window_per_lane(layout, max_lanes, layout.max_seq_len, device)
    }

    /// A cache whose pool holds one full context window per lane — every lane can reach
    /// `max_seq_len` simultaneously, like the old slab, so nothing is ever oversubscribed. The
    /// degenerate sizing: safe, and wasteful for ragged traffic in exactly the way paging exists to
    /// fix. The block-size equivalence tests drive their explicit sizes through here.
    pub fn with_window_per_lane(
        layout: KvLayout,
        max_lanes: usize,
        block_size: usize,
        device: &Device,
    ) -> Self {
        let window_per_lane = (max_lanes * layout.max_seq_len).div_ceil(block_size);
        Self::new(layout, max_lanes, block_size, window_per_lane, device)
    }

    /// The standard constructor: an explicit pool size, decoupled from the lane count — the point
    /// of paging. With `usable_blocks` below one-window-per-lane, the lanes oversubscribe the pool:
    /// more sequences can be admitted than could all simultaneously reach `max_seq_len`, and it is
    /// the serving engine's reservation accounting (worst case per sequence, against
    /// `usable_blocks`) that keeps the pool from ever running dry mid-flight.
    pub fn new(
        layout: KvLayout,
        max_lanes: usize,
        block_size: usize,
        usable_blocks: usize,
        device: &Device,
    ) -> Self {
        assert!(
            block_size >= 1 && block_size <= layout.max_seq_len,
            "block_size must be in 1..=max_seq_len"
        );
        assert!(usable_blocks >= 1, "the pool needs at least one usable block");
        let num_blocks = usable_blocks + 1; // plus the zeroed sentinel
        let layers = (0..layout.n_layers)
            .map(|_| {
                KeyValueCache::new(
                    num_blocks,
                    layout.n_kv_heads,
                    block_size,
                    layout.head_dim,
                    device,
                )
            })
            .collect::<Vec<_>>();

        Self {
            layers,
            device: device.clone(),
            layout,
            pool: BlockPool::new(block_size, num_blocks, max_lanes),
        }
    }

    /// The per-layer caches, in layer order — one entry per transformer layer, matching the
    /// `KvLayout::n_layers` this cache was built with. The model iterates these in lockstep with
    /// its layers during a forward.
    pub fn layers_mut(&mut self) -> impl Iterator<Item = &mut KeyValueCache> {
        self.layers.iter_mut()
    }

    /// Tokens per KV block.
    pub fn block_size(&self) -> usize {
        self.pool.block_size()
    }

    /// Usable blocks in the pool (the sentinel excluded) — the total an engine budgets sequence
    /// reservations against.
    pub fn usable_blocks(&self) -> usize {
        self.pool.usable_blocks()
    }

    /// Number of lanes (the model's `max_batch_size`).
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
        self.layout.max_seq_len
    }

    /// Plan one forward of `seq_len` new tokens over the given lanes, and commit it. It checks each
    /// lane has room, commits the round in the ledger (blocks and lengths together), builds the
    /// per-lane mask and the round's gather index, and returns the resulting [`LanePlan`]. A prefill
    /// calls this with a single lane; a decode round calls it with all the active lanes at once.
    pub fn prepare_lanes(
        &mut self,
        lanes: &[usize],
        seq_len: usize,
    ) -> Result<LanePlan, PagedKvError> {
        // The lanes must be in range, or the ledger lookups below would be raw indexing panics.
        // (Duplicate lanes are asserted inside `begin_round`, where a repeat would corrupt the
        // ledger by advancing a length twice.)
        debug_assert!(
            lanes.iter().all(|&lane| lane < self.pool.lane_count()),
            "prepare_lanes got a lane >= lane_count ({}): {lanes:?}",
            self.pool.lane_count()
        );

        for &lane in lanes {
            if self.pool.lane_len(lane) + seq_len > self.layout.max_seq_len {
                return Err(PagedKvError::MaxSequenceLengthExceeded {
                    actual: self.pool.lane_len(lane) + seq_len,
                    max: self.layout.max_seq_len,
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
            .map_err(|exhausted| PagedKvError::PoolExhausted {
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
            burn::tensor::TensorData::new(ids, [n * blocks_per_lane]),
            &self.device,
        );

        // ...and the round's write index: every new token's (block, offset) destination, shared by
        // every layer's K and V scatter.
        let write_ids =
            crate::block_store::write_indices(&tables, &starts, seq_len, self.pool.block_size());
        let write_idx = Tensor::<2, Int>::from_data(
            burn::tensor::TensorData::new(write_ids, [n * seq_len, 2]),
            &self.device,
        );

        Ok(LanePlan {
            lanes: lanes.to_vec(),
            starts,
            tables,
            gather_idx,
            blocks_per_lane,
            write_idx,
            l_max,
            mask,
        })
    }

    /// Reset the whole cache by freeing every lane, reused between independent generations.
    pub fn reset(&mut self) {
        for lane in 0..self.pool.lane_count() {
            self.reset_lane(lane);
        }
    }

    /// Free one lane: its blocks go back to the pool and its length zeroes. Nothing clears the
    /// block contents; the next occupant overwrites them, and every ragged read is masked (see the
    /// stale-data contract on [`BlockPool`]).
    ///
    /// A lane index past the ledger is silently ignored rather than panicking. This guards against
    /// a slot the cache never had — for instance if an engine's `max_slots` were set above the
    /// loaded lane count — reaching here and indexing out of bounds. Admission caps slots at
    /// `lane_count`, so this is a second line of defense.
    pub fn reset_lane(&mut self, lane: usize) {
        if lane >= self.pool.lane_count() {
            return;
        }
        self.pool.free_lane(lane);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane_test_layout(max_seq_len: usize) -> KvLayout {
        KvLayout {
            n_layers: 2,
            n_kv_heads: 1,
            head_dim: 2,
            max_seq_len,
        }
    }

    fn mask_rows(mask: &Tensor<4, Bool>) -> Vec<bool> {
        mask.clone().into_data().iter::<bool>().collect()
    }

    /// Decode step for two lanes at divergent positions: each lane's mask row allows exactly its
    /// own history plus the new token, and masks the tail up to the longest active lane.
    #[test]
    fn test_prepare_lanes_decode_mask_covers_exactly_the_dead_columns() {
        let mut cache =
            PagedKvCache::with_default_blocks(lane_test_layout(8), 2, &Default::default());

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

    /// Single-lane prefill: the per-lane mask reduces to the ordinary causal triangle over the
    /// lane's own (empty) history.
    #[test]
    fn test_prepare_lanes_prefill_mask_is_causal() {
        let mut cache =
            PagedKvCache::with_default_blocks(lane_test_layout(8), 2, &Default::default());

        let plan = cache.prepare_lanes(&[1], 3).unwrap();
        assert_eq!(plan.starts, vec![0]);
        assert_eq!(plan.mask.dims(), [1, 1, 3, 3]);
        assert_eq!(
            mask_rows(&plan.mask),
            vec![false, true, true, false, false, true, false, false, false]
        );
    }

    /// A lane that would exceed its capacity is an error; lane mode has no eviction.
    #[test]
    fn test_prepare_lanes_exceeded_max_seq_len() {
        let mut cache =
            PagedKvCache::with_default_blocks(lane_test_layout(4), 2, &Default::default());

        cache.prepare_lanes(&[0], 3).unwrap();
        // Lane 1 is fine on its own...
        cache.prepare_lanes(&[1], 1).unwrap();
        // ...but lane 0 cannot take 2 more positions.
        assert!(matches!(
            cache.prepare_lanes(&[0, 1], 2),
            Err(PagedKvError::MaxSequenceLengthExceeded { actual: 5, max: 4 })
        ));
        // A failed plan advances nothing.
        assert_eq!(cache.lane_len(0), 3);
        assert_eq!(cache.lane_len(1), 1);
    }

    /// `reset_lane` frees one lane, leaving the other lane untouched. With the pool's ledger the
    /// single source of length, the freed lane's next write lands at offset 0.
    #[test]
    fn test_reset_lane_isolates_one_lane() {
        let device: Device = Default::default();
        let layout = lane_test_layout(8);
        let mut cache = PagedKvCache::with_default_blocks(layout, 2, &device);

        // Write real KV rows; the plan's starts come from this cache's lengths.
        let write = |cache: &mut PagedKvCache, lanes: &[usize], seq_len: usize| {
            let x = Tensor::ones([lanes.len(), layout.n_kv_heads, seq_len, layout.head_dim], &device);
            let plan = cache.prepare_lanes(lanes, seq_len).unwrap();
            for layer in cache.layers_mut() {
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

    /// Releasing a lane outside the ledger is a no-op, not a panic — defends against an engine
    /// `max_slots` raised above the loaded lane count handing release an out-of-range slot.
    #[test]
    fn test_reset_lane_out_of_range_is_a_noop() {
        let mut cache =
            PagedKvCache::with_default_blocks(lane_test_layout(8), 2, &Default::default());
        cache.prepare_lanes(&[0], 3).unwrap();
        cache.reset_lane(5); // 5 >= lane_count 2 — must not panic
        assert_eq!(cache.lane_count(), 2);
        assert_eq!(cache.lane_len(0), 3, "an in-range lane is untouched");
    }

    /// Pool exhaustion mid-round is all-or-nothing across the WHOLE round: when one lane of a
    /// multi-lane `prepare_lanes` cannot get its block, the lanes already grown this round are
    /// unwound too — no length advances, no table grows, and the free stack is exactly as it was.
    /// The rectangle-sized pool can never run dry (the ceiling check fires first), so the test
    /// swaps in a smaller one; an oversubscribed pool inherits this contract for real.
    #[test]
    fn test_pool_exhaustion_rolls_back_the_whole_round() {
        let mut cache =
            PagedKvCache::with_default_blocks(lane_test_layout(8), 3, &Default::default());
        // Three lanes, but a pool with only two usable blocks.
        cache.pool = BlockPool::new(8, 3, 3);

        let err = cache.prepare_lanes(&[0, 1, 2], 4).unwrap_err();
        assert!(
            matches!(err, PagedKvError::PoolExhausted { short_by: 1 }),
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

    /// An oversubscribed pool: more lanes than could all reach `max_seq_len` at once. Lanes that
    /// fit run; the round that would exceed the pool fails all-or-nothing (the engine's reservation
    /// accounting exists to keep live lanes out of that branch). The explicit pool size on `new` is the
    /// load-time seam that creates this shape.
    #[test]
    fn test_oversubscribed_pool_serves_what_fits() {
        // 4 lanes over a pool of 2 window-sized blocks: half the lanes can be full at once.
        let mut cache = PagedKvCache::new(lane_test_layout(8), 4, 8, 2, &Default::default());
        assert_eq!(cache.usable_blocks(), 2);

        // Two lanes can hold a full window each; the third finds the pool dry.
        cache.prepare_lanes(&[0], 8).unwrap();
        cache.prepare_lanes(&[1], 8).unwrap();
        let err = cache.prepare_lanes(&[2], 1).unwrap_err();
        assert!(matches!(err, PagedKvError::PoolExhausted { short_by: 1 }));

        // Freeing a lane frees its blocks for the next occupant.
        cache.reset_lane(0);
        cache.prepare_lanes(&[2], 4).unwrap();
    }

    /// The unpaged configuration is the old slab layout: one block spans a lane's whole context, so
    /// growth within the window never takes another block and nothing is ever split.
    #[test]
    fn test_unpaged_holds_one_block_per_lane() {
        let mut cache = PagedKvCache::unpaged(lane_test_layout(8), 2, &Default::default());
        cache.prepare_lanes(&[0], 3).unwrap();
        let plan = cache.prepare_lanes(&[0], 5).unwrap(); // grow to the full window
        assert_eq!(plan.tables[0].len(), 1, "one block covers the whole lane");
        assert_eq!(plan.blocks_per_lane, 1);
    }
}

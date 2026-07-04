use burn::tensor::{Device, Tensor};

use super::block_store::BlockStore;
use crate::cache::LanePlan;

/// Key-value cache for autoregressive models: a pool of KV blocks, keys and values side by side.
/// One block id names the same row in both stores, so a lane's block table addresses its keys and
/// its values at once.
#[derive(Debug, Clone)]
pub struct KeyValueCache {
    key: BlockStore,
    value: BlockStore,
}

impl KeyValueCache {
    /// Create a new [key-value cache](KeyValueCache) of `num_blocks` blocks (block 0 is the zeroed
    /// sentinel) spanning `block_size` positions each.
    pub fn new(
        num_blocks: usize,
        num_heads: usize,
        block_size: usize,
        d_model: usize,
        device: &Device,
    ) -> Self {
        Self {
            key: BlockStore::new([num_blocks, num_heads, block_size, d_model], device),
            value: BlockStore::new([num_blocks, num_heads, block_size, d_model], device),
        }
    }

    /// Write one round's new keys and values, without reading back. Row `j` of `key`/`value` lands
    /// in the blocks of `tables[j]` at logical offset `starts[j]`. This is the seeding/write half of
    /// `forward_lanes`, for callers (benchmarks, tests) that fill lanes without running attention.
    pub fn write_lanes(
        &mut self,
        tables: &[Vec<u32>],
        starts: &[usize],
        key: Tensor<4>,
        value: Tensor<4>,
    ) {
        self.key.write_lanes(tables, starts, key);
        self.value.write_lanes(tables, starts, value);
    }

    /// Update the key and value caches for one round, one lane per row: write each lane's new
    /// keys/values into its blocks, then gather every active lane back out to the round's `l_max`.
    /// The plan carries all the addressing — tables and offsets for the writes, the prebuilt gather
    /// index for the reads (one index upload per round, shared by every layer's K and V) — so this
    /// cache holds no length or ownership state of its own. The caller masks each lane's stale tail.
    pub fn forward_lanes(
        &mut self,
        plan: &LanePlan,
        key: Tensor<4>,
        value: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        self.key.write_lanes(&plan.tables, &plan.starts, key);
        self.value.write_lanes(&plan.tables, &plan.starts, value);
        let k = self
            .key
            .gather(plan.gather_idx.clone(), plan.blocks_per_lane, plan.l_max);
        let v = self
            .value
            .gather(plan.gather_idx.clone(), plan.blocks_per_lane, plan.l_max);
        (k, v)
    }
}

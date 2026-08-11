use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::block_store::{write_indices, BlockStore};
use crate::cache::LanePlan;

/// Key-value cache for autoregressive models: a pool of KV blocks, keys and values side by side.
/// One block id names the same row in both stores, so a lane's block table addresses its keys and
/// its values at once. The write and read halves are deliberately separate functions — they are
/// the two kernel contracts of a paged cache (a cache-write kernel and a paged-attention kernel),
/// and each generic-op body here is written to be swapped for one.
#[derive(Debug, Clone)]
pub struct KeyValueCache {
    key: BlockStore,
    value: BlockStore,
    block_size: usize,
}

impl KeyValueCache {
    /// Create a new [key-value cache](KeyValueCache) of `num_blocks` blocks (block 0 is the zeroed
    /// sentinel) spanning `block_size` positions each.
    pub fn new(
        num_blocks: usize,
        num_heads: usize,
        block_size: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            key: BlockStore::new(num_blocks, block_size, num_heads, head_dim, device),
            value: BlockStore::new(num_blocks, block_size, num_heads, head_dim, device),
            block_size,
        }
    }

    /// Write one round's new keys and values into their blocks — one scatter per store, whatever
    /// the width, addressed by the plan's prebuilt `(block, offset)` index. This round's tokens
    /// must be written before [`paged_attention`](crate::paged_attention) reads.
    pub fn write(&mut self, plan: &LanePlan, key: Tensor<4>, value: Tensor<4>) {
        self.key.write(&plan.write_idx, key);
        self.value.write(&plan.write_idx, value);
    }

    /// Write by explicit tables and start offsets, building the destination index locally. The
    /// seeding/testing half of [`write`](Self::write), for callers (benchmarks, tests) that fill
    /// lanes without a round plan.
    pub fn write_lanes(
        &mut self,
        tables: &[Vec<u32>],
        starts: &[usize],
        key: Tensor<4>,
        value: Tensor<4>,
    ) {
        let seq_len = key.dims()[2];
        let ids = write_indices(tables, starts, seq_len, self.block_size);
        let n = ids.len() / 2;
        let idx =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [n, 2]), &key.device());
        self.key.write(&idx, key);
        self.value.write(&idx, value);
    }

    /// Gather the plan's lanes out of both stores, `[n, num_heads, l_max, head_dim]` each. The
    /// read half of the paged cache — used by the reference `paged_attention`; a dedicated kernel
    /// reads the blocks in place instead.
    pub(crate) fn gather(&self, plan: &LanePlan) -> (Tensor<4>, Tensor<4>) {
        let k = self
            .key
            .gather(plan.gather_idx.clone(), plan.blocks_per_lane, plan.l_max);
        let v = self
            .value
            .gather(plan.gather_idx.clone(), plan.blocks_per_lane, plan.l_max);
        (k, v)
    }
}

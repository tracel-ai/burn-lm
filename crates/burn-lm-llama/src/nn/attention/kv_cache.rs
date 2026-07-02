use burn::tensor::{Device, Tensor};

use super::cache::AutoregressiveCache;

/// Key-value cache for autoregressive models: a pool of KV blocks, keys and values side by side.
/// One block id names the same row in both pools, so a lane's block table addresses its keys and
/// its values at once.
#[derive(Debug, Clone)]
pub struct KeyValueCache {
    key: AutoregressiveCache,
    value: AutoregressiveCache,
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
            key: AutoregressiveCache::new([num_blocks, num_heads, block_size, d_model], device),
            value: AutoregressiveCache::new([num_blocks, num_heads, block_size, d_model], device),
        }
    }

    /// Update the key and value caches for one round, one lane per row. Row `j` of `key`/`value` is
    /// written into the blocks of `tables[j]` at logical offset `starts[j]`, the caller's length for
    /// that lane. Returns the active lanes' keys and values up to the longest active lane; the
    /// caller masks each lane's stale tail. This is the only thing this cache does — the underlying
    /// pools carry no length or ownership state of their own.
    pub fn forward_lanes(
        &mut self,
        tables: &[Vec<u32>],
        starts: &[usize],
        key: Tensor<4>,
        value: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let k = self.key.append_lanes(tables, starts, key);
        let v = self.value.append_lanes(tables, starts, value);
        (k, v)
    }
}

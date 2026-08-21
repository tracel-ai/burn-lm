use burn::tensor::{Device, Tensor};

use super::cache::AutoregressiveCache;

/// Key-value cache for autoregressive models.
#[derive(Debug, Clone)]
pub struct KeyValueCache {
    key: AutoregressiveCache<4>,
    value: AutoregressiveCache<4>,
}

impl KeyValueCache {
    /// Create a new [key-value cache](KeyValueCache).
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(
                [max_batch_size, num_heads, max_seq_len, d_model],
                2,
                device,
            ),
            value: AutoregressiveCache::new(
                [max_batch_size, num_heads, max_seq_len, d_model],
                2,
                device,
            ),
        }
    }

    /// Update the key and value caches for one round, one lane per row. Row `j` of `key`/`value` is
    /// written into buffer lane `lanes[j]` at offset `starts[j]`, the caller's length for that lane.
    /// Returns the active lanes' keys and values up to the longest active lane; the caller masks each
    /// lane's stale tail. This is the only thing this cache does — the two underlying buffers carry no
    /// length state of their own.
    pub fn forward_lanes(
        &mut self,
        lanes: &[usize],
        starts: &[usize],
        key: Tensor<4>,
        value: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let k = self.key.append_lanes(lanes, starts, key);
        let v = self.value.append_lanes(lanes, starts, value);
        (k, v)
    }
}

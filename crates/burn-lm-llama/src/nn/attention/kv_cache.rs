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

    /// Computes the complete keys and values.
    pub fn forward(&mut self, key: Tensor<4>, value: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let k = self.key.append(key);
        let v = self.value.append(value);
        (k, v)
    }

    /// Lane-sliced variant of [`Self::forward`]: row `j` of `key`/`value` is
    /// written into buffer lane `lanes[j]` at that lane's own position.
    /// Returns the active lanes' K/V up to the longest active lane; the
    /// caller masks each lane's stale tail.
    pub fn forward_lanes(
        &mut self,
        lanes: &[usize],
        key: Tensor<4>,
        value: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let k = self.key.append_lanes(lanes, key);
        let v = self.value.append_lanes(lanes, value);
        (k, v)
    }

    /// Free one lane (its buffer row is overwritten on the next use).
    pub fn reset_lane(&mut self, lane: usize) {
        self.key.reset_lane(lane);
        self.value.reset_lane(lane);
    }

    /// Returns the cached sequence length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        // We can assume key and value have the same length
        self.key.len()
    }

    /// Returns the cached sequence length of one lane.
    pub fn lane_len(&self, lane: usize) -> usize {
        // We can assume key and value have the same length
        self.key.lane_len(lane)
    }

    pub fn prepare(&mut self, num_tokens: usize) {
        self.key.prepare(num_tokens);
        self.value.prepare(num_tokens);
    }

    /// Reset key-value cache.
    /// Use between different contexts (i.e., for each new prompt).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }
}

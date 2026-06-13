use burn::tensor::{Device, Tensor};

/// Strategy for managing the autoregressive cache when its capacity is exceeded.
#[derive(Debug, Clone, Default)]
pub(crate) enum CacheStrategy {
    /// Shrinks the cache by copying the remaining tokens into a new buffer,
    /// removing the oldest tokens beyond the context limit.
    #[allow(dead_code)]
    Shrink,

    /// Shifts the remaining tokens to the start of the existing buffer in-place,
    /// overwriting the oldest tokens.
    #[default]
    Shift,
}

#[derive(Debug, Clone)]
/// Cache that keeps track of a tensor state in an autoregressive decoding process.
///
/// Bookkeeping is per lane: dimension 0 is the lane (batch row of the shared
/// buffer) and `lens[lane]` tracks how many positions that lane holds. The
/// whole-cache API (`append`/`prepare`/`len`/`reset`) delegates to lane 0,
/// which is exactly the previous single-sequence behavior.
pub(crate) struct AutoregressiveCache<const D: usize> {
    cache: Tensor<D>,
    seq_dim: usize,
    /// Per-lane sequence lengths.
    ///
    /// The lane path (`append_lanes`/`reset_lane`) is the PRODUCTION decode path: each entry is one
    /// live, independent lane's length. The whole-cache methods (`append`/`len`/`reset`) operate on
    /// `lens[0]` and are now used only by the test-only single-sequence reference forward.
    ///
    /// A given cache INSTANCE is driven by exactly one family — never mix `append` and
    /// `append_lanes` on the same instance (their offset semantics differ: `append` writes every
    /// batch row at `lens[0]`; `append_lanes` writes lane `j` at `lens[lane]`). The production
    /// decoder and the test reference build separate instances, so they never alias.
    lens: Vec<usize>,
    strategy: CacheStrategy,
}

impl<const D: usize> AutoregressiveCache<D> {
    /// Creates a new empty cache.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device) -> Self {
        // Lanes live on dimension 0; when dim 0 *is* the sequence dimension
        // (1D-style caches) there is exactly one lane.
        let n_lanes = if seq_dim == 0 { 1 } else { shape[0] };
        Self {
            cache: Tensor::empty(shape, device),
            seq_dim,
            lens: vec![0; n_lanes],
            strategy: CacheStrategy::default(),
        }
    }

    #[allow(dead_code)]
    /// Sets the cache management strategy.
    pub fn with_strategy(mut self, strategy: CacheStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Reset the cache state (all lanes).
    pub fn reset(&mut self) {
        // Note: we don't need to clear the tensor since we track the current seq length
        self.lens.iter_mut().for_each(|len| *len = 0);
    }

    /// Reset a single lane; its buffer row is overwritten on the next append.
    pub fn reset_lane(&mut self, lane: usize) {
        self.lens[lane] = 0;
    }

    /// Add the new tokens to the current cache and returns all tokens decoded since the beginning.
    ///
    /// Whole-cache path: every batch row is written at the same (lane-0)
    /// offset — single-sequence / lockstep semantics.
    ///
    /// # Shapes
    ///
    /// - input:  `[batch_size, num_heads, seq_len_input, d_model]`
    /// - output: `[batch_size, num_heads, seq_len_previous + seq_len_input, d_model]`
    pub fn append(&mut self, tokens: Tensor<D>) -> Tensor<D> {
        let shape = tokens.shape();
        let seq_len_input = shape[self.seq_dim];

        let new_seq_len = self.lens[0] + seq_len_input;

        let mut indices_added_tokens = Vec::with_capacity(shape.len());
        let mut indices_output = Vec::with_capacity(shape.len());

        for (i, shape) in shape.iter().enumerate() {
            if i == self.seq_dim {
                indices_added_tokens.push(self.lens[0]..new_seq_len);
                indices_output.push(0..new_seq_len);
            } else {
                indices_added_tokens.push(0..*shape);
                indices_output.push(0..*shape);
            }
        }
        self.cache
            .inplace(|cache| cache.slice_assign(indices_added_tokens.as_slice(), tokens));

        self.lens[0] = new_seq_len;

        self.cache.clone().slice(indices_output.as_slice())
    }

    /// Lane-sliced append: write row `j` of `tokens` into buffer row
    /// `lanes[j]` at that lane's current length, then return the active
    /// lanes' contents up to the longest active lane.
    ///
    /// Lanes are independent: each sits at its own position; columns past a
    /// lane's own length in the returned tensor are stale buffer data and
    /// MUST be masked by the caller (per-lane padding mask).
    ///
    /// # Shapes
    ///
    /// - tokens: `[n_active, num_heads, seq_len_input, d_model]`
    /// - output: `[n_active, num_heads, max(lens[lane]) + seq_len_input, d_model]`
    pub fn append_lanes(&mut self, lanes: &[usize], tokens: Tensor<D>) -> Tensor<D> {
        debug_assert_ne!(self.seq_dim, 0, "lane dimension is dim 0");
        let shape = tokens.shape();
        debug_assert_eq!(shape[0], lanes.len());
        let seq_len_input = shape[self.seq_dim];

        for (j, &lane) in lanes.iter().enumerate() {
            let start = self.lens[lane];
            let mut value_idx = Vec::with_capacity(shape.len());
            let mut cache_idx = Vec::with_capacity(shape.len());
            for (i, dim) in shape.iter().enumerate() {
                if i == 0 {
                    value_idx.push(j..j + 1);
                    cache_idx.push(lane..lane + 1);
                } else if i == self.seq_dim {
                    value_idx.push(0..seq_len_input);
                    cache_idx.push(start..start + seq_len_input);
                } else {
                    value_idx.push(0..*dim);
                    cache_idx.push(0..*dim);
                }
            }
            let row = tokens.clone().slice(value_idx.as_slice());
            self.cache
                .inplace(|cache| cache.slice_assign(cache_idx.as_slice(), row));
            self.lens[lane] = start + seq_len_input;
        }

        // Read the active lanes back, ragged tails included (mask handles them).
        let l_max = lanes.iter().map(|&lane| self.lens[lane]).max().unwrap();
        let cache_shape = self.cache.shape();
        let rows = lanes
            .iter()
            .map(|&lane| {
                let mut idx = Vec::with_capacity(cache_shape.len());
                for (i, dim) in cache_shape.iter().enumerate() {
                    if i == 0 {
                        idx.push(lane..lane + 1);
                    } else if i == self.seq_dim {
                        idx.push(0..l_max);
                    } else {
                        idx.push(0..*dim);
                    }
                }
                self.cache.clone().slice(idx.as_slice())
            })
            .collect::<Vec<_>>();
        Tensor::cat(rows, 0)
    }

    /// Prepare the cache by applying the configured strategy to make room for new tokens.
    ///
    /// `num_tokens` is the number of past tokens to discard or shift, depending on the strategy.
    pub fn prepare(&mut self, num_tokens: usize) {
        match self.strategy {
            CacheStrategy::Shrink => self.shrink(num_tokens),
            CacheStrategy::Shift => self.shift(num_tokens),
        }
    }

    /// Shrink the cache to fit in `max_seq_len` while making place for the new tokens being
    /// decoded.
    fn shrink(&mut self, num_removed: usize) {
        let old_cur_seq_len = self.lens[0];
        self.lens[0] -= num_removed;

        let shape = self.cache.shape();
        let device = self.cache.device();

        let mut slices_prev = Vec::with_capacity(shape.len());
        let mut slices_curr = Vec::with_capacity(shape.len());

        for (i, shape) in shape.iter().enumerate() {
            if i == self.seq_dim {
                slices_prev.push(num_removed..old_cur_seq_len);
                slices_curr.push(0..self.lens[0]);
            } else {
                slices_prev.push(0..*shape);
                slices_curr.push(0..*shape);
            }
        }

        self.cache.inplace(|cache| {
            let prev_slice = cache.slice(slices_prev.as_slice());
            let new_cache = Tensor::empty(shape, &device);

            new_cache.slice_assign(slices_curr.as_slice(), prev_slice)
        });
    }

    /// Shift the cache to fit in `max_seq_len` while making place for the new tokens being
    /// decoded.
    fn shift(&mut self, num_shifted: usize) {
        let old_cur_seq_len = self.lens[0];
        self.lens[0] -= num_shifted;

        let shape = self.cache.shape();

        let mut slices_prev = Vec::with_capacity(shape.len());
        let mut slices_curr = Vec::with_capacity(shape.len());

        for (i, shape) in shape.iter().enumerate() {
            if i == self.seq_dim {
                slices_prev.push(num_shifted..old_cur_seq_len);
                slices_curr.push(0..self.lens[0]);
            } else {
                slices_prev.push(0..*shape);
                slices_curr.push(0..*shape);
            }
        }

        // Shift tail -> head
        self.cache.inplace(|cache| {
            let prev_slice = cache.clone().slice(slices_prev.as_slice());

            cache.slice_assign(slices_curr.as_slice(), prev_slice)
        });
    }

    /// Returns the cached sequence length (lane 0 / whole-cache path).
    pub fn len(&self) -> usize {
        self.lens[0]
    }

    /// Returns the cached sequence length of one lane.
    pub fn lane_len(&self, lane: usize) -> usize {
        self.lens[lane]
    }

    #[allow(dead_code)]
    pub fn device(&self) -> Device {
        self.cache.device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn test_autoregressive_cache(mut cache: AutoregressiveCache<2>) {
        let device = cache.device();
        let tokens_1 = Tensor::<2>::full([4, 8], 1.0, &device);
        let tokens_2 = Tensor::<2>::full([4, 8], 2.0, &device);

        let received_1 = cache.append(tokens_1.clone());
        assert_eq!(cache.len(), 4);
        let received_2 = cache.append(tokens_2.clone());
        assert_eq!(cache.len(), 8);

        received_1.to_data().assert_eq(&tokens_1.to_data(), true);
        received_2
            .clone()
            .slice(0..4)
            .to_data()
            .assert_eq(&tokens_1.to_data(), true);
        received_2
            .slice(4..8)
            .to_data()
            .assert_eq(&tokens_2.to_data(), true);

        cache.prepare(2);
        assert_eq!(cache.len(), 6);

        let tokens_3 = Tensor::<2>::full([2, 8], 3.0, &device);
        let received_3 = cache.append(tokens_3.clone());
        assert_eq!(cache.len(), 8);

        received_3
            .clone()
            .slice(0..2)
            .to_data()
            .assert_eq(&tokens_1.slice(2..4).into_data(), true);
        received_3
            .clone()
            .slice(2..6)
            .to_data()
            .assert_eq(&tokens_2.to_data(), true);
        received_3
            .slice(6..8)
            .to_data()
            .assert_eq(&tokens_3.to_data(), true);
    }

    #[test]
    fn test_autoregressive_cache_shrink() {
        let cache = AutoregressiveCache::<2>::new([8, 8], 0, &Default::default())
            .with_strategy(CacheStrategy::Shrink);
        test_autoregressive_cache(cache);
    }

    #[test]
    fn test_autoregressive_cache_shift() {
        let cache = AutoregressiveCache::<2>::new([8, 8], 0, &Default::default())
            .with_strategy(CacheStrategy::Shift);
        test_autoregressive_cache(cache);
    }

    /// Lanes at divergent positions: ragged writes land at each lane's own
    /// offset, the read-back covers the longest active lane, and `reset_lane`
    /// frees one lane without touching its siblings.
    #[test]
    fn test_append_lanes_ragged_positions_and_reset_lane() {
        let device = Default::default();
        // [n_lanes=3, heads=1, max_seq_len=8, head_dim=2]
        let mut cache = AutoregressiveCache::<4>::new([3, 1, 8, 2], 2, &device);

        // Prefill lane 0 with 3 positions, lane 2 with 1 position.
        let p0 = Tensor::<4>::full([1, 1, 3, 2], 1.0, &device);
        cache.append_lanes(&[0], p0);
        let p2 = Tensor::<4>::full([1, 1, 1, 2], 3.0, &device);
        cache.append_lanes(&[2], p2);
        assert_eq!(cache.lane_len(0), 3);
        assert_eq!(cache.lane_len(1), 0);
        assert_eq!(cache.lane_len(2), 1);

        // Fused decode write: one new position per active lane (0 and 2).
        let step = Tensor::<4>::from_data(
            [[[[10.0, 10.0]]], [[[30.0, 30.0]]]], // rows: lane 0, lane 2
            &device,
        );
        let out = cache.append_lanes(&[0, 2], step);
        assert_eq!(cache.lane_len(0), 4);
        assert_eq!(cache.lane_len(2), 2);
        // Read-back spans the longest active lane (4 positions).
        assert_eq!(out.dims(), [2, 1, 4, 2]);
        // Lane 0 row: 3 prefill positions then the step write.
        out.clone()
            .slice([0..1, 0..1, 0..4, 0..2])
            .to_data()
            .assert_eq(
                &TensorData::from([[[[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0], [10.0, 10.0]]]]),
                false,
            );
        // Lane 2 row: its own prefill + step at ITS positions (0 and 1);
        // columns 2..4 are stale tail the mask must cover — not asserted.
        out.slice([1..2, 0..1, 0..2, 0..2])
            .to_data()
            .assert_eq(&TensorData::from([[[[3.0f32, 3.0], [30.0, 30.0]]]]), false);

        // Releasing lane 0 leaves lane 2 untouched.
        cache.reset_lane(0);
        assert_eq!(cache.lane_len(0), 0);
        assert_eq!(cache.lane_len(2), 2);

        // Lane 0 is recycled from position 0.
        let fresh = Tensor::<4>::full([1, 1, 2, 2], 7.0, &device);
        let out = cache.append_lanes(&[0], fresh);
        assert_eq!(cache.lane_len(0), 2);
        out.to_data()
            .assert_eq(&TensorData::from([[[[7.0f32, 7.0], [7.0, 7.0]]]]), false);
    }
}

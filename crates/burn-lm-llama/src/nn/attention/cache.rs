use burn::tensor::{Device, Tensor};

#[derive(Debug, Clone)]
/// Cache that keeps track of a tensor state in an autoregressive decoding process.
///
/// Bookkeeping is per lane: dimension 0 is the lane (batch row of the shared
/// buffer) and `lens[lane]` tracks how many positions that lane holds.
pub(crate) struct AutoregressiveCache<const D: usize> {
    cache: Tensor<D>,
    seq_dim: usize,
    /// Per-lane sequence lengths.
    ///
    /// The lane path (`append_lanes`/`reset_lane`) is the production decode path: each entry is one
    /// live, independent lane's length. `append_lanes` writes lane `j` at `lens[lane]`.
    lens: Vec<usize>,
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
        }
    }

    /// Reset a single lane; its buffer row is overwritten on the next append.
    pub fn reset_lane(&mut self, lane: usize) {
        self.lens[lane] = 0;
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

    /// Returns the cached sequence length of one lane.
    pub fn lane_len(&self, lane: usize) -> usize {
        self.lens[lane]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

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

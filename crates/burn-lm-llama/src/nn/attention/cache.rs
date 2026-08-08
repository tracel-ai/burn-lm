use burn::tensor::{Device, Tensor};

#[derive(Debug, Clone)]
/// A fixed-size tensor buffer that accumulates one autoregressive sequence per lane. Dimension 0 is
/// the lane (a batch row of the shared buffer); every other dimension is the per-token state. This
/// cache keeps no length counter of its own: the caller owns the per-lane lengths (in
/// `TransformerCache`) and passes each lane's write offset into `append_lanes`, so a lane can be
/// recycled simply by writing it again from offset 0.
pub(crate) struct AutoregressiveCache<const D: usize> {
    cache: Tensor<D>,
    seq_dim: usize,
}

impl<const D: usize> AutoregressiveCache<D> {
    /// Creates a new empty cache.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device) -> Self {
        Self {
            cache: Tensor::empty(shape, device),
            seq_dim,
        }
    }

    /// Write each active lane's new tokens into its own row of the buffer, then read the active lanes
    /// back. Row `j` of `tokens` lands in buffer row `lanes[j]` at offset `starts[j]`, which the
    /// caller supplies as that lane's length before this write.
    ///
    /// The lanes sit at independent positions, so the read-back spans the longest active lane and the
    /// shorter lanes come back with a stale tail past their own length. The caller MUST mask that tail
    /// with the per-lane padding mask; this cache does not zero it.
    ///
    /// # Shapes
    ///
    /// - tokens: `[n_active, num_heads, seq_len_input, d_model]`
    /// - output: `[n_active, num_heads, max(starts) + seq_len_input, d_model]`
    pub fn append_lanes(
        &mut self,
        lanes: &[usize],
        starts: &[usize],
        tokens: Tensor<D>,
    ) -> Tensor<D> {
        debug_assert_ne!(self.seq_dim, 0, "lane dimension is dim 0");
        let shape = tokens.shape();
        debug_assert_eq!(shape[0], lanes.len());
        debug_assert_eq!(starts.len(), lanes.len());
        let seq_len_input = shape[self.seq_dim];

        for (j, &lane) in lanes.iter().enumerate() {
            let start = starts[j];
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
        }

        // Read the active lanes back, ragged tails included (mask handles them).
        let l_max = starts.iter().map(|&s| s + seq_len_input).max().unwrap();
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    /// Lanes at divergent positions: ragged writes land at each lane's own
    /// caller-supplied offset, the read-back covers the longest active lane,
    /// and recycling a lane (offset 0) overwrites its row without touching its
    /// siblings.
    #[test]
    fn test_append_lanes_ragged_positions_and_reset_lane() {
        let device = Default::default();
        // [n_lanes=3, heads=1, max_seq_len=8, head_dim=2]
        let mut cache = AutoregressiveCache::<4>::new([3, 1, 8, 2], 2, &device);

        // Prefill lane 0 with 3 positions, lane 2 with 1 position.
        let p0 = Tensor::<4>::full([1, 1, 3, 2], 1.0, &device);
        cache.append_lanes(&[0], &[0], p0);
        let p2 = Tensor::<4>::full([1, 1, 1, 2], 3.0, &device);
        cache.append_lanes(&[2], &[0], p2);

        // Fused decode write: one new position per active lane (0 and 2) at
        // each lane's own offset (3 and 1).
        let step = Tensor::<4>::from_data(
            [[[[10.0, 10.0]]], [[[30.0, 30.0]]]], // rows: lane 0, lane 2
            &device,
        );
        let out = cache.append_lanes(&[0, 2], &[3, 1], step);
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

        // Lane 0 is recycled from position 0, overwriting its old row.
        let fresh = Tensor::<4>::full([1, 1, 2, 2], 7.0, &device);
        let out = cache.append_lanes(&[0], &[0], fresh);
        out.to_data()
            .assert_eq(&TensorData::from([[[[7.0f32, 7.0], [7.0, 7.0]]]]), false);
    }
}

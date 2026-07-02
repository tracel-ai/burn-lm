use burn::tensor::{Device, Int, Tensor, TensorData};

#[derive(Debug, Clone)]
/// A fixed-size pool of KV blocks that accumulates one autoregressive sequence per lane. Dimension 0
/// is the block id; every other dimension is per-token state. At this stage the mapping from lane to
/// block is hard-wired — lane `j` owns block `j + 1`, one block spans a lane's whole sequence space —
/// so the pool is still the old slab plus one row. What changed is the *spelling*: writes address a
/// block row instead of a lane row, and the read-back gathers the active lanes' blocks by index
/// instead of slicing them out one by one. That spelling is what a paged cache needs; the allocator
/// and per-lane block tables that make the mapping dynamic come next, and can then be proven
/// separately from the tensor math proven here.
///
/// Block 0 is a zeroed sentinel no lane ever owns. Once lanes span multiple smaller blocks, ragged
/// gathers pad short lanes with it, so padding can never read a live block — even a masking bug then
/// exposes zeros, not another sequence's KV. It is unused while the mapping is hard-wired, but it is
/// part of the layout contract, so it exists (and is zeroed) from the start.
///
/// This cache keeps no length counter of its own: the caller owns the per-lane lengths (in
/// `TransformerCache`) and passes each lane's write offset into `append_lanes`, so a lane can be
/// recycled simply by writing it again from offset 0.
pub(crate) struct AutoregressiveCache<const D: usize> {
    /// `[num_blocks, ...per-token state]`; row 0 is the sentinel, lane `j` owns row `j + 1`.
    pool: Tensor<D>,
    seq_dim: usize,
}

impl<const D: usize> AutoregressiveCache<D> {
    /// Creates an empty cache for `shape[0]` lanes, each spanning `shape[seq_dim]` positions. The
    /// pool allocates one block per lane plus the sentinel; only the sentinel is initialized (to
    /// zeros) — lane blocks hold garbage until written, exactly like the old slab.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device) -> Self {
        debug_assert_ne!(seq_dim, 0, "dimension 0 is the block id");
        let mut pool_shape = shape;
        pool_shape[0] += 1; // the sentinel
        let pool = Tensor::empty(pool_shape, device);

        // Zero the sentinel row. `full`'s shape is the pool row: `[1, ...per-token state]`.
        let mut row_shape = pool_shape;
        row_shape[0] = 1;
        let zeros = Tensor::zeros(row_shape, device);
        let idx: Vec<_> = pool_shape.iter().map(|&d| 0..d).collect();
        let mut sentinel_idx = idx;
        sentinel_idx[0] = 0..1;
        let pool = pool.slice_assign(sentinel_idx.as_slice(), zeros);

        Self { pool, seq_dim }
    }

    /// The block row backing `lane` while the mapping is hard-wired: row 0 is the sentinel.
    fn block_of(lane: usize) -> usize {
        lane + 1
    }

    /// Write each active lane's new tokens into its block, then gather the active lanes back. Row `j`
    /// of `tokens` lands in lane `lanes[j]`'s block at offset `starts[j]`, which the caller supplies
    /// as that lane's length before this write.
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
        debug_assert_ne!(self.seq_dim, 0, "block dimension is dim 0");
        let shape = tokens.shape();
        debug_assert_eq!(shape[0], lanes.len());
        debug_assert_eq!(starts.len(), lanes.len());
        let seq_len_input = shape[self.seq_dim];

        for (j, &lane) in lanes.iter().enumerate() {
            let start = starts[j];
            let block = Self::block_of(lane);
            let mut value_idx = Vec::with_capacity(shape.len());
            let mut pool_idx = Vec::with_capacity(shape.len());
            for (i, dim) in shape.iter().enumerate() {
                if i == 0 {
                    value_idx.push(j..j + 1);
                    pool_idx.push(block..block + 1);
                } else if i == self.seq_dim {
                    value_idx.push(0..seq_len_input);
                    pool_idx.push(start..start + seq_len_input);
                } else {
                    value_idx.push(0..*dim);
                    pool_idx.push(0..*dim);
                }
            }
            let row = tokens.clone().slice(value_idx.as_slice());
            self.pool
                .inplace(|pool| pool.slice_assign(pool_idx.as_slice(), row));
        }

        // Gather the active lanes back, ragged tails included (mask handles them). This is the read
        // spelling a paged cache needs — one indexed gather over the block dimension — kept in one
        // place so a paged-attention kernel can later replace it behind the same output contract.
        let l_max = starts.iter().map(|&s| s + seq_len_input).max().unwrap();
        self.gather_blocks(lanes, l_max)
    }

    /// Read the given lanes' blocks as one `[n, ..., l_max, ...]` tensor: select each lane's block by
    /// id, then trim the sequence dimension to `l_max`. Whole blocks are selected before the trim —
    /// the granularity cost of paging — so this leans on the backend fusing select+slice; the decode
    /// latency gate measures whether it does.
    fn gather_blocks(&self, lanes: &[usize], l_max: usize) -> Tensor<D> {
        let ids: Vec<i32> = lanes.iter().map(|&l| Self::block_of(l) as i32).collect();
        let n = ids.len();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(ids, [n]), &self.pool.device());
        let pool_shape = self.pool.shape();
        let mut trim: Vec<_> = pool_shape.iter().map(|&d| 0..d).collect();
        trim[0] = 0..n;
        trim[self.seq_dim] = 0..l_max;
        self.pool.clone().select(0, idx).slice(trim.as_slice())
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

    /// KV-contents equivalence with the slab semantics this pool replaced: a scripted mix of
    /// chunked prefills (appends at `position > 0`), interleaved lanes, and a fused decode write is
    /// read back and compared element-for-element against the tensor the slab rules dictate. The
    /// batched-equivalence suite checks logits; this checks the stored bytes directly, so a
    /// write-address or gather-order bug is caught before attention numerics can hide it.
    #[test]
    fn scripted_appends_reproduce_slab_contents_exactly() {
        let device = Default::default();
        // [n_lanes=2, heads=1, max_seq_len=6, head_dim=1]; every position holds one distinct value
        // `100·(lane+1) + position`, so any misplacement changes some element.
        let mut cache = AutoregressiveCache::<4>::new([2, 1, 6, 1], 2, &device);
        let vals = |lane: usize, range: std::ops::Range<usize>| {
            let data: Vec<f32> = range.map(|p| (100 * (lane + 1) + p) as f32).collect();
            let len = data.len();
            Tensor::<4>::from_data(TensorData::new(data, [1, 1, len, 1]), &device)
        };

        // Lane 0 prefills 4 tokens in two chunks (the chunked-prefill shape: second append lands at
        // position 2, continuing where the first ended). Lane 1 prefills 3 tokens in one shot.
        cache.append_lanes(&[0], &[0], vals(0, 0..2));
        cache.append_lanes(&[0], &[2], vals(0, 2..4));
        cache.append_lanes(&[1], &[0], vals(1, 0..3));

        // One fused decode round: lane 0 writes position 4, lane 1 writes position 3.
        let step = Tensor::<4>::from_data(
            TensorData::new(vec![104.0f32, 203.0], [2, 1, 1, 1]),
            &device,
        );
        let out = cache.append_lanes(&[0, 1], &[4, 3], step);

        // Expected, by the slab rules: lane rows in call order, each lane's own values at its own
        // positions, l_max = 5. Lane 1's column 4 is stale tail — asserted separately below, not here.
        assert_eq!(out.dims(), [2, 1, 5, 1]);
        out.clone()
            .slice([0..1, 0..1, 0..5, 0..1])
            .to_data()
            .assert_eq(
                &TensorData::from([[[[100.0f32], [101.0], [102.0], [103.0], [104.0]]]]),
                false,
            );
        out.slice([1..2, 0..1, 0..4, 0..1]).to_data().assert_eq(
            &TensorData::from([[[[200.0f32], [201.0], [202.0], [203.0]]]]),
            false,
        );
    }

    /// The sentinel block is zeroed at construction and no append may touch it: after writes to
    /// every lane, gathering the sentinel directly still reads back zeros.
    #[test]
    fn sentinel_block_stays_zeroed() {
        let device = Default::default();
        let mut cache = AutoregressiveCache::<4>::new([2, 1, 4, 1], 2, &device);
        cache.append_lanes(&[0, 1], &[0, 0], Tensor::<4>::full([2, 1, 4, 1], 9.0, &device));

        // Reach past the lane mapping and read block 0 (the sentinel) directly from the pool.
        let sentinel = cache.pool.clone().slice([0..1, 0..1, 0..4, 0..1]);
        sentinel
            .to_data()
            .assert_eq(&TensorData::from([[[[0.0f32], [0.0], [0.0], [0.0]]]]), false);
    }
}

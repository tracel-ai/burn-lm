use burn::tensor::{Device, Int, Tensor, TensorData};

use super::block_pool::SENTINEL_BLOCK;

#[derive(Debug, Clone)]
/// A fixed-size pool of KV blocks. Dimension 0 is the block id; every other dimension is per-token
/// state. This type is purely physical: it neither knows which lane owns which block nor tracks any
/// lengths — the caller (`TransformerCache`, via the `BlockPool` allocator) hands every call the
/// block table and write offset to use, so the same pool serves any lane-to-block assignment. Writes
/// address a block row; the read-back gathers whole blocks by id and trims to the requested length.
///
/// Block 0 is a zeroed sentinel no lane ever owns (the `BlockPool` never allocates it). Ragged
/// gathers pad short lanes with it, so padding can never read a live block — even a masking bug then
/// exposes zeros, not another sequence's KV.
pub(crate) struct AutoregressiveCache<const D: usize> {
    /// `[num_blocks, ...per-token state]`; row 0 is the sentinel.
    pool: Tensor<D>,
    seq_dim: usize,
}

impl<const D: usize> AutoregressiveCache<D> {
    /// Creates an empty pool of `shape[0]` blocks, each spanning `shape[seq_dim]` positions. Only
    /// the sentinel (block 0) is initialized, to zeros; every other block holds garbage until
    /// written, exactly like the old slab.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device) -> Self {
        debug_assert_ne!(seq_dim, 0, "dimension 0 is the block id");
        let pool = Tensor::empty(shape, device);

        // Zero the sentinel row.
        let mut row_shape = shape;
        row_shape[0] = 1;
        let zeros = Tensor::zeros(row_shape, device);
        let mut sentinel_idx: Vec<_> = shape.iter().map(|&d| 0..d).collect();
        sentinel_idx[0] = 0..1;
        let pool = pool.slice_assign(sentinel_idx.as_slice(), zeros);

        Self { pool, seq_dim }
    }

    /// Tokens per block: the sequence extent of one block row.
    fn block_size(&self) -> usize {
        self.pool.shape()[self.seq_dim]
    }

    /// Write each active lane's new tokens into its blocks, then gather the active lanes back. Row
    /// `j` of `tokens` lands in the blocks of `tables[j]` at logical offset `starts[j]`, which the
    /// caller supplies as that lane's length before this write; `tables[j]` must already cover
    /// `starts[j] + seq_len_input` positions (the allocator grew it before the forward).
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
        tables: &[Vec<u32>],
        starts: &[usize],
        tokens: Tensor<D>,
    ) -> Tensor<D> {
        debug_assert_ne!(self.seq_dim, 0, "block dimension is dim 0");
        let shape = tokens.shape();
        debug_assert_eq!(shape[0], tables.len());
        debug_assert_eq!(starts.len(), tables.len());
        let seq_len_input = shape[self.seq_dim];
        let bs = self.block_size();

        for (j, table) in tables.iter().enumerate() {
            let start = starts[j];
            // The write targets one block: the one covering `start`, at the block-local offset.
            // Splitting a write that crosses a block boundary arrives with the smaller block sizes
            // of the next stage; while a block spans the whole sequence space this cannot trigger.
            let offset = start % bs;
            debug_assert!(
                offset + seq_len_input <= bs,
                "write [{start}, {}) crosses a block boundary; splitting lands in the next stage",
                start + seq_len_input
            );
            let block = table[start / bs] as usize;
            let mut value_idx = Vec::with_capacity(shape.len());
            let mut pool_idx = Vec::with_capacity(shape.len());
            for (i, dim) in shape.iter().enumerate() {
                if i == 0 {
                    value_idx.push(j..j + 1);
                    pool_idx.push(block..block + 1);
                } else if i == self.seq_dim {
                    value_idx.push(0..seq_len_input);
                    pool_idx.push(offset..offset + seq_len_input);
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
        self.gather_blocks(tables, l_max)
    }

    /// Read the given lanes' blocks as one `[n, ..., l_max, ...]` tensor: select each lane's
    /// covering blocks by id — padding lanes shorter than the batch max with the zeroed sentinel —
    /// then trim the sequence dimension to `l_max`. Whole blocks are selected before the trim (the
    /// granularity cost of paging), so this leans on the backend fusing select+slice; the decode
    /// latency gate measures whether it does.
    fn gather_blocks(&self, tables: &[Vec<u32>], l_max: usize) -> Tensor<D> {
        let bs = self.block_size();
        let nb = l_max.div_ceil(bs);
        debug_assert_eq!(nb, 1, "multi-block gather lands in the next stage");
        let ids: Vec<i32> = tables
            .iter()
            .flat_map(|table| (0..nb).map(|i| *table.get(i).unwrap_or(&SENTINEL_BLOCK) as i32))
            .collect();
        let n = tables.len();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(ids, [n * nb]), &self.pool.device());
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
    /// and recycling a lane's block (offset 0) overwrites it without touching
    /// its siblings. Block ids are deliberately not `lane + 1` — the cache must
    /// follow the tables it is handed, nothing else.
    #[test]
    fn test_append_lanes_ragged_positions_and_reset_lane() {
        let device = Default::default();
        // [num_blocks=4 (sentinel + 3), heads=1, block_size=8, head_dim=2]
        let mut cache = AutoregressiveCache::<4>::new([4, 1, 8, 2], 2, &device);
        let t0 = vec![3u32]; // lane 0 -> block 3
        let t2 = vec![1u32]; // lane 2 -> block 1

        // Prefill lane 0 with 3 positions, lane 2 with 1 position.
        let p0 = Tensor::<4>::full([1, 1, 3, 2], 1.0, &device);
        cache.append_lanes(std::slice::from_ref(&t0), &[0], p0);
        let p2 = Tensor::<4>::full([1, 1, 1, 2], 3.0, &device);
        cache.append_lanes(std::slice::from_ref(&t2), &[0], p2);

        // Fused decode write: one new position per active lane at each lane's
        // own offset (3 and 1).
        let step = Tensor::<4>::from_data(
            [[[[10.0, 10.0]]], [[[30.0, 30.0]]]], // rows: lane 0, lane 2
            &device,
        );
        let out = cache.append_lanes(&[t0.clone(), t2.clone()], &[3, 1], step);
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

        // Lane 0's block is recycled from position 0, overwriting its old contents.
        let fresh = Tensor::<4>::full([1, 1, 2, 2], 7.0, &device);
        let out = cache.append_lanes(std::slice::from_ref(&t0), &[0], fresh);
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
        // [num_blocks=3, heads=1, block_size=6, head_dim=1]; every position holds one distinct value
        // `100·(lane+1) + position`, so any misplacement changes some element.
        let mut cache = AutoregressiveCache::<4>::new([3, 1, 6, 1], 2, &device);
        let t0 = vec![2u32];
        let t1 = vec![1u32];
        let vals = |lane: usize, range: std::ops::Range<usize>| {
            let data: Vec<f32> = range.map(|p| (100 * (lane + 1) + p) as f32).collect();
            let len = data.len();
            Tensor::<4>::from_data(TensorData::new(data, [1, 1, len, 1]), &device)
        };

        // Lane 0 prefills 4 tokens in two chunks (the chunked-prefill shape: second append lands at
        // position 2, continuing where the first ended). Lane 1 prefills 3 tokens in one shot.
        cache.append_lanes(std::slice::from_ref(&t0), &[0], vals(0, 0..2));
        cache.append_lanes(std::slice::from_ref(&t0), &[2], vals(0, 2..4));
        cache.append_lanes(std::slice::from_ref(&t1), &[0], vals(1, 0..3));

        // One fused decode round: lane 0 writes position 4, lane 1 writes position 3.
        let step = Tensor::<4>::from_data(
            TensorData::new(vec![104.0f32, 203.0], [2, 1, 1, 1]),
            &device,
        );
        let out = cache.append_lanes(&[t0, t1], &[4, 3], step);

        // Expected, by the slab rules: lane rows in call order, each lane's own values at its own
        // positions, l_max = 5. Lane 1's column 4 is stale tail — not asserted.
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
    /// every live block, gathering the sentinel directly still reads back zeros.
    #[test]
    fn sentinel_block_stays_zeroed() {
        let device = Default::default();
        let mut cache = AutoregressiveCache::<4>::new([3, 1, 4, 1], 2, &device);
        cache.append_lanes(
            &[vec![1u32], vec![2u32]],
            &[0, 0],
            Tensor::<4>::full([2, 1, 4, 1], 9.0, &device),
        );

        // Reach past the tables and read block 0 (the sentinel) directly from the pool.
        let sentinel = cache.pool.clone().slice([0..1, 0..1, 0..4, 0..1]);
        sentinel
            .to_data()
            .assert_eq(&TensorData::from([[[[0.0f32], [0.0], [0.0], [0.0]]]]), false);
    }
}

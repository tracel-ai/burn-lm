use burn::tensor::{Device, Int, Tensor};

#[derive(Debug, Clone)]
/// A fixed-size pool of KV blocks, shaped `[num_blocks, num_heads, block_size, head_dim]`. This type
/// is purely physical: it neither knows which lane owns which block nor tracks any lengths — the
/// caller hands every write the block tables and offsets to use, and every read the block ids to
/// gather, so the same store serves any lane-to-block assignment. A lane's positions map onto its
/// table in order: position `p` lives in block `table[p / block_size]` at offset `p % block_size`.
///
/// Block 0 is a zeroed sentinel no lane ever owns (the `BlockPool` never allocates it). Ragged
/// gathers pad short lanes with it, so padding can never read a live block — even a masking bug then
/// exposes zeros, not another sequence's KV.
pub(crate) struct BlockStore {
    /// `[num_blocks, num_heads, block_size, head_dim]`; row 0 is the sentinel.
    pool: Tensor<4>,
}

impl BlockStore {
    /// Creates an empty store of `shape[0]` blocks, each spanning `shape[2]` positions. Only the
    /// sentinel (block 0) is initialized, to zeros; every other block holds garbage until written,
    /// exactly like the old slab.
    pub fn new(shape: [usize; 4], device: &Device) -> Self {
        let pool = Tensor::empty(shape, device);

        // Zero the sentinel row.
        let [_, heads, bs, head_dim] = shape;
        let zeros = Tensor::zeros([1, heads, bs, head_dim], device);
        let pool = pool.slice_assign([0..1, 0..heads, 0..bs, 0..head_dim], zeros);

        Self { pool }
    }

    /// Tokens per block: the sequence extent of one block row.
    fn block_size(&self) -> usize {
        self.pool.shape()[2]
    }

    /// Write each active lane's new tokens into its blocks. Row `j` of `tokens` lands in the blocks
    /// of `tables[j]` starting at logical position `starts[j]`, which the caller supplies as that
    /// lane's length before this write; `tables[j]` must already cover `starts[j] + seq_len_input`
    /// positions (the allocator grew it before the forward).
    ///
    /// A write that crosses block boundaries is split at each edge, one `slice_assign` per touched
    /// block: a prefill chunk continues in the partial tail block the previous chunk left, fills it,
    /// and spills the rest into the following blocks of the table. The single-token decode write —
    /// the hot case — never splits.
    ///
    /// # Shapes
    ///
    /// - tokens: `[n_active, num_heads, seq_len_input, head_dim]`
    pub fn write_lanes(&mut self, tables: &[Vec<u32>], starts: &[usize], tokens: Tensor<4>) {
        let [n, heads, seq_len_input, head_dim] = tokens.dims();
        debug_assert_eq!(n, tables.len());
        debug_assert_eq!(starts.len(), tables.len());
        let bs = self.block_size();

        for (j, table) in tables.iter().enumerate() {
            // Walk the lane's new tokens block by block: each pass writes the largest piece that
            // fits in the block covering the current position, then moves to the next block edge.
            let mut written = 0;
            while written < seq_len_input {
                let position = starts[j] + written;
                let block = table[position / bs] as usize;
                let offset = position % bs;
                let piece = (bs - offset).min(seq_len_input - written);
                let row = tokens.clone().slice([
                    j..j + 1,
                    0..heads,
                    written..written + piece,
                    0..head_dim,
                ]);
                self.pool.inplace(|pool| {
                    pool.slice_assign(
                        [block..block + 1, 0..heads, offset..offset + piece, 0..head_dim],
                        row,
                    )
                });
                written += piece;
            }
        }
    }

    /// Read `l_max` positions for each of `n` lanes as one `[n, heads, l_max, head_dim]` tensor,
    /// from a caller-built gather index: `idx` holds `n · blocks_per_lane` block ids, each lane's
    /// covering blocks in position order, short lanes padded with the sentinel. The index is a pure
    /// function of the round's tables, so the caller (`prepare_lanes`) builds and uploads it once
    /// per round and every layer's K and V reuse the same handle.
    ///
    /// The blocks are selected in one indexed gather, stitched back into a contiguous sequence, and
    /// trimmed to `l_max`. Whole blocks are selected before the trim (the granularity cost of
    /// paging, at most `block_size - 1` wasted columns per lane), and this leans on the backend
    /// fusing the chain; the decode latency gate measures whether it does. This is the one physical
    /// read — a paged-attention kernel would replace it behind the same output contract.
    ///
    /// The lanes sit at independent positions, so shorter lanes come back with a stale tail past
    /// their own length. The caller MUST mask that tail with the per-lane padding mask; this store
    /// does not zero it.
    pub fn gather(&self, idx: Tensor<1, Int>, blocks_per_lane: usize, l_max: usize) -> Tensor<4> {
        let [_, heads, bs, head_dim] = self.pool.dims();
        let nb = blocks_per_lane;
        let n = idx.dims()[0] / nb;
        // Select every lane's covering blocks, then stitch each lane's blocks into one contiguous
        // sequence axis: [n·nb, h, bs, d] -> [n, nb, h, bs, d] -> [n, h, nb, bs, d] -> [n, h, nb·bs, d].
        //
        // The clone is a handle (refcount bump), not a copy of the pool — and it must stay AFTER the
        // writes: `write_lanes` mutates through `inplace`/`slice_assign`, which only skips a full
        // copy while the pool handle is uniquely owned. A pool clone held across the writes would
        // turn every layer's KV write into a copy-on-write of the whole pool.
        self.pool
            .clone()
            .select(0, idx)
            .reshape([n, nb, heads, bs, head_dim])
            .swap_dims(1, 2)
            .reshape([n, heads, nb * bs, head_dim])
            .slice([0..n, 0..heads, 0..l_max, 0..head_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::attention::block_pool::SENTINEL_BLOCK;
    use burn::tensor::TensorData;

    /// A `[1, 1, len, 1]` tokens tensor holding `base + position` at each position, so any
    /// misplacement changes some element.
    fn vals(base: usize, range: std::ops::Range<usize>) -> Tensor<4> {
        let data: Vec<f32> = range.map(|p| (base + p) as f32).collect();
        let len = data.len();
        Tensor::<4>::from_data(TensorData::new(data, [1, 1, len, 1]), &Default::default())
    }

    /// The expected `[1, 1, n, 1]` read-back for `base + position` over `0..n`.
    fn expect(base: usize, n: usize) -> TensorData {
        TensorData::new(
            (0..n).map(|p| (base + p) as f32).collect::<Vec<f32>>(),
            [1, 1, n, 1],
        )
    }

    /// The gather index `prepare_lanes` would build: each table's blocks in order, sentinel-padded
    /// to `nb` entries per lane.
    fn idx_for(tables: &[Vec<u32>], nb: usize) -> Tensor<1, Int> {
        let ids: Vec<i32> = tables
            .iter()
            .flat_map(|t| (0..nb).map(|i| *t.get(i).unwrap_or(&SENTINEL_BLOCK) as i32))
            .collect();
        let n = ids.len();
        Tensor::from_data(TensorData::new(ids, [n]), &Default::default())
    }

    /// Lanes at divergent positions: ragged writes land at each lane's own caller-supplied offset,
    /// the read-back covers the longest active lane, and recycling a lane's block (offset 0)
    /// overwrites it without touching its siblings. Block ids are deliberately not `lane + 1` — the
    /// store must follow the tables it is handed, nothing else.
    #[test]
    fn test_write_lanes_ragged_positions_and_reset_lane() {
        let device = Default::default();
        // [num_blocks=4 (sentinel + 3), heads=1, block_size=8, head_dim=2]
        let mut store = BlockStore::new([4, 1, 8, 2], &device);
        let t0 = vec![3u32]; // lane 0 -> block 3
        let t2 = vec![1u32]; // lane 2 -> block 1

        // Prefill lane 0 with 3 positions, lane 2 with 1 position.
        store.write_lanes(std::slice::from_ref(&t0), &[0], Tensor::full([1, 1, 3, 2], 1.0, &device));
        store.write_lanes(std::slice::from_ref(&t2), &[0], Tensor::full([1, 1, 1, 2], 3.0, &device));

        // Fused decode write: one new position per active lane at each lane's own offset (3 and 1).
        let step = Tensor::<4>::from_data(
            [[[[10.0, 10.0]]], [[[30.0, 30.0]]]], // rows: lane 0, lane 2
            &device,
        );
        let tables = [t0.clone(), t2.clone()];
        store.write_lanes(&tables, &[3, 1], step);
        let out = store.gather(idx_for(&tables, 1), 1, 4);
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
        store.write_lanes(std::slice::from_ref(&t0), &[0], Tensor::full([1, 1, 2, 2], 7.0, &device));
        let out = store.gather(idx_for(std::slice::from_ref(&t0), 1), 1, 2);
        out.to_data()
            .assert_eq(&TensorData::from([[[[7.0f32, 7.0], [7.0, 7.0]]]]), false);
    }

    /// KV-contents equivalence with the slab semantics this store replaced: a scripted mix of
    /// chunked prefills (writes at `position > 0`), interleaved lanes, and a fused decode write is
    /// read back and compared element-for-element against the tensor the slab rules dictate. The
    /// batched-equivalence suite checks logits; this checks the stored bytes directly, so a
    /// write-address or gather-order bug is caught before attention numerics can hide it.
    #[test]
    fn scripted_writes_reproduce_slab_contents_exactly() {
        let device = Default::default();
        // [num_blocks=3, heads=1, block_size=6, head_dim=1]
        let mut store = BlockStore::new([3, 1, 6, 1], &device);
        let t0 = vec![2u32];
        let t1 = vec![1u32];

        // Lane 0 prefills 4 tokens in two chunks (the chunked-prefill shape: second write lands at
        // position 2, continuing where the first ended). Lane 1 prefills 3 tokens in one shot.
        store.write_lanes(std::slice::from_ref(&t0), &[0], vals(100, 0..2));
        store.write_lanes(std::slice::from_ref(&t0), &[2], vals(100, 2..4));
        store.write_lanes(std::slice::from_ref(&t1), &[0], vals(200, 0..3));

        // One fused decode round: lane 0 writes position 4, lane 1 writes position 3.
        let step = Tensor::<4>::from_data(
            TensorData::new(vec![104.0f32, 203.0], [2, 1, 1, 1]),
            &device,
        );
        let tables = [t0, t1];
        store.write_lanes(&tables, &[4, 3], step);
        let out = store.gather(idx_for(&tables, 1), 1, 5);

        // Expected, by the slab rules: lane rows in call order, each lane's own values at its own
        // positions, l_max = 5. Lane 1's column 4 is stale tail — not asserted.
        assert_eq!(out.dims(), [2, 1, 5, 1]);
        out.clone()
            .slice([0..1, 0..1, 0..5, 0..1])
            .to_data()
            .assert_eq(&expect(100, 5), false);
        out.slice([1..2, 0..1, 0..4, 0..1])
            .to_data()
            .assert_eq(&expect(200, 4), false);
    }

    /// A prefill chunk spanning several small blocks is split at every block edge and lands intact:
    /// start offset 2 into a half-full tail block, 9 more tokens across blocks of 4 — pieces of
    /// 2 + 4 + 3 — then a decode token crossing into a fresh block. Exercises the worked example of
    /// the plan: fill the partial tail, spill into following blocks, continue at `position > 0`.
    #[test]
    fn writes_split_at_block_boundaries_and_read_back_contiguously() {
        let device = Default::default();
        // [num_blocks=5 (sentinel + 4), heads=1, block_size=4, head_dim=1]
        let mut store = BlockStore::new([5, 1, 4, 1], &device);
        // Deliberately unordered, non-contiguous ids: position i·4.. lives in table[i].
        let table = vec![3u32, 1, 4];

        // Chunk 1: positions [0, 2) — a partial tail block.
        store.write_lanes(std::slice::from_ref(&table), &[0], vals(500, 0..2));
        // Chunk 2: positions [2, 11) — fills block 3's tail (2), all of block 1 (4), part of 4 (3).
        store.write_lanes(std::slice::from_ref(&table), &[2], vals(500, 2..11));
        let out = store.gather(idx_for(std::slice::from_ref(&table), 3), 3, 11);
        assert_eq!(out.dims(), [1, 1, 11, 1]);
        out.to_data().assert_eq(&expect(500, 11), false);

        // Decode step at position 11: the last slot of table[2]; then position 12 needs a fourth
        // block — grow the table first, as the allocator does, and write again.
        store.write_lanes(std::slice::from_ref(&table), &[11], vals(500, 11..12));
        let grown = vec![3u32, 1, 4, 2];
        store.write_lanes(std::slice::from_ref(&grown), &[12], vals(500, 12..13));
        let out = store.gather(idx_for(std::slice::from_ref(&grown), 4), 4, 13);
        out.to_data().assert_eq(&expect(500, 13), false);
    }

    /// Ragged multi-lane gather with small blocks: the long lane sets `l_max`, the short lane's
    /// missing blocks come back as the zeroed sentinel — provably zeros, not another lane's data —
    /// and its stale tail inside its own last block is whatever it is (the mask's job, not asserted).
    #[test]
    fn short_lanes_pad_with_the_sentinel_never_a_live_block() {
        let device = Default::default();
        // [num_blocks=5 (sentinel + 4), heads=1, block_size=2, head_dim=1]
        let mut store = BlockStore::new([5, 1, 2, 1], &device);
        let long = vec![1u32, 2]; // positions 0..4
        let short = vec![3u32]; // positions 0..2

        store.write_lanes(std::slice::from_ref(&long), &[0], vals(700, 0..4));
        store.write_lanes(std::slice::from_ref(&short), &[0], vals(900, 0..1));

        // Fused decode: long at position 4 needs a third block (4); short at position 1.
        let long_grown = vec![1u32, 2, 4];
        let step = Tensor::<4>::from_data(TensorData::new(vec![704.0f32, 901.0], [2, 1, 1, 1]), &device);
        let tables = [long_grown, short];
        store.write_lanes(&tables, &[4, 1], step);
        // l_max = 5 -> 3 blocks per lane; `short` has one, so entries 2 and 3 of its index row are
        // the sentinel: columns [2, 5) of its row must read back exactly zero.
        let out = store.gather(idx_for(&tables, 3), 3, 5);
        assert_eq!(out.dims(), [2, 1, 5, 1]);
        out.clone()
            .slice([0..1, 0..1, 0..5, 0..1])
            .to_data()
            .assert_eq(&expect(700, 5), false);
        out.clone()
            .slice([1..2, 0..1, 0..2, 0..1])
            .to_data()
            .assert_eq(&expect(900, 2), false);
        out.slice([1..2, 0..1, 2..5, 0..1]).to_data().assert_eq(
            &TensorData::new(vec![0.0f32; 3], [1, 1, 3, 1]),
            false,
        );
    }

    /// The sentinel block is zeroed at construction and no write may touch it: after writes to
    /// every live block, gathering the sentinel directly still reads back zeros.
    #[test]
    fn sentinel_block_stays_zeroed() {
        let device = Default::default();
        let mut store = BlockStore::new([3, 1, 4, 1], &device);
        store.write_lanes(
            &[vec![1u32], vec![2u32]],
            &[0, 0],
            Tensor::<4>::full([2, 1, 4, 1], 9.0, &device),
        );

        // Reach past the tables and read block 0 (the sentinel) directly from the pool.
        let sentinel = store.pool.clone().slice([0..1, 0..1, 0..4, 0..1]);
        sentinel
            .to_data()
            .assert_eq(&TensorData::from([[[[0.0f32], [0.0], [0.0], [0.0]]]]), false);
    }
}

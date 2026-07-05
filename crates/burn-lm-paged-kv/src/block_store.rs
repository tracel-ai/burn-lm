use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor};

#[derive(Debug, Clone)]
/// A fixed-size pool of KV blocks, shaped `[num_blocks, block_size, num_heads, head_dim]` —
/// token-major, so a logical position is one row of the leading two dims and a whole round's
/// writes land in ONE indexed scatter, regardless of how many lanes wrote or how many block
/// boundaries their tokens cross. This type is purely physical: it neither knows which lane owns
/// which block nor tracks any lengths — the caller hands every write its destination indices and
/// every read the block ids to gather, so the same store serves any lane-to-block assignment.
///
/// Block 0 is a zeroed sentinel no lane ever owns (the `BlockPool` never allocates it). Ragged
/// gathers pad short lanes with it, so padding can never read a live block — even a masking bug
/// then exposes zeros, not another sequence's KV.
pub struct BlockStore {
    /// `[num_blocks, block_size, num_heads, head_dim]`; block 0 is the sentinel.
    pool: Tensor<4>,
}

impl BlockStore {
    /// Creates an empty store of `num_blocks` blocks spanning `block_size` positions each. Only
    /// the sentinel (block 0) is initialized, to zeros; every other block holds garbage until
    /// written, exactly like the old slab.
    pub(crate) fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        let pool = Tensor::empty([num_blocks, block_size, num_heads, head_dim], device);
        let zeros = Tensor::zeros([1, block_size, num_heads, head_dim], device);
        let pool = pool.slice_assign([0..1, 0..block_size, 0..num_heads, 0..head_dim], zeros);
        Self { pool }
    }

    /// Tokens per block.
    fn block_size(&self) -> usize {
        self.pool.shape()[1]
    }

    /// Write one round's new tokens in a single indexed scatter. `rows` is the round's fresh K or
    /// V, `[n, num_heads, seq_len, head_dim]` (one lane per row, as the projection produces it);
    /// `write_idx` is `[n·seq_len, 2]` of `(block, offset)` destinations in lane-major, position
    /// order — built once per round by `prepare_lanes` and shared by every layer's K and V. Tokens
    /// crossing block boundaries are nothing special: a boundary is just a different index pair.
    ///
    /// The indices are unique by construction (each destination is written exactly once per
    /// round), which is what makes `Assign` well-defined. One kernel launch per call, whatever the
    /// width — this is the generic-op stand-in for a dedicated cache-write kernel (vLLM's
    /// `reshape_and_cache`), and it already takes that kernel's exact inputs.
    pub(crate) fn write(&mut self, write_idx: &Tensor<2, Int>, rows: Tensor<4>) {
        let [n, heads, seq_len, head_dim] = rows.dims();
        // [n, heads, seq, d] -> [n·seq, heads, d]: token-major to match the pool layout.
        let rows = rows.swap_dims(1, 2).reshape([n * seq_len, heads, head_dim]);
        let idx = write_idx.clone();
        self.pool
            .inplace(|pool| pool.scatter_nd(idx, rows, IndexingUpdateOp::Assign));
    }

    /// Read `l_max` positions for each of `n` lanes as one `[n, num_heads, l_max, head_dim]`
    /// tensor, from a caller-built gather index: `idx` holds `n · blocks_per_lane` block ids, each
    /// lane's covering blocks in position order, short lanes padded with the sentinel. The index is
    /// a pure function of the round's tables, so the caller (`prepare_lanes`) builds and uploads it
    /// once per round and every layer's K and V reuse the same handle.
    ///
    /// The blocks are selected in one indexed gather, stitched back into a contiguous sequence, and
    /// trimmed to `l_max`. Whole blocks are selected before the trim (the granularity cost of
    /// paging, at most `block_size - 1` wasted columns per lane), and this leans on the backend
    /// fusing the chain; the decode latency gate measures whether it does.
    ///
    /// The lanes sit at independent positions, so shorter lanes come back with a stale tail past
    /// their own length. The caller MUST mask that tail with the per-lane padding mask; this store
    /// does not zero it.
    pub(crate) fn gather(
        &self,
        idx: Tensor<1, Int>,
        blocks_per_lane: usize,
        l_max: usize,
    ) -> Tensor<4> {
        let [_, bs, heads, head_dim] = self.pool.dims();
        let nb = blocks_per_lane;
        let n = idx.dims()[0] / nb;
        // Select every lane's covering blocks, stitch each lane's blocks into one contiguous
        // sequence axis, then put heads ahead of positions for attention:
        // [n·nb, bs, h, d] -> [n, nb·bs, h, d] -> [n, h, nb·bs, d] -> trim to l_max.
        //
        // The clone is a handle (refcount bump), not a copy of the pool — and it must stay AFTER
        // the writes: `write` mutates through `inplace`/`scatter_nd`, which only skips a full copy
        // while the pool handle is uniquely owned. A pool clone held across the writes would turn
        // every layer's KV write into a copy-on-write of the whole pool.
        self.pool
            .clone()
            .select(0, idx)
            .reshape([n, nb * bs, heads, head_dim])
            .swap_dims(1, 2)
            .slice([0..n, 0..heads, 0..l_max, 0..head_dim])
    }
}

/// Build the `(block, offset)` destination pairs for one round's writes: for each lane in order,
/// its `seq_len` new positions starting at `starts[j]`, translated through its block table. Plain
/// index arithmetic — a token past a block edge simply lands in the next table entry, which is how
/// chunked prefill's boundary crossings cost nothing special.
pub(crate) fn write_indices(
    tables: &[Vec<u32>],
    starts: &[usize],
    seq_len: usize,
    block_size: usize,
) -> Vec<i32> {
    let mut ids = Vec::with_capacity(tables.len() * seq_len * 2);
    for (table, &start) in tables.iter().zip(starts) {
        for p in start..start + seq_len {
            ids.push(table[p / block_size] as i32);
            ids.push((p % block_size) as i32);
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::SENTINEL_BLOCK;
    use burn::tensor::TensorData;

    /// A `[1, 1, len, 1]` rows tensor holding `base + position` at each position, so any
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

    /// `write` through host-built indices, as `prepare_lanes` does per round.
    fn write(store: &mut BlockStore, tables: &[Vec<u32>], starts: &[usize], rows: Tensor<4>) {
        let seq_len = rows.dims()[2];
        let ids = write_indices(tables, starts, seq_len, store.block_size());
        let n = ids.len() / 2;
        let idx = Tensor::<2, Int>::from_data(TensorData::new(ids, [n, 2]), &Default::default());
        store.write(&idx, rows);
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
    /// store must follow the indices it is handed, nothing else.
    #[test]
    fn test_writes_land_at_ragged_positions_and_blocks_recycle() {
        let device: Device = Default::default();
        // [num_blocks=4 (sentinel + 3), block_size=8, heads=1, head_dim=2]
        let mut store = BlockStore::new(4, 8, 1, 2, &device);
        let t0 = vec![3u32]; // lane 0 -> block 3
        let t2 = vec![1u32]; // lane 2 -> block 1

        write(&mut store, std::slice::from_ref(&t0), &[0], Tensor::full([1, 1, 3, 2], 1.0, &device));
        write(&mut store, std::slice::from_ref(&t2), &[0], Tensor::full([1, 1, 1, 2], 3.0, &device));

        // Fused decode write: one new position per active lane at each lane's own offset (3 and 1).
        let step = Tensor::<4>::from_data([[[[10.0, 10.0]]], [[[30.0, 30.0]]]], &device);
        let tables = [t0.clone(), t2.clone()];
        write(&mut store, &tables, &[3, 1], step);
        let out = store.gather(idx_for(&tables, 1), 1, 4);
        assert_eq!(out.dims(), [2, 1, 4, 2]);
        out.clone().slice([0..1, 0..1, 0..4, 0..2]).to_data().assert_eq(
            &TensorData::from([[[[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0], [10.0, 10.0]]]]),
            false,
        );
        // Lane 2's columns 2..4 are stale tail the mask must cover — not asserted.
        out.slice([1..2, 0..1, 0..2, 0..2])
            .to_data()
            .assert_eq(&TensorData::from([[[[3.0f32, 3.0], [30.0, 30.0]]]]), false);

        // Lane 0's block is recycled from position 0, overwriting its old contents.
        write(&mut store, std::slice::from_ref(&t0), &[0], Tensor::full([1, 1, 2, 2], 7.0, &device));
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
        let device: Device = Default::default();
        let mut store = BlockStore::new(3, 6, 1, 1, &device);
        let t0 = vec![2u32];
        let t1 = vec![1u32];

        write(&mut store, std::slice::from_ref(&t0), &[0], vals(100, 0..2));
        write(&mut store, std::slice::from_ref(&t0), &[2], vals(100, 2..4));
        write(&mut store, std::slice::from_ref(&t1), &[0], vals(200, 0..3));

        let step =
            Tensor::<4>::from_data(TensorData::new(vec![104.0f32, 203.0], [2, 1, 1, 1]), &device);
        let tables = [t0, t1];
        write(&mut store, &tables, &[4, 3], step);
        let out = store.gather(idx_for(&tables, 1), 1, 5);

        assert_eq!(out.dims(), [2, 1, 5, 1]);
        out.clone()
            .slice([0..1, 0..1, 0..5, 0..1])
            .to_data()
            .assert_eq(&expect(100, 5), false);
        out.slice([1..2, 0..1, 0..4, 0..1])
            .to_data()
            .assert_eq(&expect(200, 4), false);
    }

    /// A prefill chunk spanning several small blocks lands intact through ONE scatter: start offset
    /// 2 into a half-full tail block, 9 more tokens across blocks of 4, then decode tokens crossing
    /// into a fresh block. Boundary crossings are just different index pairs — there is no split
    /// logic left to test, only the arithmetic.
    #[test]
    fn writes_cross_block_boundaries_and_read_back_contiguously() {
        let device: Device = Default::default();
        let mut store = BlockStore::new(5, 4, 1, 1, &device);
        // Deliberately unordered, non-contiguous ids: position i·4.. lives in table[i].
        let table = vec![3u32, 1, 4];

        write(&mut store, std::slice::from_ref(&table), &[0], vals(500, 0..2));
        write(&mut store, std::slice::from_ref(&table), &[2], vals(500, 2..11));
        let out = store.gather(idx_for(std::slice::from_ref(&table), 3), 3, 11);
        assert_eq!(out.dims(), [1, 1, 11, 1]);
        out.to_data().assert_eq(&expect(500, 11), false);

        write(&mut store, std::slice::from_ref(&table), &[11], vals(500, 11..12));
        let grown = vec![3u32, 1, 4, 2];
        write(&mut store, std::slice::from_ref(&grown), &[12], vals(500, 12..13));
        let out = store.gather(idx_for(std::slice::from_ref(&grown), 4), 4, 13);
        out.to_data().assert_eq(&expect(500, 13), false);
    }

    /// Ragged multi-lane gather with small blocks: the long lane sets `l_max`, the short lane's
    /// missing blocks come back as the zeroed sentinel — provably zeros, not another lane's data.
    #[test]
    fn short_lanes_pad_with_the_sentinel_never_a_live_block() {
        let device: Device = Default::default();
        let mut store = BlockStore::new(5, 2, 1, 1, &device);
        let long = vec![1u32, 2]; // positions 0..4
        let short = vec![3u32]; // positions 0..2

        write(&mut store, std::slice::from_ref(&long), &[0], vals(700, 0..4));
        write(&mut store, std::slice::from_ref(&short), &[0], vals(900, 0..1));

        let long_grown = vec![1u32, 2, 4];
        let step =
            Tensor::<4>::from_data(TensorData::new(vec![704.0f32, 901.0], [2, 1, 1, 1]), &device);
        let tables = [long_grown, short];
        write(&mut store, &tables, &[4, 1], step);
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
        out.slice([1..2, 0..1, 2..5, 0..1])
            .to_data()
            .assert_eq(&TensorData::new(vec![0.0f32; 3], [1, 1, 3, 1]), false);
    }

    /// The sentinel block is zeroed at construction and no write may touch it.
    #[test]
    fn sentinel_block_stays_zeroed() {
        let device: Device = Default::default();
        let mut store = BlockStore::new(3, 4, 1, 1, &device);
        write(
            &mut store,
            &[vec![1u32], vec![2u32]],
            &[0, 0],
            Tensor::<4>::full([2, 1, 4, 1], 9.0, &device),
        );
        let sentinel = store.pool.clone().slice([0..1, 0..4, 0..1, 0..1]);
        sentinel
            .to_data()
            .assert_eq(&TensorData::new(vec![0.0f32; 4], [1, 4, 1, 1]), false);
    }
}

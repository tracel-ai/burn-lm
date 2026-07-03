//! Host-side bookkeeping for the paged KV cache: a shared pool of fixed-size blocks and, per lane,
//! the ledger of what the lane holds — its current length and the table mapping its logical token
//! positions onto the blocks it has been allocated.
//!
//! Under the old slab every lane owned a fixed `max_seq_len` stripe, so capacity was the rectangle
//! `max_slots × max_seq_len` — paid in full whether a sequence was a 200-token chat turn or an
//! 8000-token document. This pool replaces the stripe with on-demand blocks: a lane holds only the
//! blocks covering the tokens it has actually written, and capacity becomes the sum of what live
//! sequences actually need. This module is bookkeeping only — lengths, block ids, and tables, no
//! tensors; the storage they index into is the `BlockStore`.
//!
//! The mapping contract: logical position `p` of a lane lives in block `tables[lane][p / block_size]`
//! at offset `p % block_size`. Only the table's *index* is positional — the block ids inside a table
//! need not be contiguous or ordered, and after lanes churn they won't be. Nothing may assume id
//! contiguity.
//!
//! A lane's length and its blocks are one concept, so they live in one type: `begin_round` grows
//! tables and advances lengths together — all lanes or none — and `free_lane` clears both. There is
//! no way to move one without the other, which is the invariant everything downstream leans on.
//!
//! Block 0 is a sentinel: it never enters the free stack and is never allocated to a lane. It exists
//! so that ragged gathers can pad short lanes with a block that provably belongs to no one — even a
//! masking bug then exposes zeros, never another sequence's KV. Usable blocks are `1..num_blocks`.

/// The block id ragged gathers pad with. Zeroed at store init, owned by no lane.
pub const SENTINEL_BLOCK: u32 = 0;

/// The pool could not supply enough blocks for a round. The caller's state is untouched when this is
/// returned: `begin_round` rolls back everything it took before reporting failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolExhausted {
    /// How many more blocks the round needed than the free stack held.
    pub short_by: usize,
}

/// A shared pool of fixed-size KV blocks and the per-lane ledger (length + block table) that maps
/// each lane's positions onto them.
///
/// Allocation is a LIFO free stack: alloc = pop, free = push, both O(1). Reuse order is irrelevant to
/// correctness (the table maps positions explicitly) and to the device (blocks are rows of one
/// preallocated tensor — a freed lane's data is dead, there is no residency to preserve).
///
/// Freed blocks keep their old contents; a reallocated block may hold another lane's stale KV until
/// overwritten. That is safe only because every ragged read is masked — the same contract the slab's
/// ragged read-back relied on. If a future change ever weakens the mask, blocks must be zeroed on
/// free instead.
#[derive(Debug, Clone)]
pub struct BlockPool {
    /// Tokens per block. Fixed at construction, never per-sequence.
    block_size: usize,
    /// Free block ids, LIFO. Never contains the sentinel.
    free: Vec<u32>,
    /// Per lane, the block ids covering its tokens: `tables[lane][i]` holds positions
    /// `[i·block_size, (i+1)·block_size)`. Grows and shrinks only in lockstep with `lens`.
    tables: Vec<Vec<u32>>,
    /// Per lane, its current sequence length — the single source of truth for lane lengths. The
    /// covering table always holds exactly `lens[lane].div_ceil(block_size)` blocks.
    lens: Vec<usize>,
}

impl BlockPool {
    /// A pool of `num_blocks` blocks (block 0 is the sentinel, so `num_blocks - 1` are usable) with
    /// empty ledgers for `num_lanes` lanes.
    pub fn new(block_size: usize, num_blocks: usize, num_lanes: usize) -> Self {
        assert!(block_size >= 1, "block_size must be at least 1");
        assert!(
            num_blocks >= 2,
            "need at least one usable block besides the sentinel"
        );
        Self {
            block_size,
            // Pushed in ascending order so the first pops hand out the highest ids — order is
            // arbitrary by contract, and the tests exercise that no caller depends on it.
            free: (1..num_blocks as u32).collect(),
            tables: vec![Vec::new(); num_lanes],
            lens: vec![0; num_lanes],
        }
    }

    /// Tokens per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// How many blocks a sequence of `len` tokens occupies.
    pub fn blocks_for(&self, len: usize) -> usize {
        len.div_ceil(self.block_size)
    }

    /// How many blocks are currently free.
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Number of lanes this pool keeps ledgers for.
    pub fn lane_count(&self) -> usize {
        self.lens.len()
    }

    /// Sequence length of one lane.
    pub fn lane_len(&self, lane: usize) -> usize {
        self.lens[lane]
    }

    /// The blocks backing `lane`, in position order: entry `i` covers positions
    /// `[i·block_size, (i+1)·block_size)`.
    pub fn lane_blocks(&self, lane: usize) -> &[u32] {
        &self.tables[lane]
    }

    /// Commit one round of `seq_len` new tokens over the given lanes: grow every lane's table to
    /// cover its new length, then advance every lane's length, returning the pre-advance lengths —
    /// the round's start positions. All lanes or none: if the free stack cannot cover one lane, the
    /// blocks already taken this round go back and every ledger is exactly as the caller left it.
    /// This is the single place tables and lengths move, so they can never drift apart.
    pub fn begin_round(
        &mut self,
        lanes: &[usize],
        seq_len: usize,
    ) -> Result<Vec<usize>, PoolExhausted> {
        debug_assert!(
            lanes
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                == lanes.len(),
            "begin_round got a duplicate lane: {lanes:?}"
        );
        let mut grown: Vec<(usize, usize)> = Vec::with_capacity(lanes.len());
        for &lane in lanes {
            let before = self.tables[lane].len();
            match self.grow_table(lane, self.lens[lane] + seq_len) {
                Ok(()) => grown.push((lane, before)),
                Err(exhausted) => {
                    for &(grown_lane, keep) in &grown {
                        let excess = self.tables[grown_lane].split_off(keep);
                        self.free.extend(excess);
                    }
                    self.check_invariants();
                    return Err(exhausted);
                }
            }
        }
        let starts = lanes.iter().map(|&lane| self.lens[lane]).collect();
        for &lane in lanes {
            self.lens[lane] += seq_len;
        }
        self.check_invariants();
        Ok(starts)
    }

    /// Return every one of `lane`'s blocks to the free stack and zero its length. Block contents are
    /// not touched (see the stale-data note on the type).
    pub fn free_lane(&mut self, lane: usize) {
        let blocks = std::mem::take(&mut self.tables[lane]);
        self.free.extend(blocks);
        self.lens[lane] = 0;
        self.check_invariants();
    }

    /// Grow `lane`'s table until it covers `new_len` tokens. Idempotent — a length the table already
    /// covers is a no-op, so repeated calls across prefill chunks are safe. All or nothing for this
    /// lane — on exhaustion the blocks taken by THIS call go back and the table is as it was;
    /// `begin_round` extends that to the whole round.
    fn grow_table(&mut self, lane: usize, new_len: usize) -> Result<(), PoolExhausted> {
        let need = self.blocks_for(new_len);
        let have = self.tables[lane].len();
        if need <= have {
            return Ok(());
        }
        let short = need - have;
        if short > self.free.len() {
            return Err(PoolExhausted {
                short_by: short - self.free.len(),
            });
        }
        for _ in 0..short {
            let block = self.free.pop().expect("checked above");
            debug_assert_ne!(block, SENTINEL_BLOCK, "the sentinel must never be allocated");
            self.tables[lane].push(block);
        }
        Ok(())
    }

    /// The ledger invariants, checked after every committed mutation in debug builds: every usable
    /// block is in exactly one table or the free stack (no leaks, no double ownership), and every
    /// lane's table holds exactly the blocks its length needs — lengths and tables never drift.
    fn check_invariants(&self) {
        #[cfg(debug_assertions)]
        {
            let total_usable = self.free.len() + self.tables.iter().map(Vec::len).sum::<usize>();
            let mut seen = vec![false; total_usable + 1 + SENTINEL_BLOCK as usize];
            let mut mark = |b: u32| {
                assert_ne!(b, SENTINEL_BLOCK, "sentinel found in a table or the free stack");
                let i = b as usize;
                if i < seen.len() {
                    assert!(!seen[i], "block {b} owned twice");
                    seen[i] = true;
                }
            };
            self.free.iter().copied().for_each(&mut mark);
            self.tables.iter().flatten().copied().for_each(&mut mark);
            for (lane, len) in self.lens.iter().enumerate() {
                assert_eq!(
                    self.tables[lane].len(),
                    len.div_ceil(self.block_size),
                    "lane {lane}: table and length drifted apart"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Position → block mapping stays correct while lanes grow round by round, growth within a
    /// covered block takes nothing from the pool, and the returned starts are the pre-advance
    /// lengths — the same values decode uses as write offsets and RoPE positions.
    #[test]
    fn rounds_grow_tables_and_lengths_together() {
        let mut pool = BlockPool::new(4, 8, 2); // 7 usable blocks of 4 tokens
        let starts = pool.begin_round(&[0], 5).unwrap(); // 5 tokens -> 2 blocks
        assert_eq!(starts, vec![0]);
        assert_eq!(pool.lane_len(0), 5);
        assert_eq!(pool.lane_blocks(0).len(), 2);
        let before: Vec<u32> = pool.lane_blocks(0).to_vec();
        let free_before = pool.free_blocks();

        let starts = pool.begin_round(&[0], 3).unwrap(); // to 8 tokens: still 2 blocks
        assert_eq!(starts, vec![5]);
        assert_eq!(pool.lane_blocks(0), before.as_slice());
        assert_eq!(pool.free_blocks(), free_before);

        let starts = pool.begin_round(&[0], 1).unwrap(); // 9th token crosses into a third block
        assert_eq!(starts, vec![8]);
        assert_eq!(pool.lane_blocks(0).len(), 3);
        assert_eq!(&pool.lane_blocks(0)[..2], before.as_slice(), "existing blocks keep their slots");
    }

    /// Exhaustion is all-or-nothing across the whole round: when one lane of a multi-lane round
    /// cannot get its block, the lanes already grown are unwound too — no length advances, no table
    /// keeps a block, and the free stack is exactly as it was.
    #[test]
    fn exhaustion_rolls_back_the_whole_round() {
        let mut pool = BlockPool::new(4, 4, 3); // 3 usable blocks
        pool.begin_round(&[0], 4).unwrap(); // lane 0 takes 1, leaves 2 free

        // Lanes 1 and 2 want 2 blocks each; only 2 are free — lane 2 comes up short.
        let err = pool.begin_round(&[1, 2], 8).unwrap_err();
        assert_eq!(err.short_by, 2);
        for lane in [1, 2] {
            assert_eq!(pool.lane_len(lane), 0, "lane {lane}: no length may survive");
            assert!(pool.lane_blocks(lane).is_empty(), "lane {lane}: no block may survive");
        }
        assert_eq!(pool.free_blocks(), 2, "the free stack is exactly as it was");
        assert_eq!(pool.lane_len(0), 4, "an uninvolved lane is untouched");

        // What fits still fits: one of the two lanes alone succeeds.
        pool.begin_round(&[1], 8).unwrap();
        assert_eq!(pool.lane_len(1), 8);
    }

    /// A freed lane returns exactly its blocks and zeroes its length, and the pool can hand the
    /// blocks straight to another lane.
    #[test]
    fn freeing_a_lane_recycles_its_blocks() {
        let mut pool = BlockPool::new(4, 4, 2); // 3 usable
        pool.begin_round(&[0], 12).unwrap(); // all 3
        assert_eq!(pool.free_blocks(), 0);

        pool.free_lane(0);
        assert_eq!(pool.free_blocks(), 3);
        assert_eq!(pool.lane_len(0), 0);
        assert!(pool.lane_blocks(0).is_empty());

        pool.begin_round(&[1], 12).unwrap(); // the recycled blocks fit lane 1 whole
        assert_eq!(pool.lane_blocks(1).len(), 3);
    }

    /// After lanes churn, a table's block ids are not contiguous — and that is fine, because only
    /// the table index is positional. This is the contract the gather path relies on.
    #[test]
    fn interleaved_churn_yields_noncontiguous_ids_with_correct_mapping() {
        let mut pool = BlockPool::new(2, 8, 3); // 7 usable blocks of 2 tokens
        pool.begin_round(&[0], 4).unwrap(); // lane 0: 2 blocks
        pool.begin_round(&[1], 4).unwrap(); // lane 1: 2 blocks
        pool.free_lane(0); // lane 0's ids go back on top of the stack
        pool.begin_round(&[2], 2).unwrap(); // lane 2 takes one of lane 0's old ids
        pool.begin_round(&[1], 4).unwrap(); // lane 1 grows with whatever is on top

        let table = pool.lane_blocks(1);
        assert_eq!(table.len(), 4);
        let contiguous = table.windows(2).all(|w| w[1] == w[0] + 1);
        assert!(!contiguous, "churn should scramble ids: {table:?}");
        // The mapping stays positional regardless: position 5 lives in table[2] at offset 1.
        assert_eq!(pool.blocks_for(6) - 1, 2);
    }

    /// The sentinel is never handed out, even when the pool is drained to its last block.
    #[test]
    fn sentinel_is_never_allocated() {
        let mut pool = BlockPool::new(1, 3, 1); // 2 usable
        pool.begin_round(&[0], 2).unwrap(); // drain the pool
        assert_eq!(pool.free_blocks(), 0);
        assert!(pool.lane_blocks(0).iter().all(|&b| b != SENTINEL_BLOCK));
        assert!(pool.begin_round(&[0], 1).is_err(), "nothing left but the sentinel");
    }
}

//! Phase-2 S4 gate: batched-decode equivalence characterization harness.
//!
//! Standing acceptance test for the fused data plane: per-lane divergent
//! positions — lane-sliced `slice_assign` KV writes into a shared slab,
//! per-lane RoPE (gather primary, per-lane-loop fallback), per-lane padding
//! masks — must be numerically equivalent to independent batch-1 runs through
//! the production forward path (`LlamaDecoder::forward`).
//!
//! Gates:
//! - Correctness: byte-exact argmax token streams + tight-tolerance logits,
//!   batch=2 (lanes at divergent positions: X at 37, Y at 5) vs 2x batch=1.
//! - Throughput (Tier 1, Metal): >= 1.2x aggregate tok/s at batch 4 vs batch 1
//!   via the `#[ignore]`d benchmark below (batch 1/2/4/8). The authoritative
//!   Tier-2 gate (>= 1.5x on server CUDA) runs later via modal-rust.
//!

use burn::{
    nn::RotaryEncoding,
    prelude::*,
    tensor::{activation::softmax, Tolerance},
};

use burn_lm_inference::batching::{BatchedDecoder, DecodeRow};

use crate::{
    inference::Llama, nn::transformer::Transformer, tests::Reinitializer,
    tokenizer::byte::ByteTokenizer, LlamaConfig,
};

/// How per-lane RoPE rotations are applied in the batched path.
#[derive(Clone, Copy, Debug)]
enum RopeMode {
    /// Primary: gather rows of the precomputed `freq_complex` table by each
    /// lane's position and rotate all lanes in one batched tensor op.
    Gather,
    /// Correctness-floor fallback: loop `rope.apply(x_lane, pos_lane)` over
    /// per-lane slices and `cat`.
    PerLaneLoop,
}

/// Per-layer KV slab: `[n_lanes, n_kv_heads, max_seq_len, head_dim]`.
struct LaneSlab {
    k: Tensor<4>,
    v: Tensor<4>,
}

/// Hand-rolled multi-lane decoder over the production weights.
///
/// Each lane owns a slice of the KV slab and its own position; lanes are
/// written with lane-sliced `slice_assign` and read back with a per-lane
/// padding+causal mask. This is the tensor recipe the fused data plane
/// productionizes; this harness is its acceptance bar.
struct BatchedHarness {
    model: Transformer,
    rope: RotaryEncoding,
    slabs: Vec<LaneSlab>,
    lens: Vec<usize>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    mode: RopeMode,
}

impl BatchedHarness {
    fn new(
        model: Transformer,
        rope: RotaryEncoding,
        config: &LlamaConfig,
        n_lanes: usize,
        mode: RopeMode,
        device: &Device,
    ) -> Self {
        let n_kv_heads = config.num_key_value_heads.unwrap();
        let head_dim = config.d_model / config.num_attention_heads;
        let slabs = (0..config.num_hidden_layers)
            .map(|_| LaneSlab {
                k: Tensor::zeros([n_lanes, n_kv_heads, config.max_seq_len, head_dim], device),
                v: Tensor::zeros([n_lanes, n_kv_heads, config.max_seq_len, head_dim], device),
            })
            .collect();

        Self {
            model,
            rope,
            slabs,
            lens: vec![0; n_lanes],
            n_heads: config.num_attention_heads,
            n_kv_heads,
            head_dim,
            vocab: config.vocab_size,
            mode,
        }
    }

    /// One forward for the given lanes (all with the same input length `q`).
    /// Prefill = single lane with `q == prompt_len`; fused decode = all active
    /// lanes with `q == 1`. Returns the last-position logits, one row per lane.
    fn forward(&mut self, lanes: &[usize], x: Tensor<2, Int>) -> Tensor<2> {
        let [n, q] = x.dims();
        assert_eq!(n, lanes.len());
        let device = x.device();

        let starts: Vec<usize> = lanes.iter().map(|&l| self.lens[l]).collect();
        let l_max = starts.iter().map(|s| s + q).max().unwrap();

        // Per-lane causal + padding mask over the shared slab columns:
        // `true` = masked. Query row `r` of lane `j` may attend to columns
        // `0..=starts[j] + r`; everything beyond (other lanes' tail garbage,
        // future positions) is masked.
        let mut mask_data = Vec::with_capacity(n * q * l_max);
        for s in starts.iter() {
            for r in 0..q {
                for c in 0..l_max {
                    mask_data.push(c > s + r);
                }
            }
        }
        let mask =
            Tensor::<4, Bool>::from_data(TensorData::new(mask_data, [n, 1, q, l_max]), &device)
                .expand([n, self.n_heads, q, l_max]);

        let mut h = self.model.tok_embeddings.forward(x);

        for (layer, slab) in self.model.layers.iter().zip(self.slabs.iter_mut()) {
            let attn = &layer.attention;
            let x_norm = layer.attention_norm.forward(h.clone());

            let q_proj = attn
                .wq
                .forward(x_norm.clone())
                .reshape([n, q, self.n_heads, self.head_dim])
                .swap_dims(1, 2);
            let k_proj = attn
                .wk
                .forward(x_norm.clone())
                .reshape([n, q, self.n_kv_heads, self.head_dim])
                .swap_dims(1, 2);
            let v_proj = attn
                .wv
                .forward(x_norm)
                .reshape([n, q, self.n_kv_heads, self.head_dim])
                .swap_dims(1, 2);

            // Per-lane RoPE at divergent positions.
            let q_rot = apply_rope_lanes(&self.rope, q_proj, &starts, self.mode);
            let k_rot = apply_rope_lanes(&self.rope, k_proj, &starts, self.mode);

            // Lane-sliced KV writes into the shared slab.
            for (j, (&lane, &start)) in lanes.iter().zip(starts.iter()).enumerate() {
                let k_lane =
                    k_rot
                        .clone()
                        .slice([j..j + 1, 0..self.n_kv_heads, 0..q, 0..self.head_dim]);
                let v_lane =
                    v_proj
                        .clone()
                        .slice([j..j + 1, 0..self.n_kv_heads, 0..q, 0..self.head_dim]);
                slab.k = slab.k.clone().slice_assign(
                    [
                        lane..lane + 1,
                        0..self.n_kv_heads,
                        start..start + q,
                        0..self.head_dim,
                    ],
                    k_lane,
                );
                slab.v = slab.v.clone().slice_assign(
                    [
                        lane..lane + 1,
                        0..self.n_kv_heads,
                        start..start + q,
                        0..self.head_dim,
                    ],
                    v_lane,
                );
            }

            // Read the active lanes back (ragged tails handled by the mask).
            let k_full = Tensor::cat(
                lanes
                    .iter()
                    .map(|&lane| {
                        slab.k.clone().slice([
                            lane..lane + 1,
                            0..self.n_kv_heads,
                            0..l_max,
                            0..self.head_dim,
                        ])
                    })
                    .collect::<Vec<_>>(),
                0,
            );
            let v_full = Tensor::cat(
                lanes
                    .iter()
                    .map(|&lane| {
                        slab.v.clone().slice([
                            lane..lane + 1,
                            0..self.n_kv_heads,
                            0..l_max,
                            0..self.head_dim,
                        ])
                    })
                    .collect::<Vec<_>>(),
                0,
            );
            let k_full = attn.repeat_kv(k_full);
            let v_full = attn.repeat_kv(v_full);

            let scores = q_rot
                .matmul(k_full.swap_dims(2, 3))
                .div_scalar((self.head_dim as f32).sqrt());
            let scores = scores.mask_fill(mask.clone(), f32::NEG_INFINITY);
            let scores = softmax(scores, 3);

            let ctx =
                scores
                    .matmul(v_full)
                    .swap_dims(1, 2)
                    .reshape([n, q, self.n_heads * self.head_dim]);

            h = h + attn.wo.forward(ctx);
            h = h.clone() + layer.feed_forward.forward(layer.ffn_norm.forward(h));
        }

        for &lane in lanes {
            self.lens[lane] += q;
        }

        let h = self.model.norm.forward(h);
        let logits = self.model.output.forward(h);
        logits
            .slice([0..n, q - 1..q, 0..self.vocab])
            .reshape([n, self.vocab])
    }
}

/// Apply RoPE rotations for `n` lanes sitting at divergent `starts` positions.
fn apply_rope_lanes(
    rope: &RotaryEncoding,
    x: Tensor<4>,
    starts: &[usize],
    mode: RopeMode,
) -> Tensor<4> {
    let [n, heads, q, head_dim] = x.dims();
    let device = x.device();

    match mode {
        RopeMode::PerLaneLoop => {
            let lanes = (0..n)
                .map(|j| {
                    rope.apply(
                        x.clone().slice([j..j + 1, 0..heads, 0..q, 0..head_dim]),
                        starts[j],
                    )
                })
                .collect::<Vec<_>>();
            Tensor::cat(lanes, 0)
        }
        RopeMode::Gather => {
            // Gather the per-(lane, row) frequency rows in one `select`, then
            // rotate everything with the same batched ops `RotaryEncoding::apply`
            // uses internally.
            let mut idx = Vec::with_capacity(n * q);
            for s in starts.iter() {
                for r in 0..q {
                    idx.push((s + r) as i64);
                }
            }
            let idx = Tensor::<1, Int>::from_data(TensorData::new(idx, [n * q]), &device);
            let freqs = rope
                .freq_complex
                .clone()
                .select(0, idx) // [n * q, head_dim, 2]
                .reshape([n, 1, q, head_dim, 2])
                .expand([n, heads, q, head_dim, 2])
                .reshape([n * heads, q, head_dim, 2]);

            // 2D rotation matrix [[cos, -sin], [sin, cos]] expansion.
            let sign =
                Tensor::<2>::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], &device);

            let out = x
                .reshape([n * heads, q, head_dim / 2, 2])
                .matmul(sign.unsqueeze::<4>())
                .reshape([n * heads, q, head_dim, 2])
                * freqs;

            out.sum_dim(3).reshape([n, heads, q, head_dim])
        }
    }
}

/// Deterministic test model: `llama3_2_1b_test` config + seeded reinit, the
/// same rig as the `generate.rs` tests.
fn test_llama(device: &Device) -> Llama<ByteTokenizer> {
    let config = LlamaConfig::llama3_2_1b_test();
    let mut llama = config.init::<ByteTokenizer>(device).unwrap();
    llama.decoder.model = Reinitializer::default()
        .random_float(0, -1.0, 1.0)
        .apply(llama.decoder.model);
    llama
}

fn tokens_tensor(rows: &[Vec<u32>], device: &Device) -> Tensor<2, Int> {
    let q = rows[0].len();
    assert!(rows.iter().all(|r| r.len() == q));
    let data: Vec<i64> = rows.iter().flatten().map(|&t| t as i64).collect();
    Tensor::from_data(TensorData::new(data, [rows.len(), q]), device)
}

fn argmax_rows(logits: &Tensor<2>) -> Vec<u32> {
    logits
        .clone()
        .argmax(1)
        .into_data()
        .iter::<i64>()
        .map(|t| t as u32)
        .collect()
}

fn logits_row(logits: &Tensor<2>, row: usize, vocab: usize) -> Vec<f32> {
    logits
        .clone()
        .slice([row..row + 1, 0..vocab])
        .into_data()
        .iter::<f32>()
        .collect()
}

/// Independent batch-1 greedy run through the production forward path
/// (`LlamaDecoder::forward`: `TransformerCache::prepare` +
/// `PositionalEncodingState` + `Transformer::forward`), mirroring
/// `Llama::generate`. Returns the argmax token stream and the last-position
/// logits of every step.
fn reference_run(prompt: &[u32], steps: usize, device: &Device) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut llama = test_llama(device);
    let mut tokens = Vec::with_capacity(steps);
    let mut logits_steps = Vec::with_capacity(steps);

    let mut input: Vec<u32> = prompt.to_vec();
    for _ in 0..steps {
        let x = tokens_tensor(&[input.clone()], device);
        let logits = llama.decoder.forward(x).unwrap();
        let [b, q, v] = logits.dims();
        let last = logits.slice([0..b, q - 1..q, 0..v]).reshape([b, v]);
        let next = argmax_rows(&last)[0];
        logits_steps.push(logits_row(&last, 0, v));
        tokens.push(next);
        input = vec![next];
    }

    (tokens, logits_steps)
}

/// Multi-lane greedy run through the hand-rolled batched path: per-lane
/// unpadded prefill (staggered admits), then fused `[n, 1]` decode steps.
fn batched_run(
    prompts: &[Vec<u32>],
    steps: usize,
    mode: RopeMode,
    device: &Device,
) -> Vec<(Vec<u32>, Vec<Vec<f32>>)> {
    let n_lanes = prompts.len();
    let llama = test_llama(device);
    let mut harness = BatchedHarness::new(
        llama.decoder.model,
        llama.decoder.pos_encoding.rope,
        &LlamaConfig::llama3_2_1b_test(),
        n_lanes,
        mode,
        device,
    );
    let vocab = harness.vocab;

    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n_lanes];
    let mut logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_lanes];
    let mut last: Vec<u32> = vec![0; n_lanes];

    // Per-lane unpadded prefill: lane 1 is admitted while lane 0 already holds
    // its prompt in the slab (divergent positions from the very first step).
    for (lane, prompt) in prompts.iter().enumerate() {
        let out = harness.forward(&[lane], tokens_tensor(&[prompt.clone()], device));
        let next = argmax_rows(&out)[0];
        tokens[lane].push(next);
        logits[lane].push(logits_row(&out, 0, vocab));
        last[lane] = next;
    }

    // Fused decode: one [n_lanes, 1] forward per step.
    let all_lanes: Vec<usize> = (0..n_lanes).collect();
    for _ in 1..steps {
        let rows: Vec<Vec<u32>> = last.iter().map(|&t| vec![t]).collect();
        let out = harness.forward(&all_lanes, tokens_tensor(&rows, device));
        let next = argmax_rows(&out);
        for lane in 0..n_lanes {
            tokens[lane].push(next[lane]);
            logits[lane].push(logits_row(&out, lane, vocab));
            last[lane] = next[lane];
        }
    }

    tokens.into_iter().zip(logits).collect()
}

/// Lane X sits at position 37, lane Y at position 5 when fused decode starts.
fn divergent_prompts() -> Vec<Vec<u32>> {
    let x: Vec<u32> = "This is a long prompt for lane X in the gate"
        .bytes()
        .take(37)
        .map(|b| b as u32)
        .collect();
    let y: Vec<u32> = "Hello".bytes().map(|b| b as u32).collect();
    assert_eq!(x.len(), 37);
    assert_eq!(y.len(), 5);
    vec![x, y]
}

fn assert_equivalent(mode: RopeMode) {
    let device: Device = Default::default();
    let steps = 24;
    let prompts = divergent_prompts();

    let batched = batched_run(&prompts, steps, mode, &device);

    let tolerance = Tolerance::<f32>::rel_abs(1e-4, 1e-5);
    for (lane, prompt) in prompts.iter().enumerate() {
        let (ref_tokens, ref_logits) = reference_run(prompt, steps, &device);
        let (lane_tokens, lane_logits) = &batched[lane];

        // Byte-exact argmax token stream.
        assert_eq!(
            lane_tokens, &ref_tokens,
            "lane {lane} ({mode:?}): batched argmax stream diverged from batch-1 run"
        );

        // Tight-tolerance per-step logits.
        for (got, expected) in lane_logits.iter().zip(ref_logits.iter()) {
            let got = TensorData::new(got.clone(), [got.len()]);
            let expected = TensorData::new(expected.clone(), [expected.len()]);
            got.assert_approx_eq::<f32>(&expected, tolerance);
        }
    }
}

#[test]
fn batch2_gather_rope_equivalent_to_two_batch1_runs() {
    assert_equivalent(RopeMode::Gather);
}

#[test]
fn batch2_per_lane_loop_rope_equivalent_to_two_batch1_runs() {
    assert_equivalent(RopeMode::PerLaneLoop);
}

/// Build the tiny test model with an `n_lanes`-lane slab (slot == lane), seeded identically to
/// [`test_llama`] so its output is directly comparable to [`reference_run`].
fn test_llama_lanes(n_lanes: usize, device: &Device) -> Llama<ByteTokenizer> {
    let config = LlamaConfig::llama3_2_1b_test().with_max_batch_size(n_lanes);
    let mut llama = config.init::<ByteTokenizer>(device).unwrap();
    llama.decoder.model = Reinitializer::default()
        .random_float(0, -1.0, 1.0)
        .apply(llama.decoder.model);
    llama
}

/// Greedy multi-lane run through the REAL decoder: staggered prefill into each lane, then fused
/// `[n, 1]` decode rounds. Returns each lane's (argmax stream, per-step last-position logits).
fn real_decoder_batched_run(
    prompts: &[Vec<u32>],
    steps: usize,
    device: &Device,
) -> Vec<(Vec<u32>, Vec<Vec<f32>>)> {
    let n = prompts.len();
    let vocab = LlamaConfig::llama3_2_1b_test().vocab_size;
    let mut llama = test_llama_lanes(n, device);

    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];
    let mut last = vec![0u32; n];

    for (lane, prompt) in prompts.iter().enumerate() {
        let out = llama.decoder.prefill(lane, prompt, 0).unwrap();
        tokens[lane].push(argmax_rows(&out)[0]);
        logits[lane].push(logits_row(&out, 0, vocab));
        last[lane] = tokens[lane][0];
    }
    for _ in 1..steps {
        let rows: Vec<DecodeRow> = (0..n)
            .map(|lane| DecodeRow {
                slot: lane,
                token: last[lane],
                position: prompts[lane].len() + tokens[lane].len() - 1,
            })
            .collect();
        let out = llama.decoder.decode(&rows).unwrap();
        let next = argmax_rows(&out);
        for lane in 0..n {
            tokens[lane].push(next[lane]);
            logits[lane].push(logits_row(&out, lane, vocab));
            last[lane] = next[lane];
        }
    }
    for lane in 0..n {
        llama.decoder.release(lane);
    }

    tokens.into_iter().zip(logits).collect()
}

/// Every lane's fused real-decoder run must match its independent batch-1 reference — both
/// byte-exact argmax stream AND tight-tolerance per-step logits (the latter catches a sub-argmax
/// numerical drift that an argmax-only check would miss).
fn assert_real_decoder_equivalent(prompts: Vec<Vec<u32>>, steps: usize, device: &Device) {
    let batched = real_decoder_batched_run(&prompts, steps, device);
    let tolerance = Tolerance::<f32>::rel_abs(1e-4, 1e-5);
    for (lane, prompt) in prompts.iter().enumerate() {
        let (ref_tokens, ref_logits) = reference_run(prompt, steps, device);
        let (lane_tokens, lane_logits) = &batched[lane];
        assert_eq!(
            lane_tokens, &ref_tokens,
            "lane {lane}: fused real-decoder argmax stream diverged from the batch-1 run"
        );
        for (got, expected) in lane_logits.iter().zip(ref_logits.iter()) {
            let got = TensorData::new(got.clone(), [got.len()]);
            let expected = TensorData::new(expected.clone(), [expected.len()]);
            got.assert_approx_eq::<f32>(&expected, tolerance);
        }
    }
}

/// Helper: first `take` bytes of `s` as token ids (asserting the length so divergent positions are
/// guaranteed).
fn prompt_bytes(s: &str, take: usize) -> Vec<u32> {
    let v: Vec<u32> = s.bytes().take(take).map(|b| b as u32).collect();
    assert_eq!(v.len(), take, "prompt too short for the requested length");
    v
}

/// S7a production-path equivalence: prefill + fused `[n, 1]` decode rounds through the REAL
/// [`LlamaDecoder`] (`prepare_lanes` → `forward_lanes`, slot == lane) must match independent
/// batch-1 runs through the single-sequence forward path. Unlike the `BatchedHarness` tests above,
/// this drives the actual production seam (`prefill`/`decode`/`release`) — the standing guard the
/// hand-rolled harness becomes when it is deleted in S7c. Two lanes at divergent positions (37, 5).
#[test]
fn fused_decode_through_real_decoder_matches_batch1() {
    let device: Device = Default::default();
    assert_real_decoder_equivalent(divergent_prompts(), 24, &device);
}

/// A lane retires mid-batch and the survivors continue as a NON-CONTIGUOUS subset. The worker does
/// this whenever one request finishes before its batchmates (`decode(&rows)` over lanes `[0, 2]`
/// after lane 1 was released), but the other equivalence tests run every lane to the same length, so
/// the ragged-subset masking / RoPE / KV-slice keying off the explicit `lanes` slice (not a dense
/// `0..n`) had no real-weights coverage. Here lane 1 is released after a few rounds and lanes 0 and 2
/// must keep matching their solo runs.
#[test]
fn fused_decode_with_a_released_middle_lane_matches_batch1() {
    let device: Device = Default::default();
    let total_steps = 16;
    let drop_after = 5; // lane 1 retires before this decode round
    let prompts = vec![
        prompt_bytes("This is a long prompt for lane X in the gate", 37),
        prompt_bytes("Hello", 5),
        prompt_bytes("A medium length lane Y goes here now", 19),
    ];
    let n = prompts.len();

    // Solo references: lanes 0 and 2 run the full length; lane 1 only until it is dropped.
    let ref0 = reference_run(&prompts[0], total_steps, &device).0;
    let ref1_prefix = reference_run(&prompts[1], drop_after, &device).0;
    let ref2 = reference_run(&prompts[2], total_steps, &device).0;

    let mut llama = test_llama_lanes(n, &device);
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut last = vec![0u32; n];

    for (lane, prompt) in prompts.iter().enumerate() {
        let out = llama.decoder.prefill(lane, prompt, 0).unwrap();
        tokens[lane].push(argmax_rows(&out)[0]);
        last[lane] = tokens[lane][0];
    }

    let mut live: Vec<usize> = (0..n).collect();
    for round in 1..total_steps {
        if round == drop_after {
            llama.decoder.release(1);
            live = vec![0, 2]; // non-contiguous subset from here on
        }
        let rows: Vec<DecodeRow> = live
            .iter()
            .map(|&lane| DecodeRow {
                slot: lane,
                token: last[lane],
                position: prompts[lane].len() + tokens[lane].len() - 1,
            })
            .collect();
        let out = llama.decoder.decode(&rows).unwrap();
        let next = argmax_rows(&out);
        for (row, &lane) in live.iter().enumerate() {
            tokens[lane].push(next[row]);
            last[lane] = next[row];
        }
    }

    assert_eq!(
        tokens[0], ref0,
        "lane 0 (survivor) diverged after the middle lane was released"
    );
    assert_eq!(
        tokens[2], ref2,
        "lane 2 (survivor) diverged after the middle lane was released — its KV/RoPE must key off the explicit lane index, not a dense range"
    );
    assert_eq!(
        tokens[1], ref1_prefix,
        "lane 1 diverged before it was released"
    );
}

/// Three lanes at three distinct positions. n=2 cannot distinguish a correct per-lane mapping from
/// a row-vs-lane index swap in the mask build or the RoPE gather (both symmetric at 2), so this is
/// the test that actually pins the lane indexing at n >= 3.
#[test]
fn fused_decode_three_lanes_matches_batch1() {
    let device: Device = Default::default();
    let prompts = vec![
        prompt_bytes("This is a long prompt for lane X in the gate", 37),
        prompt_bytes("Hello", 5),
        prompt_bytes("A medium length lane Y goes here now", 19),
    ];
    assert_real_decoder_equivalent(prompts, 16, &device);
}

/// A lane reused after `release` must NOT inherit the retired occupant's KV: a second sequence
/// prefilled into the same slot (`position == 0`) self-heals the lane. This is the only test that
/// makes that self-heal load-bearing — with a clean fresh decoder the reset is a no-op, so it must
/// run on a DIRTY lane (slot previously held a longer sequence) while a sibling lane stays live.
#[test]
fn lane_reuse_after_release_starts_clean() {
    let device: Device = Default::default();
    let steps = 16;

    let a = prompt_bytes("First sequence occupies the slot for a while", 28);
    let b = prompt_bytes("Totally different", 12);
    let other = prompt_bytes("Sibling lane stays busy", 9);

    // What B should produce on its own.
    let b_solo = reference_run(&b, steps, &device).0;

    let mut llama = test_llama_lanes(2, &device);

    // Lane 1 holds an unrelated live sequence the whole time.
    llama.decoder.prefill(1, &other, 0).unwrap();

    // Sequence A fills (dirties) lane 0 with a LONGER history, decodes a few rounds, then retires.
    let mut last_a = argmax_rows(&llama.decoder.prefill(0, &a, 0).unwrap())[0];
    for k in 0..3 {
        let row = DecodeRow {
            slot: 0,
            token: last_a,
            position: a.len() + k,
        };
        last_a = argmax_rows(&llama.decoder.decode(&[row]).unwrap())[0];
    }
    llama.decoder.release(0);

    // Sequence B reused into lane 0: position 0 must wipe A's residue.
    let mut tokens_b = vec![argmax_rows(&llama.decoder.prefill(0, &b, 0).unwrap())[0]];
    for _ in 1..steps {
        let row = DecodeRow {
            slot: 0,
            token: *tokens_b.last().unwrap(),
            position: b.len() + tokens_b.len() - 1,
        };
        tokens_b.push(argmax_rows(&llama.decoder.decode(&[row]).unwrap())[0]);
    }

    assert_eq!(
        tokens_b, b_solo,
        "reused lane inherited stale KV from the released sequence"
    );
}

/// Both RoPE implementations must agree with each other token-for-token.
#[test]
fn gather_and_per_lane_loop_rope_agree() {
    let device: Device = Default::default();
    let prompts = divergent_prompts();
    let gather = batched_run(&prompts, 16, RopeMode::Gather, &device);
    let looped = batched_run(&prompts, 16, RopeMode::PerLaneLoop, &device);
    for lane in 0..prompts.len() {
        assert_eq!(
            gather[lane].0, looped[lane].0,
            "lane {lane}: token streams diverged"
        );
    }
}

/// Shared benchmark loop: hand-rolled lane decode at batch 1/2/4/8 over
/// whatever weights `make_model` produces. Prefill and a few warmup decode
/// steps (shader compile / autotune) are untimed.
fn run_bench(
    config: &LlamaConfig,
    prompt: &[u32],
    steps: usize,
    mut make_model: impl FnMut() -> (Transformer, RotaryEncoding),
) {
    use std::time::Instant;

    let device: Device = Default::default();
    println!("bench device: {device:?}");
    let warmup = 3;

    let mut baseline: Option<f64> = None;
    for batch in [1usize, 2, 4, 8] {
        let prompts: Vec<Vec<u32>> = (0..batch).map(|_| prompt.to_vec()).collect();
        let (model, rope) = make_model();
        let mut harness =
            BatchedHarness::new(model, rope, config, batch, RopeMode::Gather, &device);

        // Prefill all lanes (not timed).
        let mut last: Vec<u32> = Vec::with_capacity(batch);
        for (lane, p) in prompts.iter().enumerate() {
            let out = harness.forward(&[lane], tokens_tensor(&[p.clone()], &device));
            last.push(argmax_rows(&out)[0]);
        }

        let all_lanes: Vec<usize> = (0..batch).collect();
        let mut start = Instant::now();
        for step in 0..steps + warmup {
            if step == warmup {
                // Untimed warmup absorbs first-shape shader compilation and
                // autotune for the fused [batch, 1] decode.
                start = Instant::now();
            }
            let rows: Vec<Vec<u32>> = last.iter().map(|&t| vec![t]).collect();
            let out = harness.forward(&all_lanes, tokens_tensor(&rows, &device));
            last = argmax_rows(&out);
        }
        let elapsed = start.elapsed();

        let step_ms = elapsed.as_secs_f64() * 1e3 / steps as f64;
        let toks_per_s = (batch * steps) as f64 / elapsed.as_secs_f64();
        let speedup = baseline.map(|b| toks_per_s / b).unwrap_or(1.0);
        if baseline.is_none() {
            baseline = Some(toks_per_s);
        }
        println!(
            "batch {batch}: {step_ms:.3} ms/step, {toks_per_s:.1} tok/s aggregate, {speedup:.2}x vs batch 1"
        );
    }
}

/// Step latency / aggregate throughput at batch 1/2/4/8 on the tiny
/// reinitialized test model.
///
/// Tier-1 gate (this machine, Metal): >= 1.2x aggregate tok/s at batch 4 vs
/// batch 1. CPU (ndarray) numbers are informational only. Run with e.g.:
/// `cargo test -p burn-lm-llama --release --features test-wgpu \
///    batched_equivalence::bench -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run manually; the throughput gate needs a real GPU backend"]
fn bench_batched_decode_throughput() {
    let device: Device = Default::default();
    let prompt: Vec<u32> = "benchmark prompt".bytes().map(|b| b as u32).collect();
    run_bench(&LlamaConfig::llama3_2_1b_test(), &prompt, 50, || {
        let llama = test_llama(&device);
        (llama.decoder.model, llama.decoder.pos_encoding.rope)
    });
}

/// Same benchmark over the real downloaded Llama-3.2-1B-Instruct weights —
/// the meaningful Tier-1 number (the tiny test model overstates per-launch
/// overhead and understates bandwidth effects). Requires the weights to be
/// in `~/.cache/llama` already and the `llama3` + `pretrained` features:
/// `cargo test -p burn-lm-llama --release --features test-wgpu,llama3 \
///    batched_equivalence::bench_real -- --ignored --nocapture`
#[cfg(all(feature = "llama3", feature = "pretrained"))]
#[test]
#[ignore = "benchmark: run manually; loads ~2.8 GB of real weights"]
fn bench_real_weights_batched_decode_throughput() {
    let device: Device = Default::default();
    let max_seq_len = 128;

    // Load the real weights once; each batch config gets a cheap handle clone.
    // max_batch_size is irrelevant here: `run_bench` drives its own `BatchedHarness` slabs and
    // only borrows this model's weights + RoPE table.
    let llama = LlamaConfig::llama3_2_1b_pretrained(max_seq_len, 1, &device)
        .expect("Llama-3.2-1B-Instruct weights must already be downloaded");
    let model = llama.decoder.model;
    let rope = llama.decoder.pos_encoding.rope;

    let config = LlamaConfig::llama3_2_1b("unused").with_max_seq_len(max_seq_len);
    // Arbitrary in-vocab prompt tokens; argmax decode only needs ids.
    let prompt: Vec<u32> = (0..16).map(|i| 1000 + i * 13).collect();
    run_bench(&config, &prompt, 50, || (model.clone(), rope.clone()));
}

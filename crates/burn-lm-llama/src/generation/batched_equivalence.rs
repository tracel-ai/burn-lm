//! Production-path equivalence + throughput tests for the fused lane decoder.
//!
//! The fused multi-row `LlamaDecoder::decode` (lane-sliced KV writes into a shared slab, gather RoPE
//! at per-lane positions, per-lane padding masks) must be numerically identical to independent
//! batch-1 runs through the single-sequence forward path. These tests drive the REAL production seam
//! (`prefill`/`decode`/`release`):
//! - byte-exact argmax + tight-tolerance logits at 2 and 3 lanes at divergent positions;
//! - a non-contiguous lane subset after a mid-batch release;
//! - lane reuse after release starts clean.
//!
//! Plus an `#[ignore]`d throughput benchmark (batch 1/2/4/8) over the real decoder. Tier-1 gate
//! (this machine, Metal): >= 1.2x aggregate tok/s at batch 4 vs batch 1; the authoritative Tier-2
//! gate (>= 1.5x on server CUDA) runs later via modal-rust.

use burn::{prelude::*, tensor::Tolerance};

use burn_lm_inference::batching::{BatchedDecoder, DecodeRow};

use crate::{
    inference::{Llama, LlamaDecoder},
    tests::Reinitializer,
    tokenizer::byte::ByteTokenizer,
    LlamaConfig,
};

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
/// batch-1 runs through the single-sequence forward path, driving the actual production seam
/// (`prefill`/`decode`/`release`). Two lanes at divergent positions (37, 5). Two lanes at divergent positions (37, 5).
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
        };
        tokens_b.push(argmax_rows(&llama.decoder.decode(&[row]).unwrap())[0]);
    }

    assert_eq!(
        tokens_b, b_solo,
        "reused lane inherited stale KV from the released sequence"
    );
}

/// Shared benchmark: drive the REAL `LlamaDecoder` at batch 1/2/4/8 over a slab sized to the max
/// batch (reset between runs, using lanes `0..batch`). Prefill plus a few warmup decode steps
/// (shader compile / autotune) are untimed. This times the SHIPPED fused decode path
/// (`prefill`/`decode`), not a stand-in re-implementation.
fn run_bench(prompt: &[u32], steps: usize, decoder: &mut LlamaDecoder) {
    use std::time::Instant;

    println!("bench device: {:?}", decoder.device);
    let warmup = 3;
    let mut baseline: Option<f64> = None;

    for batch in [1usize, 2, 4, 8] {
        decoder.reset(); // fresh slab; use lanes 0..batch of the max-batch slab
        let lanes: Vec<usize> = (0..batch).collect();

        // Prefill every lane (not timed).
        let mut last: Vec<u32> = Vec::with_capacity(batch);
        for &lane in &lanes {
            let out = decoder.prefill(lane, prompt, 0).unwrap();
            last.push(argmax_rows(&out)[0]);
        }

        let mut start = Instant::now();
        for step in 0..steps + warmup {
            if step == warmup {
                // Untimed warmup absorbs first-shape shader compilation and autotune for the fused
                // [batch, 1] decode.
                start = Instant::now();
            }
            let rows: Vec<DecodeRow> = lanes
                .iter()
                .map(|&lane| DecodeRow {
                    slot: lane,
                    token: last[lane],
                })
                .collect();
            let out = decoder.decode(&rows).unwrap();
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

        for &lane in &lanes {
            decoder.release(lane);
        }
    }
}

/// Step latency / aggregate throughput at batch 1/2/4/8 on the tiny reinitialized test model,
/// through the real `LlamaDecoder`.
///
/// Tier-1 gate (this machine, Metal): >= 1.2x aggregate tok/s at batch 4 vs batch 1. CPU (ndarray)
/// numbers are informational only. Run with e.g.:
/// `cargo test -p burn-lm-llama --release --features test-wgpu \
///    batched_equivalence::bench -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run manually; the throughput gate needs a real GPU backend"]
fn bench_batched_decode_throughput() {
    let device: Device = Default::default();
    let prompt: Vec<u32> = "benchmark prompt".bytes().map(|b| b as u32).collect();
    let mut llama = test_llama_lanes(8, &device);
    run_bench(&prompt, 50, &mut llama.decoder);
}

/// Same benchmark over the real downloaded Llama-3.2-1B-Instruct weights — the meaningful Tier-1
/// number (the tiny test model overstates per-launch overhead and understates bandwidth effects).
/// Requires the weights in `~/.cache/llama` already and the `llama3` + `pretrained` features:
/// `cargo test -p burn-lm-llama --release --features test-wgpu,llama3 \
///    batched_equivalence::bench_real -- --ignored --nocapture`
#[cfg(all(feature = "llama3", feature = "pretrained"))]
#[test]
#[ignore = "benchmark: run manually; loads ~2.8 GB of real weights"]
fn bench_real_weights_batched_decode_throughput() {
    let device: Device = Default::default();
    let max_seq_len = 128;
    // An 8-lane slab so run_bench can drive batch 1..8 against the real fused decode.
    let mut llama = LlamaConfig::llama3_2_1b_pretrained(max_seq_len, 8, &device)
        .expect("Llama-3.2-1B-Instruct weights must already be downloaded");
    // Arbitrary in-vocab prompt token ids; argmax decode only needs ids.
    let prompt: Vec<u32> = (0..16).map(|i| 1000 + i * 13).collect();
    run_bench(&prompt, 50, &mut llama.decoder);
}

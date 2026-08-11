//! The paged-attention contract, with its reference implementation in generic tensor ops.
//!
//! `paged_attention` is written to be REPLACED: its signature is the contract a dedicated kernel
//! (cubecl) implements later — queries, the two block stores, and the round's plan in; attention
//! output out. Today's body is the generic-op spelling of those semantics: gather each lane's
//! blocks into a contiguous scratch, expand grouped-query heads, and run the backend's fused
//! attention under the plan's mask. A real paged-attention kernel walks the block tables in-kernel
//! instead — no scratch materialization, no head expansion — and swaps in behind this exact
//! function, gated by the same equivalence suites that gate this one. Nothing outside this module
//! may assume the scratch (or any other intermediate) exists.

use burn::tensor::{module::attention, ops::AttentionModuleOptions, Tensor};

use crate::cache::LanePlan;
use crate::kv_cache::KeyValueCache;

/// Attend `q` over the cached keys/values of the plan's lanes.
///
/// - `q`: `[n, num_heads, seq_len, head_dim]`, RoPE already applied, one row per active lane in
///   plan order.
/// - `cache`: the layer's paged K and V stores; this round's new tokens must already be written
///   (see [`KeyValueCache::write`]).
/// - `plan`: the round's addressing — prebuilt gather index, `l_max`, and the per-lane causal +
///   padding mask that hides every lane's stale tail (load-bearing: the gathered scratch contains
///   whole blocks and other lanes' ragged tails).
/// - `n_rep`: grouped-query expansion (`num_heads / num_kv_heads`); a kernel would fold this into
///   its block walk rather than materializing repeated heads.
///
/// Returns `[n, num_heads, seq_len, head_dim]`.
pub fn paged_attention(
    q: Tensor<4>,
    cache: &KeyValueCache,
    plan: &LanePlan,
    n_rep: usize,
) -> Tensor<4> {
    let (k, v) = cache.gather(plan);
    let k = repeat_kv(k, n_rep);
    let v = repeat_kv(v, n_rep);
    // plan.mask is [n, 1, seq_len, l_max]; broadcasts over heads inside the attention op.
    attention(
        q,
        k,
        v,
        Some(plan.mask.clone()),
        None,
        AttentionModuleOptions::default(),
    )
}

/// Repeat each KV head `n_rep` times for grouped-query attention. Part of the reference
/// implementation only: a paged kernel reads each KV head once and serves its query group
/// in-kernel, so this materialization disappears with it.
fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let [n, kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([n, kv_heads, n_rep, seq_len, head_dim])
        .reshape([n, kv_heads * n_rep, seq_len, head_dim])
}

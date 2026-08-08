/// Attention module.
pub mod attention;

/// Feed-forward transformation network module.
pub mod fftn;

/// Transformer module.
pub mod transformer;

/// Positional encoding module.
pub mod pos_encoding;

/// Llama architecture.
pub mod llama;

/// Run a position-independent `Linear` on a `[batch, seq, d]` activation as ONE flattened
/// `[batch·seq, d] × [d, k]` matmul. Fed the 3-D activation directly, the matmul treats `batch`
/// as a batch dimension of M=`seq` rows — during decode (`seq == 1`) that is a batch of M=1
/// products against a broadcast weight, which re-streams the weight matrix per lane instead of
/// sharing one read across the whole batch. Measured on Metal (Llama-3.2-1B fp32, width 16):
/// the LM head alone went 167 ms → 12 ms, and the full decode round 941 ms → 79 ms, with rounds
/// near-flat in width — the memory-bound behavior batching exists to buy. Rows of a linear layer
/// are independent, so the flattening is value-identical; the batched-equivalence suites gate it.
pub(crate) fn linear_flat(
    linear: &burn::nn::Linear,
    x: burn::tensor::Tensor<3>,
) -> burn::tensor::Tensor<3> {
    let [b, s, d] = x.dims();
    let y = linear.forward(x.reshape([b * s, d]));
    let k = y.dims()[1];
    y.reshape([b, s, k])
}

use burn::Tensor;
use burn_lm_inference::Backend;

pub struct MoeExpert<B: Backend> {
    num_experts: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    gate_up_proj: Tensor<B, 2>,
    gate_down_proj: Tensor<B, 2>,
    activation: (),
}

pub struct TopkRouter<B: Backend> {
    top_k: f32,
    num_experts: usize,
    norm_topk_prob: (),
    hidden_dim: usize,
    weight: Tensor<B, 1>,
}

pub struct SparseMoeLayer<B: Backend> {
    experts: Vec<MoeExpert<B>>,
    gate: TopkRouter<B>,
}

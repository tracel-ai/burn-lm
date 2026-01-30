use burn::{module::Module, Tensor};
use burn_lm_inference::Backend;

#[derive(Module, Debug)]
pub struct MoeExpert<B: Backend> {
    num_experts: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    gate_up_proj: Tensor<B, 2>,
    gate_down_proj: Tensor<B, 2>,
    activation: f32, // fixme just to satisfi module derive
}
#[derive(Module, Debug)]
pub struct TopkRouter<B: Backend> {
    top_k: f32,
    num_experts: usize,
    norm_topk_prob: f32, // fixme just to satisfi module derive
    hidden_dim: usize,
    weight: Tensor<B, 1>,
}
#[derive(Module, Debug)]
pub struct SparseMoeLayer<B: Backend> {
    experts: Vec<MoeExpert<B>>,
    gate: TopkRouter<B>,
}

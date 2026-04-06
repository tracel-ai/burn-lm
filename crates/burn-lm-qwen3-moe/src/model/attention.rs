use burn::module::Module;
use burn::Tensor;
use burn_lm_inference::Backend;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Tensor<B, 2>,
    k_proj: Tensor<B, 2>,
    v_proj: Tensor<B, 2>,
    o_proj: Tensor<B, 2>,
    q_norm: Tensor<B, 1>,
    k_norm: Tensor<B, 1>,
    head_dim: usize,
    num_key_value_groups: usize,
}

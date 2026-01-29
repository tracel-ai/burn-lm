use burn::Tensor;
use burn_lm_inference::{Backend, InferenceResult};

use crate::model::{attention::Attention, moe::SparseMoeLayer};

pub struct DecoderLayer<B: Backend> {
    attention: Attention<B>,
    input_layer_norm: Tensor<B, 1>,
    post_attention_layer_norm: Tensor<B, 1>,
    // always assigning moe, even though the transformers implementation has mlp or sparse moe, the configuration always bypasses moe
    sparse_moe: SparseMoeLayer<B>,
}
impl<B: Backend> DecoderLayer<B> {
    pub(crate) fn new(
        layer: u32,
        config: &super::config::Qwen3MoeConfig,
        tensors: &std::collections::BTreeMap<String, burn_store::TensorSnapshot>,
        device: &<B as Backend>::Device,
    ) -> InferenceResult<Self> {
        let total_experts = config.base_config.num_experts;

        let layer = Self {
            attention: todo!(),
            input_layer_norm: todo!(),
            post_attention_layer_norm: todo!(),
            sparse_moe: todo!(),
        };
        Ok(layer)
    }
}

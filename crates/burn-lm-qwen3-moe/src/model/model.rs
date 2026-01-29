use burn::Tensor;
use burn_lm_inference::{Backend, InferenceError, InferenceResult};

use crate::{error_wrapper::Wrap, model::decoder_layer::DecoderLayer};

pub struct Qwen3MoeModel<B: Backend> {
    embed_tokens: Tensor<B, 2>,
    norm: Tensor<B, 1>,
    layers: Vec<DecoderLayer<B>>,
    lm_head: Tensor<B, 2>,
    cache: (),
}
type Tensors = std::collections::BTreeMap<String, burn_store::TensorSnapshot>;

impl<B: Backend> Qwen3MoeModel<B> {
    pub(crate) fn new(
        tensors: &Tensors,
        config: &super::config::Qwen3MoeConfig,
        device_mappings: Vec<super::basic_device_mapper::DeviceMapping<<B as Backend>::Device>>,
    ) -> Result<Self, burn_lm_inference::InferenceError> {
        let embeddings = tensors
            .get("model.embed_tokens.weight")
            .ok_or(InferenceError::Custom(
                "Could not find model.embed_tokens.weight".into(),
            ))?;
        let norm = tensors
            .get("model.norm.weight")
            .ok_or(InferenceError::Custom(
                "Could not find model.norm.weight".into(),
            ))?;
        let lm_head = tensors.get("lm_head.weight").ok_or(InferenceError::Custom(
            "Could not find lm_head.weight".into(),
        ))?;
        let last_mapping = device_mappings.last().ok_or(InferenceError::Custom(
            "Device mappings cannot be empty".into(),
        ))?;
        let embeddings = Tensor::from_data(embeddings.to_data().w()?, &last_mapping.device);
        let norm = Tensor::from_data(norm.to_data().w()?, &last_mapping.device);
        let lm_head = Tensor::from_data(lm_head.to_data().w()?, &last_mapping.device);

        let layers: Vec<_> = (0..config.base_config.num_hidden_layers - 1)
            .map(|layer| -> InferenceResult<DecoderLayer<B>> {
                let device = device_mappings
                    .iter()
                    .find(|f| f.has_layer(layer))
                    .map(|d| d.device())
                    .ok_or(InferenceError::Custom(format!(
                        "No device mapping found for layer {}",
                        layer
                    )))?;
                let layer = DecoderLayer::<B>::new(layer, &config, &tensors, &device)?;
                Ok(layer)
            })
            .collect();

        let mut final_layers = Vec::with_capacity(layers.len());
        for result in layers {
            final_layers.push(result.w()?);
        }

        let model = Self {
            embed_tokens: embeddings,
            norm: norm,
            layers: final_layers,
            lm_head: lm_head,
            cache: (),
        };
        Ok(model)
    }
}

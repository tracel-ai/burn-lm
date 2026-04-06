use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    Tensor,
};
use burn_lm_inference::{Backend, InferenceError, InferenceResult};

use crate::{error_wrapper::Wrap, model::decoder_layer::DecoderLayer};

#[derive(Module, Debug)]
pub struct Qwen3MoeModel<B: Backend> {
    embed_tokens: Embedding<B>,

    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
    layers: Vec<DecoderLayer<B>>,

    cache: f32, // fixme just to satisfi module derive
}
type Tensors = std::collections::BTreeMap<String, burn_store::TensorSnapshot>;

impl<B: Backend> Qwen3MoeModel<B> {
    pub(crate) fn new(
        config: &super::config::Qwen3MoeConfig,
        device_mappings: Vec<super::basic_device_mapper::DeviceMapping<<B as Backend>::Device>>,
    ) -> Result<Self, burn_lm_inference::InferenceError> {
        let last_mapping = device_mappings.last().ok_or(InferenceError::Custom(
            "Device mappings cannot be empty".into(),
        ))?;
        let hidden_size = config.base_config.hidden_size as usize;
        let vocab_size = config.base_config.vocab_size as usize;
        let embeddings = EmbeddingConfig::new(vocab_size, hidden_size).init(&last_mapping.device);

        let norm =
            RmsNormConfig::new(config.base_config.hidden_size as usize).init(&last_mapping.device);

        let lm_head = LinearConfig::new(vocab_size, hidden_size).init(&last_mapping.device);

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
                let layer = DecoderLayer::<B>::new(layer, &config, &device)?;
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
            cache: 0.0,
        };
        Ok(model)
    }
}

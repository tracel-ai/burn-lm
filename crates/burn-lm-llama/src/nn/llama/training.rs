//! Training requirements for Llama 3.

use crate::{inference, nn::transformer::Transformer, tokenizer::Tokenizer};
use burn::{
    module::Module,
    nn::{loss::CrossEntropyLossConfig, RotaryEncoding},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Int},
    train::{
        metric::{AccuracyInput, Adaptor, LossInput},
        InferenceStep, ItemLazy, TrainOutput, TrainStep,
    },
    Tensor,
};
use tracing::debug;

/// Meta Llama large language model and tokenizer. For training uses only.
#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    /// Llama decoder-only transformer.
    pub model: Transformer<B>,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<B>,
}

#[derive(Debug, Clone)]
pub struct LlamaInput<B: Backend> {
    /// [batch_size, seq_len]
    pub tokens: Tensor<B, 2, Int>,
    /// [batch_size, seq_len]
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Debug, Clone)]
pub struct LlamaOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    /// [batch_size, seq_len, vocab_size]
    pub logits: Tensor<B, 3>,
    /// [batch_size, seq_len]
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Llama<B> {
    pub fn forward(&self, item: LlamaInput<B>) -> LlamaOutput<B> {
        let logits = self.model.forward_train(item.tokens, &self.rope);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        let logits_flattened = logits.clone().reshape([batch_size * seq_len, vocab_size]);
        let targets_flattened = item.targets.clone().reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits_flattened, targets_flattened);

        debug!(
            "logits dims {:?}, loss dims {:?}",
            logits.dims(),
            loss.dims(),
        );

        LlamaOutput {
            loss,
            logits,
            targets: item.targets,
        }
    }
}

impl<B: Backend> InferenceStep for Llama<B> {
    type Input = LlamaInput<B>;
    type Output = LlamaOutput<B>;

    fn step(&self, item: LlamaInput<B>) -> LlamaOutput<B> {
        self.forward(item)
    }
}

impl<B: AutodiffBackend> TrainStep for Llama<B> {
    type Input = LlamaInput<B>;
    type Output = LlamaOutput<B>;

    fn step(&self, item: LlamaInput<B>) -> TrainOutput<LlamaOutput<B>> {
        let output = self.forward(item);
        let grads = output.loss.backward();

        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> LlamaInput<B> {
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            tokens: self.tokens.to_device(device),
            targets: self.targets.to_device(device),
        }
    }
}

impl<B: Backend, T: Tokenizer> From<inference::Llama<B, T>> for Llama<B> {
    fn from(inference_llama: inference::Llama<B, T>) -> Self {
        Llama {
            model: inference_llama.model,
            rope: inference_llama.pos_encoding.rope,
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for LlamaOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for LlamaOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        let [batch_size, seq_len, vocab_size] = self.logits.dims();

        let logits_flattened = self
            .logits
            .clone()
            .reshape([batch_size * seq_len, vocab_size]);
        let targets_flattened = self.targets.clone().reshape([batch_size * seq_len]);

        AccuracyInput::new(logits_flattened, targets_flattened)
    }
}

impl<B: Backend> ItemLazy for LlamaOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

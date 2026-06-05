//! Training requirements for Llama 3.

use crate::{inference, nn::transformer::Transformer, tokenizer::Tokenizer};
use burn::{
    module::Module,
    nn::{loss::CrossEntropyLossConfig, RotaryEncoding},
    tensor::{Device, Int, Tensor, Transaction},
    train::{
        metric::{AccuracyInput, Adaptor, LossInput},
        InferenceStep, ItemLazy, TrainOutput, TrainStep,
    },
};
use tracing::debug;

/// Meta Llama large language model and tokenizer. For training uses only.
#[derive(Module, Debug)]
pub struct Llama {
    /// Llama decoder-only transformer.
    pub model: Transformer,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding,
}

#[derive(Debug, Clone)]
pub struct LlamaInput {
    /// [batch_size, seq_len]
    pub tokens: Tensor<2, Int>,
    /// [batch_size, seq_len]
    pub targets: Tensor<2, Int>,
}

#[derive(Debug, Clone)]
pub struct LlamaOutput {
    pub loss: Tensor<1>,
    /// [batch_size, seq_len, vocab_size]
    pub logits: Tensor<3>,
    /// [batch_size, seq_len]
    pub targets: Tensor<2, Int>,
}

impl Llama {
    pub fn forward(&self, item: LlamaInput) -> LlamaOutput {
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

impl InferenceStep for Llama {
    type Input = LlamaInput;
    type Output = LlamaOutput;

    fn step(&self, item: LlamaInput) -> LlamaOutput {
        self.forward(item)
    }
}

impl TrainStep for Llama {
    type Input = LlamaInput;
    type Output = LlamaOutput;

    fn step(&self, item: LlamaInput) -> TrainOutput<LlamaOutput> {
        let output = self.forward(item);
        let grads = output.loss.backward();

        TrainOutput::new(self, grads, output)
    }
}

impl LlamaInput {
    pub fn to_device(self, device: &Device) -> Self {
        Self {
            tokens: self.tokens.to_device(device),
            targets: self.targets.to_device(device),
        }
    }
}

impl<T: Tokenizer> From<inference::Llama<T>> for Llama {
    fn from(inference_llama: inference::Llama<T>) -> Self {
        Llama {
            model: inference_llama.model,
            rope: inference_llama.pos_encoding.rope,
        }
    }
}

impl Adaptor<LossInput> for LlamaOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<AccuracyInput> for LlamaOutput {
    fn adapt(&self) -> AccuracyInput {
        let [batch_size, seq_len, vocab_size] = self.logits.dims();

        let logits_flattened = self
            .logits
            .clone()
            .reshape([batch_size * seq_len, vocab_size]);
        let targets_flattened = self.targets.clone().reshape([batch_size * seq_len]);

        AccuracyInput::new(logits_flattened, targets_flattened)
    }
}

impl ItemLazy for LlamaOutput {
    fn sync(self) -> Self {
        let [logits, loss, targets] = Transaction::default()
            .register(self.logits)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = Device::flex();

        LlamaOutput {
            loss: Tensor::from_data(loss, &device),
            logits: Tensor::from_data(logits, &device),
            targets: Tensor::from_data(targets, &device),
        }
    }
}

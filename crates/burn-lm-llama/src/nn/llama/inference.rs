use burn::{
    module::{Module, Quantizer},
    record::{FileRecorder, RecorderError},
    tensor::{
        quantization::{Calibration, QuantScheme},
        Device, Int, Shape, Tensor, TensorData,
    },
};
use std::time::Instant;

use crate::{
    nn::{
        pos_encoding::PositionalEncodingState,
        transformer::{Transformer, TransformerCache},
    },
    tokenizer::Tokenizer,
};

/// Meta Llama large language model and tokenizer. For inference uses only.
#[derive(Debug)]
pub struct Llama<T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Llama decoder-only transformer.
    pub model: Transformer,
    /// Key-value cache for each transformer block.
    pub cache: TransformerCache,
    /// Rotary positional encoding (RoPE).
    pub pos_encoding: PositionalEncodingState,
    pub device: Device,
}

impl<T: Tokenizer> Llama<T> {
    /// Encode a string into a tensor of tokens.
    pub fn tokenize(&self, text: &str) -> Tensor<1, Int> {
        let tokens = self.tokenizer.encode(text, false, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<1, Int>::from_data(TensorData::new(tokens, shape), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder>(self, file_path: &str, recorder: &R) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {elapsed}s");

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        Ok(self)
    }

    /// Reset the model state (used between generations)
    pub fn reset(&mut self) {
        self.cache.reset();
        self.pos_encoding.reset();
    }

    /// Quantize the model weights.
    pub fn quantize(mut self, scheme: QuantScheme) -> Self {
        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };
        let device = &self.model.devices()[0];

        // TODO: improve module mapper usage for quantization (currently, this leads to additional memory usage)
        // self.model = self.model.quantize_weights(&mut quantizer);

        // Quantizing by layer reduces the peak memory usage
        let mut layers = Vec::with_capacity(self.model.layers.len());
        for layer in self.model.layers.drain(..) {
            layers.push(layer.quantize_weights(&mut quantizer));
        }
        self.model.layers = layers;
        let _ = device.sync();

        self.model.tok_embeddings = self.model.tok_embeddings.quantize_weights(&mut quantizer);
        self.model.output = self.model.output.quantize_weights(&mut quantizer);

        self
    }
}

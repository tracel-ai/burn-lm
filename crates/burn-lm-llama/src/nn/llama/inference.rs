use burn::{
    module::{Module, Quantizer},
    record::{FileRecorder, RecorderError},
    tensor::{
        quantization::{Calibration, QuantScheme},
        Device, Int, Shape, Tensor, TensorData,
    },
};
use std::time::Instant;

use burn_lm_inference::{
    batching::{BatchCapacity, BatchedDecoder, ForwardBatch, ForwardOutput},
    InferenceError, InferenceResult,
};

use crate::{
    generation::GenerationError,
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
    /// Reusable decoder state for autoregressive inference.
    pub decoder: LlamaDecoder,
}

/// Reusable Llama decoder state for autoregressive inference.
#[derive(Debug)]
pub struct LlamaDecoder {
    /// Llama decoder-only transformer.
    pub model: Transformer,
    /// Key-value cache for each transformer block.
    pub cache: TransformerCache,
    /// Rotary positional encoding (RoPE).
    pub pos_encoding: PositionalEncodingState,
    pub device: Device,
}

impl LlamaDecoder {
    /// Forward one prompt/decode chunk through the decoder, updating cache and RoPE state.
    pub fn forward(&mut self, input: Tensor<2, Int>) -> Result<Tensor<3>, GenerationError> {
        let [_, seq_len] = input.dims();

        // Prepare cache and RoPE for current sequence length and position.
        let mask = self.cache.prepare(seq_len)?;
        self.pos_encoding.prepare(seq_len);

        Ok(self
            .model
            .forward(input, &mut self.cache, &self.pos_encoding, mask))
    }

    /// Reset decoder state between independent generations.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.pos_encoding.reset();
    }
}

/// Per-sequence decoder state OWNED BY THE FRAMEWORK engine.
///
/// The KV cache and RoPE state used to live inside `LlamaDecoder`, which made it single-sequence.
/// For continuous batching the engine allocates one of these per active sequence and hands it back
/// to [`forward`](BatchedDecoder::forward) on every step; the decoder swaps it into its own slots
/// for the duration of the call. This keeps each sequence's KV history independent while the model
/// author still "just writes forward".
#[derive(Debug, Clone)]
pub struct LlamaSeqCache {
    cache: TransformerCache,
    pos_encoding: PositionalEncodingState,
}

impl BatchedDecoder for LlamaDecoder {
    type Cache = LlamaSeqCache;

    /// Allocate a fresh, empty per-sequence cache by cloning the decoder's (weights-independent)
    /// cache + RoPE templates and resetting them. `capacity` is unused for now: the round-robin
    /// stub uses one batch-1 cache per sequence rather than a slot-indexed shared cache.
    fn allocate_cache(&self, _capacity: BatchCapacity) -> Self::Cache {
        let mut cache = self.cache.clone();
        cache.reset();
        let mut pos_encoding = self.pos_encoding.clone();
        pos_encoding.reset();
        LlamaSeqCache {
            cache,
            pos_encoding,
        }
    }

    /// Forward a (currently 1-row) batch against the PASSED-IN, engine-owned cache.
    ///
    /// The decoder temporarily swaps the engine's cache + RoPE state into its own slots, runs the
    /// inner forward, then swaps them back, so the per-sequence history lives in `cache` and not in
    /// the shared decoder. `positions`/`cache_slots` are part of the shape but not yet consumed (the
    /// swapped-in state already tracks position); Phase 2's fused kernel will use them.
    fn forward(
        &mut self,
        batch: ForwardBatch,
        cache: &mut Self::Cache,
    ) -> InferenceResult<ForwardOutput> {
        std::mem::swap(&mut self.cache, &mut cache.cache);
        std::mem::swap(&mut self.pos_encoding, &mut cache.pos_encoding);

        let result = self.forward(batch.input_tokens);

        std::mem::swap(&mut self.cache, &mut cache.cache);
        std::mem::swap(&mut self.pos_encoding, &mut cache.pos_encoding);

        let logits = result.map_err(|err| match err {
            GenerationError::MaxSequenceLengthExceeded { actual, max } => {
                InferenceError::ContextLengthExceeded(actual, max)
            }
        })?;
        Ok(ForwardOutput { logits })
    }
}

impl<T: Tokenizer> Llama<T> {
    /// Encode a string into a tensor of tokens.
    pub fn tokenize(&self, text: &str) -> Tensor<1, Int> {
        let tokens = self.tokenizer.encode(text, false, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<1, Int>::from_data(TensorData::new(tokens, shape), &self.decoder.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder>(self, file_path: &str, recorder: &R) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.decoder.model.save_file(file_path, recorder)?;
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
        self.decoder.model =
            self.decoder
                .model
                .load_file(file_path, recorder, &self.decoder.device)?;
        Ok(self)
    }

    /// Reset the model state (used between generations)
    pub fn reset(&mut self) {
        self.decoder.reset()
    }

    /// Quantize the model weights.
    pub fn quantize(mut self, scheme: QuantScheme) -> Self {
        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };
        let device = &self.decoder.model.devices()[0];

        // TODO: improve module mapper usage for quantization (currently, this leads to additional memory usage)
        // self.decoder.model = self.decoder.model.quantize_weights(&mut quantizer);

        // Quantizing by layer reduces the peak memory usage
        let mut layers = Vec::with_capacity(self.decoder.model.layers.len());
        for layer in self.decoder.model.layers.drain(..) {
            layers.push(layer.quantize_weights(&mut quantizer));
        }
        self.decoder.model.layers = layers;
        let _ = device.sync();

        self.decoder.model.tok_embeddings = self
            .decoder
            .model
            .tok_embeddings
            .quantize_weights(&mut quantizer);
        self.decoder.model.output = self.decoder.model.output.quantize_weights(&mut quantizer);

        self
    }
}

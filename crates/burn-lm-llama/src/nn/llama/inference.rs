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
    batching::{BatchedDecoder, DecodeRow},
    InferenceError, InferenceResult,
};
use std::collections::HashMap;

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
    /// One private KV/RoPE state per engine slot (see [`BatchedDecoder`]), created on first use
    /// and dropped on [`release`](BatchedDecoder::release). Today each slot is an independent
    /// batch-1 cache; replacing this map with one shared, slot-indexed cache is the next step.
    pub(crate) slots: HashMap<usize, LlamaSeqCache>,
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
        // Slot caches are normally all released by the engine before this is called; clearing
        // them too keeps "reset" meaning what it says — no decoding state survives.
        self.slots.clear();
    }

    /// A fresh, empty per-slot state: the decoder's (weights-independent) cache + RoPE templates,
    /// cloned and reset.
    fn fresh_slot_state(&self) -> LlamaSeqCache {
        let mut cache = self.cache.clone();
        cache.reset();
        let mut pos_encoding = self.pos_encoding.clone();
        pos_encoding.reset();
        LlamaSeqCache {
            cache,
            pos_encoding,
        }
    }

    /// Forward `input` through the given slot's cache, returning the last position's logits
    /// (`[1, vocab]`).
    ///
    /// The slot's state is taken OUT of the map (created fresh on first use), swapped into the
    /// decoder's own cache/RoPE fields for the duration of the inner forward, then swapped back.
    /// On success it is put back into the map; on error it is dropped, which leaves the slot as
    /// if it had never been used (the contract `prefill` promises).
    fn forward_slot(&mut self, slot: usize, input: Tensor<2, Int>) -> InferenceResult<Tensor<2>> {
        let mut state = self
            .slots
            .remove(&slot)
            .unwrap_or_else(|| self.fresh_slot_state());

        std::mem::swap(&mut self.cache, &mut state.cache);
        std::mem::swap(&mut self.pos_encoding, &mut state.pos_encoding);

        let result = self.forward(input);

        std::mem::swap(&mut self.cache, &mut state.cache);
        std::mem::swap(&mut self.pos_encoding, &mut state.pos_encoding);

        let logits = result.map_err(|err| match err {
            GenerationError::MaxSequenceLengthExceeded { actual, max } => {
                InferenceError::ContextLengthExceeded(actual, max)
            }
        })?;
        self.slots.insert(slot, state);

        let [batch, seq_len, vocab] = logits.dims();
        Ok(logits
            .slice([0..batch, seq_len - 1..seq_len, 0..vocab])
            .reshape([batch, vocab]))
    }
}

/// One slot's decoding state: the KV cache and RoPE position for a single sequence. Kept inside
/// [`LlamaDecoder`] behind the engine's slot numbers; nothing outside this module touches it.
#[derive(Debug, Clone)]
pub(crate) struct LlamaSeqCache {
    cache: TransformerCache,
    pos_encoding: PositionalEncodingState,
}

impl BatchedDecoder for LlamaDecoder {
    /// Run a whole prompt into one slot. A `position == 0` prompt starts the slot fresh; the slot's
    /// own cache/RoPE state tracks everything else, and the engine only prefills at position 0 today.
    /// The parameter exists for the shared, slot-indexed cache that replaces the per-slot map
    /// next.
    fn prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
        position: usize,
    ) -> InferenceResult<Tensor<2>> {
        // A prompt starting at position 0 is a NEW sequence: drop whatever a previous occupant
        // left in this slot. Normally `release` already did, but this makes the slot self-healing
        // even if a retire path missed it (e.g. a hypothetical server whose `decoder()` fails
        // transiently while staying loaded) — a fresh sequence can never resume a dead one's
        // cache. Chunked prompts (position > 0, future work) legitimately continue the state.
        if position == 0 {
            self.slots.remove(&slot);
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [1, tokens.len()]), &self.device);
        self.forward_slot(slot, input)
    }

    /// Advance each row's slot by one token. Still one inner forward per row (the fused
    /// multi-row call comes next); an error on any row fails the whole call, which is fine
    /// because the framework only passes single-row slices today.
    fn decode(&mut self, rows: &[DecodeRow]) -> InferenceResult<Tensor<2>> {
        let mut outputs = Vec::with_capacity(rows.len());
        for row in rows {
            let input = Tensor::<2, Int>::from_data(
                TensorData::new(vec![row.token as i32], [1, 1]),
                &self.device,
            );
            outputs.push(self.forward_slot(row.slot, input)?);
        }
        Ok(Tensor::cat(outputs, 0))
    }

    fn release(&mut self, slot: usize) {
        self.slots.remove(&slot);
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

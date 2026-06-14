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
///
/// The KV/RoPE state is a single shared slab: `cache` holds `max_batch_size` lanes (one per engine
/// slot, see [`BatchCapacity`](burn_lm_inference::batching::BatchCapacity)), and a whole round of
/// active sequences advances through ONE lane-aware forward. Slot numbers index lanes directly
/// (the server sets `max_batch_size == max_slots`).
#[derive(Debug)]
pub struct LlamaDecoder {
    /// Llama decoder-only transformer.
    pub model: Transformer,
    /// Shared, lane-indexed KV cache (one lane per engine slot). Writes/reads are lane-sliced and
    /// [`release`](BatchedDecoder::release) resets a lane.
    pub cache: TransformerCache,
    /// Rotary positional encoding (RoPE). Lane decode reads `rope` directly at each lane's absolute
    /// position (no table shift).
    pub pos_encoding: PositionalEncodingState,
    pub device: Device,
}

impl LlamaDecoder {
    /// Reset decoder state between independent generations.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.pos_encoding.reset();
    }

    /// Forward `seq_len` new tokens for each lane in `lanes` (all the same length) through the
    /// shared slab, returning each lane's last-position logits, `[lanes.len(), vocab]`.
    ///
    /// Prefill is one lane with `seq_len == prompt_len`; fused decode is N active lanes with
    /// `seq_len == 1`. [`prepare_lanes`](TransformerCache::prepare_lanes) snapshots each lane's
    /// start position, builds the per-lane causal+padding mask, and advances the lane lengths; the
    /// model then writes/reads each lane sliced and rotates each at its own absolute position. A
    /// lane past `max_seq_len` is an error (lane mode has no cache eviction), surfaced as
    /// `ContextLengthExceeded`.
    fn forward_lanes(
        &mut self,
        lanes: &[usize],
        input: Tensor<2, Int>,
    ) -> InferenceResult<Tensor<2>> {
        let seq_len_in = input.dims()[1];
        // The cache's per-lane lengths (RoPE positions + mask) must agree with each layer's KV write
        // offset before we forward. They only ever move together, but check it here so a future
        // wiring regression fails loudly instead of silently attending to the wrong KV columns.
        debug_assert!(
            self.cache.lanes_in_lockstep(lanes),
            "lane bookkeepers desynced before forward (cache lens vs layer KV)"
        );
        let plan = self
            .cache
            .prepare_lanes(lanes, seq_len_in)
            .map_err(|err| match err {
                GenerationError::MaxSequenceLengthExceeded { actual, max } => {
                    InferenceError::ContextLengthExceeded(actual, max)
                }
            })?;
        let logits =
            self.model
                .forward_lanes(input, &mut self.cache, &self.pos_encoding.rope, &plan);
        let [n, seq_len, vocab] = logits.dims();
        Ok(logits
            .slice([0..n, seq_len - 1..seq_len, 0..vocab])
            .reshape([n, vocab]))
    }
}

impl BatchedDecoder for LlamaDecoder {
    /// Run a whole prompt into one lane, returning the last position's logits (`[1, vocab]`).
    ///
    /// A `position == 0` prompt is a NEW sequence: reset the lane first so a fresh sequence can
    /// never resume a previous occupant's KV (normally `release` already did this — resetting again
    /// makes the lane self-healing). Chunked prompts (`position > 0`, future work) legitimately
    /// continue the lane's state.
    fn prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
        position: usize,
    ) -> InferenceResult<Tensor<2>> {
        if position == 0 {
            self.cache.reset_lane(slot);
        }
        debug_assert_eq!(
            position,
            self.cache.lane_len(slot),
            "prefill position must equal the lane's current length"
        );
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [1, tokens.len()]), &self.device);
        self.forward_lanes(&[slot], input)
    }

    /// Advance every row's lane by one token in ONE fused forward, returning logits
    /// `[rows.len(), vocab]` where row `i` belongs to `rows[i]`. Each lane sits at its own absolute
    /// position; the lane-aware forward writes/reads each sliced and masks each lane's stale tail.
    fn decode(&mut self, rows: &[DecodeRow]) -> InferenceResult<Tensor<2>> {
        let lanes: Vec<usize> = rows.iter().map(|row| row.slot).collect();
        let ids: Vec<i32> = rows.iter().map(|row| row.token as i32).collect();
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [rows.len(), 1]), &self.device);
        self.forward_lanes(&lanes, input)
    }

    fn release(&mut self, slot: usize) {
        self.cache.reset_lane(slot);
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

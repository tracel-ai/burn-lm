use std::ops::Range;

use burn::tensor::{backend::Backend, Device, Shape, Tensor};

#[derive(Debug, Clone)]
/// Cache that keeps track of a tensor state in an autoregressive decoding process.
pub(crate) struct AutoregressiveCache<B: Backend, const D: usize> {
    cache: Tensor<B, D>,
    seq_dim: usize,
    cur_seq_len: usize,
    pub(crate) max_seq_len: usize,
}

impl<const D: usize, B: Backend> AutoregressiveCache<B, D> {
    /// Creates a new empty cache.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device<B>) -> Self {
        let max_seq_len = shape[seq_dim];
        Self {
            cache: Tensor::empty(shape, device),
            seq_dim,
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        let shape = self.cache.shape();
        let device = self.cache.device();

        self.cache.inplace(|cache| {
            core::mem::drop(cache);
            Tensor::empty(shape, &device)
        });

        self.cur_seq_len = 0;
    }

    /// Add the new tokens to the current cache and returns all tokens decoded since the beginning.
    ///
    /// # Shapes
    ///
    /// - input:  [batch_size, num_heads, seq_len_input, d_model]
    /// - output: [batch_size, num_heads, seq_len_previous + seq_len_input, d_model]
    pub fn append(&mut self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let shape = tokens.shape();
        let seq_len_input = shape.dims[self.seq_dim];

        if self.cur_seq_len + seq_len_input > self.max_seq_len {
            self.shrink(&shape);
        }

        let new_seq_len = self.cur_seq_len + seq_len_input;

        let mut indices_added_tokens = Vec::with_capacity(shape.dims.len());
        let mut indices_output = Vec::with_capacity(shape.dims.len());

        for (i, shape) in shape.dims.iter().enumerate() {
            if i == self.seq_dim {
                indices_added_tokens.push(self.cur_seq_len..new_seq_len);
                indices_output.push(0..new_seq_len);
            } else {
                indices_added_tokens.push(0..*shape);
                indices_output.push(0..*shape);
            }
        }
        self.cache.inplace(|cache| {
            cache.slice_assign::<D>(indices_added_tokens.try_into().unwrap(), tokens)
        });

        self.cur_seq_len += seq_len_input;

        self.cache
            .clone()
            .slice::<D, [Range<usize>; D]>(indices_output.try_into().unwrap())
    }

    /// Shrink the cache to fit in `max_seq_len` while making place for the new tokens being
    /// decoded.
    fn shrink(&mut self, shape_tokens: &Shape) {
        let seq_len_input = shape_tokens.dims[self.seq_dim];
        self.cur_seq_len = self.max_seq_len - seq_len_input;

        let mut slices_prev = Vec::with_capacity(shape_tokens.dims.len());
        let mut slices_curr = Vec::with_capacity(shape_tokens.dims.len());

        for (i, shape) in shape_tokens.dims.iter().enumerate() {
            if i == self.seq_dim {
                slices_prev.push(seq_len_input..self.max_seq_len);
                slices_curr.push(0..self.cur_seq_len);
            } else {
                slices_prev.push(0..*shape);
                slices_curr.push(0..*shape);
            }
        }

        let prev_slice = self
            .cache
            .clone()
            .slice::<D, [Range<usize>; D]>(slices_prev.try_into().unwrap());

        let new_cache = Tensor::empty(self.cache.shape(), &self.cache.device());
        self.cache = new_cache.slice_assign::<D>(slices_curr.try_into().unwrap(), prev_slice);
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}

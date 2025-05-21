use std::ops::Range;

use burn::tensor::{backend::Backend, Device, Tensor};

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
        let mut new_seq_len = self.cur_seq_len + seq_len_input;

        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len_input;

            let mut slices_prev = Vec::with_capacity(shape.dims.len());
            let mut slices_curr = Vec::with_capacity(shape.dims.len());

            for (i, shape) in shape.dims.iter().enumerate() {
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
            new_seq_len = self.max_seq_len;
        }

        let mut slices_assign = Vec::with_capacity(shape.dims.len());
        let mut slices_output = Vec::with_capacity(shape.dims.len());

        for (i, shape) in shape.dims.iter().enumerate() {
            if i == self.seq_dim {
                slices_assign.push(self.cur_seq_len..new_seq_len);
                slices_output.push(0..self.cur_seq_len + seq_len_input);
            } else {
                slices_assign.push(0..*shape);
                slices_output.push(0..*shape);
            }
        }
        self.cache
            .inplace(|cache| cache.slice_assign::<D>(slices_assign.try_into().unwrap(), tokens));

        self.cur_seq_len += seq_len_input;

        self.cache
            .clone()
            .slice::<D, [Range<usize>; D]>(slices_output.try_into().unwrap())
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}

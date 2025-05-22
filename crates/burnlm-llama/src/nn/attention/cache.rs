use std::ops::Range;

use burn::tensor::{backend::Backend, Device, Tensor};

#[derive(Debug, Clone)]
/// Cache that keeps track of a tensor state in an autoregressive decoding process.
pub(crate) struct AutoregressiveCache<B: Backend, const D: usize> {
    cache: Tensor<B, D>,
    seq_dim: usize,
    cur_seq_len: usize,
}

impl<const D: usize, B: Backend> AutoregressiveCache<B, D> {
    /// Creates a new empty cache.
    pub fn new(shape: [usize; D], seq_dim: usize, device: &Device<B>) -> Self {
        Self {
            cache: Tensor::empty(shape, device),
            seq_dim,
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
    pub fn shrink(&mut self, num_removed: usize) {
        let old_cur_seq_len = self.cur_seq_len;
        self.cur_seq_len -= num_removed;

        let shape = self.cache.shape();
        let device = self.cache.device();

        let mut slices_prev = Vec::with_capacity(shape.dims.len());
        let mut slices_curr = Vec::with_capacity(shape.dims.len());

        for (i, shape) in shape.dims.iter().enumerate() {
            if i == self.seq_dim {
                slices_prev.push(num_removed..old_cur_seq_len);
                slices_curr.push(0..self.cur_seq_len);
            } else {
                slices_prev.push(0..*shape);
                slices_curr.push(0..*shape);
            }
        }

        self.cache.inplace(|cache| {
            let prev_slice = cache.slice::<D, [Range<usize>; D]>(slices_prev.try_into().unwrap());
            let new_cache = Tensor::empty(shape, &device);

            new_cache.slice_assign::<D>(slices_curr.try_into().unwrap(), prev_slice)
        });
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn test_autoregressive_cache() {
        let device: Device<TestBackend> = Default::default();
        let mut cache = AutoregressiveCache::<TestBackend, 2>::new([8, 8], 0, &device);

        let tokens_1 = Tensor::<TestBackend, 2>::full([4, 8], 1.0, &device);
        let tokens_2 = Tensor::<TestBackend, 2>::full([4, 8], 2.0, &device);

        let received_1 = cache.append(tokens_1.clone());
        assert_eq!(cache.len(), 4);
        let received_2 = cache.append(tokens_2.clone());
        assert_eq!(cache.len(), 8);

        received_1.to_data().assert_eq(&tokens_1.to_data(), true);
        received_2
            .clone()
            .slice(0..4)
            .to_data()
            .assert_eq(&tokens_1.to_data(), true);
        received_2
            .slice(4..8)
            .to_data()
            .assert_eq(&tokens_2.to_data(), true);

        cache.shrink(2);
        assert_eq!(cache.len(), 6);

        let tokens_3 = Tensor::<TestBackend, 2>::full([2, 8], 3.0, &device);
        let received_3 = cache.append(tokens_3.clone());
        assert_eq!(cache.len(), 8);

        received_3
            .clone()
            .slice(0..2)
            .to_data()
            .assert_eq(&tokens_1.slice(2..4).into_data(), true);
        received_3
            .clone()
            .slice(2..6)
            .to_data()
            .assert_eq(&tokens_2.to_data(), true);
        received_3
            .slice(6..8)
            .to_data()
            .assert_eq(&tokens_3.to_data(), true);
    }
}

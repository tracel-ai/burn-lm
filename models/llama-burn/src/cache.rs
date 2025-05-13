use burn::tensor::{backend::Backend, Device, Tensor};

#[derive(Debug, Clone)]
pub(crate) struct AutoregressiveCache<B: Backend> {
    /// Tensor cache with shape `[batch_size, num_heads, seq_len, d_model]`
    cache: Option<Tensor<B, 4>>,
    pub(crate) max_seq_len: usize,
    cur_seq_len: usize,
}

impl<B: Backend> AutoregressiveCache<B> {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            cache: Some(Tensor::zeros(
                [max_batch_size, num_heads, max_seq_len, d_model],
                device,
            )),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        let old_cache = self.cache.take().expect("Cache to be allocated");
        let shape = old_cache.shape();
        let device = old_cache.device();

        core::mem::drop(old_cache);

        self.cache = Some(Tensor::zeros(shape, &device));
        self.cur_seq_len = 0;
    }

    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut cache = self.cache.take().expect("Cache to be allocated");

        let [batch_size, num_heads, seq_len, d_model] = tensor.dims();
        let mut new_seq_len = self.cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = cache.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..d_model,
            ]);
            cache = cache.slice_assign(
                [0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        cache = cache.slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..d_model,
            ],
            tensor,
        );

        self.cur_seq_len += seq_len;

        let result =
            cache
                .clone()
                .slice([0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model]);

        self.cache = Some(cache);

        result
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}

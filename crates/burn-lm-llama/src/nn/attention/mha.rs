use burn::{
    nn::{Linear, LinearConfig, RotaryEncoding},
    prelude::*,
    tensor::activation::softmax,
};

use crate::nn::pos_encoding::PositionalEncodingState;

use super::kv_cache::KeyValueCache;

/// Configuration to create a [multi-head attention](MultiHeadAttention) module.
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    /// Query projection.
    wq: Linear,
    /// Key projection.
    wk: Linear,
    /// Value projection.
    wv: Linear,
    /// Output projection.
    wo: Linear,

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Applies masked self-attention in a non-cached (non-incremental) setting.
    ///
    /// This function is intended for scenarios where the entire input sequence
    /// is available.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward_masked(&self, input: Tensor<3>, rope: &RotaryEncoding) -> Tensor<3> {
        let device = input.device();
        let [batch_size, seq_len, hidden_size] = input.dims();

        let (q, k, v) = self.forward_projection(input);

        // Start position is 0
        let q = rope.forward(q);
        let k = rope.forward(k);

        let mask = if seq_len > 1 {
            let mask = Tensor::<2, Bool>::tril_mask([seq_len, seq_len], 0, &device);
            Some(mask.unsqueeze::<4>())
        } else {
            None
        };

        let output = self.forward_attention(q, k, v, mask, batch_size, seq_len, hidden_size);
        self.wo.forward(output)
    }

    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward_cache(
        &self,
        input: Tensor<3>,
        cache: &mut KeyValueCache,
        pos_encoding: &PositionalEncodingState,
        mask: Option<Tensor<4, Bool>>,
    ) -> Tensor<3> {
        let device = input.device();
        let [batch_size, seq_len, hidden_size] = input.dims();

        let (q, k, v) = self.forward_projection(input);

        let q = pos_encoding.apply(q);
        let k = pos_encoding.apply(k);

        // Key-value caching
        let (k, v) = cache.forward(k, v);

        let mask = if seq_len > 1 {
            match mask {
                Some(mask) => Some(mask),
                None => {
                    // We ensure that the correct mask is applied
                    let cache_seq_len = cache.len();
                    let mask = Tensor::<2, Bool>::tril_mask(
                        [seq_len, cache_seq_len],
                        (cache_seq_len - seq_len) as i64, // offset
                        &device,
                    );

                    Some(mask.unsqueeze::<4>())
                }
            }
        } else {
            None
        };

        let output = self.forward_attention(q, k, v, mask, batch_size, seq_len, hidden_size);

        self.wo.forward(output)
    }

    fn forward_projection(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch_size, seq_len, _hidden_size] = input.dims();

        let q = self.wq.forward(input.clone());
        let k = self.wk.forward(input.clone());
        let v = self.wv.forward(input);

        // [batch_size, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        (q, k, v)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        mask: Option<Tensor<4, Bool>>,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Tensor<3> {
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Attention scores
        let mut scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt());

        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }

        let scores = softmax(scores, 3);
        let output = scores.matmul(v);

        output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, hidden_size])
    }

    /// Repeats a key or value tensor for grouped query attention.
    fn repeat_kv(&self, x: Tensor<4>) -> Tensor<4> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            x
        } else {
            let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();

            x.unsqueeze_dim::<5>(2)
                .expand([batch_size, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([batch_size, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multi-head attention](MultiHeadAttention) module.
    pub fn init(&self, device: &Device) -> MultiHeadAttention {
        let head_dim = self.d_model / self.n_heads;

        let wq = LinearConfig::new(self.d_model, self.n_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wk = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wv = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * head_dim, self.d_model)
            .with_bias(false)
            .init(device);

        MultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::reinit_uniform;
    use burn::{nn::RotaryEncodingConfig, tensor::Tolerance};

    #[test]
    pub fn test_attention_empty_cache() {
        let seq_length = 3;
        let batch_size = 2;
        let config = MultiHeadAttentionConfig::new(32, 2, 2);
        let device: Device = Default::default();
        let mha = config.init(&device);

        let mha = reinit_uniform(mha, -2.0, 2.0);

        let shape = Shape::from([batch_size, seq_length, config.d_model]);
        let input = Tensor::arange(0..shape.num_elements() as i64, &device)
            .reshape(shape)
            .float();

        let mut cache = KeyValueCache::new(
            batch_size,
            config.n_heads,
            seq_length,
            config.d_model,
            &device,
        );

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);
        let rope = PositionalEncodingState::new(rope);

        let output = mha.forward_cache(input, &mut cache, &rope, None);
        let expected = arange_mha_masked_expected_value();

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }

    #[test]
    pub fn test_attention_masked() {
        let seq_length = 3;
        let batch_size = 2;
        let config = MultiHeadAttentionConfig::new(32, 2, 2);
        let device: Device = Default::default();
        let mha = config.init(&device);

        let mha = reinit_uniform(mha, -2.0, 2.0);

        let shape = Shape::from([batch_size, seq_length, config.d_model]);
        let input = Tensor::arange(0..shape.num_elements() as i64, &device)
            .reshape(shape)
            .float();

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);

        let output = mha.forward_masked(input, &rope);
        let expected = arange_mha_masked_expected_value();

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }

    #[test]
    pub fn test_attention_decoding() {
        let seq_length = 3;
        let batch_size = 2;
        let config = MultiHeadAttentionConfig::new(32, 2, 2);
        let device: Device = Default::default();
        let mha = config.init(&device);

        let mha = reinit_uniform(mha, -2.0, 2.0);

        let shape = Shape::from([batch_size, seq_length, config.d_model]);
        let input = Tensor::arange(0..shape.num_elements() as i64, &device)
            .reshape(shape)
            .float();

        let rope = RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads)
            .init(&device);
        let rope = PositionalEncodingState::new(rope);

        let mut cache = KeyValueCache::new(
            batch_size,
            config.n_heads,
            seq_length,
            config.d_model,
            &device,
        );

        let out_1 = mha.forward_cache(
            input
                .clone()
                .slice([0..batch_size, 0..1, 0..config.d_model]),
            &mut cache,
            &rope,
            None,
        );
        let out_2 = mha.forward_cache(
            input
                .clone()
                .slice([0..batch_size, 1..2, 0..config.d_model]),
            &mut cache,
            &rope,
            None,
        );
        let out_3 = mha.forward_cache(
            input
                .clone()
                .slice([0..batch_size, 2..3, 0..config.d_model]),
            &mut cache,
            &rope,
            None,
        );

        let output = Tensor::cat(vec![out_1, out_2, out_3], 1);

        let expected = arange_mha_decoding_expected_value();

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }

    fn arange_mha_masked_expected_value() -> TensorData {
        TensorData::from([
            [
                [
                    90.5116, 547.54144, -1703.8286, 729.67267, -1223.3955, -808.56445, 144.42798,
                    164.06868, -495.59302, -266.84998, -751.40497, 610.7481, -1286.2684, 452.2409,
                    -636.4697, 122.76248, 863.5856, 1135.9159, 719.8953, 1248.2205, -1573.8358,
                    235.662, 1466.8099, 1222.4841, -307.24506, -1887.6362, -419.21533, -441.77,
                    315.9563, 698.2246, 103.88786, 55.314255,
                ],
                [
                    -65.781685, 1003.6326, -3940.6128, 2394.3564, -2243.464, -809.18884, -45.47723,
                    1835.6915, -1770.7511, -272.3688, -1517.1707, 520.0995, -1833.5188, 1609.8396,
                    -1294.0443, -850.968, 304.09576, 744.3857, 1936.7549, 2347.9138, -2375.1675,
                    1238.5516, 2195.7654, 3416.9504, -1255.3962, -3201.321, -538.0854, -824.5863,
                    170.30212, 1377.9116, -220.8315, 937.8362,
                ],
                [
                    -222.07532, 1459.7244, -6177.3975, 4059.041, -3263.5322, -809.81366,
                    -235.38222, 3507.3142, -3045.9087, -277.88712, -2282.9368, 429.45065,
                    -2380.7686, 2767.4392, -1951.6188, -1824.6986, -255.394, 352.85568, 3153.6145,
                    3447.607, -3176.4993, 2241.441, 2924.7207, 5611.416, -2203.5476, -4515.0054,
                    -656.9555, -1207.4027, 24.648682, 2057.5981, -545.55145, 1820.3583,
                ],
            ],
            [
                [
                    -2005.0234,
                    6449.864,
                    -10454.1045,
                    2914.761,
                    -8109.836,
                    -3940.267,
                    2928.1294,
                    2582.8,
                    462.07996,
                    -1644.6171,
                    -6235.0845,
                    1430.9061,
                    -4518.3496,
                    4767.4746,
                    14.1672945,
                    -320.83813,
                    607.05505,
                    4557.583,
                    7000.466,
                    8613.178,
                    -9793.199,
                    2747.376,
                    8880.409,
                    8671.414,
                    848.31537,
                    -11634.635,
                    -4033.9194,
                    -477.09555,
                    1183.0527,
                    3689.6943,
                    971.4231,
                    2270.056,
                ],
                [
                    -2703.5347, 8417.301, -13370.862, 3643.1228, -10405.316, -4984.167, 3856.027,
                    3389.044, 781.3019, -2103.8726, -8062.9775, 1704.2921, -5595.711, 6205.8857,
                    231.0452, -468.7061, 521.54425, 5698.139, 9093.989, 11068.161, -12532.983,
                    3584.6143, 11351.608, 11154.392, 1233.502, -14883.637, -5238.82, -488.86993,
                    1472.085, 4686.852, 1260.6014, 3008.302,
                ],
                [
                    -3402.047, 10384.742, -16287.621, 4371.487, -12700.798, -6028.069, 4783.931,
                    4195.287, 1100.5276, -2563.1282, -9890.871, 1977.6786, -6673.072, 7644.296,
                    447.925, -616.5714, 436.03445, 6838.695, 11187.512, 13523.148, -15272.77,
                    4421.8525, 13822.809, 13637.365, 1618.6896, -18132.635, -6443.721, -500.64493,
                    1761.1156, 5684.0083, 1549.7786, 3746.55,
                ],
            ],
        ])
    }

    fn arange_mha_decoding_expected_value() -> TensorData {
        TensorData::from([
            [
                [
                    90.51162, 547.5413, -1703.8289, 729.6727, -1223.3956, -808.5644, 144.42801,
                    164.06871, -495.5932, -266.85004, -751.4048, 610.74805, -1286.2683, 452.2407,
                    -636.4697, 122.76244, 863.5855, 1135.9158, 719.8953, 1248.2205, -1573.8358,
                    235.66185, 1466.8098, 1222.4841, -307.24512, -1887.6364, -419.2152, -441.77002,
                    315.95633, 698.2246, 103.887955, 55.31427,
                ],
                [
                    -65.78191, 1003.6329, -3940.6133, 2394.3567, -2243.4644, -809.18884,
                    -45.477036, 1835.6917, -1770.7513, -272.36874, -1517.1708, 520.0992,
                    -1833.5187, 1609.84, -1294.0444, -850.968, 304.09583, 744.3856, 1936.7551,
                    2347.9136, -2375.1675, 1238.5514, 2195.765, 3416.9507, -1255.397, -3201.321,
                    -538.0853, -824.5863, 170.30243, 1377.9113, -220.83176, 937.83624,
                ],
                [
                    -222.07547, 1459.7246, -6177.3984, 4059.04, -3263.5322, -809.81384, -235.38206,
                    3507.3142, -3045.9092, -277.88736, -2282.9368, 429.45026, -2380.7693,
                    2767.4385, -1951.6189, -1824.6982, -255.3937, 352.85568, 3153.615, 3447.607,
                    -3176.4993, 2241.4407, 2924.7205, 5611.4165, -2203.5479, -4515.006, -656.95557,
                    -1207.4023, 24.648346, 2057.598, -545.5519, 1820.3585,
                ],
            ],
            [
                [
                    -2005.0233, 6449.863, -10454.104, 2914.76, -8109.834, -3940.2686, 2928.129,
                    2582.7998, 462.0785, -1644.6161, -6235.085, 1430.9066, -4518.3506, 4767.474,
                    14.167358, -320.8379, 607.0547, 4557.5825, 7000.466, 8613.18, -9793.199,
                    2747.3762, 8880.409, 8671.414, 848.3146, -11634.635, -4033.919, -477.09497,
                    1183.0525, 3689.6948, 971.42194, 2270.0557,
                ],
                [
                    -2161.3164, 6905.954, -12690.888, 4579.4443, -9129.906, -3940.8914, 2738.2227,
                    4254.423, -813.07996, -1650.1348, -7000.8506, 1340.2574, -5065.6016, 5925.0728,
                    -643.4073, -1294.5688, 47.56482, 4166.053, 8217.325, 9712.869, -10594.531,
                    3750.266, 9609.365, 10865.881, -99.83569, -12948.323, -4152.789, -859.9115,
                    1037.3982, 4369.382, 646.70325, 3152.5762,
                ],
                [
                    -2317.6104, 7362.045, -14927.674, 6244.13, -10149.975, -3941.5168, 2548.3186,
                    5926.045, -2088.2378, -1655.6528, -7766.617, 1249.6083, -5612.8516, 7082.672,
                    -1300.9817, -2268.299, -511.92505, 3774.5227, 9434.186, 10812.566, -11395.861,
                    4753.1553, 10338.315, 13060.347, -1047.9883, -14262.004, -4271.659, -1242.7279,
                    891.745, 5049.0684, 321.98376, 4035.0996,
                ],
            ],
        ])
    }
}

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, SwiGlu, SwiGluConfig};
use burn::tensor::{Device, Tensor};

#[derive(Config, Debug)]
/// Configuration to create a [feed-forward transformation network](FeedForward).
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
}

/// Feed-forward transformation network.
#[derive(Module, Debug)]
pub struct FeedForward {
    // Swish gated linear unit with trainable parameters.
    swiglu: SwiGlu,
    /// Outer linear.
    w2: Linear,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init(&self, device: &Device) -> FeedForward {
        let swiglu = SwiGluConfig::new(self.d_model, self.hidden_size)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_size, self.d_model)
            .with_bias(false)
            .init(device);

        FeedForward { swiglu, w2 }
    }
}
impl FeedForward {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        self.w2.forward(self.swiglu.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{TensorData, Tolerance};

    use crate::tests::reinit_uniform;

    use super::*;

    #[test]
    fn test_fftn() {
        let device: Device = Default::default();
        let batch_size = 2;
        let seq_length = 2;
        let d_model = 4;
        let hidden_size = 8;

        let config = FeedForwardConfig::new(d_model, hidden_size);
        let transformer: FeedForward = config.init(&device);

        let input = Tensor::arange(0..(batch_size * seq_length * d_model) as i64, &device)
            .reshape([batch_size, seq_length, d_model])
            .float();

        let nn = reinit_uniform(transformer, 0.0, 5.0);

        let output = nn.forward(input);

        let expected = TensorData::from([
            [
                [6205.8174, 8136.716, 3293.2156, 6027.6084],
                [79597.02, 106373.56, 44417.668, 82802.47],
            ],
            [
                [235638.03, 316002.06, 132491.31, 247838.31],
                [474329.06, 637021.25, 267513.06, 501135.06],
            ],
        ]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.1));
    }
}

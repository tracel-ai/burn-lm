# New Model Implementation

To add a new model implementation, you can use the command :

```sh
cargo burn-lm new {model_name}
```

This will create a new crate and register it in the model registry.

## Project Structure

For the project structure, you can draw inspiration from the `burn-lm-llama` crate. The
implementations are still evolving rapidly, so detailed documentation may not keep pace with the
code changes.

## Import Weights

To support a new model in `burn-lm`, you typically want to import pre-trained weights from an
existing open-source model release. The process involves mapping the external model definition to a
Burn-native one and converting the weights.

This can be broken down into three main steps:

1. Define the model with Burn
1. Add configs for each variant
1. Import the weights

### Burn Model Definition

Implementing your model with Burn is pretty straightforward thanks to the provided building blocks.

Start by defining the model in Burn according to the architecture used in the original
implementation. For example, we referred to the
[open-source PyTorch definition provided by Meta](https://github.com/meta-llama/llama3/tree/main/llama)
(moved to [llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py))
for `burn-lm-llama` models.

In Burn, custom modules are defined by creating structs or enums deriving the `Module` trait:

```rust
/// Feed-forward transformation network.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    // Swish gated linear unit with trainable parameters.
    swiglu: SwiGlu<B>,
    /// Outer linear.
    w2: Linear<B>,
}
```

Subsequently, the forward pass can be implemented for the module:

```rust
impl<B: Backend> FeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2.forward(self.swiglu.forward(input))
    }
}
```

While the method is typically named forward for clarity and consistency, this is only by convention.
There is no restriction on the method name.

Once your blocks are defined, compose them into a top-level module that reflects the overall
architecture of the model. Here's a simplified example from the `burn-lm-llama` implementation,
which closely mirrors the original Llama model layout:

```rust
/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    output: Linear<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        cache: &mut TransformerCache<B>,
        pos_encoding: &PositionalEncodingState<B>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let mut h = self.tok_embeddings.forward(input);

        for (layer, c) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(h, c, pos_encoding, mask.clone());
        }

        let h = self.norm.forward(h);
        self.output.forward(h)
    }
}
```

<div class="warning">

While matching the architecture and naming structure of the source model is not mandatory, closely
following it makes importing weights significantly easier. The more your model diverges from the
original structure, the more remapping logic you'll need to add during the import step.

</div>

Most modules are quite straightforward to implement, but special care should be taken with
tokenization and postional encoding. These components are closely tied to the model's input
representation, and even small discrepancies can lead to subtle bugs. Thorough testing of their
correctness is especially important.

### Model Configurations

Once the module structure is in place, the next step is to define how each part of the model should
be initialized to construct a concrete variant (e.g., Llama 3.2 3B). This is done using a config
type for each module along with an initialization that builds the actual module.

Burn provides a `#[derive(Config)]` macro that simplifies this process. It supports:

- Default values using the `#[config(default = ...)]` attribute.
- Automatic constructors such as `new(...)` and `with_` methods for each field.
- Serialization, which helps makes config files easy to save and load.

Here's a minimal example:

```rust
#[derive(Config, Debug)]
/// Configuration to create a [feed-forward transformation network](FeedForward).
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let swiglu = SwiGluConfig::new(self.d_model, self.hidden_size)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_size, self.d_model)
            .with_bias(false)
            .init(device);

        FeedForward { swiglu, w2 }
    }
}
```

At the top level, transformer-based models like Llama are typically parameterized by several
hyperparameters, which are captured in a composite config:

```rust
/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config, Debug)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
}
```

This allows instantiating different variants of the transformer-based model by simply providing
different values. The top-level config serves as the entry point for model initialization. It
delegates the construction of individual blocks to their respective config types:

```rust
impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Transformer<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}
```

### Load the Weights

[TODO]

Once the pre-trained weights have been loaded successfully, we recommend saving the weights to
Burn's native format. This avoids unnecessary deseriliazation and conversion every time you want to
use the model.

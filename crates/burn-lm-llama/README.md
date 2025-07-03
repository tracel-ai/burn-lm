# Llama Burn

<img src="./assets/llama.jpeg" alt="An image of a llama surrounded by fiery colors and a gust of fire" width="500px"/>

The popular Llama LLM is here!

This repository contains the [Llama 3.2](https://github.com/meta-llama/llama-models/),
[Llama 3.1](https://github.com/meta-llama/llama-models/),
[Llama 3](https://github.com/meta-llama/llama3) and
[TinyLlama](https://github.com/jzhang38/TinyLlama) implementations with their corresponding
tokenizers. You can find the [Burn](https://github.com/tracel-ai/burn) implementation for the Llama
variants in [src/nn/llama/base.rs](src/nn/llama/base.rs).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
llama = { git = "https://github.com/tracel-ai/models", package = "llama", default-features = false }
```

If you want to use Llama 3 or TinyLlama (including pre-trained weights if default features are
active), enable the corresponding feature flag.

> **Important:** these features require `std`.

#### Llama 3

```toml
[dependencies]
llama = { git = "https://github.com/tracel-ai/models", package = "llama", features = ["llama3"] }
```

**Built with Llama 3.** This implementation uses the
[Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),
[Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
instruction-tuned models.

#### TinyLlama

```toml
[dependencies]
llama = { git = "https://github.com/tracel-ai/models", package = "llama", features = ["tiny"] }
```

This implementation uses the
[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
instruction-tuned model based on the Llama2 architecture and tokenizer.

## Known Issues

Based on your hardware and the model selected, the `wgpu` backend might not be able to successfully
run the model due to the current memory management strategy.

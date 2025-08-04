<div align="center">

<img src="./assets/burn-blue.webp" width="350px"/>
<h1>Burn LM</h1>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

---

**Burn-LM aims at democratizing large model inference and training on any device.**

<br/>
</div>

## Quick Start

Launch a Burn LM shell with:

```sh
git clone https://github.com/tracel-ai/burn-lm.git
cd burn-lm
cargo burn-lm
```

Type `help` to get a list of commands.

## Available Models

The list of models is very small at the moment since we're focused on performance optimization.
Still we're accepting high quality contributions to port open-source models to Burn-LM.

Here's the current list of supported models:

| Model     | Size   |
| --------- | ------ |
| Llama 3   | 8B     |
| Llama 3.1 | 8B     |
| Llama 3.2 | 1B, 3B |
| TinyLlama | 1.1B   |

### Adding a New Model

Models can be easily integrated with Burn LM by implementing the `InferenceServer`
trait to create a pluggable server that can be added to the Burn LM registry.

To bootstrap a new model server you can use the dedicated command `new`:

```sh
cargo burn-lm new "my-model"
```

This will create a new crate named `burn-lm-inference-my-model` and automatically
register it in `burn-lm-registry`.

The bootstraped server is a model-less server that just repeat the prompt it is
given. You can also get inspiration from the other crate with the crate `burn-lm-llama`.

<div align="center">

<h1>Burn LM</h1>

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
[![Burn](https://img.shields.io/badge/DL_Framework-Burn-f45b16)](https://github.com/tracel-ai/burn)
[![CubeCL](https://img.shields.io/badge/Compute_Language-CubeCL-3c83c2)](https://github.com/tracel-ai/cubecl)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

---

**Burn Large Models Repository.**

<br/>
</div>

# Quick Start

Launch a Burn LM shell with:

```sh
git clone https://github.com/tracel-ai/burn-lm.git
cd burn-lm
cargo burnlm
```

Type `help` to get a list of commands.

# Usage

## Models management

The list of installed models is displayed with:

```sh
cargo burnlm models
```

To download a model use the `download` command. This is will give you
the list of all downloadable models:

```sh
cargo burnlm download
```

To delete a downloaded model use the `delete` command.

## Inference

Run a single inference with the command `run`:

```sh
cargo burnlm run llama32 "Name a famous Quebecois dish."
```

## Chat

Burn LM allows to chat with LLMs both in the terminal or in the browser.

### Chat in the terminal

Start a chat session with a chosen model using the `chat` command:

```sh
cargo burnlm chat llama32
```

Some slash commands are available, you can get the list of them by typing `/help`
as a prompt.

### Chat in Open WebUI

First make sure `docker` and `docker-compose` are available on your system.

Then execute the command:

```sh
cargo burnlm web start
```

Head your browser to http://localhost:3000 and enjoy.

## Plugins

Models can be easily integrated with Burn LM by implementing the `InferenceServer`
trait to create a pluggable server that can be added to the Burn LM registry.

To bootstrap a new model server you can use the dedicated command `new`:

```sh
cargo burnlm new "my-model"
```

This will create a new crate named `burnlm-inference-my-model` and automatically
register it in `burnlm-registry`.

The bootstraped server is a model-less server that just repeat the prompt it is
given. You can also get inspiration from the other crate with name starting with
`burnlm-inference-`.




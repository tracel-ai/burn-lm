# burn-lm

Large Models Forge Repository powered by Burn.

## Quick Start

### Run inference from the terminal

```sh
cargo burnlm run tinyllama "Name a famous Quebecois dish."
```

List available models with:

```sh
cargo burnlm models
```

### Run inference from the browser

First make sure `docker` is available on your system.

Then execute the command:

```sh
cargo burnlm web start
```

Open http://localhost:3000 in your browser.

## Register a new model

To register a new model first create a new crate with:

```sh
cargo burnlm new "crate-name"
```

Then verify that your new model is registered with:

```sh
cargo burnlm models
```



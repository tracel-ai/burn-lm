use burn::{
    nn::RotaryEncodingConfig,
    tensor::{backend::Backend, Distribution, Element, Int, Tensor},
};
use burn_lm_llama::nn::{
    pos_encoding::PositionalEncodingState,
    transformer::{Transformer, TransformerCache, TransformerConfig},
};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

pub struct TransformerBenchmark<B: Backend> {
    seq_length: usize,
    batch_size: usize,
    config: Config,
    config_transformer: TransformerConfig,
    device: B::Device,
    transformer: Transformer<B>,
    pos_encoding: PositionalEncodingState<B>,
}

impl<B: Backend> Benchmark for TransformerBenchmark<B> {
    type Input = (Tensor<B, 2, Int>, TransformerCache<B>);
    type Output = Tensor<B, 3>;

    fn name(&self) -> String {
        format!(
            "transformer-{}-{:?}",
            self.config.name,
            B::FloatElem::dtype()
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.config.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Input) -> Self::Output {
        self.transformer
            .forward(input, &mut cache, &self.pos_encoding, None)
    }

    fn prepare(&self) -> Self::Input {
        let input = Tensor::<B, 2>::random(
            [self.batch_size, self.seq_length],
            Distribution::Uniform(0., 10000.0),
            &self.device,
        )
        .int();

        let cache = TransformerCache::new(&self.config_transformer, self.batch_size, &self.device);

        (input, cache)
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

struct Config {
    n_layers: usize,
    n_heads: usize,
    n_heads_kv: usize,
    d_model: usize,
    vocab_size: usize,
    hidden_size: usize,
    name: &'static str,
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let max_seq_length = 512;

    let mut results = Vec::new();

    for (batch_size, seq_length) in [(32, 1), (1, max_seq_length)] {
        // Layer of 1 for now.
        for config in [
            Config {
                vocab_size: 128256,
                n_heads: 32,
                n_heads_kv: 8,
                n_layers: 1,
                d_model: 2048,
                hidden_size: 8192,
                name: "llama-3.2-1B",
            },
            Config {
                vocab_size: 128256,
                n_heads: 24,
                n_heads_kv: 8,
                n_layers: 1,
                d_model: 3072,
                hidden_size: 8192,
                name: "llama-3.2-3B",
            },
            Config {
                vocab_size: 128256,
                n_heads: 32,
                n_heads_kv: 8,
                n_layers: 1,
                d_model: 4096,
                hidden_size: 14336,
                name: "llama-8B",
            },
        ] {
            let config_transformer = TransformerConfig::new(
                config.vocab_size,
                config.n_layers,
                config.d_model,
                config.hidden_size,
                config.n_heads,
                config.n_heads_kv,
            );
            let transformer = config_transformer.init(device);
            let rope =
                RotaryEncodingConfig::new(max_seq_length * 2, config.d_model / config.n_heads)
                    .init(device);
            let benchmark = TransformerBenchmark::<B> {
                batch_size,
                seq_length,
                config,
                config_transformer,
                device: device.clone(),
                transformer,
                pos_encoding: PositionalEncodingState::new(rope),
            };
            let result = run_benchmark(benchmark);
            results.push(result);
        }
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}

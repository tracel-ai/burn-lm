use burn::{
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{backend::Backend, Distribution, Element, Int, Tensor},
};
use burn_common::benchmark::{run_benchmark, Benchmark, BenchmarkResult};
use burnlm_llama::transformer::{KeyValueCache, Transformer, TransformerConfig};

pub struct TransformerBenchmark<B: Backend> {
    seq_length: usize,
    batch_size: usize,
    config: Config,
    device: B::Device,
    transformer: Transformer<B>,
    rope: RotaryEncoding<B>,
}

impl<B: Backend> Benchmark for TransformerBenchmark<B> {
    type Args = (Tensor<B, 2, Int>, Vec<KeyValueCache<B>>);

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

    fn execute(&self, (input, mut caches): Self::Args) {
        self.transformer.forward(input, &mut caches, &self.rope);
    }

    fn prepare(&self) -> Self::Args {
        let input = Tensor::<B, 2>::random(
            [self.batch_size, self.seq_length],
            Distribution::Uniform(0., 10000.0),
            &self.device,
        )
        .int();

        let caches = (0..self.config.n_layers)
            .map(|_| {
                KeyValueCache::new(
                    self.batch_size,
                    self.config.n_heads,
                    self.seq_length,
                    self.config.d_model,
                    &self.device,
                )
            })
            .collect();

        (input, caches)
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
    let batch_size = 1;
    let seq_length = 512;

    let mut results = Vec::new();

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
        let transformer = TransformerConfig::new(
            config.vocab_size,
            config.n_layers,
            config.d_model,
            config.hidden_size,
            config.n_heads,
            config.n_heads_kv,
        )
        .init(device);
        let rope =
            RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads).init(device);
        let benchmark = TransformerBenchmark::<B> {
            batch_size,
            seq_length,
            config,
            device: device.clone(),
            transformer,
            rope,
        };
        let result = run_benchmark(benchmark);
        results.push(result);
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}

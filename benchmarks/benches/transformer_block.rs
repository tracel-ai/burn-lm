use burn::{
    nn::RotaryEncodingConfig,
    tensor::{backend::Backend, Distribution, Element, Tensor},
};
use burn_common::benchmark::{run_benchmark, Benchmark, BenchmarkResult};
use burnlm_llama::{
    nn::{
        attention::KeyValueCache,
        transformer::{TransformerBlock, TransformerBlockConfig},
    },
    PositionalEncodingState,
};

pub struct TransformerBlockBenchmark<B: Backend> {
    seq_length: usize,
    batch_size: usize,
    config: Config,
    device: B::Device,
    block: TransformerBlock<B>,
    rope: PositionalEncodingState<B>,
}

impl<B: Backend> Benchmark for TransformerBlockBenchmark<B> {
    type Args = (Tensor<B, 3>, KeyValueCache<B>);

    fn name(&self) -> String {
        format!(
            "transformer-block-{}-{:?}",
            self.config.name,
            B::FloatElem::dtype()
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.config.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Args) {
        self.block.forward(input, &mut cache, &self.rope, None);
    }

    fn prepare(&self) -> Self::Args {
        let input = Tensor::<B, 3>::random(
            [self.batch_size, self.seq_length, self.config.d_model],
            Distribution::Default,
            &self.device,
        );
        let cache = KeyValueCache::new(
            self.batch_size,
            self.config.n_heads,
            self.seq_length,
            self.config.d_model,
            &self.device,
        );

        (input, cache)
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

struct Config {
    n_heads: usize,
    n_heads_kv: usize,
    d_model: usize,
    hidden_size: usize,
    name: &'static str,
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let batch_size = 1;
    let n_layers = 1;
    let seq_length = 512;
    let norm_eps = 1e-5;

    let mut results = Vec::new();

    for config in [
        Config {
            n_heads: 32,
            n_heads_kv: 8,
            d_model: 2048,
            hidden_size: 8192,
            name: "llama-3.2-1B",
        },
        Config {
            n_heads: 24,
            n_heads_kv: 8,
            d_model: 3072,
            hidden_size: 8192,
            name: "llama-3.2-3B",
        },
        Config {
            n_heads: 32,
            n_heads_kv: 8,
            d_model: 4096,
            hidden_size: 14336,
            name: "llama-8B",
        },
    ] {
        let block = TransformerBlockConfig::new(
            n_layers,
            config.d_model,
            config.hidden_size,
            config.n_heads,
            config.n_heads_kv,
            norm_eps,
        )
        .init(device);
        let rope =
            RotaryEncodingConfig::new(seq_length * 2, config.d_model / config.n_heads).init(device);
        let benchmark = TransformerBlockBenchmark::<B> {
            batch_size,
            seq_length,
            config,
            device: device.clone(),
            block,
            rope: PositionalEncodingState::new(rope),
        };
        let result = run_benchmark(benchmark);
        results.push(result);
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}

use burn::{
    nn::RotaryEncodingConfig,
    tensor::{DType, Device, Distribution, Tensor},
};
use burn_lm_llama::nn::{
    attention::KeyValueCache,
    pos_encoding::PositionalEncodingState,
    transformer::{TransformerBlock, TransformerBlockConfig},
};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

pub struct TransformerBlockBenchmark {
    seq_length: usize,
    batch_size: usize,
    config: Config,
    device: Device,
    block: TransformerBlock,
    pos_encoding: PositionalEncodingState,
    dtype: DType,
}

impl Benchmark for TransformerBlockBenchmark {
    type Input = (Tensor<3>, KeyValueCache);
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!("transformer-block-{}-{:?}", self.config.name, self.dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.config.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Input) -> Self::Output {
        self.block
            .forward(input, &mut cache, &self.pos_encoding, None)
    }

    fn prepare(&self) -> Self::Input {
        let input = Tensor::<3>::random(
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
        self.device.sync().unwrap();
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
fn bench(device: &Device, dtype: DType) -> Vec<BenchmarkResult> {
    let n_layers = 1;
    let max_seq_length = 512;
    let norm_eps = 1e-5;

    let mut results = Vec::new();

    for (batch_size, seq_length) in [(32, 1), (32, 1), (1, max_seq_length)] {
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
                RotaryEncodingConfig::new(max_seq_length * 2, config.d_model / config.n_heads)
                    .init(device);
            let benchmark = TransformerBlockBenchmark {
                batch_size,
                seq_length,
                config,
                device: device.clone(),
                block,
                pos_encoding: PositionalEncodingState::new(rope),
                dtype,
            };
            let result = run_benchmark(benchmark);
            results.push(result);
        }
    }

    results
}

fn main() {
    // needs and update to burnbench
    // burnbench::bench_on_backend!();
    todo!()
}

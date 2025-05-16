use burn::{
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Distribution, Element, Int, Tensor},
};
use burn_common::benchmark::{run_benchmark, Benchmark, BenchmarkResult};

pub struct EmbeddingBenchmark<B: Backend> {
    seq_length: usize,
    batch_size: usize,
    config: Config,
    device: B::Device,
    embedding: Embedding<B>,
}

impl<B: Backend> Benchmark for EmbeddingBenchmark<B> {
    type Args = Tensor<B, 2, Int>;

    fn name(&self) -> String {
        format!("embedding-{}-{:?}", self.config.name, B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.config.d_model]]
    }

    fn execute(&self, input: Self::Args) {
        self.embedding.forward(input);
    }

    fn prepare(&self) -> Self::Args {
        let input = Tensor::<B, 2>::random(
            [self.batch_size, self.seq_length],
            Distribution::Uniform(0., 10000.0),
            &self.device,
        );
        input.int()
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

struct Config {
    d_model: usize,
    vocab_size: usize,
    name: &'static str,
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let batch_size = 1;
    let seq_length = 512;

    let mut results = Vec::new();

    for config in [
        Config {
            d_model: 2048,
            vocab_size: 128256,
            name: "llama-3.2-1B",
        },
        Config {
            d_model: 5632,
            vocab_size: 32000,
            name: "tinyllama-1.1",
        },
    ] {
        let embedding = EmbeddingConfig::new(config.vocab_size, config.d_model).init(device);
        let benchmark = EmbeddingBenchmark::<B> {
            batch_size,
            seq_length,
            config,
            device: device.clone(),
            embedding,
        };
        let result = run_benchmark(benchmark);
        results.push(result);
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}

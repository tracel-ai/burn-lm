use burn::{
    nn::{Embedding, EmbeddingConfig},
    tensor::{DType, Device, Distribution, Int, Tensor},
};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

pub struct EmbeddingBenchmark {
    seq_length: usize,
    batch_size: usize,
    config: Config,
    device: Device,
    embedding: Embedding,
    dtype: DType,
}

impl Benchmark for EmbeddingBenchmark {
    type Input = Tensor<2, Int>;
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!("embedding-{}-{:?}", self.config.name, self.dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.config.d_model]]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.embedding.forward(input)
    }

    fn prepare(&self) -> Self::Input {
        let input = Tensor::<2>::random(
            [self.batch_size, self.seq_length],
            Distribution::Uniform(0., 10000.0),
            &self.device,
        );
        input.int()
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

struct Config {
    d_model: usize,
    vocab_size: usize,
    name: &'static str,
}

#[allow(dead_code)]
fn bench(device: &Device, dtype: DType) -> Vec<BenchmarkResult> {
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
        let benchmark = EmbeddingBenchmark {
            batch_size,
            seq_length,
            config,
            device: device.clone(),
            embedding,
            dtype,
        };
        let result = run_benchmark(benchmark);
        results.push(result);
    }

    results
}

fn main() {
    // needs and update to burnbench
    // burnbench::bench_on_backend!();
    todo!()
}

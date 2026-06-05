use burn::{
    nn::RotaryEncodingConfig,
    tensor::{DType, Device, Distribution, Tensor},
};
use burn_lm_llama::nn::{
    attention::{KeyValueCache, MultiHeadAttention, MultiHeadAttentionConfig},
    pos_encoding::PositionalEncodingState,
};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

pub struct AttentionBenchmark {
    seq_length: usize,
    batch_size: usize,
    d_model: usize,
    n_heads: usize,
    device: Device,
    attn: MultiHeadAttention,
    rope: PositionalEncodingState,
    dtype: DType,
}

impl Benchmark for AttentionBenchmark {
    type Input = (Tensor<3>, KeyValueCache);
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!("llama-attention-{:?}", self.dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Input) -> Self::Output {
        self.attn.forward_cache(input, &mut cache, &self.rope, None)
    }

    fn prepare(&self) -> Self::Input {
        let input = Tensor::<3>::random(
            [self.batch_size, self.seq_length, self.d_model],
            Distribution::Default,
            &self.device,
        );
        let cache = KeyValueCache::new(
            self.batch_size,
            self.n_heads,
            self.seq_length,
            self.d_model,
            &self.device,
        );

        (input, cache)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device, dtype: DType) -> Vec<BenchmarkResult> {
    let n_heads = 32;

    let max_seq_length = 512;
    let d_model = 4096;

    let mut results = Vec::new();

    for (batch_size, seq_length) in [(32, 1), (1, max_seq_length)] {
        let attn = MultiHeadAttentionConfig::new(d_model, n_heads, n_heads).init(device);
        let rope = RotaryEncodingConfig::new(max_seq_length * 2, d_model / n_heads).init(device);
        let benchmark = AttentionBenchmark {
            batch_size,
            n_heads,
            seq_length,
            d_model,
            device: device.clone(),
            attn,
            rope: PositionalEncodingState::new(rope),
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

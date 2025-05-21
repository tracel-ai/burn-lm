use burn::{
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{backend::Backend, Distribution, Element, Tensor},
};
use burn_common::benchmark::{run_benchmark, Benchmark, BenchmarkResult};
use burnlm_llama::nn::attention::{KeyValueCache, MultiHeadAttention, MultiHeadAttentionConfig};

pub struct AttentionBenchmark<B: Backend> {
    seq_length: usize,
    batch_size: usize,
    d_model: usize,
    n_heads: usize,
    device: B::Device,
    attn: MultiHeadAttention<B>,
    rope: RotaryEncoding<B>,
}

impl<B: Backend> Benchmark for AttentionBenchmark<B> {
    type Args = (Tensor<B, 3>, KeyValueCache<B>);

    fn name(&self) -> String {
        format!("llama-attention-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, self.seq_length, self.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Args) {
        self.attn.forward_cache(input, &mut cache, &self.rope, None);
    }

    fn prepare(&self) -> Self::Args {
        let input = Tensor::<B, 3>::random(
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
        B::sync(&self.device);
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let batch_size = 1;
    let n_heads = 32;

    let seq_length = 512;
    let d_model = 4096;

    let attn = MultiHeadAttentionConfig::new(d_model, n_heads, n_heads).init(device);
    let rope = RotaryEncodingConfig::new(seq_length * 2, d_model / n_heads).init(device);
    let benchmark = AttentionBenchmark::<B> {
        batch_size,
        n_heads,
        seq_length,
        d_model,
        device: device.clone(),
        attn,
        rope,
    };

    vec![run_benchmark(benchmark)]
}

fn main() {
    burnbench::bench_on_backend!();
}

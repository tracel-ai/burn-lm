use burn::{
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{DType, Device, Distribution, Int, Tensor},
};
use burn_lm_llama::nn::attention::{LanePlan, PagedKvCache};
use burn_lm_llama::nn::transformer::{Transformer, TransformerConfig};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

// Lane-aware whole-transformer decode benchmark, swept over the lane count (batch) 1, 2, 4, 8. Each
// run times one `Transformer::forward_lanes` (token ids in, logits out) over a slab prefilled to a
// fixed prompt length, every active lane contributing one new token. See `attention.rs` for the
// shared shape of these benches.

/// Prompt length each lane is prefilled to before the timed decode step.
const PROMPT_LEN: usize = 64;

pub struct TransformerBenchmark {
    batch_size: usize,
    config: Config,
    device: Device,
    transformer: Transformer,
    rope: RotaryEncoding,
    cache: PagedKvCache,
    plan: LanePlan,
    dtype: DType,
}

impl Benchmark for TransformerBenchmark {
    type Input = (Tensor<2, Int>, PagedKvCache);
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!(
            "transformer-lanes-{}-{}-{:?}",
            self.batch_size, self.config.name, self.dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, 1, self.config.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Input) -> Self::Output {
        self.transformer
            .forward_lanes(input, &mut cache, &self.rope, &self.plan)
    }

    fn prepare(&self) -> Self::Input {
        // One new token id per lane: an `[n, 1]` decode input. The cache is cloned so each sample
        // starts from the prefilled slab.
        let input = Tensor::<2>::random(
            [self.batch_size, 1],
            Distribution::Uniform(0., self.config.vocab_size as f64),
            &self.device,
        )
        .int();
        (input, self.cache.clone())
    }

    fn sync(&self) {
        self.device.sync().unwrap();
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
fn bench(device: &Device, dtype: DType) -> Vec<BenchmarkResult> {
    let max_seq_length = 512;

    let mut results = Vec::new();

    for batch_size in [1usize, 2, 4, 8] {
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
            )
            .with_max_seq_len(max_seq_length);
            let transformer = config_transformer.init(device);
            let rope =
                RotaryEncodingConfig::new(max_seq_length * 2, config.d_model / config.n_heads)
                    .init(device);

            // Prefill every lane to PROMPT_LEN through the real lane forward so the cache holds true
            // KV, then build the decode-round plan (one new token per lane).
            let mut cache = PagedKvCache::with_default_blocks(config_transformer.kv_layout(), batch_size, device);
            for lane in 0..batch_size {
                let prefill_plan = cache.prepare_lanes(&[lane], PROMPT_LEN).unwrap();
                let prompt = Tensor::<2>::random(
                    [1, PROMPT_LEN],
                    Distribution::Uniform(0., config.vocab_size as f64),
                    device,
                )
                .int();
                transformer.forward_lanes(prompt, &mut cache, &rope, &prefill_plan);
            }
            let lanes: Vec<usize> = (0..batch_size).collect();
            let plan = cache.prepare_lanes(&lanes, 1).unwrap();

            let benchmark = TransformerBenchmark {
                batch_size,
                config,
                device: device.clone(),
                transformer,
                rope,
                cache,
                plan,
                dtype,
            };
            results.push(run_benchmark(benchmark));
        }
    }

    results
}

fn main() {
    let device = Device::default();
    for result in bench(&device, DType::F32) {
        println!("{}: mean {:?}, median {:?}", result.name, result.computed.mean, result.computed.median);
    }
}

use burn::{
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{DType, Device, Distribution, Tensor},
};
use burn_lm_llama::nn::{
    attention::KeyValueCache,
    transformer::{LanePlan, TransformerBlock, TransformerBlockConfig, TransformerCache, TransformerConfig},
};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

// Lane-aware single-block decode benchmark, swept over the lane count (batch) 1, 2, 4, 8. Each run
// times one `TransformerBlock::forward_lanes` over a slab prefilled to a fixed prompt length, every
// active lane contributing one new token. See `attention.rs` for the shared shape of these benches.

/// Prompt length each lane is prefilled to before the timed decode step.
const PROMPT_LEN: usize = 64;

pub struct TransformerBlockBenchmark {
    batch_size: usize,
    config: Config,
    device: Device,
    block: TransformerBlock,
    rope: RotaryEncoding,
    cache: KeyValueCache,
    plan: LanePlan,
    dtype: DType,
}

impl Benchmark for TransformerBlockBenchmark {
    type Input = (Tensor<3>, KeyValueCache);
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!(
            "transformer-block-lanes-{}-{}-{:?}",
            self.batch_size, self.config.name, self.dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, 1, self.config.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Input) -> Self::Output {
        self.block
            .forward_lanes(input, &mut cache, &self.rope, &self.plan)
    }

    fn prepare(&self) -> Self::Input {
        let input = Tensor::<3>::random(
            [self.batch_size, 1, self.config.d_model],
            Distribution::Default,
            &self.device,
        );
        (input, self.cache.clone())
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

    for batch_size in [1usize, 2, 4, 8] {
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
            let head_dim = config.d_model / config.n_heads;
            let block = TransformerBlockConfig::new(
                n_layers,
                config.d_model,
                config.hidden_size,
                config.n_heads,
                config.n_heads_kv,
                norm_eps,
            )
            .init(device);
            let rope = RotaryEncodingConfig::new(max_seq_length * 2, head_dim).init(device);

            // A single-layer transformer cache supplies the lane-length bookkeeping for the plan.
            let tcfg = TransformerConfig::new(
                128256,
                1,
                config.d_model,
                config.hidden_size,
                config.n_heads,
                config.n_heads_kv,
            )
            .with_max_seq_len(max_seq_length);
            let mut tcache = TransformerCache::new(&tcfg, batch_size, device);
            // Keep each lane's prefill plan: its block table is where the standalone KV cache
            // below must seed, so the seeding lands in the same blocks the decode plan addresses.
            let mut tables: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
            for lane in 0..batch_size {
                let plan = tcache.prepare_lanes(&[lane], PROMPT_LEN).unwrap();
                tables.push(plan.tables[0].clone());
            }

            // Seed the block's own KV buffer to the same prefilled state.
            let mut cache = KeyValueCache::new(
                batch_size + 1,
                config.n_heads_kv,
                max_seq_length,
                head_dim,
                device,
            );
            let lanes: Vec<usize> = (0..batch_size).collect();
            let prompt_kv = Tensor::<4>::random(
                [batch_size, config.n_heads_kv, PROMPT_LEN, head_dim],
                Distribution::Default,
                device,
            );
            cache.write_lanes(
                &tables,
                &vec![0usize; batch_size],
                prompt_kv.clone(),
                prompt_kv,
            );

            let plan = tcache.prepare_lanes(&lanes, 1).unwrap();

            let benchmark = TransformerBlockBenchmark {
                batch_size,
                config,
                device: device.clone(),
                block,
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

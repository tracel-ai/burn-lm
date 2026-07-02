use burn::{
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{DType, Device, Distribution, Tensor},
};
use burn_lm_llama::nn::{
    attention::{KeyValueCache, MultiHeadAttention, MultiHeadAttentionConfig},
    transformer::{LanePlan, TransformerCache, TransformerConfig},
};
use burnbench::{run_benchmark, Benchmark, BenchmarkResult};

// These three nn benches measure the production decode path: the lane-aware forward swept over the
// lane count (batch) 1, 2, 4, 8. Each run times one decode step — every active lane contributes one
// new token, an `[n, 1]` forward — over a slab whose lanes were prefilled to a fixed prompt length
// so the step attends over real KV. The per-module batch-scaling numbers decompose the end-to-end
// throughput speedup the batched decoder reports.

/// Prompt length each lane is prefilled to before the timed decode step.
const PROMPT_LEN: usize = 64;

pub struct AttentionBenchmark {
    batch_size: usize,
    d_model: usize,
    device: Device,
    attn: MultiHeadAttention,
    rope: RotaryEncoding,
    // The KV slab and the decode plan are built once at construction: the slab is prefilled to
    // `PROMPT_LEN` on every lane and the plan describes the one-token-per-lane decode round. The
    // timed `execute` writes the same KV slots on every sample, which is consistent across runs.
    cache: KeyValueCache,
    plan: LanePlan,
    dtype: DType,
}

impl Benchmark for AttentionBenchmark {
    type Input = (Tensor<3>, KeyValueCache);
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!("llama-attention-lanes-{}-{:?}", self.batch_size, self.dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.batch_size, 1, self.d_model]]
    }

    fn execute(&self, (input, mut cache): Self::Input) -> Self::Output {
        self.attn
            .forward_cache_lanes(input, &mut cache, &self.rope, &self.plan)
    }

    fn prepare(&self) -> Self::Input {
        // One new token per lane: an `[n, 1, d_model]` decode input. The cache is cloned so each
        // sample starts from the prefilled slab.
        let input = Tensor::<3>::random(
            [self.batch_size, 1, self.d_model],
            Distribution::Default,
            &self.device,
        );
        (input, self.cache.clone())
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device, dtype: DType) -> Vec<BenchmarkResult> {
    let n_heads = 32;
    let n_kv_heads = 8;
    let max_seq_length = 512;
    let d_model = 4096;
    let head_dim = d_model / n_heads;

    let mut results = Vec::new();

    for batch_size in [1usize, 2, 4, 8] {
        let attn = MultiHeadAttentionConfig::new(d_model, n_heads, n_kv_heads).init(device);
        let rope = RotaryEncodingConfig::new(max_seq_length * 2, head_dim).init(device);

        // A single-layer transformer cache gives us one lane-aware KV buffer plus the lane-length
        // bookkeeping `prepare_lanes` needs to build the decode plan.
        let cfg = TransformerConfig::new(128256, 1, d_model, 14336, n_heads, n_kv_heads)
            .with_max_seq_len(max_seq_length);
        let mut tcache = TransformerCache::new(&cfg, batch_size, device);

        // Prefill every lane to PROMPT_LEN so the decode step attends over real history.
        // Keep each lane's prefill plan: its block table is where the standalone KV cache below
        // must seed, so the seeding lands in the same blocks the decode plan will address.
        let mut tables: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        for lane in 0..batch_size {
            let plan = tcache.prepare_lanes(&[lane], PROMPT_LEN).unwrap();
            tables.push(plan.tables[0].clone());
        }
        let mut cache =
            KeyValueCache::new(
            batch_size + 1, n_kv_heads, max_seq_length, head_dim, device);
        let lanes: Vec<usize> = (0..batch_size).collect();
        let prompt_kv = Tensor::<4>::random(
            [batch_size, n_kv_heads, PROMPT_LEN, head_dim],
            Distribution::Default,
            device,
        );
        // Seed every lane's KV from offset 0, mirroring the TransformerCache prefill above so the
        // two stay in lockstep: the plan's per-lane starts (PROMPT_LEN) are exactly where the next
        // token writes.
        cache.write_lanes(
            &tables,
            &vec![0usize; batch_size],
            prompt_kv.clone(),
            prompt_kv,
        );

        // The decode-round plan: one new token per lane, each lane starting at PROMPT_LEN.
        let plan = tcache.prepare_lanes(&lanes, 1).unwrap();

        let benchmark = AttentionBenchmark {
            batch_size,
            d_model,
            device: device.clone(),
            attn,
            rope,
            cache,
            plan,
            dtype,
        };
        results.push(run_benchmark(benchmark));
    }

    results
}

fn main() {
    let device = Device::default();
    for result in bench(&device, DType::F32) {
        println!("{}: mean {:?}, median {:?}", result.name, result.computed.mean, result.computed.median);
    }
}

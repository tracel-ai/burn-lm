#![recursion_limit = "256"]

pub mod tokenizer;

/// Neural network components.
pub mod nn;

/// Text generation components.
pub mod generation;

#[cfg(feature = "inference-server")]
pub mod server;

pub use nn::llama::*;

#[cfg(test)]
mod tests {
    use burn::module::{Module, ModuleMapper, Param};
    use burn::tensor::{Device, Shape, Tensor, TensorData};

    // Burn 0.22 stores the backend behind `Device`, so test tensors no longer
    // carry a backend type parameter.
    pub type TestTensor<const D: usize> = burn::tensor::Tensor<D>;

    fn fold_path_hash(mut hash: u64, segment: &str) -> u64 {
        for byte in segment.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    // Test-only replacement for the removed Burn `Reinitializer` helper.
    //
    // A naive replacement using `NdArray::seed(0)` + `Initializer::Uniform`
    // made golden values change across runs: NdArray's RNG state is global, so
    // parallel tests can interleave random draws and perturb each other's fake
    // weights. Keep this generator local and path-seeded instead.
    fn deterministic_uniform<const D: usize>(
        shape: impl Into<Shape>,
        min: f64,
        max: f64,
        device: &Device,
        seed: u64,
    ) -> Tensor<D> {
        let shape = shape.into();
        let n = shape.num_elements();
        let mut state = seed;
        let span = max - min;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 11) as f64 / ((1u64 << 53) - 1) as f64;
                (min + unit * span) as f32
            })
            .collect();
        Tensor::from_data(TensorData::new(data, shape), device)
    }

    struct UniformReinit {
        min: f64,
        max: f64,
        path_hashes: [u64; 64],
        depth: usize,
    }

    impl ModuleMapper for UniformReinit {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            // Seed each parameter from its traversal path so fake test weights
            // are stable regardless of global RNG state or test scheduling.
            let parent = if self.depth == 0 {
                0
            } else {
                self.path_hashes[self.depth - 1]
            };
            self.path_hashes[self.depth] = fold_path_hash(parent, name);
            self.depth += 1;
        }

        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            if self.depth > 0 {
                self.depth -= 1;
            }
        }

        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let seed = if self.depth == 0 {
                0
            } else {
                self.path_hashes[self.depth - 1]
            };
            param.map(|tensor| {
                let shape = tensor.dims();
                let device = tensor.device();
                deterministic_uniform(shape, self.min, self.max, &device, seed)
            })
        }
    }

    pub fn reinit_uniform<M: Module>(module: M, min: f64, max: f64) -> M {
        // Keep golden tests reproducible without depending on backend-global RNG
        // state or backend-specific initialization behavior.
        module.map(&mut UniformReinit {
            min,
            max,
            path_hashes: [0; 64],
            depth: 0,
        })
    }

    #[allow(dead_code)]
    pub fn dump_golden_f32(name: &str, data: &burn::tensor::TensorData) {
        let values: Vec<f32> = data
            .clone()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 tensor data");
        eprintln!("// {name}\n{values:#?}");
    }
}

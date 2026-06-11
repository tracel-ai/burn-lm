use burn::{
    config::Config,
    nn::RotaryEncoding,
    tensor::{Int, Tensor, TensorData},
};

/// Tracks the state of rotary positional encodings during autoregressive inference.
///
/// Manages shifting of precomputed frequency tables when the sequence length exceeds
/// the initially allocated range. Used to avoid recomputing RoPE values on-the-fly
/// while maintaining correct positional alignment across decoding steps.
#[derive(Debug, Clone)]
pub struct PositionalEncodingState {
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding,
    /// RoPE maximum sequence length.
    pub max_seq_len: usize,
    /// The next position.
    pub next_position: usize,
    /// The current sequence length.
    pub curr_seq_len: usize,
    /// The index start offset.
    pub start_offset: usize,
}

impl PositionalEncodingState {
    pub fn new(rope: RotaryEncoding) -> Self {
        // Initial max position corresponds to the RoPE max seq len on initialization
        let max_seq_len = rope.freq_complex.dims()[0];
        Self {
            rope,
            max_seq_len,
            next_position: 0,
            curr_seq_len: 0,
            start_offset: 0,
        }
    }

    pub fn prepare(&mut self, seq_len: usize) {
        self.curr_seq_len = seq_len;
        self.next_position += seq_len;
        if self.next_position > self.max_seq_len + self.start_offset {
            let start = self.position();
            self.rope.shift(start);
            self.start_offset = start;
        }
    }

    pub fn reset(&mut self) {
        // A counter-only reset is valid until the RoPE table has shifted. After a shift,
        // table index 0 represents `start_offset` instead of absolute position 0, so
        // stateless generation must restore the original position window.
        self.rope.reset();

        self.next_position = 0;
        self.curr_seq_len = 0;
        self.start_offset = 0;
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.rope.forward(x)
    }

    pub fn apply<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.rope.apply(x, self.index())
    }

    /// Returns the absolute sequence position since the beginning,
    /// regardless of shifting.
    pub fn position(&self) -> usize {
        // The absolute sequence position should not include the current sequence
        // input so we subtract to get the current generation position.
        self.next_position - self.curr_seq_len
    }

    /// Returns the next index position for the pre-computed frequencies.
    pub fn index(&self) -> usize {
        let mut index = self.position();
        if self.start_offset > 0 {
            index -= self.start_offset
        }
        index
    }
}

/// Apply RoPE rotations for `n` lanes sitting at divergent `starts` positions.
///
/// Row `r` of lane `j` is rotated at absolute position `starts[j] + r`: the
/// per-(lane, row) frequency rows are gathered from the precomputed
/// `freq_complex` table in one `select`, then every lane is rotated with the
/// same batched ops `RotaryEncoding::apply` uses internally.
///
/// Positions are ABSOLUTE table indices: lane mode never shifts the table
/// (per-lane lengths are bounded by `max_seq_len`, well inside the
/// precomputed window).
///
/// # Shapes
///
/// - x: `[n_lanes, heads, seq_len, head_dim]`
/// - output: same.
// The decoder switch-over onto the lane-aware path lands in the next change.
#[allow(dead_code)]
pub(crate) fn apply_rope_lanes(rope: &RotaryEncoding, x: Tensor<4>, starts: &[usize]) -> Tensor<4> {
    let [n, heads, q, head_dim] = x.dims();
    debug_assert_eq!(n, starts.len());
    // Every gathered index must fall inside the precomputed rotation table. Callers guarantee
    // this through the per-lane capacity check in `prepare_lanes` (a full lane FINISHES instead
    // of sliding the window — a slide re-bases the table globally, which cannot coexist with
    // lanes at different positions). This assert keeps the invariant from silently depending on
    // every future call site remembering that pairing.
    let table_rows = rope.freq_complex.dims()[0];
    debug_assert!(
        starts.iter().all(|s| s + q <= table_rows),
        "lane position past the rotation table: need row {}, table has {table_rows}",
        starts.iter().map(|s| s + q).max().unwrap_or(0),
    );
    let device = x.device();

    let mut idx = Vec::with_capacity(n * q);
    for s in starts.iter() {
        for r in 0..q {
            idx.push((s + r) as i64);
        }
    }
    let idx = Tensor::<1, Int>::from_data(TensorData::new(idx, [n * q]), &device);
    let freqs = rope
        .freq_complex
        .clone()
        .select(0, idx) // [n * q, head_dim, 2]
        .reshape([n, 1, q, head_dim, 2])
        .expand([n, heads, q, head_dim, 2])
        .reshape([n * heads, q, head_dim, 2]);

    // 2D rotation matrix [[cos, -sin], [sin, cos]] expansion.
    let sign = Tensor::<2>::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], &device);

    let out = x
        .reshape([n * heads, q, head_dim / 2, 2])
        .matmul(sign.unsqueeze::<4>())
        .reshape([n * heads, q, head_dim, 2])
        * freqs;

    out.sum_dim(3).reshape([n, heads, q, head_dim])
}

/// Correctness-floor fallback for [`apply_rope_lanes`]: loop
/// `rope.apply(x_lane, pos)` over per-lane slices and `cat`. Kept test-only
/// to cross-validate the gather path against the production single-lane op.
#[cfg(test)]
pub(crate) fn apply_rope_lanes_looped(
    rope: &RotaryEncoding,
    x: Tensor<4>,
    starts: &[usize],
) -> Tensor<4> {
    let [n, heads, q, head_dim] = x.dims();
    let lanes = (0..n)
        .map(|j| {
            rope.apply(
                x.clone().slice([j..j + 1, 0..heads, 0..q, 0..head_dim]),
                starts[j],
            )
        })
        .collect::<Vec<_>>();
    Tensor::cat(lanes, 0)
}

/// Rotary positional encoding (RoPE)
#[derive(Config, Debug)]
pub struct RopeConfig {
    pub theta: f32,
    #[config(default = "None")]
    pub scaled: Option<RopeFrequencyScaling>,
}

/// RoPE frequency scaling.
#[derive(Config, Debug)]
pub struct RopeFrequencyScaling {
    #[config(default = "8.")]
    pub scale_factor: f32,
    #[config(default = "1.")]
    pub low_freq_factor: f32,
    #[config(default = "4.")]
    pub high_freq_factor: f32,
    #[config(default = "8192.")]
    pub old_context_len: f32,
}

impl RopeFrequencyScaling {
    /// Applies frequency scaling by parts following Llama 3.1's scheme.
    ///
    /// Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45
    pub fn freq_scaling_by_parts(&self, freqs: Tensor<1>) -> Tensor<1> {
        let low_freq_wavelen = self.old_context_len / self.low_freq_factor;
        let high_freq_wavelen = self.old_context_len / self.high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(self.old_context_len)
            .sub_scalar(self.low_freq_factor)
            .div_scalar(self.high_freq_factor - self.low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(self.scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(self.scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);

        new_freqs.mask_where(cond, freqs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    use burn::{
        nn::RotaryEncodingConfig,
        tensor::{Device, TensorData, Tolerance},
    };

    #[test]
    fn test_rope() {
        let device = Default::default();

        let max_seq_len = 16;
        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len * 2, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling(freq_scaling_fn, &device);

        let input = TestTensor::<4>::from([[
            [[-0.60253906, -0.035308838], [0.41357422, 0.15100098]],
            [[-0.044677734, -0.094177246], [0.60546875, 0.2442627]],
        ]]);

        let output = rope.apply(input, 0);
        let expected = TensorData::from([[
            [[-0.60253906, -0.035308838], [0.09643555, 0.42944336]],
            [[-0.044677734, -0.094177246], [0.12194824, 0.64160156]],
        ]]);

        output
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }

    /// The gather RoPE path equals both the per-lane-loop fallback and the
    /// production `RotaryEncoding::apply` at every lane's position.
    #[test]
    fn test_rope_lanes_gather_matches_loop_and_production_apply() {
        let device: Device = Default::default();
        let rope = RotaryEncodingConfig::new(64, 4).init(&device);

        // 3 lanes at divergent positions, q = 2.
        let starts = [37usize, 5, 0];
        let x = TestTensor::<4>::random(
            [3, 2, 2, 4],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let gathered = apply_rope_lanes(&rope, x.clone(), &starts);
        let looped = apply_rope_lanes_looped(&rope, x.clone(), &starts);
        gathered
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&looped.into_data(), Tolerance::rel_abs(1e-5, 1e-6));

        for (j, &start) in starts.iter().enumerate() {
            let lane = x.clone().slice([j..j + 1, 0..2, 0..2, 0..4]);
            let expected = rope.apply(lane, start);
            gathered
                .clone()
                .slice([j..j + 1, 0..2, 0..2, 0..4])
                .into_data()
                .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-6));
        }
    }

    #[test]
    fn test_rope_shift() {
        let device: Device = Default::default();

        let max_seq_len = 16;
        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling(freq_scaling_fn, &device);

        let mut pos_encoding = PositionalEncodingState::new(rope);
        assert_eq!(pos_encoding.max_seq_len, max_seq_len);

        // Input prompt
        pos_encoding.prepare(14);
        assert_eq!(pos_encoding.position(), 0);
        assert_eq!(pos_encoding.index(), 0);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 14);
        assert_eq!(pos_encoding.index(), 14);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 15);
        assert_eq!(pos_encoding.index(), 15);

        // Next prompt
        pos_encoding.prepare(8); // should apply shift
        assert_eq!(pos_encoding.position(), 16);
        assert_eq!(pos_encoding.index(), 0);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 24);
        assert_eq!(pos_encoding.index(), 8);

        // Next prompt
        pos_encoding.prepare(8);
        assert_eq!(pos_encoding.position(), 25);
        assert_eq!(pos_encoding.index(), 0);

        // Next token
        pos_encoding.prepare(1);
        assert_eq!(pos_encoding.position(), 33);
        assert_eq!(pos_encoding.index(), 8);
    }

    #[test]
    fn test_rope_reset_after_shift() {
        let device: Device = Default::default();

        let max_seq_len = 16;
        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling(freq_scaling_fn, &device);

        let mut pos_encoding = PositionalEncodingState::new(rope);
        let input = TestTensor::<4>::from([[
            [[-0.60253906, -0.035308838], [0.41357422, 0.15100098]],
            [[-0.044677734, -0.094177246], [0.60546875, 0.2442627]],
        ]]);
        let expected = pos_encoding.apply(input.clone()).into_data();

        // Drive the state far enough to shift the cached RoPE frequencies.
        pos_encoding.prepare(14);
        pos_encoding.prepare(1);
        pos_encoding.prepare(1);
        pos_encoding.prepare(8);
        assert_eq!(pos_encoding.start_offset, 16);

        pos_encoding.reset();
        assert_eq!(pos_encoding.position(), 0);
        assert_eq!(pos_encoding.index(), 0);
        assert_eq!(pos_encoding.start_offset, 0);

        pos_encoding
            .apply(input.clone())
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));

        // Shift and reset a second time to ensure the saved original window wasn't
        // aliased and mutated by the first shift.
        pos_encoding.prepare(14);
        pos_encoding.prepare(1);
        pos_encoding.prepare(1);
        pos_encoding.prepare(8);
        assert_eq!(pos_encoding.start_offset, 16);

        pos_encoding.reset();
        pos_encoding
            .apply(input.clone())
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(0.05));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum, strum::Display, strum::EnumIter)]
pub(crate) enum BackendValues {
    // candle ----------------------------------------------------------------
    #[strum(to_string = "candle-accelerate")]
    CandleAccelerate,

    #[strum(to_string = "candle-cpu")]
    CandleCpu,

    #[cfg(not(target_os = "macos"))]
    #[strum(to_string = "candle-cuda")]
    CandleCuda,

    #[cfg(target_os = "macos")]
    #[strum(to_string = "candle-metal")]
    CandleMetal,

    // cuda ------------------------------------------------------------------
    #[cfg(not(target_os = "macos"))]
    #[strum(to_string = "cuda")]
    Cuda,

    // rocm -------------------------------------------------------------------
    #[cfg(target_os = "linux")]
    #[strum(to_string = "rocm")]
    Rocm,

    // ndarray ---------------------------------------------------------------
    #[strum(to_string = "ndarray")]
    Ndarray,

    #[strum(to_string = "ndarray-blas-accelerate")]
    NdarrayBlasAccelerate,

    #[strum(to_string = "ndarray-blas-netlib")]
    NdarrayBlasNetlib,

    #[strum(to_string = "ndarray-blas-openblas")]
    NdarrayBlasOpenblas,

    // libtorch --------------------------------------------------------------
    #[strum(to_string = "libtorch-cpu")]
    LibTorchCpu,

    #[strum(to_string = "libtorch")]
    LibTorch,

    // vulkan ----------------------------------------------------------------
    #[strum(to_string = "vulkan")]
    Vulkan,

    // Metal -----------------------------------------------------------------
    #[cfg(target_os = "macos")]
    #[strum(to_string = "metal")]
    Metal,

    // wgpu ------------------------------------------------------------------
    #[strum(to_string = "wgpu")]
    Wgpu,

    #[strum(to_string = "wgpu-cpu")]
    WgpuCpu,
}

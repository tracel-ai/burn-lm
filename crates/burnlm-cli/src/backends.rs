#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum, strum::Display, strum::EnumIter)]
pub(crate) enum BackendValues {

    // candle ----------------------------------------------------------------
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

    #[cfg(not(target_os = "macos"))]
    #[strum(to_string = "cuda-fusion")]
    CudaFusion,

    // hip -------------------------------------------------------------------
    #[cfg(target_os = "linux")]
    #[strum(to_string = "hip")]
    Hip,

    // ndarray ---------------------------------------------------------------
    #[strum(to_string = "ndarray")]
    Ndarray,

    #[strum(to_string = "ndarray-blas-accelerate")]
    NdarrayBlasAccelerate,

    #[strum(to_string = "ndarray-blas-netlib")]
    NdarrayBlasNetlib,

    #[strum(to_string = "ndarray-blas-openblas")]
    NdarrayBlasOpenblas,

    // torch -----------------------------------------------------------------
    #[strum(to_string = "tch-cpu")]
    TchCpu,

    #[strum(to_string = "tch-gpu")]
    TchGpu,

    // wgpu ------------------------------------------------------------------
    #[strum(to_string = "wgpu")]
    Wgpu,

    #[strum(to_string = "wgpu-fusion")]
    WgpuFusion,

    #[strum(to_string = "wgpu-spirv")]
    WgpuSpirv,

    #[strum(to_string = "wgpu-spirv-fusion")]
    WgpuSpirvFusion,
}

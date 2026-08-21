//! In-process CPU profiler exposed over HTTP (feature = "profiling").
//!
//! Mounted on the SAME port the server already binds, so a Modal `@web_server` deployment —
//! which only proxies that one port — can be profiled remotely with no second listener:
//!
//! ```text
//! # raw pprof protobuf for the interactive flamegraph / call graph:
//! curl "https://<workspace>--burn-lm-bench-serve.modal.run/debug/pprof/profile?seconds=30" > prof.pb
//! pprof -http=:8080 prof.pb
//!
//! # or a ready-made flamegraph SVG straight in the browser:
//! curl "https://<workspace>--.../debug/pprof/flamegraph?seconds=30" > flame.svg
//! ```
//!
//! This is a CPU (on-CPU, SIGPROF-sampled) profile of the whole process, so it shows where
//! HOST time goes — the single batching worker's per-sequence detok/emit + SSE pushes,
//! tokenization, lock waits surfacing as on-CPU spins — which the GPU counters and the
//! batching logs can't see. GPU kernel time is async and won't appear here (use nsys for
//! that); the point is exactly to separate host-bound from device-bound.

use std::time::Duration;

use axum::{
    extract::Query,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ProfileParams {
    /// Sampling window in seconds (default 30).
    #[serde(default = "default_seconds")]
    seconds: u64,
    /// Sampling frequency in Hz (default 99 — coprime with common timers to avoid lockstep).
    #[serde(default = "default_frequency")]
    frequency: i32,
}

fn default_seconds() -> u64 {
    30
}
fn default_frequency() -> i32 {
    99
}

/// The `/debug/pprof/*` routes. Merge into the top-level router (not under `/v1`) so the
/// paths match the conventional Go pprof layout that `pprof -http` expects.
pub fn router() -> Router {
    Router::new()
        .route("/debug/pprof/profile", get(pprof_profile))
        .route("/debug/pprof/flamegraph", get(pprof_flamegraph))
}

/// Sample the process for `seconds` at `frequency` Hz and build a report. The blocklist keeps
/// the unwinder out of frames that are unsafe to walk (the usual pprof-rs recommendation).
async fn sample(seconds: u64, frequency: i32) -> Result<pprof::Report, String> {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(frequency)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .map_err(|e| format!("profiler start failed: {e}"))?;
    tokio::time::sleep(Duration::from_secs(seconds)).await;
    guard
        .report()
        .build()
        .map_err(|e| format!("report build failed: {e}"))
}

/// `GET /debug/pprof/profile?seconds=30&frequency=99` -> pprof protobuf (`pprof -http`).
async fn pprof_profile(Query(p): Query<ProfileParams>) -> impl IntoResponse {
    let report = match sample(p.seconds, p.frequency).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };
    use pprof::protos::Message;
    let profile = match report.pprof() {
        Ok(p) => p,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("pprof encode failed: {e}"))
                .into_response()
        }
    };
    match profile.write_to_bytes() {
        Ok(body) => ([(header::CONTENT_TYPE, "application/octet-stream")], body).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("protobuf write failed: {e}"))
            .into_response(),
    }
}

/// `GET /debug/pprof/flamegraph?seconds=30&frequency=99` -> flamegraph SVG (view in browser).
async fn pprof_flamegraph(Query(p): Query<ProfileParams>) -> impl IntoResponse {
    let report = match sample(p.seconds, p.frequency).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    };
    let mut svg = Vec::new();
    match report.flamegraph(&mut svg) {
        Ok(()) => ([(header::CONTENT_TYPE, "image/svg+xml")], svg).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("flamegraph failed: {e}"))
            .into_response(),
    }
}

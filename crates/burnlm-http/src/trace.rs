use std::{fmt, time::Duration};

use axum::http::header::HeaderName;
use tower_http::LatencyUnit;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

lazy_static! {
    pub static ref X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
}

pub struct Latency {
    pub unit: LatencyUnit,
    pub duration: Duration,
}

impl fmt::Display for Latency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unit {
            LatencyUnit::Seconds => write!(f, "{}s", self.duration.as_secs_f64()),
            LatencyUnit::Micros => write!(f, "{}μs", self.duration.as_micros()),
            LatencyUnit::Nanos => write!(f, "{}ns", self.duration.as_nanos()),
            // defaults to millis
            _ => write!(f, "{}ms", self.duration.as_millis()),
        }
    }
}

pub fn init() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();
}

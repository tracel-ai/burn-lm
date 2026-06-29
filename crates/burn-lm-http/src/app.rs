use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use axum::{
    http::{HeaderName, Request},
    response::Response,
    routing::get,
    Router,
};
use tokio::net::TcpListener;
use tower_http::{
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    trace::TraceLayer,
};
use tracing::{info, Span};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{
    openapi::ApiDoc,
    routers::{chat_routers, model_routers},
    stores::chat_store::ChatStore,
    trace::{self, Latency},
};

lazy_static! {
    pub static ref X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
}

/// Application
#[derive(Debug)]
pub struct App {
    host: IpAddr,
    port: u16,
}

impl Default for App {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 3000,
        }
    }
}

/// Load-time server overrides a caller can pass to [`App::new_with_config`] — the programmatic
/// equivalent of the `BURN_LM_MAX_SLOTS` / `BURN_LM_MAX_SEQ_LEN` env vars, for code that embeds the
/// server (e.g. a benchmark harness) and wants to size the KV slab without the environment. Each
/// `None` leaves that knob to its env var or compiled default.
#[derive(Debug, Default, Clone)]
pub struct AppConfig {
    /// The batched decoder's KV-slab lane count (the concurrency cap, `max_slots`).
    pub max_slots: Option<usize>,
    /// The context window each lane reserves (`max_seq_len`); lower it to fit a high `max_slots`.
    pub max_seq_len: Option<usize>,
    /// The chunked-prefill width (`prefill_chunk_size`): prompt tokens prefilled per round. `0` is
    /// unbounded (the whole prompt in one round).
    pub prefill_chunk_size: Option<usize>,
}

impl App {
    pub fn new(host: IpAddr, port: u16) -> Self {
        dotenvy::from_filename(".env").ok();
        trace::init();
        Self { host, port }
    }

    /// Like [`App::new`], but applies load-time config from code. It works by setting the same env
    /// vars the server reads when it sizes the KV slab at load (`config_usize` in the Llama server),
    /// so it shares one mechanism with deployment env config — and an env var already set in the
    /// process wins, so an operator can still override a baked-in code value. Call this before
    /// `serve()`; the model loads lazily on the first request, after which the values are fixed until
    /// reload.
    pub fn new_with_config(host: IpAddr, port: u16, config: AppConfig) -> Self {
        let set_if_unset = |var: &str, value: Option<usize>| {
            if let Some(value) = value {
                if std::env::var(var).is_err() {
                    std::env::set_var(var, value.to_string());
                }
            }
        };
        set_if_unset("BURN_LM_MAX_SLOTS", config.max_slots);
        set_if_unset("BURN_LM_MAX_SEQ_LEN", config.max_seq_len);
        set_if_unset("BURN_LM_PREFILL_CHUNK_SIZE", config.prefill_chunk_size);
        Self::new(host, port)
    }
}

impl App {
    /// Define application service (router)
    async fn app(&self) -> Router {
        let version_prefix = "/v1";
        let model_store = ChatStore::create_state();
        let openapi = ApiDoc::openapi();
        let public_routes = Router::new()
            .route("/", get(|| async { "Home" }))
            .merge(chat_routers::public_router(model_store.clone()))
            .merge(model_routers::public_router(model_store.clone()));
        let router = Router::new().merge(public_routes);
        let base = Router::new()
            .nest(version_prefix, router)
            .merge(SwaggerUi::new("/v1/swagger-ui").url("/v1/api-docs/openapi.json", openapi));
        // Mount the CPU profiler routes (`/debug/pprof/*`) on the same port when built with
        // `--features profiling`. They sit ahead of the layers so they get request-id/trace too.
        #[cfg(feature = "profiling")]
        let base = base.merge(crate::profiling::router());
        base
            // Propagate request ID header from requests to responses
            .layer(PropagateRequestIdLayer::new(X_REQUEST_ID.clone()))
            // Log requests
            .layer(
                TraceLayer::new_for_http()
                    .make_span_with(|request: &Request<_>| {
                        // define a span only when debug level is set
                        tracing::debug_span!(
                            "http_request",
                            headers = ?request.headers(),
                            version = ?request.version(),
                        )
                    })
                    .on_request(move |request: &Request<_>, _span: &Span| {
                        tracing::debug!(
                            request_id = ?request.headers()[X_REQUEST_ID.clone()],
                            method = %request.method(),
                            uri = %request.uri(),
                            "incoming request",
                        );
                    })
                    .on_response(
                        move |response: &Response, latency: Duration, _span: &Span| {
                            let latency = Latency {
                                unit: tower_http::LatencyUnit::Millis,
                                duration: latency,
                            };
                            tracing::info!(
                                request_id = ?response.headers()[X_REQUEST_ID.clone()],
                                %latency,
                                status = response.status().as_u16(),
                                "sent response",
                            );
                        },
                    ),
            )
            // Create Request ID
            .layer(SetRequestIdLayer::new(
                X_REQUEST_ID.clone(),
                MakeRequestUuid,
            ))
    }

    /// Create and start the application HTTP server
    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error>> {
        // Bind address is operator-configured (see the `--host` flag); it defaults to 0.0.0.0 (all
        // interfaces) so the server is reachable from outside the container — required by Modal's
        // @web_server proxy — and can be set to 127.0.0.1 to keep it local-only.
        let addr = SocketAddr::from((self.host, self.port));
        let banner = r#"

  ██████╗ ██╗   ██╗██████╗ ███╗   ██╗    ██╗     ███╗   ███╗
  ██╔══██╗██║   ██║██╔══██╗████╗  ██║    ██║     ████╗ ████║
  ██████╔╝██║   ██║██████╔╝██╔██╗ ██║    ██║     ██╔████╔██║
  ██╔══██╗██║   ██║██╔══██╗██║╚██╗██║    ██║     ██║╚██╔╝██║
  ██████╔╝╚██████╔╝██║  ██║██║ ╚████║    ███████╗██║ ╚═╝ ██║
  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝    ╚══════╝╚═╝     ╚═╝

       ███████╗███████╗██████╗ ██╗   ██╗███████╗██████╗
       ██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗
       ███████╗█████╗  ██████╔╝██║   ██║█████╗  ██████╔╝
       ╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██╔══╝  ██╔══██╗
       ███████║███████╗██║  ██║ ╚████╔╝ ███████╗██║  ██║
       ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝
"#;
        info!("{banner}");
        info!("Starting server on '{addr}'...");
        let listener = TcpListener::bind(addr)
            .await
            .expect("Server should bind to address successfully");
        // Serve the application
        let app = self.app().await;
        info!("Server started! (press CTRL+C to exit)");
        axum::serve(listener, app).await?;
        Ok(())
    }
}

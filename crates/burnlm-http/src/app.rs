use std::{net::SocketAddr, time::Duration};

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
    stores::model_store::ModelStore,
    trace::{self, Latency},
};

lazy_static! {
    pub static ref X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
}

/// Application
#[derive(Debug)]
pub struct App {
    port: u16,
}

impl Default for App {
    fn default() -> Self {
        Self { port: 3000 }
    }
}

impl App {
    pub fn new(port: u16) -> Self {
        dotenvy::from_filename(".env").ok();
        trace::init();
        Self { port }
    }
}

impl App {
    /// Define application service (router)
    async fn app(&self) -> Router {
        let version_prefix = "/v1";
        let model_store = ModelStore::create_state();
        let openapi = ApiDoc::openapi();
        let public_routes = Router::new()
            .route("/", get(|| async { "Home" }))
            .merge(chat_routers::public_router(model_store.clone()))
            .merge(model_routers::public_router(model_store.clone()));
        let router = Router::new().merge(public_routes);
        Router::new()
            .nest(version_prefix, router)
            .merge(SwaggerUi::new("/v1/swagger-ui").url("/v1/api-docs/openapi.json", openapi))
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
        let addr = SocketAddr::from(([127, 0, 0, 1], self.port));
        info!("Starting server on port '{addr}'...");
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

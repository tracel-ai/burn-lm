use std::net::SocketAddr;

use axum::{routing::get, Router};
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::{filter, fmt, layer::SubscriberExt, util::SubscriberInitExt, Layer};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::{openapi::ApiDoc, routers::model_routers, stores::model_store::ModelStore};

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
        tracing_subscriber::registry()
            .with(fmt::layer().with_filter(filter::LevelFilter::INFO))
            .init();
        Self { port }
    }
}

impl App {
    /// Define application service (router)
    async fn app(&self) -> Router {
        let version_prefix = "/v1";
        let model_store = ModelStore::new();
        let openapi = ApiDoc::openapi();
        let public_routes = Router::new()
            .route("/", get(|| async { "Home" }))
            .merge(model_routers::public_router(model_store));
        let router = Router::new().merge(public_routes);
        Router::new().nest(version_prefix, router)
            .merge(SwaggerUi::new("/v1/swagger-ui").url("/v1/api-docs/openapi.json", openapi))
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
        info!("Server started!");
        axum::serve(listener, app).await?;
        Ok(())
    }
}

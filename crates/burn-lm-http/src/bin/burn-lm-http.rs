use std::net::IpAddr;

use burn_lm_http::App;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the Axum server.
    Run {
        /// Host/IP address to bind to. Defaults to `0.0.0.0` (all interfaces) so the server is
        /// reachable from outside the container; pass `127.0.0.1` to keep it local-only.
        #[arg(long, default_value = "0.0.0.0")]
        host: IpAddr,
        /// Listening port for the server.
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { host, port } => run_server(host, port).await,
    }
}

async fn run_server(host: IpAddr, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = App::new(host, port);
    app.serve().await
}

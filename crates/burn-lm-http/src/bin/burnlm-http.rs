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
        /// Listening port for the server.
        #[arg(short, long, default_value_t = 3000)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { port } => run_server(port).await,
    }
}

async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = App::new(port);
    app.serve().await
}

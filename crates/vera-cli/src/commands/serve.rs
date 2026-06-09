//! `vera serve` — Start the Vera HTTP API server.

use anyhow::Result;
use vera_core::config::{InferenceBackend, VeraConfig};

/// Run the `vera serve` command.
pub fn run(
    host: &str,
    port: u16,
    api_key: Option<String>,
    backend: InferenceBackend,
    config: VeraConfig,
    idle_timeout_secs: i64,
) -> Result<()> {
    let idle_timeout = match idle_timeout_secs {
        0 => None,
        -1 => Some(std::time::Duration::MAX),
        n if n > 0 => Some(std::time::Duration::from_secs(n as u64)),
        _ => None,
    };
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(vera_serve::run_server(
        config,
        backend,
        api_key,
        host,
        port,
        idle_timeout,
    ))
}

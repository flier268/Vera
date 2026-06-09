//! Vera HTTP API server.
//!
//! Exposes OpenAI-compatible inference endpoints for standard vera clients:
//!
//! ```text
//! POST /v1/embeddings   OpenAI format  (EMBEDDING_MODEL_BASE_URL)
//! POST /v1/rerank       Cohere/Jina format  (RERANKER_MODEL_BASE_URL)
//! GET  /v1/health       liveness + model info
//! ```
//!
//! A regular vera client configured with `vera setup --api` pointing at
//! `http://host:port/v1` will work without any modifications.

mod handlers;
pub mod types;

use std::sync::Arc;

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use vera_core::config::{InferenceBackend, VeraConfig};

/// Shared state injected into every handler.
pub struct AppState {
    pub api_key: Option<String>,
    /// Config used to create providers on-demand per request.
    pub config: VeraConfig,
    pub backend: InferenceBackend,
    /// Human-readable model name reported in /v1/health and embeddings responses.
    pub model_name: String,
    /// Whether a reranker is available (probed at startup).
    pub reranker_available: bool,
}

/// Start the Vera HTTP server.
///
/// Loads the embedding model and reranker once at startup, then listens for
/// connections on `host:port`.
///
/// - `config`   — vera retrieval/embedding config
/// - `backend`  — compute backend (API, CPU, GPU)
/// - `api_key`  — optional bearer token; `None` disables auth
/// - `host`     — bind address (e.g. `"127.0.0.1"` or `"0.0.0.0"`)
/// - `port`     — TCP port to listen on
pub async fn run_server(
    config: VeraConfig,
    backend: InferenceBackend,
    api_key: Option<String>,
    host: &str,
    port: u16,
) -> Result<()> {
    eprintln!(
        "vera serve: initializing {} backend…",
        backend_label(backend)
    );

    // Probe-load to validate config and obtain the model name, then release immediately.
    let (probe, model_name) =
        vera_core::embedding::create_dynamic_provider(&config, backend)
            .await
            .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?;
    drop(probe);

    eprintln!("vera serve: embedding model ready ({})", model_name);

    let reranker_available = vera_core::retrieval::create_dynamic_reranker(&config, backend)
        .await
        .unwrap_or_else(|e| {
            eprintln!("vera serve: reranker unavailable ({e}), reranking disabled");
            None
        })
        .is_some();

    if reranker_available {
        eprintln!("vera serve: reranker ready");
    }

    let api_key = api_key.filter(|k| !k.is_empty());

    if api_key.is_some() {
        eprintln!("vera serve: API key authentication enabled");
    } else {
        eprintln!("vera serve: no API key set — unauthenticated access allowed");
    }

    let state = Arc::new(AppState {
        api_key,
        config,
        backend,
        model_name,
        reranker_available,
    });

    let app = Router::new()
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/rerank", post(handlers::rerank))
        .route("/v1/health", get(handlers::health))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("vera serve: listening on http://{addr}");
    eprintln!();
    eprintln!("  Client setup:");
    eprintln!("    vera setup --api  (then set EMBEDDING_MODEL_BASE_URL=http://{addr}/v1)");
    axum::serve(listener, app).await?;
    Ok(())
}

fn backend_label(backend: InferenceBackend) -> &'static str {
    use vera_core::config::OnnxExecutionProvider;
    match backend {
        InferenceBackend::Api => "api",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cpu) => "cpu",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Cuda) => "cuda (GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::Rocm) => "rocm (AMD GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::DirectMl) => "directml (GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::CoreMl) => "coreml (Apple GPU)",
        InferenceBackend::OnnxJina(OnnxExecutionProvider::OpenVino) => "openvino (Intel GPU)",
    }
}

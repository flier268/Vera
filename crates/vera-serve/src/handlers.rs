//! Axum route handlers for the Vera HTTP API.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use vera_core::embedding::EmbeddingProvider;
use vera_core::retrieval::Reranker;

use crate::{
    AppState,
    types::{
        ApiError, EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage,
        HealthResponse, RerankRequest, RerankResponse, RerankResult,
    },
};

// ── Auth ──────────────────────────────────────────────────────────────────────

/// Constant-time byte comparison (prevents timing-oracle prefix attacks).
fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Option<(StatusCode, Json<ApiError>)> {
    let Some(ref key) = state.api_key else {
        return None;
    };
    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if !constant_time_eq(provided, key) {
        Some((
            StatusCode::UNAUTHORIZED,
            Json(ApiError {
                error: "invalid or missing API key".into(),
            }),
        ))
    } else {
        None
    }
}

// ── OpenAI-compatible /v1/embeddings ─────────────────────────────────────────

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingsRequest>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    if req.input.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiError {
                error: "input must not be empty".into(),
            }),
        )
            .into_response();
    }

    let provider = Arc::clone(&state.embedding_provider);
    let texts = req.input.clone();
    let total_chars: usize = texts.iter().map(|s| s.len()).sum();

    match provider.embed_batch(&texts).await {
        Ok(vecs) => {
            let data: Vec<EmbeddingObject> = vecs
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| EmbeddingObject {
                    object: "embedding",
                    embedding,
                    index,
                })
                .collect();
            Json(EmbeddingsResponse {
                object: "list",
                data,
                model: state.model_name.clone(),
                usage: EmbeddingsUsage {
                    prompt_tokens: total_chars / 4,
                    total_tokens: total_chars / 4,
                },
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError { error: e.to_string() }),
        )
            .into_response(),
    }
}

// ── Cohere/Jina-compatible /v1/rerank ────────────────────────────────────────

pub async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RerankRequest>,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    let Some(ref reranker) = state.reranker else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError {
                error: "reranker not available on this server".into(),
            }),
        )
            .into_response();
    };

    let reranker = Arc::clone(reranker);
    let query = req.query.clone();
    let documents = req.documents.clone();
    let top_n = req.top_n;

    match reranker.rerank(&query, &documents).await {
        Ok(mut scores) => {
            scores.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
            if let Some(n) = top_n {
                scores.truncate(n);
            }
            let results = scores
                .into_iter()
                .map(|s| RerankResult {
                    index: s.index,
                    relevance_score: s.relevance_score,
                })
                .collect();
            Json(RerankResponse { results }).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError { error: e.to_string() }),
        )
            .into_response(),
    }
}

// ── /v1/health ────────────────────────────────────────────────────────────────

pub async fn health(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Some(err) = check_auth(&state, &headers) {
        return err.into_response();
    }
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        model: state.model_name.clone(),
        backend: format!("{}", state.backend),
    })
    .into_response()
}

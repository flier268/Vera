//! Axum route handlers for the Vera HTTP API.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use tokio_util::sync::CancellationToken;
use tracing::error;
use vera_core::embedding::{DynamicProvider, EmbeddingError, EmbeddingProvider};
use vera_core::retrieval::{DynamicReranker, Reranker, RerankerError};

use crate::{
    AppState, CachedProviders,
    types::{
        ApiError, EmbeddingObject, EmbeddingsRequest, EmbeddingsResponse, EmbeddingsUsage,
        HealthResponse, RerankRequest, RerankResponse, RerankResult,
    },
};

// ── Provider cache helpers ────────────────────────────────────────────────────

type ProviderPair = (Arc<DynamicProvider>, Option<Arc<DynamicReranker>>);
type AcquireError = (StatusCode, Json<ApiError>);

async fn load_fresh(state: &AppState) -> Result<ProviderPair, AcquireError> {
    let (embedding, _) =
        vera_core::embedding::create_dynamic_provider(&state.config, state.backend)
            .await
            .map_err(|e| {
                error!(error = %e, "failed to load embedding model");
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ApiError {
                        error: "embedding model unavailable".into(),
                    }),
                )
            })?;
    let reranker = if state.reranker_available {
        vera_core::retrieval::create_dynamic_reranker(&state.config, state.backend)
            .await
            .unwrap_or_else(|e| {
                error!(error = %e, "failed to load reranker");
                None
            })
            .map(Arc::new)
    } else {
        None
    };
    Ok((Arc::new(embedding), reranker))
}

/// Acquire providers: per-request (no cache) or from the idle cache.
async fn acquire_providers(state: &AppState) -> Result<ProviderPair, AcquireError> {
    if state.idle_timeout.is_none() {
        // Cache disabled — load fresh, drop when handler returns.
        return load_fresh(state).await;
    }

    let mut guard = state.provider_cache.lock().await;
    if guard.is_none() {
        let (embedding, reranker) = load_fresh(state).await?;
        *guard = Some(CachedProviders {
            embedding,
            reranker,
            last_used: Instant::now(),
        });
    }
    let cached = guard.as_mut().unwrap();
    cached.last_used = Instant::now();
    Ok((
        Arc::clone(&cached.embedding),
        cached.reranker.as_ref().map(Arc::clone),
    ))
}

// ── Auth ──────────────────────────────────────────────────────────────────────

/// Constant-time byte comparison (prevents timing-oracle prefix attacks).
fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Option<(StatusCode, Json<ApiError>)> {
    let key = state.api_key.as_ref()?;
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

    let total_chars: usize = req.input.iter().map(|s| s.len()).sum();

    let (provider, _) = match acquire_providers(&state).await {
        Ok(p) => p,
        Err((status, body)) => return (status, body).into_response(),
    };

    let cancel = CancellationToken::new();
    let _guard = cancel.clone().drop_guard();

    match provider.embed_batch_cancellable(&req.input, &cancel).await {
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
        Err(EmbeddingError::Cancelled) => (
            StatusCode::from_u16(499).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(ApiError {
                error: "request cancelled".into(),
            }),
        )
            .into_response(),
        Err(e) => {
            error!(error = %e, "embedding inference failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError {
                    error: "internal server error".into(),
                }),
            )
                .into_response()
        }
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
    if !state.reranker_available {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError {
                error: "reranker not available on this server".into(),
            }),
        )
            .into_response();
    }

    let (_, reranker_opt) = match acquire_providers(&state).await {
        Ok(p) => p,
        Err((status, body)) => return (status, body).into_response(),
    };
    let Some(reranker) = reranker_opt else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError {
                error: "reranker not available on this server".into(),
            }),
        )
            .into_response();
    };

    let top_n = req.top_n;

    let cancel = CancellationToken::new();
    let _guard = cancel.clone().drop_guard();

    match reranker
        .rerank_cancellable(&req.query, &req.documents, &cancel)
        .await
    {
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
        Err(RerankerError::Cancelled) => (
            StatusCode::from_u16(499).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(ApiError {
                error: "request cancelled".into(),
            }),
        )
            .into_response(),
        Err(e) => {
            error!(error = %e, "rerank inference failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError {
                    error: "internal server error".into(),
                }),
            )
                .into_response()
        }
    }
}

// ── /v1/health ────────────────────────────────────────────────────────────────

pub async fn health(State(state): State<Arc<AppState>>, headers: HeaderMap) -> impl IntoResponse {
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

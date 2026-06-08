//! Request and response types for the Vera HTTP API.
//!
//! POST /v1/embeddings  — same format as OpenAI /v1/embeddings
//! POST /v1/rerank      — same format as Cohere/Jina /v1/rerank
//! GET  /v1/health      — liveness check

use serde::{Deserialize, Deserializer, Serialize};

// ── OpenAI-compatible embeddings ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model identifier (informational only; server uses its loaded model).
    #[serde(default)]
    pub model: Option<String>,
    /// Texts to embed — accepts both a bare string and an array of strings.
    #[serde(deserialize_with = "string_or_seq")]
    pub input: Vec<String>,
}

fn string_or_seq<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(s) => Ok(vec![s]),
        OneOrMany::Many(v) => Ok(v),
    }
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// ── Cohere/Jina-compatible reranker ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: Option<bool>,
}

#[derive(Serialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
}

#[derive(Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
}

// ── Shared ────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub model: String,
    pub backend: String,
}

#[derive(Serialize)]
pub struct ApiError {
    pub error: String,
}

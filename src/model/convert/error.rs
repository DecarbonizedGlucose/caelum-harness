use serde_json::Value;

use crate::error::{ErrorCode, HarnessError};
use crate::model::protocol::Protocol;

/// Wraps a JSON decode failure with provider and context.
pub(crate) fn decode_error(
    context: &str,
    protocol: Protocol,
    error: serde_json::Error,
) -> HarnessError {
    if error.is_syntax() || error.is_eof() {
        HarnessError::json(context, error)
    } else {
        HarnessError::protocol(protocol, "$", format!("could not decode payload: {error}"))
    }
}

/// Classifies a provider error payload into a stable error code.
pub(crate) fn provider_error(
    protocol: Protocol,
    message: Option<&str>,
    kind: Option<&str>,
    code: Option<&Value>,
    status: Option<u16>,
) -> HarnessError {
    let code_text = code
        .map(|value| match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    let detail = [
        message.unwrap_or(""),
        kind.unwrap_or(""),
        code_text.as_str(),
    ]
    .join(" ")
    .to_lowercase()
    .replace(['_', '-'], " ");

    let error_code = if detail.contains("content policy")
        || detail.contains("content filter")
        || detail.contains("moderation")
    {
        ErrorCode::ContentFilter
    } else if status == Some(429)
        || detail.contains("rate limit")
        || detail.contains("too many requests")
    {
        ErrorCode::RateLimit
    } else if status == Some(401)
        || status == Some(403)
        || detail.contains("auth")
        || detail.contains("invalid api key")
    {
        ErrorCode::Auth
    } else if status == Some(529)
        || detail.contains("overloaded")
        || detail.contains("server error")
        || detail.contains("unavailable")
    {
        ErrorCode::Server
    } else if detail.contains("context length")
        || detail.contains("context window")
        || detail.contains("too many tokens")
    {
        ErrorCode::ContextWindowExceeded
    } else if detail.contains("quota") || detail.contains("insufficient balance") {
        ErrorCode::Quota
    } else if status.is_some_and(|status| status >= 500) {
        ErrorCode::Server
    } else {
        ErrorCode::Unknown
    };

    let mut text = message.unwrap_or("provider error").to_string();
    if let Some(status) = status {
        text.push_str(&format!(" (HTTP {status})"));
    }
    if !kind.unwrap_or("").is_empty() {
        text.push_str(&format!(" [{}]", kind.unwrap_or("")));
    }
    HarnessError::new(error_code, format!("{protocol}: {text}"))
}

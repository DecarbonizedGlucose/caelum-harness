use std::fmt;

use crate::model::protocol::Protocol;

/// Stable machine-readable error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    Json,
    Protocol,
    UnsupportedContent,
    UnsupportedOption,
    EmptyResponse,
    ContentFilter,
    Refusal,
    StreamClosed,
    InvalidRequest,
    InvalidConfig,
    ContextWindowExceeded,
    MissingCredential,
    InvalidCredential,
    Auth,
    Quota,
    AccountQuota,
    UnknownModel,
    RateLimit,
    Server,
    Timeout,
    Transport,
    Http,
    Aborted,
    StepLimit,
    Internal,
    Unknown,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "JSON",
            Self::Protocol => "PROTOCOL",
            Self::UnsupportedContent => "UNSUPPORTED_CONTENT",
            Self::UnsupportedOption => "UNSUPPORTED_OPTION",
            Self::EmptyResponse => "EMPTY_RESPONSE",
            Self::ContentFilter => "CONTENT_FILTER",
            Self::Refusal => "REFUSAL",
            Self::StreamClosed => "STREAM_CLOSED",
            Self::InvalidRequest => "INVALID_REQUEST",
            Self::InvalidConfig => "INVALID_CONFIG",
            Self::ContextWindowExceeded => "CONTEXT_WINDOW_EXCEEDED",
            Self::MissingCredential => "MISSING_CREDENTIAL",
            Self::InvalidCredential => "INVALID_CREDENTIAL",
            Self::Auth => "AUTH",
            Self::Quota => "QUOTA",
            Self::AccountQuota => "ACCOUNT_QUOTA",
            Self::UnknownModel => "UNKNOWN_MODEL",
            Self::RateLimit => "RATE_LIMIT",
            Self::Server => "SERVER",
            Self::Timeout => "TIMEOUT",
            Self::Transport => "TRANSPORT",
            Self::Http => "HTTP",
            Self::Aborted => "ABORTED",
            Self::StepLimit => "STEP_LIMIT",
            Self::Internal => "INTERNAL",
            Self::Unknown => "UNKNOWN",
        }
    }

    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::EmptyResponse | Self::RateLimit | Self::Server | Self::Timeout | Self::Transport
        )
    }
}

/// Runtime error with a stable code, a readable message, and an optional cause.
#[derive(Debug)]
pub struct HarnessError {
    code: ErrorCode,
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl HarnessError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        let context = context.into();
        let message = format!("failed to parse {context} as JSON: {source}");
        Self {
            code: ErrorCode::Json,
            message,
            source: Some(Box::new(source)),
        }
    }

    pub fn protocol(
        protocol: Protocol,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let path = path.into();
        let message = message.into();
        Self::new(
            ErrorCode::Protocol,
            format!("{protocol} response is invalid at {path}: {message}"),
        )
    }

    pub fn unsupported(protocol: Protocol, feature: impl Into<String>) -> Self {
        let feature = feature.into();
        Self::new(
            ErrorCode::UnsupportedContent,
            format!("{protocol} does not support {feature}"),
        )
    }

    pub fn empty_response(protocol: Protocol) -> Self {
        Self::new(
            ErrorCode::EmptyResponse,
            format!("{protocol} returned an empty response"),
        )
    }

    pub fn stream_closed(protocol: Protocol) -> Self {
        Self::new(
            ErrorCode::StreamClosed,
            format!("{protocol} stream closed before completion"),
        )
    }

    pub fn content_filter(protocol: Protocol) -> Self {
        Self::new(
            ErrorCode::ContentFilter,
            format!("{protocol} response was blocked by a content filter"),
        )
    }

    pub fn refusal(protocol: Protocol) -> Self {
        Self::new(
            ErrorCode::Refusal,
            format!("{protocol} response was refused"),
        )
    }

    pub fn request(
        protocol: Protocol,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let path = path.into();
        let message = message.into();
        Self::new(
            ErrorCode::InvalidRequest,
            format!("{protocol} request is invalid at {path}: {message}"),
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::new(
            ErrorCode::Internal,
            format!("internal harness error: {message}"),
        )
    }

    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn code_str(&self) -> &'static str {
        self.code.as_str()
    }

    pub fn is_retryable(&self) -> bool {
        self.code.is_retryable()
    }
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HarnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

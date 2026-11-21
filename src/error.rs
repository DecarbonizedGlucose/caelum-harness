use std::fmt;

use crate::model::protocol::Protocol;

/// Runtime error categories used by the harness.
#[derive(Debug)]
pub enum HarnessError {
    Json {
        context: String,
        source: serde_json::Error,
    },
    Protocol {
        protocol: Protocol,
        path: String,
        message: String,
    },
    Unsupported {
        protocol: Protocol,
        feature: String,
    },
    Internal {
        message: String,
    },
}

impl HarnessError {
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    pub fn protocol(
        protocol: Protocol,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::Protocol {
            protocol,
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn unsupported(protocol: Protocol, feature: impl Into<String>) -> Self {
        Self::Unsupported {
            protocol,
            feature: feature.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Stable machine-readable error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Json { .. } => "JSON",
            Self::Protocol { .. } => "PROTOCOL",
            Self::Unsupported { .. } => "UNSUPPORTED",
            Self::Internal { .. } => "INTERNAL",
        }
    }
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json { context, source } => {
                write!(f, "failed to parse {context} as JSON: {source}")
            }
            Self::Protocol {
                protocol,
                path,
                message,
            } => write!(f, "{protocol} response is invalid at {path}: {message}"),
            Self::Unsupported { protocol, feature } => {
                write!(f, "{protocol} does not support {feature}")
            }
            Self::Internal { message } => write!(f, "internal harness error: {message}"),
        }
    }
}

impl std::error::Error for HarnessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}

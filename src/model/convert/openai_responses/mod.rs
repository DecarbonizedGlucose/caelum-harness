use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::HarnessError;
use crate::model::convert::{ProtocolCodec, StreamDecoder};
use crate::model::message::{Message, Origin, StopReason, Usage};
use crate::model::protocol::ChatRequest;
use crate::model::wire::openai_responses as wire;

mod request;
mod response;
mod stream;

pub(super) const MESSAGE_OPAQUE_KIND: &str = "openai.responses.message";
pub(super) const REASONING_OPAQUE_KIND: &str = "openai.responses.reasoning";

/// Provider-specific compatibility for OpenAI Responses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiResponsesCompat {
    pub include_encrypted_reasoning: bool,
    pub replay_reasoning_items: bool,
    pub drop_sampling_with_reasoning: bool,
}

impl OpenAiResponsesCompat {
    pub fn openai() -> Self {
        Self {
            include_encrypted_reasoning: true,
            replay_reasoning_items: true,
            drop_sampling_with_reasoning: true,
        }
    }

    pub fn deepseek() -> Self {
        Self {
            include_encrypted_reasoning: false,
            replay_reasoning_items: false,
            drop_sampling_with_reasoning: false,
        }
    }

    pub fn openrouter() -> Self {
        Self {
            include_encrypted_reasoning: false,
            replay_reasoning_items: false,
            drop_sampling_with_reasoning: false,
        }
    }
}

/// OpenAI Responses protocol codec.
pub struct OpenAiResponsesCodec {
    origin: Origin,
    compat: OpenAiResponsesCompat,
}

impl OpenAiResponsesCodec {
    pub fn new(origin: Origin, compat: OpenAiResponsesCompat) -> Self {
        Self { origin, compat }
    }
}

impl ProtocolCodec for OpenAiResponsesCodec {
    fn origin(&self) -> &Origin {
        &self.origin
    }

    fn encode_request(&self, request: &ChatRequest, stream: bool) -> Result<Value, HarnessError> {
        request::encode_request(&self.origin, request, &self.compat, stream)
    }

    fn decode_response(&self, raw: &str) -> Result<Message, HarnessError> {
        response::decode_response(raw, &self.origin)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::OpenAiResponsesStream::new(self.origin.clone()))
    }
}

/// Metadata replayed on an assistant message item.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MessageMeta {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub phase: Option<String>,
}

pub(super) fn decode_usage(usage: Option<&wire::Usage>) -> Option<Usage> {
    let usage = usage?;
    let details = usage.input_tokens_details.as_ref();
    let cached = details.and_then(|details| details.cached_tokens);
    let written = details.and_then(|details| details.cache_write_tokens);
    let parsed = Usage {
        input_tokens: usage.input_tokens.map(|input| {
            input
                .saturating_sub(cached.unwrap_or(0))
                .saturating_sub(written.unwrap_or(0))
        }),
        output_tokens: usage.output_tokens,
        cache_read_tokens: cached,
        cache_write_tokens: written,
        reasoning_tokens: usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        total_tokens: usage.total_tokens,
        cost: usage.cost,
    };

    (!parsed.is_empty()).then_some(parsed)
}

pub(super) fn terminal_stop_reason(
    response: &wire::Response,
    has_tool_calls: bool,
) -> Option<StopReason> {
    if response.status.as_deref() == Some("cancelled") {
        return Some(StopReason::Aborted);
    }
    if response.status.as_deref() == Some("incomplete") {
        let reason = response
            .incomplete_details
            .as_ref()
            .and_then(|details| details.reason.as_deref());
        return Some(match reason {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::ContentFilter,
            Some(other) => StopReason::Other(other.to_string()),
            None => StopReason::Other("incomplete".to_string()),
        });
    }

    Some(if has_tool_calls {
        StopReason::ToolCall
    } else {
        StopReason::EndTurn
    })
}

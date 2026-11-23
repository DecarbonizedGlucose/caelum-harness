use serde_json::Value;

use crate::error::HarnessError;
use crate::model::convert::{ProtocolCodec, StreamDecoder};
use crate::model::message::{Message, Origin, StopReason, Usage};
use crate::model::protocol::ChatRequest;
use crate::model::wire::anthropic as wire;

mod request;
mod response;
mod stream;

const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Stateless Anthropic Messages codec.
#[derive(Debug, Clone)]
pub struct AnthropicCodec {
    origin: Origin,
    default_max_tokens: u32,
}

impl AnthropicCodec {
    pub fn new(origin: Origin) -> Self {
        Self {
            origin,
            default_max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub fn with_default_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }
}

impl ProtocolCodec for AnthropicCodec {
    fn origin(&self) -> &Origin {
        &self.origin
    }

    fn encode_request(&self, request: &ChatRequest, stream: bool) -> Result<Value, HarnessError> {
        request::encode_request(&self.origin.model, self.default_max_tokens, request, stream)
    }

    fn decode_response(&self, raw: &str) -> Result<Message, HarnessError> {
        response::decode_response(raw, &self.origin)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::AnthropicStream::new(self.origin.clone()))
    }
}

pub(super) fn decode_usage(usage: Option<&wire::Usage>) -> Option<Usage> {
    let usage = usage?;
    let parsed = Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_tokens: usage.cache_read_input_tokens,
        cache_write_tokens: usage.cache_creation_input_tokens,
        reasoning_tokens: usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.thinking_tokens),
        total_tokens: None,
        cost: None,
    };

    (!parsed.is_empty()).then_some(parsed)
}

pub(super) fn stop_reason(raw: &str) -> StopReason {
    match raw {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" | "model_context_window_exceeded" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolCall,
        "stop_sequence" => StopReason::StopSequence,
        "refusal" => StopReason::Refusal,
        "pause_turn" => StopReason::Pause,
        other => StopReason::Other(other.to_string()),
    }
}

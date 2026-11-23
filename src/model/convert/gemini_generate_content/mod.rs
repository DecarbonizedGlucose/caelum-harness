use serde_json::Value;

use crate::error::HarnessError;
use crate::model::convert::{ProtocolCodec, StreamDecoder};
use crate::model::message::{Message, Origin, StopReason, Usage};
use crate::model::protocol::ChatRequest;
use crate::model::wire::gemini_generate_content as wire;

mod request;
mod response;
mod stream;

pub(super) const THOUGHT_SIGNATURE_KIND: &str = "gemini_generate_content.thought_signature";

/// How a Gemini model configures reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    None,
    Budget,
    Level,
}

/// Provider-specific compatibility for Gemini GenerateContent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiGenerateContentCompat {
    pub replay_thought_signatures: bool,
    pub thinking_mode: ThinkingMode,
}

impl GeminiGenerateContentCompat {
    pub fn google() -> Self {
        Self {
            replay_thought_signatures: true,
            thinking_mode: ThinkingMode::Budget,
        }
    }

    pub fn for_model(model: &str) -> Self {
        let short = model
            .rsplit('/')
            .next()
            .unwrap_or(model)
            .to_ascii_lowercase();
        let thinking_mode = if short.starts_with("gemini-2.5") {
            ThinkingMode::Budget
        } else if short.starts_with("gemini-3") || short.starts_with("gemini-4") {
            ThinkingMode::Level
        } else {
            ThinkingMode::None
        };
        Self {
            replay_thought_signatures: true,
            thinking_mode,
        }
    }
}

/// Gemini GenerateContent protocol codec.
pub struct GeminiGenerateContentCodec {
    origin: Origin,
    compat: GeminiGenerateContentCompat,
}

impl GeminiGenerateContentCodec {
    pub fn new(origin: Origin, compat: GeminiGenerateContentCompat) -> Self {
        Self { origin, compat }
    }
}

impl ProtocolCodec for GeminiGenerateContentCodec {
    fn origin(&self) -> &Origin {
        &self.origin
    }

    fn encode_request(&self, request: &ChatRequest, _stream: bool) -> Result<Value, HarnessError> {
        request::encode_request(&self.origin, request, &self.compat)
    }

    fn decode_response(&self, raw: &str) -> Result<Message, HarnessError> {
        response::decode_response(raw, &self.origin)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::GeminiGenerateContentStream::new(
            self.origin.clone(),
        ))
    }
}

/// Accumulates cumulative Gemini usage snapshots without losing omitted counters.
#[derive(Default)]
pub(super) struct UsageAccumulator {
    prompt: Option<u64>,
    candidates: Option<u64>,
    thoughts: Option<u64>,
    cached: Option<u64>,
    total: Option<u64>,
}

impl UsageAccumulator {
    pub(super) fn update(&mut self, metadata: Option<&wire::UsageMetadata>) -> Option<Usage> {
        let metadata = metadata?;
        if metadata.prompt_token_count.is_some() {
            self.prompt = metadata.prompt_token_count;
        }
        if metadata.candidates_token_count.is_some() {
            self.candidates = metadata.candidates_token_count;
        }
        if metadata.thoughts_token_count.is_some() {
            self.thoughts = metadata.thoughts_token_count;
        }
        if metadata.cached_content_token_count.is_some() {
            self.cached = metadata.cached_content_token_count;
        }
        if metadata.total_token_count.is_some() {
            self.total = metadata.total_token_count;
        }

        let cache_read_tokens = self
            .cached
            .map(|cached| self.prompt.map_or(cached, |prompt| cached.min(prompt)));
        let input_tokens = self
            .prompt
            .map(|prompt| prompt.saturating_sub(cache_read_tokens.unwrap_or(0)));
        let output_tokens = match (self.candidates, self.thoughts) {
            (Some(candidates), Some(thoughts)) => Some(candidates.saturating_add(thoughts)),
            (Some(candidates), None) => Some(candidates),
            (None, Some(thoughts)) => Some(thoughts),
            (None, None) => None,
        };
        let total_tokens = self.total.or_else(|| match (self.prompt, output_tokens) {
            (Some(prompt), Some(output)) => Some(prompt.saturating_add(output)),
            _ => None,
        });

        let usage = Usage {
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens: None,
            reasoning_tokens: self.thoughts,
            total_tokens,
            cost: None,
        };

        (!usage.is_empty()).then_some(usage)
    }
}

pub(super) fn map_finish_reason(raw: &str, has_tool_calls: bool) -> StopReason {
    match raw {
        "STOP" => {
            if has_tool_calls {
                StopReason::ToolCall
            } else {
                StopReason::EndTurn
            }
        }
        "FUNCTION_CALL" => StopReason::ToolCall,
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY"
        | "RECITATION"
        | "LANGUAGE"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "MODEL_ARMOR"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_OTHER"
        | "NO_IMAGE"
        | "IMAGE_RECITATION" => StopReason::ContentFilter,
        other => StopReason::Other(other.to_string()),
    }
}

pub(super) fn is_protocol_failure_reason(raw: &str) -> bool {
    matches!(
        raw,
        "MALFORMED_FUNCTION_CALL"
            | "MALFORMED_RESPONSE"
            | "MISSING_THOUGHT_SIGNATURE"
            | "UNEXPECTED_TOOL_CALL"
            | "TOO_MANY_TOOL_CALLS"
    )
}

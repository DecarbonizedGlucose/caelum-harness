use serde_json::Value;

use crate::error::HarnessError;
use crate::model::convert::{ProtocolCodec, StreamDecoder};
use crate::model::message::{Message, Origin, StopReason, Usage};
use crate::model::protocol::ChatRequest;
use crate::model::wire::openai_chat as wire;

mod request;
mod response;
mod stream;

/// Non-standard reasoning field used by an OpenAI-compatible provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningField {
    ReasoningContent,
    Reasoning,
    ReasoningText,
}

/// Output token limit field used by an OpenAI-compatible provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokensField {
    MaxTokens,
    MaxCompletionTokens,
}

/// Provider-specific compatibility for OpenAI Chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiChatCompat {
    pub reasoning_field: Option<ReasoningField>,
    pub max_tokens_field: MaxTokensField,
    pub developer_role: bool,
    pub replay_reasoning: bool,
}

impl OpenAiChatCompat {
    pub fn openai() -> Self {
        Self {
            reasoning_field: None,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            developer_role: true,
            replay_reasoning: false,
        }
    }

    pub fn deepseek() -> Self {
        Self {
            reasoning_field: Some(ReasoningField::ReasoningContent),
            max_tokens_field: MaxTokensField::MaxTokens,
            developer_role: false,
            replay_reasoning: true,
        }
    }

    pub fn openrouter() -> Self {
        Self {
            reasoning_field: Some(ReasoningField::Reasoning),
            max_tokens_field: MaxTokensField::MaxTokens,
            developer_role: false,
            replay_reasoning: true,
        }
    }

    pub(crate) fn reasoning_text<'a>(
        &self,
        reasoning_content: Option<&'a str>,
        reasoning: Option<&'a str>,
        reasoning_text: Option<&'a str>,
    ) -> Option<&'a str> {
        let order = match self.reasoning_field {
            Some(ReasoningField::ReasoningContent) => {
                [reasoning_content, reasoning, reasoning_text]
            }
            Some(ReasoningField::Reasoning) => [reasoning, reasoning_content, reasoning_text],
            Some(ReasoningField::ReasoningText) => [reasoning_text, reasoning_content, reasoning],
            None => return None,
        };
        order.into_iter().flatten().find(|text| !text.is_empty())
    }
}

/// OpenAI Chat protocol codec.
pub struct OpenAiChatCodec {
    origin: Origin,
    compat: OpenAiChatCompat,
}

impl OpenAiChatCodec {
    pub fn new(origin: Origin, compat: OpenAiChatCompat) -> Self {
        Self { origin, compat }
    }
}

impl ProtocolCodec for OpenAiChatCodec {
    fn origin(&self) -> &Origin {
        &self.origin
    }

    fn encode_request(&self, request: &ChatRequest, stream: bool) -> Result<Value, HarnessError> {
        request::encode_request(&self.origin.model, request, &self.compat, stream)
    }

    fn decode_response(&self, raw: &str) -> Result<Message, HarnessError> {
        response::decode_response(raw, &self.origin, &self.compat)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(stream::OpenAiChatStream::new(
            self.origin.clone(),
            self.compat.clone(),
        ))
    }
}

fn decode_usage(usage: Option<&wire::Usage>) -> Option<Usage> {
    let usage = usage?;
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .or(usage.prompt_cache_hit_tokens);
    let parsed = Usage {
        input_tokens: usage
            .prompt_tokens
            .map(|prompt| prompt.saturating_sub(cached.unwrap_or(0))),
        output_tokens: usage.completion_tokens,
        cache_read_tokens: cached,
        cache_write_tokens: None,
        reasoning_tokens: usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        total_tokens: usage.total_tokens,
        cost: usage.cost,
    };

    (!parsed.is_empty()).then_some(parsed)
}

fn stop_reason(raw: &str) -> StopReason {
    match raw {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolCall,
        "stop_sequence" => StopReason::StopSequence,
        "content_filter" => StopReason::ContentFilter,
        "refusal" => StopReason::Refusal,
        "pause_turn" => StopReason::Pause,
        "aborted" => StopReason::Aborted,
        other => StopReason::Other(other.to_string()),
    }
}

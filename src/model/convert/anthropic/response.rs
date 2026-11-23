use serde_json::Value;

use crate::error::HarnessError;
use crate::model::convert::anthropic::{decode_usage, stop_reason};
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::message::{
    Block, Message, Opaque, Origin, ReasoningBlock, TextBlock, ToolCallBlock,
};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, StreamEvent};
use crate::model::wire::anthropic as wire;

const THINKING_OPAQUE_KIND: &str = "anthropic.thinking";
const REDACTED_THINKING_OPAQUE_KIND: &str = "anthropic.redacted_thinking";

/// Decodes an Anthropic Messages response body.
pub(super) fn decode_response(raw: &str, origin: &Origin) -> Result<Message, HarnessError> {
    let response: wire::MessagesResponse = serde_json::from_str(raw)
        .map_err(|error| decode_error("Anthropic response", Protocol::Anthropic, error))?;
    if let Some(error) = &response.error {
        return Err(provider_error(
            Protocol::Anthropic,
            error.message.as_deref(),
            error.kind.as_deref(),
            error.code.as_ref(),
            error.status,
        ));
    }

    let events = decode_response_events(&response, origin)?;
    let mut assembler = Assembler::new(origin.clone());
    for event in events {
        assembler.push(event)?;
    }
    assembler.finish()
}

fn decode_response_events(
    response: &wire::MessagesResponse,
    origin: &Origin,
) -> Result<Vec<StreamEvent>, HarnessError> {
    let mut blocks = Vec::new();

    for content in &response.content {
        match content {
            wire::ContentBlock::Text { text } => {
                if !text.is_empty() {
                    blocks.push(Block::Text(TextBlock {
                        text: text.clone(),
                        opaque: None,
                    }));
                }
            }
            wire::ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                let opaque = signature
                    .as_deref()
                    .filter(|sig| !sig.is_empty())
                    .map(|sig| Opaque {
                        origin: origin.clone(),
                        kind: THINKING_OPAQUE_KIND.to_string(),
                        data: sig.to_string(),
                        item_id: None,
                    });
                if !thinking.is_empty() || opaque.is_some() {
                    blocks.push(Block::Reasoning(ReasoningBlock {
                        text: thinking.clone(),
                        opaque,
                    }));
                }
            }
            wire::ContentBlock::RedactedThinking { data } => {
                blocks.push(Block::Opaque(Opaque {
                    origin: origin.clone(),
                    kind: REDACTED_THINKING_OPAQUE_KIND.to_string(),
                    data: data.clone(),
                    item_id: None,
                }));
            }
            wire::ContentBlock::ToolUse { id, name, input } => {
                blocks.push(Block::ToolCall(decode_tool_call(id, name, input)?));
            }
            wire::ContentBlock::Unknown => {}
        }
    }

    let stop_reason = response.stop_reason.as_deref().map(stop_reason);
    let usage = decode_usage(response.usage.as_ref());

    let mut events = Vec::new();
    events.push(StreamEvent::MessageStart {
        response_id: response.id.clone(),
    });
    for (index, block) in blocks.into_iter().enumerate() {
        events.push(StreamEvent::CompleteBlock { index, block });
    }
    if let Some(usage) = usage {
        events.push(StreamEvent::Usage(usage));
    }
    events.push(StreamEvent::Finish { stop_reason });
    Ok(events)
}

fn decode_tool_call(id: &str, name: &str, input: &Value) -> Result<ToolCallBlock, HarnessError> {
    let raw_args = serde_json::to_string(input).map_err(|error| {
        HarnessError::internal(format!("could not serialize tool arguments: {error}"))
    })?;
    let (args, args_error) = if input.is_object() {
        (input.clone(), None)
    } else {
        (
            Value::Null,
            Some("tool arguments must be a JSON object".to_string()),
        )
    };

    Ok(ToolCallBlock {
        id: id.to_string(),
        name: name.to_string(),
        args,
        raw_args: Some(raw_args),
        args_error,
        provider_item_id: None,
        opaque: None,
    })
}

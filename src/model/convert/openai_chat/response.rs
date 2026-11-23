use serde_json::Value;

use crate::error::HarnessError;
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::convert::openai_chat::{OpenAiChatCompat, decode_usage, stop_reason};
use crate::model::convert::util::{as_object, required_str};
use crate::model::message::{Block, Message, Origin, ReasoningBlock, TextBlock, ToolCallBlock};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, StreamEvent};
use crate::model::wire::openai_chat as wire;

/// Decodes an OpenAI Chat Completions response body.
pub(super) fn decode_response(
    raw: &str,
    origin: &Origin,
    compat: &OpenAiChatCompat,
) -> Result<Message, HarnessError> {
    let response: wire::ChatResponse = serde_json::from_str(raw)
        .map_err(|error| decode_error("OpenAI Chat response", Protocol::OpenaiChat, error))?;
    if let Some(error) = &response.error {
        return Err(provider_error(
            Protocol::OpenaiChat,
            error.message.as_deref(),
            error.kind.as_deref(),
            error.code.as_ref(),
            error.status,
        ));
    }
    let events = decode_response_events(&response, compat)?;
    let mut assembler = Assembler::new(origin.clone());
    for event in events {
        assembler.push(event)?;
    }
    assembler.finish()
}

fn decode_response_events(
    response: &wire::ChatResponse,
    compat: &OpenAiChatCompat,
) -> Result<Vec<StreamEvent>, HarnessError> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| protocol("choices", "expected at least one choice"))?;
    let message = &choice.message;

    let mut blocks = Vec::new();

    if let Some(reasoning) = compat.reasoning_text(
        message.reasoning_content.as_deref(),
        message.reasoning.as_deref(),
        message.reasoning_text.as_deref(),
    ) {
        blocks.push(Block::Reasoning(ReasoningBlock {
            text: reasoning.to_string(),
            opaque: None,
        }));
    }

    if let Some(content) = &message.content {
        decode_content(content, &mut blocks)?;
    }

    if let Some(refusal) = message.refusal.as_deref().filter(|text| !text.is_empty()) {
        blocks.push(Block::Text(TextBlock {
            text: refusal.to_string(),
            opaque: None,
        }));
    }

    if let Some(tool_calls) = &message.tool_calls {
        for call in tool_calls {
            blocks.push(Block::ToolCall(decode_tool_call(call)?));
        }
    }

    let stop_reason = choice.finish_reason.as_deref().map(stop_reason);
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

fn decode_content(content: &Value, blocks: &mut Vec<Block>) -> Result<(), HarnessError> {
    match content {
        Value::Null => Ok(()),
        Value::String(text) => {
            if !text.is_empty() {
                blocks.push(Block::Text(TextBlock {
                    text: text.clone(),
                    opaque: None,
                }));
            }
            Ok(())
        }
        Value::Array(parts) => {
            for (index, part) in parts.iter().enumerate() {
                let path = format!("choices[0].message.content[{index}]");
                let part = as_object(part, Protocol::OpenaiChat, &path)?;
                let kind = required_str(part, "type", Protocol::OpenaiChat, &path)?;
                match kind {
                    "text" => {
                        let text = required_str(part, "text", Protocol::OpenaiChat, &path)?;
                        if !text.is_empty() {
                            blocks.push(Block::Text(TextBlock {
                                text: text.to_string(),
                                opaque: None,
                            }));
                        }
                    }
                    "refusal" => {
                        let refusal = required_str(part, "refusal", Protocol::OpenaiChat, &path)?;
                        blocks.push(Block::Text(TextBlock {
                            text: refusal.to_string(),
                            opaque: None,
                        }));
                    }
                    other => {
                        return Err(HarnessError::unsupported(
                            Protocol::OpenaiChat,
                            format!("content part `{other}`"),
                        ));
                    }
                }
            }
            Ok(())
        }
        _ => Err(protocol(
            "choices[0].message.content",
            "expected a string, an array, or null",
        )),
    }
}

fn decode_tool_call(call: &wire::ToolCall) -> Result<ToolCallBlock, HarnessError> {
    if call.kind.as_deref().unwrap_or("function") != "function" {
        return Err(HarnessError::unsupported(
            Protocol::OpenaiChat,
            format!("tool call type `{}`", call.kind.as_deref().unwrap_or("")),
        ));
    }

    let (args, raw_args, args_error) =
        crate::model::convert::util::decode_tool_arguments(&call.function.arguments)?;

    Ok(ToolCallBlock {
        id: call.id.clone(),
        name: call.function.name.clone(),
        args,
        raw_args,
        args_error,
        provider_item_id: None,
        opaque: None,
    })
}

fn protocol(path: impl Into<String>, message: impl Into<String>) -> HarnessError {
    HarnessError::protocol(Protocol::OpenaiChat, path, message)
}

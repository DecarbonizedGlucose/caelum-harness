use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, HarnessError};
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::convert::openai_responses::{
    MESSAGE_OPAQUE_KIND, MessageMeta, REASONING_OPAQUE_KIND, decode_usage, terminal_stop_reason,
};
use crate::model::convert::util::decode_tool_arguments;
use crate::model::message::{
    Block, Message, Opaque, Origin, ReasoningBlock, TextBlock, ToolCallBlock,
};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, StreamEvent};
use crate::model::wire::openai_responses as wire;

/// Decodes an OpenAI Responses response body.
pub(super) fn decode_response(raw: &str, origin: &Origin) -> Result<Message, HarnessError> {
    let response: wire::Response = serde_json::from_str(raw)
        .map_err(|error| decode_error("OpenAI Responses response", Protocol::OpenaiResp, error))?;

    if let Some(error) = &response.error {
        return Err(provider_error(
            Protocol::OpenaiResp,
            error.message.as_deref(),
            error.kind.as_deref(),
            error.code.as_ref(),
            error.status,
        ));
    }
    if response.status.as_deref() == Some("failed") {
        return Err(HarnessError::new(
            ErrorCode::Server,
            format!("{}: response failed", Protocol::OpenaiResp),
        ));
    }

    let mut blocks = Vec::new();
    for (index, item) in response.output.iter().enumerate() {
        if let Some(block) = decode_output_item(item, origin, index)? {
            blocks.push(block);
        }
    }

    let has_tool_calls = blocks
        .iter()
        .any(|block| matches!(block, Block::ToolCall(_)));
    let stop_reason = terminal_stop_reason(&response, has_tool_calls);
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

    let mut assembler = Assembler::new(origin.clone());
    for event in events {
        assembler.push(event)?;
    }
    assembler.finish()
}

/// Converts one output item into a provider-neutral block.
pub(super) fn decode_output_item(
    item: &wire::OutputItem,
    origin: &Origin,
    index: usize,
) -> Result<Option<Block>, HarnessError> {
    match item {
        wire::OutputItem::Message {
            id,
            status,
            phase,
            content,
        } => {
            let mut text = String::new();
            for part in content {
                match part {
                    wire::ContentPart::OutputText { text: value } => text.push_str(value),
                    wire::ContentPart::Refusal { refusal } => text.push_str(refusal),
                    wire::ContentPart::Unknown => {}
                }
            }
            if text.is_empty() {
                return Ok(None);
            }
            let opaque = id
                .as_ref()
                .filter(|id| !id.is_empty() && id.len() <= 64)
                .map(|id| Opaque {
                    origin: origin.clone(),
                    kind: MESSAGE_OPAQUE_KIND.to_string(),
                    data: serde_json::to_string(&MessageMeta {
                        id: id.clone(),
                        status: status.clone(),
                        phase: phase.clone(),
                    })
                    .unwrap_or_default(),
                    item_id: Some(id.clone()),
                });
            Ok(Some(Block::Text(TextBlock { text, opaque })))
        }
        wire::OutputItem::Reasoning {
            id,
            status,
            content,
            summary,
            encrypted_content,
        } => {
            let mut text = content
                .iter()
                .filter_map(|part| match part {
                    wire::ReasoningPart::ReasoningText { text } => Some(text.as_str()),
                    wire::ReasoningPart::Unknown => None,
                })
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                text = summary
                    .iter()
                    .filter_map(|part| match part {
                        wire::SummaryPart::SummaryText { text } => Some(text.as_str()),
                        wire::SummaryPart::Unknown => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
            }

            let opaque = encrypted_content
                .as_deref()
                .filter(|encrypted| !encrypted.is_empty())
                .map(|encrypted| Opaque {
                    origin: origin.clone(),
                    kind: REASONING_OPAQUE_KIND.to_string(),
                    data: reasoning_item_json(id, status, content, summary, encrypted).to_string(),
                    item_id: id.clone(),
                });

            if text.is_empty() && opaque.is_none() {
                return Ok(None);
            }
            Ok(Some(Block::Reasoning(ReasoningBlock { text, opaque })))
        }
        wire::OutputItem::FunctionCall {
            id,
            status,
            call_id,
            name,
            arguments,
        } => {
            let (args, raw_args, mut args_error) = decode_tool_arguments(arguments)?;
            if args_error.is_none()
                && let Some(status) = status.as_deref()
                && status != "completed"
            {
                args_error = Some(format!("function call status was {status}"));
            }

            Ok(Some(Block::ToolCall(ToolCallBlock {
                id: call_id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("call-{index}")),
                name: name.clone().unwrap_or_default(),
                args,
                raw_args,
                args_error,
                provider_item_id: id.clone(),
                opaque: None,
            })))
        }
        wire::OutputItem::Unknown => Ok(None),
    }
}

fn reasoning_item_json(
    id: &Option<String>,
    status: &Option<String>,
    content: &[wire::ReasoningPart],
    summary: &[wire::SummaryPart],
    encrypted_content: &str,
) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), json!("reasoning"));
    if let Some(id) = id {
        object.insert("id".to_string(), json!(id));
    }
    if let Some(status) = status {
        object.insert("status".to_string(), json!(status));
    }
    let content = content
        .iter()
        .filter_map(|part| match part {
            wire::ReasoningPart::ReasoningText { text } => {
                Some(json!({ "type": "reasoning_text", "text": text }))
            }
            wire::ReasoningPart::Unknown => None,
        })
        .collect::<Vec<_>>();
    if !content.is_empty() {
        object.insert("content".to_string(), Value::Array(content));
    }
    let summary = summary
        .iter()
        .filter_map(|part| match part {
            wire::SummaryPart::SummaryText { text } => {
                Some(json!({ "type": "summary_text", "text": text }))
            }
            wire::SummaryPart::Unknown => None,
        })
        .collect::<Vec<_>>();
    if !summary.is_empty() {
        object.insert("summary".to_string(), Value::Array(summary));
    }
    object.insert("encrypted_content".to_string(), json!(encrypted_content));
    Value::Object(object)
}

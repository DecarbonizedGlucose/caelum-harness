use serde_json::{Value, json};

use crate::error::HarnessError;
use crate::model::convert::openai_responses::{
    MESSAGE_OPAQUE_KIND, MessageMeta, OpenAiResponsesCompat, REASONING_OPAQUE_KIND,
};
use crate::model::convert::util::{data_url, encode_tool_arguments};
use crate::model::message::{Block, Message, MessageType, Opaque, Origin, ToolResultBlock};
use crate::model::protocol::{ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef};
use crate::model::wire::openai_responses as wire;

/// Encodes a provider-neutral request as an OpenAI Responses body.
pub(super) fn encode_request(
    origin: &Origin,
    request: &ChatRequest,
    compat: &OpenAiResponsesCompat,
    stream: bool,
) -> Result<Value, HarnessError> {
    if !request.stop.is_empty() {
        return Err(HarnessError::unsupported_option(
            Protocol::OpenaiResp,
            "stop",
        ));
    }

    let (instructions, input) = encode_messages(origin, &request.messages, compat)?;
    let tools =
        (!request.tools.is_empty()).then(|| request.tools.iter().map(encode_tool).collect());
    let tool_choice = (!request.tools.is_empty()).then(|| encode_tool_choice(&request.tool_choice));
    let reasoning = request.reasoning_effort.map(encode_reasoning);
    let include = compat
        .include_encrypted_reasoning
        .then(|| vec!["reasoning.encrypted_content"]);
    let drop_sampling = compat.drop_sampling_with_reasoning
        && reasoning.is_some()
        && request.reasoning_effort != Some(ReasoningEffort::None);

    let body = wire::ResponsesRequest {
        model: origin.model.clone(),
        input,
        instructions,
        temperature: (!drop_sampling).then_some(request.temperature).flatten(),
        top_p: (!drop_sampling).then_some(request.top_p).flatten(),
        max_output_tokens: request.max_tokens,
        tools,
        tool_choice,
        reasoning,
        include,
        stream: stream.then_some(true),
        store: false,
    };

    serde_json::to_value(body)
        .map_err(|error| HarnessError::internal(format!("could not encode request: {error}")))
}

fn encode_tool(tool: &ToolDef) -> wire::Tool {
    wire::Tool {
        kind: "function",
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({ "type": "function", "name": name }),
    }
}

fn encode_reasoning(effort: ReasoningEffort) -> wire::Reasoning {
    wire::Reasoning {
        effort: match effort {
            ReasoningEffort::None => "none",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::Max => "max",
        },
        summary: (effort != ReasoningEffort::None).then_some("auto"),
    }
}

fn encode_messages(
    origin: &Origin,
    messages: &[Message],
    compat: &OpenAiResponsesCompat,
) -> Result<(Option<String>, Vec<wire::InputItem>), HarnessError> {
    let mut instructions = Vec::new();
    let mut input = Vec::new();

    for message in messages {
        match message.mtype {
            MessageType::System => encode_system(message, &mut instructions)?,
            MessageType::User => encode_user(message, &mut input)?,
            MessageType::Assistant => encode_assistant(origin, message, compat, &mut input)?,
            MessageType::Tool => encode_tool_message(message, &mut input)?,
        }
    }

    let instructions = (!instructions.is_empty()).then(|| instructions.join("\n\n"));
    Ok((instructions, input))
}

fn encode_system(message: &Message, instructions: &mut Vec<String>) -> Result<(), HarnessError> {
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    instructions.push(text.text.clone());
                }
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiResp,
                    "non-text content in system messages",
                ));
            }
        }
    }
    Ok(())
}

fn encode_user(message: &Message, out: &mut Vec<wire::InputItem>) -> Result<(), HarnessError> {
    let mut content = Vec::new();
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    content.push(wire::InputContent::InputText {
                        text: text.text.clone(),
                    });
                }
            }
            Block::Image(image) => {
                content.push(wire::InputContent::InputImage {
                    image_url: data_url(&image.source),
                });
            }
            Block::Reasoning(_) | Block::Opaque(_) => {}
            Block::ToolCall(_) => {
                return Err(HarnessError::request(
                    Protocol::OpenaiResp,
                    "input",
                    "tool calls must use the assistant role",
                ));
            }
            Block::ToolResult(_) => {
                return Err(HarnessError::request(
                    Protocol::OpenaiResp,
                    "input",
                    "tool results must use the tool role",
                ));
            }
        }
    }

    if !content.is_empty() {
        out.push(wire::InputItem::Message {
            role: "user",
            content,
        });
    }
    Ok(())
}

fn encode_assistant(
    origin: &Origin,
    message: &Message,
    compat: &OpenAiResponsesCompat,
    out: &mut Vec<wire::InputItem>,
) -> Result<(), HarnessError> {
    let mut pending_text = String::new();

    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if let Some(meta) = message_meta(text.opaque.as_ref(), origin) {
                    flush_text(&mut pending_text, out);
                    append_assistant_message(out, meta, &text.text);
                } else if !text.text.is_empty() {
                    pending_text.push_str(&text.text);
                }
            }
            Block::Reasoning(reasoning) => {
                if compat.replay_reasoning_items
                    && let Some(item) = reasoning_item(reasoning.opaque.as_ref(), origin)
                {
                    flush_text(&mut pending_text, out);
                    out.push(wire::InputItem::RawItem(item));
                }
            }
            Block::ToolCall(call) => {
                flush_text(&mut pending_text, out);
                let arguments = encode_tool_arguments(
                    &call.args,
                    call.raw_args.as_deref(),
                    call.args_error.as_deref(),
                )?;
                let id = call
                    .provider_item_id
                    .as_deref()
                    .filter(|id| id.starts_with("fc_") && id.len() <= 64)
                    .map(ToString::to_string);
                out.push(wire::InputItem::FunctionCall {
                    kind: "function_call",
                    id,
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    arguments,
                });
            }
            Block::Image(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiResp,
                    "image content in assistant messages",
                ));
            }
            Block::ToolResult(_) => {
                return Err(HarnessError::request(
                    Protocol::OpenaiResp,
                    "input",
                    "tool results must use the tool role",
                ));
            }
            Block::Opaque(_) => {}
        }
    }

    flush_text(&mut pending_text, out);
    Ok(())
}

fn flush_text(pending: &mut String, out: &mut Vec<wire::InputItem>) {
    if pending.is_empty() {
        return;
    }
    out.push(wire::InputItem::Message {
        role: "assistant",
        content: vec![wire::InputContent::OutputText {
            text: std::mem::take(pending),
        }],
    });
}

fn append_assistant_message(out: &mut Vec<wire::InputItem>, meta: MessageMeta, text: &str) {
    let part = wire::InputContent::OutputText {
        text: text.to_string(),
    };

    if let Some(wire::InputItem::AssistantMessage { id, content, .. }) = out.last_mut()
        && *id == meta.id
    {
        content.push(part);
        return;
    }

    out.push(wire::InputItem::AssistantMessage {
        kind: "message",
        role: "assistant",
        id: meta.id,
        status: meta.status.unwrap_or_else(|| "completed".to_string()),
        phase: meta.phase,
        content: vec![part],
    });
}

fn encode_tool_message(
    message: &Message,
    out: &mut Vec<wire::InputItem>,
) -> Result<(), HarnessError> {
    for block in &message.blocks {
        let Block::ToolResult(result) = block else {
            return Err(HarnessError::request(
                Protocol::OpenaiResp,
                "input",
                "tool role contains a non-tool-result block",
            ));
        };
        out.push(wire::InputItem::FunctionCallOutput {
            kind: "function_call_output",
            call_id: result.call_id.clone(),
            output: encode_tool_output(result),
        });
    }
    Ok(())
}

fn encode_tool_output(result: &ToolResultBlock) -> wire::FunctionOutput {
    let has_image = result
        .content
        .iter()
        .any(|block| matches!(block, Block::Image(_)));

    if has_image {
        let parts = result
            .content
            .iter()
            .filter_map(|block| match block {
                Block::Text(text) => Some(wire::InputContent::InputText {
                    text: text.text.clone(),
                }),
                Block::Image(image) => Some(wire::InputContent::InputImage {
                    image_url: data_url(&image.source),
                }),
                _ => None,
            })
            .collect();
        return wire::FunctionOutput::Content(parts);
    }

    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            Block::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    wire::FunctionOutput::Text(text)
}

fn message_meta(opaque: Option<&Opaque>, origin: &Origin) -> Option<MessageMeta> {
    let opaque = opaque?;
    if opaque.kind != MESSAGE_OPAQUE_KIND || !same_origin(opaque, origin) {
        return None;
    }
    serde_json::from_str(&opaque.data).ok()
}

fn reasoning_item(opaque: Option<&Opaque>, origin: &Origin) -> Option<Value> {
    let opaque = opaque?;
    if opaque.kind != REASONING_OPAQUE_KIND || !same_origin(opaque, origin) {
        return None;
    }
    let value: Value = serde_json::from_str(&opaque.data).ok()?;
    (value.get("type").and_then(Value::as_str) == Some("reasoning")).then_some(value)
}

fn same_origin(opaque: &Opaque, origin: &Origin) -> bool {
    opaque.origin.protocol == Protocol::OpenaiResp
        && opaque.origin.vendor == origin.vendor
        && opaque.origin.model == origin.model
}

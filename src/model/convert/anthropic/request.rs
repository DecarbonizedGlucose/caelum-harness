use serde_json::{Value, json};

use crate::error::HarnessError;
use crate::model::convert::util::encode_base64;
use crate::model::message::{Block, ImageSource, Message, MessageType, Opaque, ToolResultBlock};
use crate::model::protocol::{ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef};
use crate::model::wire::anthropic as wire;

const THINKING_OPAQUE_KIND: &str = "anthropic.thinking";
const REDACTED_THINKING_OPAQUE_KIND: &str = "anthropic.redacted_thinking";

/// Encodes a provider-neutral request as an Anthropic Messages body.
pub(super) fn encode_request(
    model: &str,
    default_max_tokens: u32,
    request: &ChatRequest,
    stream: bool,
) -> Result<Value, HarnessError> {
    let (system, messages) = encode_messages(&request.messages)?;
    let tools =
        (!request.tools.is_empty()).then(|| request.tools.iter().map(encode_tool).collect());
    let tool_choice = (!request.tools.is_empty())
        .then(|| encode_tool_choice(&request.tool_choice))
        .flatten();
    let thinking = request.reasoning_effort.and_then(encode_thinking);
    let mut max_tokens = request.max_tokens.unwrap_or(default_max_tokens);
    if let Some(config) = &thinking
        && max_tokens <= config.budget_tokens
    {
        max_tokens = config.budget_tokens + 4096;
    }
    let (temperature, top_p) = if thinking.is_some() {
        (None, None)
    } else {
        (request.temperature, request.top_p)
    };

    let body = wire::MessagesRequest {
        model: model.to_string(),
        max_tokens,
        system,
        messages,
        tools,
        tool_choice,
        temperature,
        top_p,
        stop_sequences: (!request.stop.is_empty()).then(|| request.stop.clone()),
        thinking,
        stream: stream.then_some(true),
    };

    serde_json::to_value(body)
        .map_err(|error| HarnessError::internal(format!("could not encode request: {error}")))
}

fn encode_tool(tool: &ToolDef) -> wire::Tool {
    wire::Tool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        input_schema: tool.parameters.clone(),
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(json!({ "type": "auto" })),
        ToolChoice::None => Some(json!({ "type": "none" })),
        ToolChoice::Required => Some(json!({ "type": "any" })),
        ToolChoice::Tool(name) => Some(json!({ "type": "tool", "name": name })),
    }
}

fn encode_thinking(effort: ReasoningEffort) -> Option<wire::ThinkingConfig> {
    let budget_tokens = match effort {
        ReasoningEffort::None => return None,
        ReasoningEffort::Low => 1024,
        ReasoningEffort::Medium => 4096,
        ReasoningEffort::High => 8192,
        ReasoningEffort::Max => 16384,
    };
    Some(wire::ThinkingConfig {
        kind: "enabled",
        budget_tokens,
    })
}

fn encode_messages(
    messages: &[Message],
) -> Result<(Option<String>, Vec<wire::RequestMessage>), HarnessError> {
    let mut system_parts = Vec::new();
    let mut out = Vec::new();

    for message in messages {
        match message.mtype {
            MessageType::System => encode_system(message, &mut system_parts)?,
            MessageType::User => encode_user(message, &mut out)?,
            MessageType::Assistant => encode_assistant(message, &mut out)?,
            MessageType::Tool => encode_tool_message(message, &mut out)?,
        }
    }

    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    Ok((system, out))
}

fn encode_system(message: &Message, system_parts: &mut Vec<String>) -> Result<(), HarnessError> {
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    system_parts.push(text.text.clone());
                }
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::Anthropic,
                    "non-text content in system messages",
                ));
            }
        }
    }
    Ok(())
}

fn encode_user(message: &Message, out: &mut Vec<wire::RequestMessage>) -> Result<(), HarnessError> {
    let mut blocks = Vec::new();
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    blocks.push(wire::RequestContent::Text {
                        text: text.text.clone(),
                    });
                }
            }
            Block::Image(image) => {
                blocks.push(wire::RequestContent::Image {
                    source: encode_image(&image.source),
                });
            }
            Block::ToolResult(result) => {
                blocks.push(encode_tool_result_block(result)?);
            }
            Block::Reasoning(_) | Block::Opaque(_) => {}
            Block::ToolCall(_) => {
                return Err(HarnessError::request(
                    Protocol::Anthropic,
                    "messages",
                    "tool calls must use the assistant role",
                ));
            }
        }
    }

    push_role_blocks(out, "user", blocks);
    Ok(())
}

fn encode_assistant(
    message: &Message,
    out: &mut Vec<wire::RequestMessage>,
) -> Result<(), HarnessError> {
    let mut blocks = Vec::new();
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    blocks.push(wire::RequestContent::Text {
                        text: text.text.clone(),
                    });
                }
            }
            Block::Reasoning(reasoning) => {
                if let Some(opaque) = reasoning
                    .opaque
                    .as_ref()
                    .filter(|opaque| is_anthropic(opaque))
                {
                    if opaque.kind == THINKING_OPAQUE_KIND && !opaque.data.is_empty() {
                        blocks.push(wire::RequestContent::Thinking {
                            thinking: reasoning.text.clone(),
                            signature: opaque.data.clone(),
                        });
                    } else if opaque.kind == REDACTED_THINKING_OPAQUE_KIND {
                        blocks.push(wire::RequestContent::RedactedThinking {
                            data: opaque.data.clone(),
                        });
                    }
                }
            }
            Block::Opaque(opaque) => {
                if is_anthropic(opaque) && opaque.kind == REDACTED_THINKING_OPAQUE_KIND {
                    blocks.push(wire::RequestContent::RedactedThinking {
                        data: opaque.data.clone(),
                    });
                }
            }
            Block::ToolCall(call) => {
                let input = if call.args_error.is_none() && call.args.is_object() {
                    call.args.clone()
                } else {
                    json!({})
                };
                blocks.push(wire::RequestContent::ToolUse {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input,
                });
            }
            Block::Image(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::Anthropic,
                    "image content in assistant messages",
                ));
            }
            Block::ToolResult(_) => {
                return Err(HarnessError::request(
                    Protocol::Anthropic,
                    "messages",
                    "tool results must use the tool role",
                ));
            }
        }
    }

    push_role_blocks(out, "assistant", blocks);
    Ok(())
}

fn encode_tool_message(
    message: &Message,
    out: &mut Vec<wire::RequestMessage>,
) -> Result<(), HarnessError> {
    let mut blocks = Vec::new();
    for block in &message.blocks {
        let Block::ToolResult(result) = block else {
            return Err(HarnessError::request(
                Protocol::Anthropic,
                "messages",
                "tool role contains a non-tool-result block",
            ));
        };
        blocks.push(encode_tool_result_block(result)?);
    }

    push_role_blocks(out, "user", blocks);
    Ok(())
}

fn encode_tool_result_block(
    result: &ToolResultBlock,
) -> Result<wire::RequestContent, HarnessError> {
    let mut content = Vec::new();
    for block in &result.content {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    content.push(wire::RequestToolResultContent::Text {
                        text: text.text.clone(),
                    });
                }
            }
            Block::Image(image) => {
                content.push(wire::RequestToolResultContent::Image {
                    source: encode_image(&image.source),
                });
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::Anthropic,
                    "non-text content in tool results",
                ));
            }
        }
    }

    Ok(wire::RequestContent::ToolResult {
        tool_use_id: result.call_id.clone(),
        content,
        is_error: result.is_error,
    })
}

fn encode_image(source: &ImageSource) -> wire::ImageSource {
    match source {
        ImageSource::Base64 { mime, data } => wire::ImageSource::Base64 {
            media_type: mime.clone(),
            data: encode_base64(data),
        },
        ImageSource::Url(url) => wire::ImageSource::Url { url: url.clone() },
    }
}

fn push_role_blocks(
    out: &mut Vec<wire::RequestMessage>,
    role: &'static str,
    mut blocks: Vec<wire::RequestContent>,
) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = out.last_mut()
        && last.role == role
    {
        last.content.append(&mut blocks);
        return;
    }
    out.push(wire::RequestMessage {
        role,
        content: blocks,
    });
}

fn is_anthropic(opaque: &Opaque) -> bool {
    opaque.origin.protocol == Protocol::Anthropic
}

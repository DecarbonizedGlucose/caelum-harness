use serde_json::{Map, Value, json};

use crate::error::HarnessError;
use crate::model::convert::openai_chat::{MaxTokensField, OpenAiChatCompat, ReasoningField};
use crate::model::convert::util::data_url;
use crate::model::message::{Block, Message, MessageType, ToolResultBlock};
use crate::model::protocol::{ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef};
use crate::model::wire::openai_chat as wire;

/// Encodes a provider-neutral request as an OpenAI Chat Completions body.
pub(super) fn encode_request(
    model: &str,
    request: &ChatRequest,
    compat: &OpenAiChatCompat,
    stream: bool,
) -> Result<Value, HarnessError> {
    let messages = encode_messages(&request.messages, compat)?;
    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(request.tools.iter().map(encode_tool).collect())
    };
    let tool_choice = tools
        .as_ref()
        .map(|_| encode_tool_choice(&request.tool_choice));
    let (max_tokens, max_completion_tokens) = match compat.max_tokens_field {
        MaxTokensField::MaxTokens => (request.max_tokens, None),
        MaxTokensField::MaxCompletionTokens => (None, request.max_tokens),
    };

    let body = wire::ChatRequest {
        model: model.to_string(),
        messages,
        tools,
        tool_choice,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens,
        max_completion_tokens,
        stop: (!request.stop.is_empty()).then(|| request.stop.clone()),
        reasoning_effort: request
            .reasoning_effort
            .map(|effort| reasoning_effort(effort).to_string()),
        stream: stream.then_some(true),
        stream_options: stream.then_some(wire::StreamOptions {
            include_usage: true,
        }),
    };

    serde_json::to_value(body)
        .map_err(|error| HarnessError::internal(format!("could not encode request: {error}")))
}

fn encode_tool(tool: &ToolDef) -> wire::Tool {
    wire::Tool {
        kind: "function",
        function: wire::ToolFunction {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        },
    }
}

fn encode_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_string()),
        ToolChoice::None => Value::String("none".to_string()),
        ToolChoice::Required => Value::String("required".to_string()),
        ToolChoice::Tool(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

fn reasoning_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Max => "max",
    }
}

fn reasoning_field(field: ReasoningField) -> &'static str {
    match field {
        ReasoningField::ReasoningContent => "reasoning_content",
        ReasoningField::Reasoning => "reasoning",
        ReasoningField::ReasoningText => "reasoning_text",
    }
}

fn encode_messages(
    messages: &[Message],
    compat: &OpenAiChatCompat,
) -> Result<Vec<wire::RequestMessage>, HarnessError> {
    let mut out = Vec::new();
    for message in messages {
        match message.mtype {
            MessageType::System => encode_system(message, compat, &mut out)?,
            MessageType::User => encode_user(message, &mut out)?,
            MessageType::Assistant => encode_assistant(message, compat, &mut out)?,
            MessageType::Tool => encode_tool_results(message, &mut out)?,
        }
    }
    Ok(out)
}

fn encode_system(
    message: &Message,
    compat: &OpenAiChatCompat,
    out: &mut Vec<wire::RequestMessage>,
) -> Result<(), HarnessError> {
    let text = only_text(&message.blocks, "non-text content in system messages")?;
    if !text.is_empty() {
        out.push(wire::RequestMessage {
            role: if compat.developer_role {
                "developer".to_string()
            } else {
                "system".to_string()
            },
            content: Value::String(text),
            tool_calls: None,
            tool_call_id: None,
            extra: Map::new(),
        });
    }
    Ok(())
}

fn encode_user(message: &Message, out: &mut Vec<wire::RequestMessage>) -> Result<(), HarnessError> {
    let mut text = String::new();
    let mut parts = Vec::new();
    let mut has_image = false;

    for block in &message.blocks {
        match block {
            Block::Text(block) => {
                text.push_str(&block.text);
                parts.push(json!({ "type": "text", "text": block.text }));
            }
            Block::Image(block) => {
                has_image = true;
                parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": data_url(&block.source) }
                }));
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiChat,
                    "non-text content in user messages",
                ));
            }
        }
    }

    let content = if has_image {
        Value::Array(parts)
    } else if !text.is_empty() {
        Value::String(text)
    } else {
        return Ok(());
    };

    out.push(wire::RequestMessage {
        role: "user".to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        extra: Map::new(),
    });
    Ok(())
}

fn encode_assistant(
    message: &Message,
    compat: &OpenAiChatCompat,
    out: &mut Vec<wire::RequestMessage>,
) -> Result<(), HarnessError> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut extra = Map::new();

    for block in &message.blocks {
        match block {
            Block::Text(block) => text.push_str(&block.text),
            Block::Reasoning(block) => {
                if compat.replay_reasoning
                    && !block.text.is_empty()
                    && let Some(field) = compat.reasoning_field
                {
                    extra.insert(
                        reasoning_field(field).to_string(),
                        Value::String(block.text.clone()),
                    );
                }
            }
            Block::ToolCall(call) => {
                let arguments = if call.args_error.is_some() {
                    "{}".to_string()
                } else {
                    match &call.raw_args {
                        Some(raw) => raw.clone(),
                        None => serde_json::to_string(&call.args).map_err(|error| {
                            HarnessError::internal(format!(
                                "could not serialize tool arguments: {error}"
                            ))
                        })?,
                    }
                };
                tool_calls.push(wire::RequestToolCall {
                    id: call.id.clone(),
                    kind: "function",
                    function: wire::RequestFunction {
                        name: call.name.clone(),
                        arguments,
                    },
                });
            }
            Block::Image(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiChat,
                    "image content in assistant messages",
                ));
            }
            Block::ToolResult(_) => {
                return Err(HarnessError::request(
                    Protocol::OpenaiChat,
                    "messages",
                    "tool results must use the tool role",
                ));
            }
            Block::Opaque(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiChat,
                    "opaque content in assistant messages",
                ));
            }
        }
    }

    if text.is_empty() && tool_calls.is_empty() {
        return Ok(());
    }

    out.push(wire::RequestMessage {
        role: "assistant".to_string(),
        content: if text.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        },
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
        extra,
    });
    Ok(())
}

fn encode_tool_results(
    message: &Message,
    out: &mut Vec<wire::RequestMessage>,
) -> Result<(), HarnessError> {
    for block in &message.blocks {
        match block {
            Block::ToolResult(result) => {
                let text = tool_result_text(result)?;
                out.push(wire::RequestMessage {
                    role: "tool".to_string(),
                    content: Value::String(text),
                    tool_calls: None,
                    tool_call_id: Some(result.call_id.clone()),
                    extra: Map::new(),
                });
            }
            _ => {
                return Err(HarnessError::request(
                    Protocol::OpenaiChat,
                    "messages",
                    "tool role contains a non-tool-result block",
                ));
            }
        }
    }
    Ok(())
}

fn only_text(blocks: &[Block], feature: &str) -> Result<String, HarnessError> {
    let mut text = String::new();
    for block in blocks {
        match block {
            Block::Text(block) => text.push_str(&block.text),
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiChat,
                    feature.to_string(),
                ));
            }
        }
    }
    Ok(text)
}

fn tool_result_text(result: &ToolResultBlock) -> Result<String, HarnessError> {
    let mut text = String::new();
    for block in &result.content {
        match block {
            Block::Text(block) => text.push_str(&block.text),
            Block::Image(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiChat,
                    "image content in tool results",
                ));
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::OpenaiChat,
                    "non-text content in tool results",
                ));
            }
        }
    }
    Ok(text)
}

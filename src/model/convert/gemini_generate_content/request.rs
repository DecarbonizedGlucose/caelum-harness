use std::collections::HashMap;

use serde_json::{Value, json};

use crate::error::HarnessError;
use crate::model::convert::gemini_generate_content::{
    GeminiGenerateContentCompat, THOUGHT_SIGNATURE_KIND, ThinkingMode,
};
use crate::model::convert::util::encode_base64;
use crate::model::message::{
    Block, ImageSource, Message, MessageType, Opaque, Origin, ToolResultBlock,
};
use crate::model::protocol::{ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef};
use crate::model::wire::gemini_generate_content as wire;

const MAX_STOP_SEQUENCES: usize = 5;

/// Encodes a provider-neutral request as a Gemini GenerateContent body.
pub(super) fn encode_request(
    origin: &Origin,
    request: &ChatRequest,
    compat: &GeminiGenerateContentCompat,
) -> Result<Value, HarnessError> {
    if request.stop.len() > MAX_STOP_SEQUENCES {
        return Err(HarnessError::unsupported_option(
            Protocol::GeminiGenerateContent,
            "stop (maximum 5 sequences)",
        ));
    }

    for tool in &request.tools {
        validate_tool_schema(&tool.parameters)?;
    }

    let (system_instruction, contents) = encode_messages(origin, &request.messages, compat)?;
    let tools = (!request.tools.is_empty()).then(|| {
        vec![wire::Tool {
            function_declarations: request.tools.iter().map(encode_tool).collect(),
        }]
    });
    let tool_config = (!request.tools.is_empty()).then(|| encode_tool_config(&request.tool_choice));
    let thinking_config = request
        .reasoning_effort
        .and_then(|effort| encode_thinking_config(effort, compat.thinking_mode));

    let generation_config = Some(wire::GenerationConfig {
        max_output_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences: (!request.stop.is_empty()).then(|| request.stop.clone()),
        candidate_count: Some(1),
        thinking_config,
    });

    let body = wire::GenerateContentRequest {
        contents,
        system_instruction,
        tools,
        tool_config,
        generation_config,
    };

    serde_json::to_value(body)
        .map_err(|error| HarnessError::internal(format!("could not encode request: {error}")))
}

fn validate_tool_schema(schema: &Value) -> Result<(), HarnessError> {
    const UNSUPPORTED: &[&str] = &[
        "$schema",
        "additionalProperties",
        "$id",
        "definitions",
        "$defs",
        "oneOf",
    ];
    match schema {
        Value::Object(map) => {
            for key in UNSUPPORTED {
                if map.contains_key(*key) {
                    return Err(HarnessError::unsupported_option(
                        Protocol::GeminiGenerateContent,
                        format!("tool schema keyword `{key}`"),
                    ));
                }
            }
            for value in map.values() {
                validate_tool_schema(value)?;
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                validate_tool_schema(item)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn encode_tool(tool: &ToolDef) -> wire::FunctionDeclaration {
    wire::FunctionDeclaration {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
    }
}

fn encode_tool_config(choice: &ToolChoice) -> wire::ToolConfig {
    let function_calling_config = match choice {
        ToolChoice::Auto => wire::FunctionCallingConfig {
            mode: "AUTO",
            allowed_function_names: None,
        },
        ToolChoice::None => wire::FunctionCallingConfig {
            mode: "NONE",
            allowed_function_names: None,
        },
        ToolChoice::Required => wire::FunctionCallingConfig {
            mode: "ANY",
            allowed_function_names: None,
        },
        ToolChoice::Tool(name) => wire::FunctionCallingConfig {
            mode: "ANY",
            allowed_function_names: Some(vec![name.clone()]),
        },
    };
    wire::ToolConfig {
        function_calling_config,
    }
}

fn encode_thinking_config(
    effort: ReasoningEffort,
    mode: ThinkingMode,
) -> Option<wire::ThinkingConfig> {
    let include_thoughts = effort != ReasoningEffort::None;
    match mode {
        ThinkingMode::None => None,
        ThinkingMode::Budget => {
            let thinking_budget = match effort {
                ReasoningEffort::None => 0,
                ReasoningEffort::Low => 1024,
                ReasoningEffort::Medium => 4096,
                ReasoningEffort::High => 8192,
                ReasoningEffort::Max => 24576,
            };
            Some(wire::ThinkingConfig {
                include_thoughts,
                thinking_budget: Some(thinking_budget),
                thinking_level: None,
            })
        }
        ThinkingMode::Level => {
            let thinking_level = match effort {
                ReasoningEffort::None => "MINIMAL",
                ReasoningEffort::Low => "LOW",
                ReasoningEffort::Medium => "MEDIUM",
                ReasoningEffort::High | ReasoningEffort::Max => "HIGH",
            };
            Some(wire::ThinkingConfig {
                include_thoughts,
                thinking_budget: None,
                thinking_level: Some(thinking_level),
            })
        }
    }
}

fn encode_messages(
    origin: &Origin,
    messages: &[Message],
    compat: &GeminiGenerateContentCompat,
) -> Result<(Option<wire::RequestContent>, Vec<wire::RequestContent>), HarnessError> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut pending_tool_results = Vec::new();
    let mut call_order = HashMap::new();
    let mut next_call_ordinal = 0usize;

    for message in messages {
        if message.mtype == MessageType::Tool {
            for block in &message.blocks {
                let Block::ToolResult(result) = block else {
                    return Err(HarnessError::request(
                        Protocol::GeminiGenerateContent,
                        "contents",
                        "tool role contains a non-tool-result block",
                    ));
                };
                pending_tool_results.push(result);
            }
            continue;
        }

        flush_tool_results(&mut contents, &mut pending_tool_results, &call_order)?;

        match message.mtype {
            MessageType::System => encode_system(message, &mut system_parts)?,
            MessageType::User => encode_user(message, &mut contents)?,
            MessageType::Assistant => {
                for call in message.tool_calls() {
                    call_order.insert(call.id.as_str(), next_call_ordinal);
                    next_call_ordinal += 1;
                }
                encode_assistant(origin, message, compat, &mut contents)?;
            }
            MessageType::Tool => unreachable!(),
        }
    }

    flush_tool_results(&mut contents, &mut pending_tool_results, &call_order)?;

    let system_instruction = (!system_parts.is_empty()).then_some(wire::RequestContent {
        role: None,
        parts: system_parts,
    });
    Ok((system_instruction, contents))
}

fn flush_tool_results(
    contents: &mut Vec<wire::RequestContent>,
    results: &mut Vec<&ToolResultBlock>,
    call_order: &HashMap<&str, usize>,
) -> Result<(), HarnessError> {
    if results.is_empty() {
        return Ok(());
    }

    let mut ordered: Vec<(usize, &ToolResultBlock)> = results
        .iter()
        .filter_map(|result| {
            call_order
                .get(result.call_id.as_str())
                .map(|&ordinal| (ordinal, *result))
        })
        .collect();
    ordered.sort_by_key(|(ordinal, _)| *ordinal);
    for (slot, (_, result)) in results
        .iter_mut()
        .filter(|result| call_order.contains_key(result.call_id.as_str()))
        .zip(ordered)
    {
        *slot = result;
    }

    let mut parts = Vec::with_capacity(results.len());
    for result in results.drain(..) {
        parts.push(encode_tool_result(result)?);
    }
    push_content(contents, "user", parts);
    Ok(())
}

fn encode_system(
    message: &Message,
    parts: &mut Vec<wire::RequestPart>,
) -> Result<(), HarnessError> {
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    parts.push(wire::RequestPart::Text {
                        text: text.text.clone(),
                        thought_signature: None,
                    });
                }
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::GeminiGenerateContent,
                    "non-text content in system messages",
                ));
            }
        }
    }
    Ok(())
}

fn encode_user(
    message: &Message,
    contents: &mut Vec<wire::RequestContent>,
) -> Result<(), HarnessError> {
    let mut parts = Vec::new();
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                if !text.text.is_empty() {
                    parts.push(wire::RequestPart::Text {
                        text: text.text.clone(),
                        thought_signature: None,
                    });
                }
            }
            Block::Image(image) => {
                parts.push(wire::RequestPart::InlineData {
                    inline_data: encode_image(&image.source)?,
                });
            }
            Block::ToolResult(result) => {
                parts.push(encode_tool_result(result)?);
            }
            Block::Reasoning(_) | Block::Opaque(_) => {}
            Block::ToolCall(_) => {
                return Err(HarnessError::request(
                    Protocol::GeminiGenerateContent,
                    "contents",
                    "tool calls must use the assistant role",
                ));
            }
        }
    }

    push_content(contents, "user", parts);
    Ok(())
}

fn encode_assistant(
    origin: &Origin,
    message: &Message,
    compat: &GeminiGenerateContentCompat,
    contents: &mut Vec<wire::RequestContent>,
) -> Result<(), HarnessError> {
    let mut parts = Vec::new();
    for block in &message.blocks {
        match block {
            Block::Text(text) => {
                let sig = replay_signature(text.opaque.as_ref(), origin, compat);
                if !text.text.is_empty() || sig.is_some() {
                    parts.push(wire::RequestPart::Text {
                        text: text.text.clone(),
                        thought_signature: sig,
                    });
                }
            }
            Block::Reasoning(reasoning) => {
                if let Some(sig) = replay_signature(reasoning.opaque.as_ref(), origin, compat) {
                    parts.push(wire::RequestPart::Thought {
                        text: reasoning.text.clone(),
                        thought: true,
                        thought_signature: Some(sig),
                    });
                }
            }
            Block::ToolCall(call) => {
                let args = if call.args_error.is_none() && call.args.is_object() {
                    call.args.clone()
                } else {
                    json!({})
                };
                let sig = replay_signature(call.opaque.as_ref(), origin, compat);
                parts.push(wire::RequestPart::FunctionCall {
                    function_call: wire::RequestFunctionCall {
                        name: call.name.clone(),
                        args,
                    },
                    thought_signature: sig,
                });
            }
            Block::Image(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::GeminiGenerateContent,
                    "image content in assistant messages",
                ));
            }
            Block::ToolResult(_) => {
                return Err(HarnessError::request(
                    Protocol::GeminiGenerateContent,
                    "contents",
                    "tool results must use the tool role",
                ));
            }
            Block::Opaque(_) => {}
        }
    }

    push_content(contents, "model", parts);
    Ok(())
}

fn encode_tool_result(result: &ToolResultBlock) -> Result<wire::RequestPart, HarnessError> {
    let mut text_parts = Vec::new();
    for block in &result.content {
        match block {
            Block::Text(text) => text_parts.push(text.text.as_str()),
            Block::Image(_) => {
                return Err(HarnessError::unsupported(
                    Protocol::GeminiGenerateContent,
                    "image content in tool results",
                ));
            }
            _ => {
                return Err(HarnessError::unsupported(
                    Protocol::GeminiGenerateContent,
                    "non-text content in tool results",
                ));
            }
        }
    }

    let joined = text_parts.join("\n");
    let response = if result.is_error {
        json!({ "error": joined })
    } else {
        json!({ "result": joined })
    };

    Ok(wire::RequestPart::FunctionResponse {
        function_response: wire::FunctionResponse {
            name: result.name.clone(),
            response,
        },
    })
}

fn encode_image(source: &ImageSource) -> Result<wire::Blob, HarnessError> {
    match source {
        ImageSource::Base64 { mime, data } => Ok(wire::Blob {
            mime_type: mime.clone(),
            data: encode_base64(data),
        }),
        ImageSource::Url(_) => Err(HarnessError::unsupported(
            Protocol::GeminiGenerateContent,
            "image URLs in inlineData",
        )),
    }
}

fn push_content(
    contents: &mut Vec<wire::RequestContent>,
    role: &'static str,
    mut parts: Vec<wire::RequestPart>,
) {
    if parts.is_empty() {
        return;
    }
    if let Some(last) = contents.last_mut()
        && last.role == Some(role)
    {
        last.parts.append(&mut parts);
        return;
    }
    contents.push(wire::RequestContent {
        role: Some(role),
        parts,
    });
}

fn replay_signature(
    opaque: Option<&Opaque>,
    origin: &Origin,
    compat: &GeminiGenerateContentCompat,
) -> Option<String> {
    if !compat.replay_thought_signatures {
        return None;
    }
    let opaque = opaque?;
    if opaque.kind == THOUGHT_SIGNATURE_KIND
        && opaque.origin.protocol == Protocol::GeminiGenerateContent
        && opaque.origin.vendor == origin.vendor
        && opaque.origin.model == origin.model
        && !opaque.data.is_empty()
    {
        Some(opaque.data.clone())
    } else {
        None
    }
}

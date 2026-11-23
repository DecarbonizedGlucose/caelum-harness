use caelum_harness::error::ErrorCode;
use caelum_harness::model::convert::ProtocolCodec;
use caelum_harness::model::convert::gemini_generate_content::{
    GeminiGenerateContentCodec, GeminiGenerateContentCompat, ThinkingMode,
};
use caelum_harness::model::message::{
    AssistantMeta, Block, ImageBlock, ImageSource, Message, MessageType, Opaque, Origin,
    ReasoningBlock, StopReason, TextBlock, ToolCallBlock, ToolResultBlock, Usage,
};
use caelum_harness::model::protocol::{
    ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef,
};
use caelum_harness::model::stream::StreamEvent;
use serde_json::{Value, json};

fn origin(model: &str) -> Origin {
    Origin {
        protocol: Protocol::GeminiGenerateContent,
        vendor: "google".to_string(),
        model: model.to_string(),
    }
}

fn codec() -> GeminiGenerateContentCodec {
    GeminiGenerateContentCodec::new(
        origin("gemini-2.5-flash"),
        GeminiGenerateContentCompat::for_model("gemini-2.5-flash"),
    )
}

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/gemini/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("could not read {path}: {error}"))
}

fn meta(message: &Message) -> &AssistantMeta {
    message.meta.as_ref().expect("assistant metadata")
}

fn decode_stream(codec: &GeminiGenerateContentCodec, name: &str) -> (Vec<StreamEvent>, Message) {
    let raw = fixture(name);
    let mut decoder = codec.stream_decoder();
    let mut events = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        events.extend(decoder.push(line).expect("stream event should decode"));
    }
    let message = decoder.finish().expect("stream should finish");
    (events, message)
}

fn sig_opaque(model: &str, sig: &str) -> Opaque {
    Opaque {
        origin: origin(model),
        kind: "gemini_generate_content.thought_signature".to_string(),
        data: sig.to_string(),
        item_id: None,
    }
}

#[test]
fn decodes_reference_text_response() {
    let message = codec()
        .decode_response(&fixture("text.json"))
        .expect("response should decode");

    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
    let meta = meta(&message);
    assert_eq!(meta.response_id.as_deref(), Some("gemini-fixture-text"));
    assert_eq!(meta.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(
        meta.usage,
        Some(Usage {
            input_tokens: Some(9),
            output_tokens: Some(1),
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
            total_tokens: Some(10),
            cost: None,
        })
    );
}

#[test]
fn decodes_reference_thinking_response_and_combines_output_tokens() {
    let message = codec()
        .decode_response(&fixture("thinking.json"))
        .expect("response should decode");

    assert_eq!(message.blocks.len(), 2);
    assert_eq!(
        message.blocks[0],
        Block::Reasoning(ReasoningBlock {
            text: "We need answer only number. 17*3=51.".to_string(),
            opaque: Some(sig_opaque("gemini-2.5-flash", "gemini-thought-signature-1")),
        })
    );
    assert_eq!(
        message.blocks[1],
        Block::Text(TextBlock {
            text: "51".to_string(),
            opaque: None,
        })
    );

    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(44));
    assert_eq!(usage.output_tokens, Some(28));
    assert_eq!(usage.reasoning_tokens, Some(13));
    assert_eq!(usage.total_tokens, Some(72));
}

#[test]
fn decodes_reference_tool_call_response() {
    let message = codec()
        .decode_response(&fixture("tool_call.json"))
        .expect("response should decode");

    assert_eq!(
        message.blocks,
        vec![Block::ToolCall(ToolCallBlock {
            id: "gemini-call-1".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Paris" }),
            raw_args: Some(r#"{"city":"Paris"}"#.to_string()),
            args_error: None,
            provider_item_id: None,
            opaque: None,
        })]
    );
    assert_eq!(meta(&message).stop_reason, Some(StopReason::ToolCall));
}

#[test]
fn streams_reference_text_and_tool_call() {
    let (_, text_msg) = decode_stream(&codec(), "stream_text.jsonl");
    assert_eq!(
        text_msg.blocks,
        vec![Block::Text(TextBlock {
            text: "Hello world".to_string(),
            opaque: None,
        })]
    );
    assert_eq!(meta(&text_msg).stop_reason, Some(StopReason::EndTurn));

    let (events, tool_msg) = decode_stream(&codec(), "stream_tool_call.jsonl");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallStart { name, .. } if name == "get_weather"
    )));
    assert_eq!(
        tool_msg.blocks,
        vec![Block::ToolCall(ToolCallBlock {
            id: "gemini-call-1".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Paris" }),
            raw_args: Some(r#"{"city":"Paris"}"#.to_string()),
            args_error: None,
            provider_item_id: None,
            opaque: None,
        })]
    );
    assert_eq!(meta(&tool_msg).stop_reason, Some(StopReason::ToolCall));
}

#[test]
fn seals_signed_parts_and_attaches_trailing_signature_chunks() {
    let mut decoder = codec().stream_decoder();
    decoder
        .push(
            &json!({
                "candidates": [{ "content": { "parts": [{ "text": "Hello " }] } }]
            })
            .to_string(),
        )
        .unwrap();
    // Trailing empty text chunk carrying thoughtSignature seals the open text block.
    decoder
        .push(
            &json!({
                "candidates": [{ "content": { "parts": [{ "text": "", "thoughtSignature": "sig-1" }] } }]
            })
            .to_string(),
        )
        .unwrap();
    // Subsequent text must start a fresh block, not merge into the sealed one.
    decoder
        .push(
            &json!({
                "candidates": [{
                    "content": { "parts": [{ "text": "next" }] },
                    "finishReason": "STOP"
                }]
            })
            .to_string(),
        )
        .unwrap();

    let message = decoder.finish().expect("streamed");
    assert_eq!(
        message.blocks,
        vec![
            Block::Text(TextBlock {
                text: "Hello ".to_string(),
                opaque: Some(sig_opaque("gemini-2.5-flash", "sig-1")),
            }),
            Block::Text(TextBlock {
                text: "next".to_string(),
                opaque: None,
            }),
        ]
    );
}

#[test]
fn preserves_empty_signed_parts_without_matching_open_block() {
    let mut standalone = codec().stream_decoder();
    standalone
        .push(
            &json!({
                "candidates": [{
                    "content": { "parts": [{ "text": "", "thoughtSignature": "sig-only" }] },
                    "finishReason": "STOP"
                }]
            })
            .to_string(),
        )
        .unwrap();
    assert_eq!(
        standalone.finish().unwrap().blocks,
        vec![Block::Text(TextBlock {
            text: String::new(),
            opaque: Some(sig_opaque("gemini-2.5-flash", "sig-only")),
        })]
    );

    let mut mismatched = codec().stream_decoder();
    mismatched
        .push(
            &json!({
                "candidates": [{ "content": { "parts": [{ "text": "hello" }] } }]
            })
            .to_string(),
        )
        .unwrap();
    mismatched
        .push(
            &json!({
                "candidates": [{
                    "content": { "parts": [{ "text": "", "thought": true, "thoughtSignature": "sig-thought" }] },
                    "finishReason": "STOP"
                }]
            })
            .to_string(),
        )
        .unwrap();
    assert_eq!(
        mismatched.finish().unwrap().blocks,
        vec![
            Block::Text(TextBlock {
                text: "hello".to_string(),
                opaque: None,
            }),
            Block::Reasoning(ReasoningBlock {
                text: String::new(),
                opaque: Some(sig_opaque("gemini-2.5-flash", "sig-thought")),
            }),
        ]
    );
}

#[test]
fn accumulates_partial_usage_after_finish_and_subtracts_cached() {
    let mut decoder = codec().stream_decoder();
    decoder
        .push(
            &json!({
                "candidates": [{ "content": { "parts": [{ "text": "ok" }] } }],
                "usageMetadata": { "promptTokenCount": 100, "cachedContentTokenCount": 40 }
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .push(
            &json!({
                "candidates": [{ "finishReason": "STOP" }]
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .push(
            &json!({
                "usageMetadata": { "candidatesTokenCount": 10, "thoughtsTokenCount": 5 }
            })
            .to_string(),
        )
        .unwrap();

    let message = decoder.finish().expect("streamed");
    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(60));
    assert_eq!(usage.cache_read_tokens, Some(40));
    assert_eq!(usage.output_tokens, Some(15));
    assert_eq!(usage.reasoning_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(115));
}

#[test]
fn ignores_unknown_parts_and_done_marker() {
    let raw = json!({
        "candidates": [{
            "content": {
                "parts": [
                    { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                    { "text": "ok" }
                ]
            },
            "finishReason": "STOP"
        }]
    })
    .to_string();

    let message = codec().decode_response(&raw).expect("response");
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );

    let mut decoder = codec().stream_decoder();
    assert!(decoder.push("[DONE]").unwrap().is_empty());
    decoder.push(&raw).unwrap();
    assert_eq!(decoder.finish().unwrap().blocks, message.blocks);
}

#[test]
fn encodes_request_with_system_tools_reordered_parallel_results_and_signatures() {
    let request = ChatRequest {
        messages: vec![
            Message::system("Be concise."),
            Message::user("Check Paris and London."),
            Message::assistant(vec![
                Block::Reasoning(ReasoningBlock {
                    text: "Calling both weather tools.".to_string(),
                    opaque: Some(sig_opaque("gemini-2.5-flash", "thought-sig")),
                }),
                Block::ToolCall(ToolCallBlock {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({ "city": "Paris" }),
                    raw_args: None,
                    args_error: None,
                    provider_item_id: None,
                    opaque: Some(sig_opaque("gemini-2.5-flash", "call-sig")),
                }),
                Block::ToolCall(ToolCallBlock {
                    id: "call_2".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({ "city": "London" }),
                    raw_args: None,
                    args_error: None,
                    provider_item_id: None,
                    opaque: None,
                }),
            ]),
            // Completed out of order; encoder must reorder to match call_1, call_2.
            Message::tool_result("call_2", "get_weather", "timeout", true),
            Message::tool_result("call_1", "get_weather", "20C", false),
        ],
        tools: vec![ToolDef {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        }],
        tool_choice: ToolChoice::Tool("get_weather".to_string()),
        temperature: Some(0.2),
        top_p: Some(0.9),
        max_tokens: Some(512),
        stop: vec!["DONE".to_string()],
        reasoning_effort: Some(ReasoningEffort::Medium),
    };

    let body = codec()
        .encode_request(&request, true)
        .expect("request should encode");

    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be concise.");
    assert_eq!(
        body["toolConfig"]["functionCallingConfig"],
        json!({ "mode": "ANY", "allowedFunctionNames": ["get_weather"] })
    );
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": 4096 })
    );
    assert_eq!(body["generationConfig"]["maxOutputTokens"], 512);
    assert_eq!(body["generationConfig"]["stopSequences"], json!(["DONE"]));

    let contents = body["contents"].as_array().expect("contents array");
    assert_eq!(contents.len(), 3);
    assert_eq!(contents[1]["role"], "model");
    assert_eq!(contents[1]["parts"][0]["thought"], true);
    assert_eq!(contents[1]["parts"][0]["thoughtSignature"], "thought-sig");
    assert_eq!(contents[1]["parts"][1]["thoughtSignature"], "call-sig");
    assert!(contents[1]["parts"][2].get("thoughtSignature").is_none());

    // Parallel tool responses merged into a single user turn in call order.
    assert_eq!(contents[2]["role"], "user");
    let parts = contents[2]["parts"].as_array().expect("response parts");
    assert_eq!(parts.len(), 2);
    assert_eq!(
        parts[0]["functionResponse"]["response"],
        json!({ "result": "20C" })
    );
    assert_eq!(
        parts[1]["functionResponse"]["response"],
        json!({ "error": "timeout" })
    );
}

#[test]
fn gemini_3_compat_uses_thinking_level() {
    let codec = GeminiGenerateContentCodec::new(
        origin("gemini-3-pro-preview"),
        GeminiGenerateContentCompat::for_model("gemini-3-pro-preview"),
    );
    let mut request = ChatRequest::new(vec![Message::user("hi")]);
    request.reasoning_effort = Some(ReasoningEffort::High);

    let body = codec.encode_request(&request, false).expect("request");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingLevel": "HIGH" })
    );
    assert!(
        body["generationConfig"]["thinkingConfig"]
            .get("thinkingBudget")
            .is_none()
    );

    let unknown_codec = GeminiGenerateContentCodec::new(
        origin("custom-gemini"),
        GeminiGenerateContentCompat::for_model("custom-gemini"),
    );
    let body = unknown_codec
        .encode_request(&request, false)
        .expect("request");
    assert!(body["generationConfig"].get("thinkingConfig").is_none());
}

#[test]
fn skips_foreign_signatures_and_invalid_tool_args() {
    let mut compat = GeminiGenerateContentCompat::google();
    compat.thinking_mode = ThinkingMode::None;
    let codec = GeminiGenerateContentCodec::new(origin("gemini-2.5-flash"), compat);

    let request = ChatRequest::new(vec![
        Message::user("hi"),
        Message::assistant(vec![
            Block::Reasoning(ReasoningBlock {
                text: "foreign".to_string(),
                opaque: Some(Opaque {
                    origin: Origin {
                        protocol: Protocol::Anthropic,
                        vendor: "anthropic".to_string(),
                        model: "claude".to_string(),
                    },
                    kind: "anthropic.thinking".to_string(),
                    data: "sig".to_string(),
                    item_id: None,
                }),
            }),
            Block::ToolCall(ToolCallBlock {
                id: "call_1".to_string(),
                name: "ping".to_string(),
                args: Value::Null,
                raw_args: Some("{bad".to_string()),
                args_error: Some("bad json".to_string()),
                provider_item_id: None,
                opaque: None,
            }),
        ]),
    ]);

    let body = codec.encode_request(&request, false).expect("request");
    let parts = body["contents"][1]["parts"].as_array().expect("parts");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["functionCall"]["args"], json!({}));
}

#[test]
fn rejects_unsupported_options_and_content() {
    let mut too_many_stops = ChatRequest::new(vec![Message::user("hi")]);
    too_many_stops.stop = (1..=6).map(|i| format!("S{i}")).collect();
    assert_eq!(
        codec()
            .encode_request(&too_many_stops, false)
            .unwrap_err()
            .code(),
        ErrorCode::UnsupportedOption
    );

    let mut unsupported_schema = ChatRequest::new(vec![Message::user("hi")]);
    unsupported_schema.tools.push(ToolDef {
        name: "ping".to_string(),
        description: "ping".to_string(),
        parameters: json!({ "type": "object", "additionalProperties": false }),
    });
    assert_eq!(
        codec()
            .encode_request(&unsupported_schema, false)
            .unwrap_err()
            .code(),
        ErrorCode::UnsupportedOption
    );

    let url_image = ChatRequest::new(vec![Message {
        mtype: MessageType::User,
        blocks: vec![Block::Image(ImageBlock {
            source: ImageSource::Url("https://example.com/a.png".to_string()),
            opaque: None,
        })],
        meta: None,
    }]);
    assert_eq!(
        codec()
            .encode_request(&url_image, false)
            .unwrap_err()
            .code(),
        ErrorCode::UnsupportedContent
    );

    let tool_image = ChatRequest::new(vec![
        Message::user("hi"),
        Message {
            mtype: MessageType::Tool,
            blocks: vec![Block::ToolResult(ToolResultBlock {
                call_id: "call_1".to_string(),
                name: "ping".to_string(),
                content: vec![Block::Image(ImageBlock {
                    source: ImageSource::Base64 {
                        mime: "image/png".to_string(),
                        data: vec![1, 2, 3],
                    },
                    opaque: None,
                })],
                is_error: false,
            })],
            meta: None,
        },
    ]);
    assert_eq!(
        codec()
            .encode_request(&tool_image, false)
            .unwrap_err()
            .code(),
        ErrorCode::UnsupportedContent
    );
}

#[test]
fn maps_errors_safety_protocol_failures_and_max_tokens() {
    let blocked = json!({
        "promptFeedback": { "blockReason": "SAFETY" }
    })
    .to_string();
    let err = codec().decode_response(&blocked).unwrap_err();
    assert_eq!(err.code(), ErrorCode::ContentFilter);
    assert!(!err.is_retryable());

    let malformed = json!({
        "candidates": [{ "finishReason": "MALFORMED_FUNCTION_CALL" }]
    })
    .to_string();
    assert_eq!(
        codec().decode_response(&malformed).unwrap_err().code(),
        ErrorCode::Protocol
    );

    let http_err = json!({
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "status": "RESOURCE_EXHAUSTED"
        }
    })
    .to_string();
    let rate_err = codec().decode_response(&http_err).unwrap_err();
    assert_eq!(rate_err.code(), ErrorCode::RateLimit);
    assert!(rate_err.is_retryable());

    let truncated = json!({
        "candidates": [{
            "content": { "parts": [{ "functionCall": { "name": "ping", "args": {} } }] },
            "finishReason": "MAX_TOKENS"
        }]
    })
    .to_string();
    let msg = codec().decode_response(&truncated).expect("truncated");
    assert_eq!(meta(&msg).stop_reason, Some(StopReason::MaxTokens));
    let Block::ToolCall(call) = &msg.blocks[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.id, "call-0");
    assert!(call.args_error.is_some());
}

#[test]
fn missing_finish_reason_reports_stream_closed_and_decoders_are_isolated() {
    let mut incomplete = codec().stream_decoder();
    incomplete
        .push(r#"{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}"#)
        .unwrap();
    assert_eq!(
        incomplete.finish().unwrap_err().code(),
        ErrorCode::StreamClosed
    );

    let c = codec();
    let mut first = c.stream_decoder();
    let mut second = c.stream_decoder();
    first
        .push(r#"{"candidates":[{"content":{"parts":[{"text":"A"}]}}]}"#)
        .unwrap();
    second
        .push(r#"{"candidates":[{"content":{"parts":[{"text":"B"}]}}]}"#)
        .unwrap();
    first
        .push(r#"{"candidates":[{"finishReason":"STOP"}]}"#)
        .unwrap();
    second
        .push(r#"{"candidates":[{"finishReason":"STOP"}]}"#)
        .unwrap();
    assert_eq!(
        first.finish().unwrap().blocks[0],
        Block::Text(TextBlock {
            text: "A".to_string(),
            opaque: None,
        })
    );
    assert_eq!(
        second.finish().unwrap().blocks[0],
        Block::Text(TextBlock {
            text: "B".to_string(),
            opaque: None,
        })
    );
}

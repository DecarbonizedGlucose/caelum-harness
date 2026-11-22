use caelum_harness::error::ErrorCode;
use caelum_harness::model::convert::ProtocolCodec;
use caelum_harness::model::convert::openai_chat::{
    OpenAiChatCodec, OpenAiChatCompat, ReasoningField,
};
use caelum_harness::model::message::{
    AssistantMeta, Block, Message, MessageType, Origin, ReasoningBlock, StopReason, TextBlock,
    ToolCallBlock, ToolResultBlock, Usage,
};
use caelum_harness::model::protocol::{
    ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef,
};
use caelum_harness::model::stream::StreamEvent;
use serde_json::{Value, json};

fn origin(vendor: &str, model: &str) -> Origin {
    Origin {
        protocol: Protocol::OpenaiChat,
        vendor: vendor.to_string(),
        model: model.to_string(),
    }
}

fn codec() -> OpenAiChatCodec {
    OpenAiChatCodec::new(
        origin("deepseek", "deepseek-flash"),
        OpenAiChatCompat::deepseek(),
    )
}

fn decode(raw: &str) -> Message {
    codec()
        .decode_response(raw)
        .expect("response should decode")
}

fn meta(message: &Message) -> &AssistantMeta {
    message.meta.as_ref().expect("assistant metadata")
}

fn stream_events(raw: &str, codec: &OpenAiChatCodec) -> (Vec<StreamEvent>, Message) {
    let mut decoder = codec.stream_decoder();
    let mut events = Vec::new();
    for line in raw.lines() {
        events.extend(decoder.push(line).expect("stream event should decode"));
    }
    let message = decoder.finish().expect("stream should finish");
    (events, message)
}

#[test]
fn decodes_deepseek_text_response() {
    let message = decode(include_str!("fixtures/openai_chat/deepseek_text.json"));

    assert_eq!(message.mtype, MessageType::Assistant);
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );

    let meta = meta(&message);
    assert_eq!(meta.origin, origin("deepseek", "deepseek-flash"));
    assert_eq!(
        meta.response_id.as_deref(),
        Some("d64dda02-007f-49e8-8143-da2520383d95")
    );
    assert_eq!(meta.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(
        meta.usage,
        Some(Usage {
            input_tokens: Some(9),
            output_tokens: Some(1),
            cache_read_tokens: Some(0),
            cache_write_tokens: None,
            reasoning_tokens: None,
            total_tokens: Some(10),
            cost: None,
        })
    );
}

#[test]
fn decodes_deepseek_reasoning_response() {
    let message = decode(include_str!("fixtures/openai_chat/deepseek_reasoning.json"));

    assert_eq!(message.blocks.len(), 2);
    assert!(matches!(
        &message.blocks[0],
        Block::Reasoning(ReasoningBlock { text, .. })
            if text == "We need answer only number. 17*3=51."
    ));
    assert_eq!(
        message.blocks[1],
        Block::Text(TextBlock {
            text: "51".to_string(),
            opaque: None,
        })
    );

    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(44));
    assert_eq!(usage.output_tokens, Some(15));
    assert_eq!(usage.cache_read_tokens, Some(0));
    assert_eq!(usage.reasoning_tokens, Some(13));
}

#[test]
fn decodes_deepseek_tool_call_response() {
    let message = decode(include_str!("fixtures/openai_chat/deepseek_tool_call.json"));

    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected a tool call block");
    };
    assert_eq!(call.id, "call_00_Hz1r0WD6zyURoLbQqqb08125");
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.args, json!({ "city": "Paris" }));
    assert_eq!(call.raw_args.as_deref(), Some("{\"city\": \"Paris\"}"));
    assert_eq!(call.args_error, None);
    assert_eq!(call.provider_item_id, None);

    let meta = meta(&message);
    assert_eq!(meta.stop_reason, Some(StopReason::ToolCall));
    assert_eq!(meta.usage.expect("usage").input_tokens, Some(283));
}

#[test]
fn decodes_openrouter_text_response() {
    let codec = OpenAiChatCodec::new(
        origin("openrouter", "mistralai/mistral-nemo"),
        OpenAiChatCompat::openrouter(),
    );
    let message = codec
        .decode_response(include_str!("fixtures/openai_chat/openrouter_text.json"))
        .expect("response should decode");

    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
    let usage = message.meta.expect("metadata").usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(9));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.reasoning_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(11));
    assert!(usage.cost.is_some_and(|cost| cost > 0.0));
}

#[test]
fn invalid_tool_arguments_are_preserved_as_args_error() {
    let raw = json!({
        "id": "invalid-tool-args",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{"
                    }
                }]
            }
        }]
    })
    .to_string();

    let message = codec()
        .decode_response(&raw)
        .expect("a malformed tool call must not fail the whole response");

    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected a tool call block");
    };
    assert_eq!(call.args, Value::Null);
    assert_eq!(call.raw_args.as_deref(), Some("{"));
    assert!(call.args_error.is_some());
}

#[test]
fn empty_response_reports_empty_response_code() {
    let raw = json!({
        "id": "empty",
        "choices": [{
            "finish_reason": "stop",
            "message": { "role": "assistant", "content": null }
        }]
    })
    .to_string();

    let error = codec().decode_response(&raw).expect_err("empty response");
    assert_eq!(error.code(), ErrorCode::EmptyResponse);
}

#[test]
fn invalid_json_reports_json_code_and_context() {
    let error = codec().decode_response("{").expect_err("invalid json");

    assert_eq!(error.code(), ErrorCode::Json);
    assert!(error.to_string().contains("OpenAI Chat response"));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn encodes_basic_request() {
    let request = ChatRequest {
        messages: vec![Message::system("be brief"), Message::user("hello")],
        tools: vec![ToolDef {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({ "type": "object" }),
        }],
        tool_choice: ToolChoice::Required,
        temperature: Some(0.2),
        top_p: None,
        max_tokens: Some(64),
        stop: vec!["END".to_string()],
        reasoning_effort: Some(ReasoningEffort::Low),
    };

    let body = codec()
        .encode_request(&request, false)
        .expect("request should encode");

    assert_eq!(body["model"], "deepseek-flash");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["content"], "hello");
    assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["max_tokens"], 64);
    assert_eq!(body["stop"][0], "END");
    assert_eq!(body["reasoning_effort"], "low");
}

#[test]
fn streams_text_events() {
    let codec = codec();
    let (events, message) = stream_events(
        include_str!("fixtures/openai_chat/deepseek_stream_text.jsonl"),
        &codec,
    );

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::MessageStart { response_id: Some(id) }
            if id == "c3ab1c88-5a15-4b5f-9675-6fb10b09be93"
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::TextDelta { text, .. } if text == "ok"))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::Finish {
            stop_reason: Some(StopReason::EndTurn)
        }
    )));
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
    assert_eq!(meta(&message).stop_reason, Some(StopReason::EndTurn));
}

#[test]
fn streams_reasoning_events() {
    let codec = codec();
    let (events, message) = stream_events(
        include_str!("fixtures/openai_chat/deepseek_stream_reasoning.jsonl"),
        &codec,
    );

    let reasoning: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning,
        "We need answer only number. 17*3=51. Final only."
    );

    let text: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "51");

    assert_eq!(message.blocks.len(), 2);
    assert!(matches!(message.blocks[0], Block::Reasoning(_)));
    assert_eq!(
        message.blocks[1],
        Block::Text(TextBlock {
            text: "51".to_string(),
            opaque: None,
        })
    );
    assert_eq!(
        meta(&message).usage.expect("usage").reasoning_tokens,
        Some(16)
    );
}

#[test]
fn streams_tool_call_arguments() {
    let codec = codec();
    let (events, message) = stream_events(
        include_str!("fixtures/openai_chat/deepseek_stream_tool_call.jsonl"),
        &codec,
    );

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallStart { name, .. } if name == "get_weather"
    )));

    let arguments: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ToolArgsDelta { fragment, .. } => Some(fragment.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(arguments, "{\"city\": \"Paris\"}");

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::Finish {
            stop_reason: Some(StopReason::ToolCall)
        }
    )));
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.args, json!({ "city": "Paris" }));
    assert_eq!(meta(&message).stop_reason, Some(StopReason::ToolCall));
}

#[test]
fn stream_without_finish_reports_stream_closed() {
    let codec = codec();
    let event = json!({
        "id": "missing-finish",
        "choices": [{ "delta": { "content": "hello" } }]
    });
    let mut decoder = codec.stream_decoder();

    decoder
        .push(&event.to_string())
        .expect("delta should decode");
    let error = decoder.finish().expect_err("missing finish");
    assert_eq!(error.code(), ErrorCode::StreamClosed);
}

#[test]
fn encodes_stream_request_with_usage() {
    let request = ChatRequest {
        messages: vec![Message::user("hello")],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens: Some(64),
        stop: Vec::new(),
        reasoning_effort: None,
    };

    let body = codec()
        .encode_request(&request, true)
        .expect("stream request should encode");

    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["max_tokens"], 64);
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn openai_compat_uses_completion_tokens_and_developer_role() {
    let codec = OpenAiChatCodec::new(origin("openai", "gpt-5-mini"), OpenAiChatCompat::openai());
    let request = ChatRequest {
        messages: vec![Message::system("be brief"), Message::user("hello")],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens: Some(32),
        stop: Vec::new(),
        reasoning_effort: None,
    };

    let body = codec
        .encode_request(&request, false)
        .expect("request should encode");

    assert_eq!(body["messages"][0]["role"], "developer");
    assert_eq!(body["max_completion_tokens"], 32);
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn replays_reasoning_only_when_compat_requests_it() {
    let compat = OpenAiChatCompat {
        reasoning_field: Some(ReasoningField::ReasoningContent),
        replay_reasoning: true,
        ..OpenAiChatCompat::deepseek()
    };
    let codec = OpenAiChatCodec::new(origin("deepseek", "deepseek-flash"), compat);
    let request = ChatRequest {
        messages: vec![
            Message::user("hello"),
            Message::assistant(vec![
                Block::Reasoning(ReasoningBlock {
                    text: "previous thought".to_string(),
                    opaque: None,
                }),
                Block::Text(TextBlock {
                    text: "answer".to_string(),
                    opaque: None,
                }),
            ]),
        ],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: Vec::new(),
        reasoning_effort: None,
    };

    let body = codec
        .encode_request(&request, false)
        .expect("request should encode");

    assert_eq!(body["messages"][1]["reasoning_content"], "previous thought");
}

#[test]
fn invalid_tool_arguments_are_not_replayed_verbatim() {
    let request = ChatRequest {
        messages: vec![
            Message::user("hello"),
            Message::assistant(vec![Block::ToolCall(ToolCallBlock {
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                args: Value::Null,
                raw_args: Some("{".to_string()),
                args_error: Some("invalid json".to_string()),
                provider_item_id: None,
                opaque: None,
            })]),
        ],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: Vec::new(),
        reasoning_effort: None,
    };

    let body = codec()
        .encode_request(&request, false)
        .expect("request should encode");

    assert_eq!(
        body["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

#[test]
fn accepts_tool_arguments_as_json_objects() {
    let raw = json!({
        "id": "object-args",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": { "city": "Paris" }
                    }
                }]
            }
        }]
    })
    .to_string();

    let message = codec()
        .decode_response(&raw)
        .expect("response should decode");
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.args, json!({ "city": "Paris" }));
    assert!(call.raw_args.is_some());
}

#[test]
fn streams_usage_from_a_chunk_after_finish() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    let content = json!({
        "id": "usage-after-finish",
        "choices": [{ "delta": { "content": "ok" } }]
    });
    let finish = json!({
        "id": "usage-after-finish",
        "choices": [{ "delta": {}, "finish_reason": "stop" }]
    });
    let usage = json!({
        "id": "usage-after-finish",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": { "cached_tokens": 4 }
        }
    });

    let mut events = decoder.push(&content.to_string()).expect("content chunk");
    events.extend(decoder.push(&finish.to_string()).expect("finish chunk"));
    events.extend(decoder.push(&usage.to_string()).expect("usage chunk"));
    let message = decoder.finish().expect("stream should finish");

    let usage = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::Usage(usage) => Some(*usage),
            _ => None,
        })
        .expect("usage event");
    assert_eq!(usage.input_tokens, Some(6));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.cache_read_tokens, Some(4));
    assert_eq!(meta(&message).usage.expect("assembled usage"), usage);
}

#[test]
fn decoders_do_not_share_stream_state() {
    let codec = codec();
    let mut first = codec.stream_decoder();
    let mut second = codec.stream_decoder();

    first
        .push(
            &json!({
                "id": "first",
                "choices": [{ "delta": { "content": "A" } }]
            })
            .to_string(),
        )
        .expect("first delta");
    second
        .push(
            &json!({
                "id": "second",
                "choices": [{ "delta": { "content": "B" } }]
            })
            .to_string(),
        )
        .expect("second delta");

    let finish = json!({
        "id": "finish",
        "choices": [{ "delta": {}, "finish_reason": "stop" }]
    });
    first.push(&finish.to_string()).expect("first finish");
    second.push(&finish.to_string()).expect("second finish");

    let first = first.finish().expect("first message");
    let second = second.finish().expect("second message");

    assert_eq!(
        first.blocks,
        vec![Block::Text(TextBlock {
            text: "A".to_string(),
            opaque: None,
        })]
    );
    assert_eq!(
        second.blocks,
        vec![Block::Text(TextBlock {
            text: "B".to_string(),
            opaque: None,
        })]
    );
}

#[test]
fn missing_choices_reports_protocol_error() {
    let raw = json!({
        "id": "missing-choices",
        "choices": []
    })
    .to_string();

    let error = codec().decode_response(&raw).expect_err("missing choices");
    assert_eq!(error.code(), ErrorCode::Protocol);
}

#[test]
fn maps_stream_error_chunk_to_rate_limit() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    let error = decoder
        .push(
            &json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded"
                }
            })
            .to_string(),
        )
        .expect_err("error chunk");

    assert_eq!(error.code(), ErrorCode::RateLimit);
    assert!(error.is_retryable());
}

#[test]
fn malformed_stream_payload_reports_json_error() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    let error = decoder.push("{").expect_err("invalid stream json");

    assert_eq!(error.code(), ErrorCode::Json);
    assert!(error.to_string().contains("OpenAI Chat stream event"));
}

#[test]
fn refusal_delta_is_visible_text() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    decoder
        .push(
            &json!({
                "id": "refusal",
                "choices": [{ "delta": { "refusal": "I cannot help with that." } }]
            })
            .to_string(),
        )
        .expect("refusal delta");
    decoder
        .push(
            &json!({
                "id": "refusal",
                "choices": [{ "delta": {}, "finish_reason": "content_filter" }]
            })
            .to_string(),
        )
        .expect("finish");

    let message = decoder.finish().expect("refusal message");
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "I cannot help with that.".to_string(),
            opaque: None,
        })]
    );
    assert_eq!(meta(&message).stop_reason, Some(StopReason::ContentFilter));
}

#[test]
fn parallel_tool_calls_without_index_do_not_merge() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    decoder
        .push(
            &json!({
                "id": "parallel",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call-a",
                            "type": "function",
                            "function": { "name": "first", "arguments": "{\"a\":1}" }
                        }]
                    }
                }]
            })
            .to_string(),
        )
        .expect("first tool call");
    decoder
        .push(
            &json!({
                "id": "parallel",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call-b",
                            "type": "function",
                            "function": { "name": "second", "arguments": "{\"b\":2}" }
                        }]
                    }
                }]
            })
            .to_string(),
        )
        .expect("second tool call");
    decoder
        .push(
            &json!({
                "id": "parallel",
                "choices": [{ "delta": {}, "finish_reason": "tool_calls" }]
            })
            .to_string(),
        )
        .expect("finish");

    let message = decoder.finish().expect("tool message");
    assert_eq!(message.blocks.len(), 2);
    let Block::ToolCall(first) = &message.blocks[0] else {
        panic!("expected first tool call");
    };
    let Block::ToolCall(second) = &message.blocks[1] else {
        panic!("expected second tool call");
    };
    assert_eq!(first.name, "first");
    assert_eq!(second.name, "second");
    assert_eq!(first.args, json!({ "a": 1 }));
    assert_eq!(second.args, json!({ "b": 2 }));
}

#[test]
fn openrouter_compat_prefers_reasoning_delta() {
    let codec = OpenAiChatCodec::new(
        origin("openrouter", "mistralai/mistral-nemo"),
        OpenAiChatCompat::openrouter(),
    );
    let mut decoder = codec.stream_decoder();
    let events = decoder
        .push(
            &json!({
                "id": "reasoning",
                "choices": [{ "delta": { "reasoning": "thinking" } }]
            })
            .to_string(),
        )
        .expect("reasoning delta");

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ReasoningDelta { text, .. } if text == "thinking"
    )));
}

#[test]
fn request_errors_use_request_context() {
    let request = ChatRequest {
        messages: vec![Message::assistant(vec![Block::ToolResult(
            ToolResultBlock {
                call_id: "call-1".to_string(),
                name: "get_weather".to_string(),
                content: vec![Block::Text(TextBlock {
                    text: "sunny".to_string(),
                    opaque: None,
                })],
                is_error: false,
            },
        )])],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: Vec::new(),
        reasoning_effort: None,
    };

    let error = codec()
        .encode_request(&request, false)
        .expect_err("invalid history");
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert!(error.to_string().contains("request is invalid"));
}

#[test]
fn maps_non_streaming_error_body_to_rate_limit() {
    let raw = json!({
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    })
    .to_string();

    let error = codec().decode_response(&raw).expect_err("error body");
    assert_eq!(error.code(), ErrorCode::RateLimit);
    assert!(error.is_retryable());
}

#[test]
fn stream_shape_error_reports_protocol() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    let error = decoder
        .push(r#"{"choices":"not-an-array"}"#)
        .expect_err("shape error");

    assert_eq!(error.code(), ErrorCode::Protocol);
    assert!(error.to_string().contains("could not decode payload"));
}

#[test]
fn openai_compat_ignores_reasoning_fields() {
    let codec = OpenAiChatCodec::new(origin("openai", "gpt-5-mini"), OpenAiChatCompat::openai());
    let raw = json!({
        "id": "openai-reasoning",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "ok",
                "reasoning": "this must not become a reasoning block"
            }
        }]
    })
    .to_string();

    let message = codec.decode_response(&raw).expect("response");
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
}

#[test]
fn openrouter_compat_parses_non_streaming_reasoning() {
    let codec = OpenAiChatCodec::new(
        origin("openrouter", "mistralai/mistral-nemo"),
        OpenAiChatCompat::openrouter(),
    );
    let raw = json!({
        "id": "openrouter-reasoning",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "ok",
                "reasoning": "thinking"
            }
        }]
    })
    .to_string();

    let message = codec.decode_response(&raw).expect("response");
    assert_eq!(
        message.blocks,
        vec![
            Block::Reasoning(ReasoningBlock {
                text: "thinking".to_string(),
                opaque: None,
            }),
            Block::Text(TextBlock {
                text: "ok".to_string(),
                opaque: None,
            }),
        ]
    );
}

#[test]
fn deepseek_preset_replays_reasoning() {
    let request = ChatRequest {
        messages: vec![
            Message::user("hello"),
            Message::assistant(vec![
                Block::Reasoning(ReasoningBlock {
                    text: "previous thought".to_string(),
                    opaque: None,
                }),
                Block::Text(TextBlock {
                    text: "answer".to_string(),
                    opaque: None,
                }),
            ]),
        ],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        top_p: None,
        max_tokens: None,
        stop: Vec::new(),
        reasoning_effort: None,
    };

    let body = codec().encode_request(&request, false).expect("request");
    assert_eq!(body["messages"][1]["reasoning_content"], "previous thought");
}

use caelum_harness::error::ErrorCode;
use caelum_harness::model::convert::ProtocolCodec;
use caelum_harness::model::convert::anthropic::AnthropicCodec;
use caelum_harness::model::message::{
    AssistantMeta, Block, ImageBlock, ImageSource, Message, Opaque, Origin, ReasoningBlock,
    StopReason, TextBlock, ToolCallBlock, ToolResultBlock, Usage,
};
use caelum_harness::model::protocol::{
    ChatRequest, Protocol, ReasoningEffort, ToolChoice, ToolDef,
};
use caelum_harness::model::stream::StreamEvent;
use serde_json::{Value, json};

fn origin(vendor: &str, model: &str) -> Origin {
    Origin {
        protocol: Protocol::Anthropic,
        vendor: vendor.to_string(),
        model: model.to_string(),
    }
}

fn codec() -> AnthropicCodec {
    AnthropicCodec::new(origin("anthropic", "claude-sonnet-4-5"))
}

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/anthropic/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("could not read {path}: {error}"))
}

fn meta(message: &Message) -> &AssistantMeta {
    message.meta.as_ref().expect("assistant metadata")
}

fn decode_stream(codec: &AnthropicCodec, name: &str) -> (Vec<StreamEvent>, Message) {
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

#[test]
fn decodes_deepseek_text_response() {
    let codec = AnthropicCodec::new(origin("deepseek", "deepseek-flash"));
    let message = codec
        .decode_response(&fixture("deepseek_text.json"))
        .expect("response should decode");

    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
    let meta = meta(&message);
    assert_eq!(meta.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(
        meta.usage,
        Some(Usage {
            input_tokens: Some(9),
            output_tokens: Some(1),
            cache_read_tokens: Some(0),
            cache_write_tokens: Some(0),
            reasoning_tokens: None,
            total_tokens: None,
            cost: None,
        })
    );
}

#[test]
fn decodes_deepseek_thinking_response_with_signature_opaque() {
    let codec = AnthropicCodec::new(origin("deepseek", "deepseek-flash"));
    let message = codec
        .decode_response(&fixture("deepseek_thinking.json"))
        .expect("response should decode");

    assert_eq!(message.blocks.len(), 2);
    let Block::Reasoning(reasoning) = &message.blocks[0] else {
        panic!("expected reasoning block");
    };
    assert_eq!(
        reasoning.text,
        "We need answer only number. 17*3=51. Must reply number only."
    );
    assert_eq!(
        reasoning.opaque,
        Some(Opaque {
            origin: origin("deepseek", "deepseek-flash"),
            kind: "anthropic.thinking".to_string(),
            data: "317079b1-4f93-44e5-b70f-0dd7cbeedb0a".to_string(),
            item_id: None,
        })
    );
    assert_eq!(
        meta(&message).usage.expect("usage").reasoning_tokens,
        Some(18)
    );
}

#[test]
fn decodes_deepseek_tool_use_response() {
    let codec = AnthropicCodec::new(origin("deepseek", "deepseek-flash"));
    let message = codec
        .decode_response(&fixture("deepseek_tool_use.json"))
        .expect("response should decode");

    assert_eq!(
        message.blocks,
        vec![Block::ToolCall(ToolCallBlock {
            id: "call_00_UcEOFm5O0gT7jfxB3tdM3328".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Paris" }),
            raw_args: Some(r#"{"city":"Paris"}"#.to_string()),
            args_error: None,
            provider_item_id: None,
            opaque: None,
        })]
    );
    let meta = meta(&message);
    assert_eq!(meta.stop_reason, Some(StopReason::ToolCall));
    let usage = meta.usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(155));
    assert_eq!(usage.cache_read_tokens, Some(128));
    assert_eq!(usage.output_tokens, Some(33));
}

#[test]
fn streams_text_and_signature_delta() {
    let codec = AnthropicCodec::new(origin("deepseek", "deepseek-flash"));
    let (events, message) = decode_stream(&codec, "deepseek_stream_text.jsonl");

    assert!(matches!(
        events.first(),
        Some(StreamEvent::MessageStart { response_id: Some(id) }) if id == "65d729c5-c0df-4a54-a795-58b806e98521"
    ));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::Finish {
            stop_reason: Some(StopReason::EndTurn)
        })
    ));

    assert_eq!(message.blocks.len(), 2);
    let Block::Reasoning(reasoning) = &message.blocks[0] else {
        panic!("expected reasoning block");
    };
    assert_eq!(
        reasoning.opaque,
        Some(Opaque {
            origin: origin("deepseek", "deepseek-flash"),
            kind: "anthropic.thinking".to_string(),
            data: "65d729c5-c0df-4a54-a795-58b806e98521".to_string(),
            item_id: None,
        })
    );
    assert_eq!(
        message.blocks[1],
        Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })
    );
    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(35));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.reasoning_tokens, Some(0));
}

#[test]
fn streams_tool_use_input_json_delta() {
    let codec = AnthropicCodec::new(origin("deepseek", "deepseek-flash"));
    let (events, message) = decode_stream(&codec, "deepseek_stream_tool_use.jsonl");

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallStart { name, .. } if name == "get_weather"
    )));
    assert_eq!(
        message.blocks,
        vec![Block::ToolCall(ToolCallBlock {
            id: "call_00_SVjPYHe5Fh9WVvFzE7RU4151".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Paris" }),
            raw_args: Some(r#"{"city": "Paris"}"#.to_string()),
            args_error: None,
            provider_item_id: None,
            opaque: None,
        })]
    );
    assert_eq!(meta(&message).stop_reason, Some(StopReason::ToolCall));
}

#[test]
fn encodes_request_with_system_tools_thinking_and_merged_user_turns() {
    let request = ChatRequest {
        messages: vec![
            Message::system("Be concise."),
            Message::user("What is the weather in Paris?"),
            Message::assistant(vec![
                Block::Reasoning(ReasoningBlock {
                    text: "Need to call weather tool.".to_string(),
                    opaque: Some(Opaque {
                        origin: origin("anthropic", "claude-sonnet-4-5"),
                        kind: "anthropic.thinking".to_string(),
                        data: "sig-123".to_string(),
                        item_id: None,
                    }),
                }),
                Block::ToolCall(ToolCallBlock {
                    id: "toolu_1".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({ "city": "Paris" }),
                    raw_args: None,
                    args_error: None,
                    provider_item_id: None,
                    opaque: None,
                }),
            ]),
            Message::tool_result("toolu_1", "get_weather", "20C", false),
            Message::user("Also check London."),
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
        tool_choice: ToolChoice::Required,
        temperature: Some(0.2),
        top_p: Some(0.9),
        max_tokens: None,
        stop: vec!["DONE".to_string()],
        reasoning_effort: Some(ReasoningEffort::Medium),
    };

    let body = codec()
        .encode_request(&request, true)
        .expect("request should encode");

    assert_eq!(body["model"], "claude-sonnet-4-5");
    assert_eq!(body["max_tokens"], 8192);
    assert_eq!(body["system"], "Be concise.");
    assert_eq!(body["stream"], true);
    assert_eq!(body["tool_choice"], json!({ "type": "any" }));
    assert_eq!(
        body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 4096 })
    );
    assert_eq!(body["stop_sequences"], json!(["DONE"]));
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");

    // Tool result and trailing user message merge into one user turn.
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"][0]["type"], "thinking");
    assert_eq!(messages[1]["content"][0]["signature"], "sig-123");
    assert_eq!(messages[1]["content"][1]["type"], "tool_use");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"][0]["type"], "tool_result");
    assert_eq!(messages[2]["content"][1]["type"], "text");
}

#[test]
fn replays_redacted_thinking_and_skips_foreign_reasoning() {
    let request = ChatRequest::new(vec![
        Message::user("hi"),
        Message::assistant(vec![
            Block::Reasoning(ReasoningBlock {
                text: "foreign thought".to_string(),
                opaque: Some(Opaque {
                    origin: Origin {
                        protocol: Protocol::OpenaiResp,
                        vendor: "openai".to_string(),
                        model: "gpt-5".to_string(),
                    },
                    kind: "openai.responses.reasoning".to_string(),
                    data: "enc".to_string(),
                    item_id: None,
                }),
            }),
            Block::Opaque(Opaque {
                origin: origin("anthropic", "claude-sonnet-4-5"),
                kind: "anthropic.redacted_thinking".to_string(),
                data: "b3BhcXVlCg==".to_string(),
                item_id: None,
            }),
            Block::Text(TextBlock {
                text: "hello".to_string(),
                opaque: None,
            }),
        ]),
    ]);

    let body = codec().encode_request(&request, false).expect("request");
    let content = body["messages"][1]["content"]
        .as_array()
        .expect("assistant content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "redacted_thinking");
    assert_eq!(content[0]["data"], "b3BhcXVlCg==");
    assert_eq!(content[1]["type"], "text");
}

#[test]
fn encodes_images_in_user_messages() {
    let request = ChatRequest::new(vec![Message {
        mtype: caelum_harness::model::message::MessageType::User,
        blocks: vec![
            Block::Image(ImageBlock {
                source: ImageSource::Base64 {
                    mime: "image/png".to_string(),
                    data: vec![1, 2, 3],
                },
                opaque: None,
            }),
            Block::Image(ImageBlock {
                source: ImageSource::Url("https://example.com/a.png".to_string()),
                opaque: None,
            }),
        ],
        meta: None,
    }]);

    let body = codec().encode_request(&request, false).expect("request");
    let content = body["messages"][0]["content"].as_array().expect("content");
    assert_eq!(content[0]["source"]["type"], "base64");
    assert_eq!(content[0]["source"]["media_type"], "image/png");
    assert_eq!(content[0]["source"]["data"], "AQID");
    assert_eq!(content[1]["source"]["type"], "url");
    assert_eq!(content[1]["source"]["url"], "https://example.com/a.png");
}

#[test]
fn invalid_tool_arguments_are_not_replayed_verbatim() {
    let request = ChatRequest::new(vec![
        Message::user("hi"),
        Message::assistant(vec![Block::ToolCall(ToolCallBlock {
            id: "toolu_1".to_string(),
            name: "search".to_string(),
            args: Value::Null,
            raw_args: Some("{bad".to_string()),
            args_error: Some("bad json".to_string()),
            provider_item_id: None,
            opaque: None,
        })]),
    ]);

    let body = codec().encode_request(&request, false).expect("request");
    assert_eq!(body["messages"][1]["content"][0]["input"], json!({}));
}

#[test]
fn decodes_redacted_thinking_in_response_and_stream() {
    let raw = json!({
        "id": "msg_redacted",
        "content": [
            { "type": "redacted_thinking", "data": "b3BhcXVlCg==" },
            { "type": "text", "text": "done" }
        ],
        "stop_reason": "end_turn"
    })
    .to_string();

    let message = codec().decode_response(&raw).expect("response");
    assert_eq!(
        message.blocks[0],
        Block::Opaque(Opaque {
            origin: origin("anthropic", "claude-sonnet-4-5"),
            kind: "anthropic.redacted_thinking".to_string(),
            data: "b3BhcXVlCg==".to_string(),
            item_id: None,
        })
    );

    let mut decoder = codec().stream_decoder();
    decoder
        .push(r#"{"type":"message_start","message":{"id":"msg_redacted"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"b3BhcXVlCg=="}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"content_block_stop","index":0}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#)
        .unwrap();
    decoder.push(r#"{"type":"message_stop"}"#).unwrap();

    let streamed = decoder.finish().expect("streamed");
    assert_eq!(streamed.blocks, message.blocks[..1].to_vec());
}

#[test]
fn non_object_tool_input_sets_args_error() {
    let raw = json!({
        "id": "msg_bad_tool",
        "content": [
            { "type": "tool_use", "id": "toolu_1", "name": "ping", "input": "oops" }
        ],
        "stop_reason": "tool_use"
    })
    .to_string();

    let message = codec().decode_response(&raw).expect("response");
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.args, Value::Null);
    assert!(call.args_error.is_some());
}

#[test]
fn refusal_empty_response_is_not_retryable() {
    let raw = json!({
        "id": "msg_refusal",
        "content": [],
        "stop_reason": "refusal"
    })
    .to_string();

    let error = codec().decode_response(&raw).expect_err("refusal");
    assert_eq!(error.code(), ErrorCode::Refusal);
    assert!(!error.is_retryable());
}

#[test]
fn maps_non_streaming_and_stream_errors() {
    let raw = json!({
        "error": {
            "type": "rate_limit_error",
            "message": "Rate limit exceeded"
        }
    })
    .to_string();
    let error = codec().decode_response(&raw).expect_err("error response");
    assert_eq!(error.code(), ErrorCode::RateLimit);
    assert!(error.is_retryable());

    let codec = codec();
    let mut decoder = codec.stream_decoder();
    let stream_err = decoder
        .push(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)
        .expect_err("stream error");
    assert_eq!(stream_err.code(), ErrorCode::Server);
    assert!(stream_err.is_retryable());
}

#[test]
fn distinguishes_stream_json_and_protocol_errors_and_missing_stop() {
    let codec = codec();
    let mut decoder = codec.stream_decoder();
    let json_err = decoder.push("{").expect_err("syntax error");
    assert_eq!(json_err.code(), ErrorCode::Json);

    let proto_err = decoder
        .push(r#"{"type":"message_start","message":"bad"}"#)
        .expect_err("shape error");
    assert_eq!(proto_err.code(), ErrorCode::Protocol);

    let mut incomplete = codec.stream_decoder();
    incomplete
        .push(r#"{"type":"message_start","message":{"id":"m1"}}"#)
        .unwrap();
    incomplete
        .push(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        )
        .unwrap();
    incomplete
        .push(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .unwrap();
    incomplete
        .push(r#"{"type":"content_block_stop","index":0}"#)
        .unwrap();
    incomplete
        .push(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#)
        .unwrap();
    let closed_err = incomplete.finish().expect_err("missing message_stop");
    assert_eq!(closed_err.code(), ErrorCode::StreamClosed);
}

#[test]
fn decoders_do_not_share_stream_state() {
    let codec = codec();
    let mut first = codec.stream_decoder();
    let mut second = codec.stream_decoder();

    first
        .push(r#"{"type":"message_start","message":{"id":"first"}}"#)
        .unwrap();
    second
        .push(r#"{"type":"message_start","message":{"id":"second"}}"#)
        .unwrap();

    first
        .push(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        )
        .unwrap();
    second
        .push(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        )
        .unwrap();

    first
        .push(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"A"}}"#,
        )
        .unwrap();
    second
        .push(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"B"}}"#,
        )
        .unwrap();

    for decoder in [&mut first, &mut second] {
        decoder
            .push(r#"{"type":"content_block_stop","index":0}"#)
            .unwrap();
        decoder
            .push(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#)
            .unwrap();
        decoder.push(r#"{"type":"message_stop"}"#).unwrap();
    }

    let first_msg = first.finish().unwrap();
    let second_msg = second.finish().unwrap();
    assert_eq!(
        first_msg.blocks,
        vec![Block::Text(TextBlock {
            text: "A".to_string(),
            opaque: None,
        })]
    );
    assert_eq!(
        second_msg.blocks,
        vec![Block::Text(TextBlock {
            text: "B".to_string(),
            opaque: None,
        })]
    );
}

#[test]
fn request_errors_are_reported_clearly() {
    let request = ChatRequest::new(vec![Message::assistant(vec![Block::ToolResult(
        ToolResultBlock {
            call_id: "toolu_1".to_string(),
            name: "ping".to_string(),
            content: Vec::new(),
            is_error: false,
        },
    )])]);

    let error = codec()
        .encode_request(&request, false)
        .expect_err("invalid history");
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
}

#[test]
fn ignores_unknown_blocks_deltas_and_events() {
    let raw = json!({
        "id": "msg_unknown",
        "content": [
            { "type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {} },
            { "type": "text", "text": "ok" }
        ],
        "stop_reason": "end_turn"
    })
    .to_string();
    let decoded = codec().decode_response(&raw).expect("response");
    assert_eq!(
        decoded.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );

    let mut decoder = codec().stream_decoder();
    decoder
        .push(r#"{"type":"message_start","message":{"id":"msg_unknown"}}"#)
        .unwrap();
    decoder.push(r#"{"type":"future_event","x":1}"#).unwrap();
    decoder
        .push(r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srv_1"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"content_block_stop","index":0}"#)
        .unwrap();
    decoder
        .push(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        )
        .unwrap();
    decoder
        .push(r#"{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":{}}}"#)
        .unwrap();
    decoder
        .push(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}"#,
        )
        .unwrap();
    decoder
        .push(r#"{"type":"content_block_stop","index":1}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#)
        .unwrap();
    decoder.push(r#"{"type":"message_stop"}"#).unwrap();

    let streamed = decoder.finish().expect("streamed");
    assert_eq!(streamed.blocks, decoded.blocks);
}

#[test]
fn thinking_expands_max_tokens_and_omits_sampling_params() {
    let mut request = ChatRequest::new(vec![Message::user("hi")]);
    request.reasoning_effort = Some(ReasoningEffort::High);
    request.max_tokens = Some(4096);
    request.temperature = Some(0.2);
    request.top_p = Some(0.9);

    let body = codec().encode_request(&request, false).expect("request");
    assert_eq!(
        body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 8192 })
    );
    assert_eq!(body["max_tokens"], 8192 + 4096);
    assert!(body.get("temperature").is_none());
    assert!(body.get("top_p").is_none());
}

#[test]
fn filters_empty_text_inside_tool_result() {
    let request = ChatRequest::new(vec![
        Message::user("hi"),
        Message::assistant(vec![Block::ToolCall(ToolCallBlock {
            id: "toolu_1".to_string(),
            name: "ping".to_string(),
            args: json!({}),
            raw_args: None,
            args_error: None,
            provider_item_id: None,
            opaque: None,
        })]),
        Message::tool_result("toolu_1", "ping", "", false),
    ]);

    let body = codec().encode_request(&request, false).expect("request");
    assert_eq!(body["messages"][2]["content"][0]["content"], json!([]));
}

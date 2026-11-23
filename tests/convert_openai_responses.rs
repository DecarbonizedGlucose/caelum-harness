use caelum_harness::error::ErrorCode;
use caelum_harness::model::convert::ProtocolCodec;
use caelum_harness::model::convert::openai_responses::{
    OpenAiResponsesCodec, OpenAiResponsesCompat,
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

fn origin(vendor: &str, model: &str) -> Origin {
    Origin {
        protocol: Protocol::OpenaiResp,
        vendor: vendor.to_string(),
        model: model.to_string(),
    }
}

fn codec(vendor: &str, model: &str, compat: OpenAiResponsesCompat) -> OpenAiResponsesCodec {
    OpenAiResponsesCodec::new(origin(vendor, model), compat)
}

fn openai() -> OpenAiResponsesCodec {
    codec("openai", "gpt-5", OpenAiResponsesCompat::openai())
}

fn deepseek() -> OpenAiResponsesCodec {
    codec(
        "deepseek",
        "deepseek-flash",
        OpenAiResponsesCompat::deepseek(),
    )
}

fn fixture(name: &str) -> String {
    let path = format!("tests/fixtures/openai_responses/{name}");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("could not read {path}: {error}"))
}

fn meta(message: &Message) -> &AssistantMeta {
    message.meta.as_ref().expect("assistant metadata")
}

fn decode_stream(codec: &OpenAiResponsesCodec, name: &str) -> (Vec<StreamEvent>, Message) {
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

fn message_meta(id: &str) -> Opaque {
    Opaque {
        origin: origin("openai", "gpt-5"),
        kind: "openai.responses.message".to_string(),
        data: json!({
            "id": id,
            "status": "completed",
            "phase": "final_answer"
        })
        .to_string(),
        item_id: Some(id.to_string()),
    }
}

fn reasoning_opaque(id: &str, encrypted: &str) -> Opaque {
    Opaque {
        origin: origin("openai", "gpt-5"),
        kind: "openai.responses.reasoning".to_string(),
        data: json!({
            "type": "reasoning",
            "id": id,
            "encrypted_content": encrypted
        })
        .to_string(),
        item_id: Some(id.to_string()),
    }
}

#[test]
fn decodes_deepseek_text_response() {
    let message = deepseek()
        .decode_response(&fixture("deepseek_text.json"))
        .expect("response should decode");

    assert_eq!(message.blocks.len(), 1);
    let Block::Text(text) = &message.blocks[0] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "ok");
    let opaque = text.opaque.as_ref().expect("message opaque");
    assert_eq!(opaque.kind, "openai.responses.message");
    assert_eq!(
        opaque.item_id.as_deref(),
        Some("783dcf73-335e-4d57-989b-f5eeaf9bcb05")
    );
    let meta = meta(&message);
    assert_eq!(meta.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(
        meta.usage,
        Some(Usage {
            input_tokens: Some(9),
            output_tokens: Some(1),
            cache_read_tokens: Some(0),
            cache_write_tokens: None,
            reasoning_tokens: Some(0),
            total_tokens: Some(10),
            cost: None,
        })
    );
}

#[test]
fn decodes_deepseek_reasoning_response_with_opaque() {
    let message = deepseek()
        .decode_response(&fixture("deepseek_reasoning.json"))
        .expect("response should decode");

    assert_eq!(message.blocks.len(), 2);
    let Block::Reasoning(reasoning) = &message.blocks[0] else {
        panic!("expected reasoning block");
    };
    assert!(reasoning.text.contains("17*3=51"));
    let opaque = reasoning.opaque.as_ref().expect("reasoning opaque");
    assert_eq!(opaque.kind, "openai.responses.reasoning");
    assert_eq!(
        opaque.item_id.as_deref(),
        Some("01617861-f5c2-4c11-8912-89c08ff2ec5a")
    );
    assert!(
        opaque
            .data
            .contains("07cab2aa-2c90-47d9-9f3a-badc40859936-0")
    );
    let Block::Text(text) = &message.blocks[1] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "51");
    assert_eq!(
        text.opaque
            .as_ref()
            .and_then(|opaque| opaque.item_id.as_deref()),
        Some("6a25bc35-a9f1-4a2c-a0a9-d9fe0e245a75")
    );
    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.output_tokens, Some(21));
    assert_eq!(usage.reasoning_tokens, Some(19));
    assert_eq!(usage.total_tokens, Some(65));
}

#[test]
fn decodes_deepseek_tool_call_with_separate_ids() {
    let message = deepseek()
        .decode_response(&fixture("deepseek_tool_call.json"))
        .expect("response should decode");

    assert_eq!(
        message.blocks,
        vec![Block::ToolCall(ToolCallBlock {
            id: "call_00_Y90z5A0cMPShHvWo0usy3731".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Paris" }),
            raw_args: Some(r#"{"city": "Paris"}"#.to_string()),
            args_error: None,
            provider_item_id: Some("c08e0ff8-3c68-4ee5-ad79-4e99e4cdd433".to_string()),
            opaque: None,
        })]
    );
    let meta = meta(&message);
    assert_eq!(meta.stop_reason, Some(StopReason::ToolCall));
    let usage = meta.usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(155));
    assert_eq!(usage.cache_read_tokens, Some(128));
    assert_eq!(usage.output_tokens, Some(33));
    assert_eq!(usage.total_tokens, Some(316));
}

#[test]
fn decodes_openrouter_text_with_cost() {
    let codec = codec(
        "openrouter",
        "mistralai/mistral-nemo",
        OpenAiResponsesCompat::openrouter(),
    );
    let message = codec
        .decode_response(&fixture("openrouter_text.json"))
        .expect("response should decode");

    let Block::Text(text) = &message.blocks[0] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "ok");
    assert_eq!(
        text.opaque
            .as_ref()
            .and_then(|opaque| opaque.item_id.as_deref()),
        Some("msg_tmp_9o9rx80110j")
    );
    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.total_tokens, Some(11));
    assert!(
        usage
            .cost
            .is_some_and(|cost| (cost - 3.21e-07).abs() < 1e-12)
    );
}

#[test]
fn streams_text_and_reasoning_with_opaque() {
    let (events, message) = decode_stream(&deepseek(), "deepseek_stream_text.jsonl");

    let reasoning_deltas = events
        .iter()
        .filter(|event| matches!(event, StreamEvent::ReasoningDelta { .. }))
        .count();
    assert!(reasoning_deltas > 1);

    assert!(matches!(
        events.first(),
        Some(StreamEvent::MessageStart { response_id: Some(id) }) if id == "8e704abb-d47d-49e8-a165-0cdad333ad58"
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
        reasoning.text,
        "The user just wants \"ok\". Need exactly that."
    );
    let opaque = reasoning.opaque.as_ref().expect("reasoning opaque");
    assert_eq!(opaque.kind, "openai.responses.reasoning");
    assert!(
        opaque
            .data
            .contains("8e704abb-d47d-49e8-a165-0cdad333ad58-0")
    );
    let Block::Text(text) = &message.blocks[1] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "ok");
    let opaque = text.opaque.as_ref().expect("message opaque");
    assert_eq!(opaque.kind, "openai.responses.message");
    assert_eq!(
        opaque.item_id.as_deref(),
        Some("9727edc8-4b55-4b76-9c7c-3a4ac67f0df8")
    );
    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(35));
    assert_eq!(usage.output_tokens, Some(13));
    assert_eq!(usage.reasoning_tokens, Some(11));
    assert_eq!(usage.total_tokens, Some(48));
}

#[test]
fn streams_tool_arguments() {
    let (events, message) = decode_stream(&deepseek(), "deepseek_stream_tool_call.jsonl");

    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ToolCallStart { name, provider_item_id: Some(id), .. }
            if name == "get_weather" && id == "d26cd15d-dfb6-42e0-bafb-eea9d4a63eb7"
    )));
    assert_eq!(
        message.blocks,
        vec![Block::ToolCall(ToolCallBlock {
            id: "call_00_sQvBJxbdDrKQKFfnNgVh6895".to_string(),
            name: "get_weather".to_string(),
            args: json!({ "city": "Paris" }),
            raw_args: Some(r#"{"city": "Paris"}"#.to_string()),
            args_error: None,
            provider_item_id: Some("d26cd15d-dfb6-42e0-bafb-eea9d4a63eb7".to_string()),
            opaque: None,
        })]
    );
    assert_eq!(meta(&message).stop_reason, Some(StopReason::ToolCall));
}

#[test]
fn encodes_request_with_instructions_and_reasoning() {
    let request = ChatRequest {
        messages: vec![
            Message::system("Be concise."),
            Message::user("What is the weather in Paris?"),
            Message::assistant(vec![
                Block::Reasoning(ReasoningBlock {
                    text: "Need the weather tool.".to_string(),
                    opaque: Some(reasoning_opaque("rs_1", "encrypted-1")),
                }),
                Block::ToolCall(ToolCallBlock {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    args: json!({ "city": "Paris" }),
                    raw_args: None,
                    args_error: None,
                    provider_item_id: Some("fc_1".to_string()),
                    opaque: None,
                }),
            ]),
            Message::tool_result("call_1", "get_weather", "20C", false),
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
        max_tokens: Some(256),
        stop: Vec::new(),
        reasoning_effort: Some(ReasoningEffort::Medium),
    };

    let body = openai()
        .encode_request(&request, true)
        .expect("request should encode");

    assert_eq!(body["model"], "gpt-5");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["instructions"], "Be concise.");
    assert_eq!(body["max_output_tokens"], 256);
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert!(body["tools"][0].get("function").is_none());
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(
        body["reasoning"],
        json!({ "effort": "medium", "summary": "auto" })
    );
    assert!(body.get("temperature").is_none());
    assert!(body.get("top_p").is_none());

    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 5);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[1]["encrypted_content"], "encrypted-1");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["id"], "fc_1");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[3]["type"], "function_call_output");
    assert_eq!(input[3]["call_id"], "call_1");
    assert_eq!(input[3]["output"], "20C");
    assert_eq!(input[4]["role"], "user");
}

#[test]
fn deepseek_compat_keeps_sampling_and_skips_reasoning_replay() {
    let request = ChatRequest {
        messages: vec![
            Message::user("hi"),
            Message::assistant(vec![
                Block::Reasoning(ReasoningBlock {
                    text: "thought".to_string(),
                    opaque: Some(Opaque {
                        origin: origin("deepseek", "deepseek-flash"),
                        kind: "openai.responses.reasoning".to_string(),
                        data: json!({
                            "type": "reasoning",
                            "id": "rs_1",
                            "encrypted_content": "enc"
                        })
                        .to_string(),
                        item_id: Some("rs_1".to_string()),
                    }),
                }),
                Block::Text(TextBlock {
                    text: "hello".to_string(),
                    opaque: None,
                }),
            ]),
        ],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: Some(0.5),
        top_p: None,
        max_tokens: None,
        stop: Vec::new(),
        reasoning_effort: Some(ReasoningEffort::Low),
    };

    let body = deepseek()
        .encode_request(&request, false)
        .expect("request should encode");

    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["reasoning"]["effort"], "low");
    assert!(body.get("include").is_none());
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2);
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[1]["content"][0]["text"], "hello");
}

#[test]
fn replays_message_item_with_id_status_and_phase() {
    let request = ChatRequest::new(vec![
        Message::user("hi"),
        Message::assistant(vec![Block::Text(TextBlock {
            text: "hello".to_string(),
            opaque: Some(message_meta("msg_1")),
        })]),
    ]);

    let body = openai()
        .encode_request(&request, false)
        .expect("request should encode");
    let item = &body["input"][1];
    assert_eq!(item["type"], "message");
    assert_eq!(item["id"], "msg_1");
    assert_eq!(item["role"], "assistant");
    assert_eq!(item["status"], "completed");
    assert_eq!(item["phase"], "final_answer");
    assert_eq!(item["content"][0]["type"], "output_text");
    assert_eq!(item["content"][0]["text"], "hello");
}

#[test]
fn skips_foreign_reasoning_and_non_fc_item_ids() {
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
                args: json!({}),
                raw_args: Some("{}".to_string()),
                args_error: None,
                provider_item_id: Some("not-an-fc-id".to_string()),
                opaque: None,
            }),
        ]),
    ]);

    let body = openai()
        .encode_request(&request, false)
        .expect("request should encode");
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2);
    assert_eq!(input[1]["type"], "function_call");
    assert!(input[1].get("id").is_none());
}

#[test]
fn stop_option_is_unsupported() {
    let mut request = ChatRequest::new(vec![Message::user("hi")]);
    request.stop = vec!["END".to_string()];

    let error = openai()
        .encode_request(&request, false)
        .expect_err("stop should be rejected");
    assert_eq!(error.code(), ErrorCode::UnsupportedOption);
}

#[test]
fn encodes_images_and_tool_result_images() {
    let request = ChatRequest::new(vec![
        Message {
            mtype: MessageType::User,
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
        },
        Message::assistant(vec![Block::ToolCall(ToolCallBlock {
            id: "call_1".to_string(),
            name: "screenshot".to_string(),
            args: json!({}),
            raw_args: None,
            args_error: None,
            provider_item_id: None,
            opaque: None,
        })]),
        Message {
            mtype: MessageType::Tool,
            blocks: vec![Block::ToolResult(ToolResultBlock {
                call_id: "call_1".to_string(),
                name: "screenshot".to_string(),
                content: vec![
                    Block::Text(TextBlock {
                        text: "image".to_string(),
                        opaque: None,
                    }),
                    Block::Image(ImageBlock {
                        source: ImageSource::Base64 {
                            mime: "image/png".to_string(),
                            data: vec![4, 5, 6],
                        },
                        opaque: None,
                    }),
                ],
                is_error: false,
            })],
            meta: None,
        },
    ]);

    let body = openai()
        .encode_request(&request, false)
        .expect("request should encode");
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input[0]["content"][0]["type"], "input_image");
    assert_eq!(
        input[0]["content"][0]["image_url"],
        "data:image/png;base64,AQID"
    );
    assert_eq!(input[0]["content"][1]["type"], "input_image");
    assert_eq!(
        input[0]["content"][1]["image_url"],
        "https://example.com/a.png"
    );
    assert_eq!(input[2]["type"], "function_call_output");
    assert_eq!(input[2]["output"][0]["type"], "input_text");
    assert_eq!(input[2]["output"][1]["type"], "input_image");
    assert_eq!(
        input[2]["output"][1]["image_url"],
        "data:image/png;base64,BAUG"
    );
}

#[test]
fn maps_error_bodies_and_stream_errors() {
    let raw = json!({
        "error": {
            "message": "Rate limit exceeded",
            "code": "rate_limit_exceeded"
        }
    })
    .to_string();
    let error = openai().decode_response(&raw).expect_err("error body");
    assert_eq!(error.code(), ErrorCode::RateLimit);
    assert!(error.is_retryable());

    let mut decoder = openai().stream_decoder();
    let stream_error = decoder
        .push(r#"{"type":"error","message":"context_length_exceeded","code":"context_length_exceeded"}"#)
        .expect_err("stream error");
    assert_eq!(stream_error.code(), ErrorCode::ContextWindowExceeded);

    let mut decoder = openai().stream_decoder();
    let policy_error = decoder
        .push(r#"{"type":"error","message":"content_policy_violation","code":"content_policy_violation"}"#)
        .expect_err("policy error");
    assert_eq!(policy_error.code(), ErrorCode::ContentFilter);
    assert!(!policy_error.is_retryable());
}

#[test]
fn maps_failed_response_error() {
    let mut decoder = openai().stream_decoder();
    let error = decoder
        .push(
            &json!({
                "type": "response.failed",
                "response": {
                    "id": "resp_failed",
                    "status": "failed",
                    "error": { "code": "server_error", "message": "boom" }
                }
            })
            .to_string(),
        )
        .expect_err("failed response");
    assert_eq!(error.code(), ErrorCode::Server);
    assert!(error.is_retryable());
}

#[test]
fn incomplete_max_output_tokens_marks_truncated_tool_call() {
    let raw = json!({
        "id": "resp_incomplete",
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}"
            }
        ]
    })
    .to_string();

    let message = openai().decode_response(&raw).expect("incomplete response");
    assert_eq!(meta(&message).stop_reason, Some(StopReason::MaxTokens));
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert!(call.args_error.is_some());
}

#[test]
fn incomplete_content_filter_empty_response_is_not_retryable() {
    let raw = json!({
        "id": "resp_incomplete",
        "status": "incomplete",
        "incomplete_details": { "reason": "content_filter" },
        "output": []
    })
    .to_string();

    let error = openai().decode_response(&raw).expect_err("content filter");
    assert_eq!(error.code(), ErrorCode::ContentFilter);
    assert!(!error.is_retryable());
}

#[test]
fn ignores_unknown_output_items_and_stream_events() {
    let raw = json!({
        "id": "resp_unknown",
        "status": "completed",
        "output": [
            { "type": "web_search_call", "id": "ws_1", "status": "completed" },
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "ok" }]
            }
        ]
    })
    .to_string();

    let message = openai()
        .decode_response(&raw)
        .expect("unknown item should be ignored");
    assert_eq!(message.blocks.len(), 1);
    let Block::Text(text) = &message.blocks[0] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "ok");
    assert_eq!(
        text.opaque
            .as_ref()
            .and_then(|opaque| opaque.item_id.as_deref()),
        Some("msg_1")
    );

    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_unknown"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"web_search_call","id":"ws_1"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"web_search_call","id":"ws_1","status":"completed"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","status":"in_progress","content":[]}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_text.delta","output_index":1,"delta":"ok"}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"ok"}]}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.completed","response":{"id":"resp_unknown","status":"completed","output":[{"type":"web_search_call","id":"ws_1","status":"completed"},{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}}"#)
        .unwrap();

    let streamed = decoder.finish().expect("streamed");
    assert_eq!(streamed.blocks.len(), 1);
    let Block::Text(text) = &streamed.blocks[0] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "ok");
    assert_eq!(
        text.opaque
            .as_ref()
            .and_then(|opaque| opaque.item_id.as_deref()),
        Some("msg_1")
    );
}

#[test]
fn falls_back_to_done_snapshots_without_deltas() {
    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_snapshot"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","content":[],"summary":[],"encrypted_content":"enc"}}"#)
        .unwrap();
    decoder
        .push(
            r#"{"type":"response.reasoning_summary_text.done","output_index":0,"text":"summary"}"#,
        )
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","status":"completed","content":[],"summary":[{"type":"summary_text","text":"summary"}],"encrypted_content":"enc"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","status":"in_progress","content":[]}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_text.done","output_index":1,"text":"hello"}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"hello"}]}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.completed","response":{"id":"resp_snapshot","status":"completed","output":[]}}"#)
        .unwrap();

    let message = decoder.finish().expect("streamed");
    assert_eq!(message.blocks.len(), 2);
    let Block::Reasoning(reasoning) = &message.blocks[0] else {
        panic!("expected reasoning");
    };
    assert_eq!(reasoning.text, "summary");
    assert!(reasoning.opaque.is_some());
    let Block::Text(text) = &message.blocks[1] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "hello");
    assert_eq!(
        text.opaque
            .as_ref()
            .and_then(|opaque| opaque.item_id.as_deref()),
        Some("msg_1")
    );
}

#[test]
fn usage_subtracts_cached_and_cache_write_tokens() {
    let raw = json!({
        "id": "resp_usage",
        "status": "completed",
        "output": [{ "type": "message", "content": [{ "type": "output_text", "text": "ok" }] }],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 20, "cache_write_tokens": 30 },
            "output_tokens": 50,
            "output_tokens_details": { "reasoning_tokens": 10 },
            "total_tokens": 150
        }
    })
    .to_string();

    let message = openai().decode_response(&raw).expect("usage response");
    let usage = meta(&message).usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(50));
    assert_eq!(usage.cache_read_tokens, Some(20));
    assert_eq!(usage.cache_write_tokens, Some(30));
    assert_eq!(usage.output_tokens, Some(50));
    assert_eq!(usage.reasoning_tokens, Some(10));
    assert_eq!(usage.total_tokens, Some(150));
}

#[test]
fn openai_compat_includes_encrypted_reasoning_without_effort() {
    let request = ChatRequest::new(vec![Message::user("hi")]);
    let body = openai()
        .encode_request(&request, false)
        .expect("request should encode");

    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert!(body.get("reasoning").is_none());
}

#[test]
fn explicit_no_reasoning_keeps_sampling() {
    let mut request = ChatRequest::new(vec![Message::user("hi")]);
    request.reasoning_effort = Some(ReasoningEffort::None);
    request.temperature = Some(0.3);

    let body = openai()
        .encode_request(&request, false)
        .expect("request should encode");
    assert_eq!(body["reasoning"], json!({ "effort": "none" }));
    assert_eq!(body["temperature"], 0.3);
}

#[test]
fn tool_call_done_without_arguments_keeps_streamed_deltas() {
    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_args"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#)
        .unwrap();
    decoder
        .push(
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{\"city\":\"Paris\"}"
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","status":"completed","call_id":"call_1","name":"get_weather"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.completed","response":{"id":"resp_args","status":"completed","output":[]}}"#)
        .unwrap();

    let message = decoder.finish().expect("streamed");
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.args, json!({ "city": "Paris" }));
    assert!(call.args_error.is_none());
}

#[test]
fn terminal_output_fills_missing_item_done() {
    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_terminal"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","content":[]}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_text.delta","output_index":0,"delta":"ok"}"#)
        .unwrap();
    decoder
        .push(
            &json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_terminal",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "content": [{ "type": "output_text", "text": "ok" }]
                    }]
                }
            })
            .to_string(),
        )
        .unwrap();

    let message = decoder.finish().expect("streamed");
    let Block::Text(text) = &message.blocks[0] else {
        panic!("expected text");
    };
    assert_eq!(text.text, "ok");
    assert_eq!(
        text.opaque
            .as_ref()
            .and_then(|opaque| opaque.item_id.as_deref()),
        Some("msg_1")
    );
}

#[test]
fn response_done_alias_is_terminal() {
    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_done_alias"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_text.delta","output_index":0,"delta":"ok"}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.done","response":{"id":"resp_done_alias","status":"completed","output":[]}}"#)
        .unwrap();

    let message = decoder.finish().expect("streamed");
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
}

#[test]
fn stream_incomplete_marks_tool_call_truncated() {
    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_incomplete"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#)
        .unwrap();
    decoder
        .push(
            &json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": "{\"city\":\"Par"
            })
            .to_string(),
        )
        .unwrap();
    decoder
        .push(
            &json!({
                "type": "response.incomplete",
                "response": {
                    "id": "resp_incomplete",
                    "status": "incomplete",
                    "incomplete_details": { "reason": "max_output_tokens" },
                    "output": []
                }
            })
            .to_string(),
        )
        .unwrap();

    let message = decoder.finish().expect("streamed");
    assert_eq!(meta(&message).stop_reason, Some(StopReason::MaxTokens));
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert!(call.args_error.is_some());
    assert_eq!(call.provider_item_id.as_deref(), Some("fc_1"));
}

#[test]
fn missing_terminal_event_reports_stream_closed() {
    let mut decoder = openai().stream_decoder();
    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_open"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_text.delta","output_index":0,"delta":"hi"}"#)
        .unwrap();

    let error = decoder.finish().expect_err("missing terminal");
    assert_eq!(error.code(), ErrorCode::StreamClosed);
}

#[test]
fn done_payload_is_ignored() {
    let mut decoder = openai().stream_decoder();
    let events = decoder.push("[DONE]").expect("done marker");
    assert!(events.is_empty());

    decoder
        .push(r#"{"type":"response.created","response":{"id":"resp_done"}}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.output_text.delta","output_index":0,"delta":"ok"}"#)
        .unwrap();
    decoder
        .push(r#"{"type":"response.completed","response":{"id":"resp_done","status":"completed","output":[]}}"#)
        .unwrap();

    let message = decoder.finish().expect("streamed");
    assert_eq!(
        message.blocks,
        vec![Block::Text(TextBlock {
            text: "ok".to_string(),
            opaque: None,
        })]
    );
}

#[test]
fn decoders_do_not_share_stream_state() {
    let codec = openai();
    let mut first = codec.stream_decoder();
    let mut second = codec.stream_decoder();

    first
        .push(r#"{"type":"response.created","response":{"id":"first"}}"#)
        .unwrap();
    second
        .push(r#"{"type":"response.created","response":{"id":"second"}}"#)
        .unwrap();
    first
        .push(r#"{"type":"response.output_text.delta","output_index":0,"delta":"A"}"#)
        .unwrap();
    second
        .push(r#"{"type":"response.output_text.delta","output_index":0,"delta":"B"}"#)
        .unwrap();
    first
        .push(r#"{"type":"response.completed","response":{"id":"first","status":"completed","output":[]}}"#)
        .unwrap();
    second
        .push(r#"{"type":"response.completed","response":{"id":"second","status":"completed","output":[]}}"#)
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

#[test]
fn non_object_tool_arguments_set_args_error() {
    let raw = json!({
        "id": "resp_bad_args",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "ping",
            "arguments": "[1,2]"
        }]
    })
    .to_string();

    let message = openai().decode_response(&raw).expect("response");
    let Block::ToolCall(call) = &message.blocks[0] else {
        panic!("expected tool call");
    };
    assert_eq!(call.args, Value::Null);
    assert!(call.args_error.is_some());
}

#[test]
fn invalid_tool_arguments_are_not_replayed_verbatim() {
    let request = ChatRequest::new(vec![
        Message::user("hi"),
        Message::assistant(vec![Block::ToolCall(ToolCallBlock {
            id: "call_1".to_string(),
            name: "ping".to_string(),
            args: Value::Null,
            raw_args: Some("{bad".to_string()),
            args_error: Some("bad json".to_string()),
            provider_item_id: None,
            opaque: None,
        })]),
    ]);

    let body = openai()
        .encode_request(&request, false)
        .expect("request should encode");
    assert_eq!(body["input"][1]["arguments"], "{}");
}

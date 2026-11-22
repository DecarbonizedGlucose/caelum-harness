use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::HarnessError;
use crate::model::message::{
    AssistantMeta, Block, Message, MessageType, Opaque, Origin, ReasoningBlock, StopReason,
    TextBlock, ToolCallBlock, Usage,
};

/// Content block kinds that can be streamed from a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Text,
    Image,
    Reasoning,
    ToolCall,
}

/// Provider-neutral model stream event.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    MessageStart {
        response_id: Option<String>,
    },
    BlockStart {
        index: usize,
        kind: BlockKind,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    ReasoningDelta {
        index: usize,
        text: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
        provider_item_id: Option<String>,
    },
    ToolArgsDelta {
        index: usize,
        fragment: String,
    },
    BlockEnd {
        index: usize,
        opaque: Option<Opaque>,
    },
    CompleteBlock {
        index: usize,
        block: Block,
    },
    Usage(Usage),
    Finish {
        stop_reason: Option<StopReason>,
    },
}

struct PartialBlock {
    kind: BlockKind,
    text: String,
    tool_id: Option<String>,
    tool_name: Option<String>,
    provider_item_id: Option<String>,
    raw_args: String,
    opaque: Option<Opaque>,
    complete: Option<Block>,
}

impl PartialBlock {
    fn new(kind: BlockKind) -> Self {
        Self {
            kind,
            text: String::new(),
            tool_id: None,
            tool_name: None,
            provider_item_id: None,
            raw_args: String::new(),
            opaque: None,
            complete: None,
        }
    }
}

/// Incrementally assembles model stream events into one assistant message.
pub struct Assembler {
    origin: Origin,
    blocks: BTreeMap<usize, PartialBlock>,
    response_id: Option<String>,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
    finish_seen: bool,
}

impl Assembler {
    pub fn new(origin: Origin) -> Self {
        Self {
            origin,
            blocks: BTreeMap::new(),
            response_id: None,
            usage: None,
            stop_reason: None,
            finish_seen: false,
        }
    }

    pub fn push(&mut self, event: StreamEvent) -> Result<(), HarnessError> {
        let protocol = self.origin.protocol;
        match event {
            StreamEvent::MessageStart { response_id } => {
                if response_id.is_some() {
                    self.response_id = response_id;
                }
            }
            StreamEvent::BlockStart { index, kind } => {
                self.ensure(index, kind);
            }
            StreamEvent::TextDelta { index, text } => {
                let partial = self.ensure(index, BlockKind::Text);
                if partial.complete.is_none() {
                    partial.text.push_str(&text);
                }
            }
            StreamEvent::ReasoningDelta { index, text } => {
                let partial = self.ensure(index, BlockKind::Reasoning);
                if partial.complete.is_none() {
                    partial.text.push_str(&text);
                }
            }
            StreamEvent::ToolCallStart {
                index,
                id,
                name,
                provider_item_id,
            } => {
                let partial = self.ensure(index, BlockKind::ToolCall);
                if partial.complete.is_none() {
                    if !id.is_empty() {
                        partial.tool_id = Some(id);
                    }
                    if !name.is_empty() {
                        partial.tool_name = Some(name);
                    }
                    partial.provider_item_id = provider_item_id;
                }
            }
            StreamEvent::ToolArgsDelta { index, fragment } => {
                let partial = self.ensure(index, BlockKind::ToolCall);
                if partial.complete.is_none() {
                    partial.raw_args.push_str(&fragment);
                }
            }
            StreamEvent::BlockEnd { index, opaque } => {
                let Some(partial) = self.blocks.get_mut(&index) else {
                    return Err(HarnessError::protocol(
                        protocol,
                        format!("blocks[{index}]"),
                        "block end without a block start",
                    ));
                };
                if partial.complete.is_none() {
                    partial.opaque = opaque;
                    let block = assemble_partial(protocol, index, partial)?;
                    partial.complete = Some(block);
                }
            }
            StreamEvent::CompleteBlock { index, block } => {
                let partial = self
                    .blocks
                    .entry(index)
                    .or_insert_with(|| PartialBlock::new(BlockKind::Text));
                partial.complete = Some(block);
            }
            StreamEvent::Usage(usage) => {
                if let Some(current) = &mut self.usage {
                    current.merge(usage);
                } else if !usage.is_empty() {
                    self.usage = Some(usage);
                }
            }
            StreamEvent::Finish { stop_reason } => {
                self.finish_seen = true;
                self.stop_reason = stop_reason;
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Message, HarnessError> {
        let protocol = self.origin.protocol;
        if !self.finish_seen {
            return Err(HarnessError::stream_closed(protocol));
        }

        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (index, partial) in self.blocks {
            blocks.push(assemble_partial(protocol, index, &partial)?);
        }

        if blocks.is_empty() {
            return match self.stop_reason {
                Some(StopReason::ContentFilter) => Err(HarnessError::content_filter(protocol)),
                Some(StopReason::Refusal) => Err(HarnessError::refusal(protocol)),
                _ => Err(HarnessError::empty_response(protocol)),
            };
        }

        if self.stop_reason == Some(StopReason::MaxTokens) {
            for block in &mut blocks {
                if let Block::ToolCall(call) = block
                    && call.args_error.is_none()
                {
                    call.args_error = Some(
                        "response stopped by max_tokens; tool arguments may be truncated"
                            .to_string(),
                    );
                }
            }
        }

        Ok(Message {
            mtype: MessageType::Assistant,
            blocks,
            meta: Some(AssistantMeta {
                origin: self.origin,
                response_id: self.response_id,
                stop_reason: self.stop_reason,
                usage: self.usage,
                opaque: None,
            }),
        })
    }

    pub fn interrupted(self) -> Message {
        let protocol = self.origin.protocol;
        let blocks = self
            .blocks
            .into_iter()
            .filter_map(|(index, partial)| {
                let block = assemble_partial(protocol, index, &partial).ok()?;
                match &block {
                    Block::Text(text) if text.text.trim().is_empty() => None,
                    Block::Text(_) | Block::Reasoning(_) => Some(block),
                    _ => None,
                }
            })
            .collect();

        Message {
            mtype: MessageType::Assistant,
            blocks,
            meta: Some(AssistantMeta {
                origin: self.origin,
                response_id: self.response_id,
                stop_reason: Some(StopReason::Aborted),
                usage: self.usage,
                opaque: None,
            }),
        }
    }

    fn ensure(&mut self, index: usize, kind: BlockKind) -> &mut PartialBlock {
        self.blocks
            .entry(index)
            .or_insert_with(|| PartialBlock::new(kind))
    }
}

fn assemble_partial(
    protocol: crate::model::protocol::Protocol,
    index: usize,
    partial: &PartialBlock,
) -> Result<Block, HarnessError> {
    if let Some(block) = &partial.complete {
        return Ok(block.clone());
    }

    match partial.kind {
        BlockKind::Text => Ok(Block::Text(TextBlock {
            text: partial.text.clone(),
            opaque: partial.opaque.clone(),
        })),
        BlockKind::Reasoning => Ok(Block::Reasoning(ReasoningBlock {
            text: partial.text.clone(),
            opaque: partial.opaque.clone(),
        })),
        BlockKind::ToolCall => {
            let raw_args = partial.raw_args.clone();
            let (args, args_error) = if raw_args.trim().is_empty() {
                (Value::Object(Default::default()), None)
            } else {
                match serde_json::from_str::<Value>(&raw_args) {
                    Ok(value) if value.is_object() => (value, None),
                    Ok(_) => (
                        Value::Null,
                        Some("tool arguments must be a JSON object".to_string()),
                    ),
                    Err(error) => (Value::Null, Some(error.to_string())),
                }
            };

            Ok(Block::ToolCall(ToolCallBlock {
                id: partial
                    .tool_id
                    .clone()
                    .filter(|id| !id.is_empty())
                    .unwrap_or_else(|| format!("call-{index}")),
                name: partial.tool_name.clone().unwrap_or_default(),
                args,
                raw_args: if raw_args.is_empty() {
                    None
                } else {
                    Some(raw_args)
                },
                args_error,
                provider_item_id: partial.provider_item_id.clone(),
                opaque: partial.opaque.clone(),
            }))
        }
        BlockKind::Image => Err(HarnessError::unsupported(
            protocol,
            "streamed image block without complete content",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::protocol::Protocol;
    use serde_json::json;

    fn origin() -> Origin {
        Origin {
            protocol: Protocol::OpenaiChat,
            vendor: "test".to_string(),
            model: "test-model".to_string(),
        }
    }

    fn assembler() -> Assembler {
        Assembler::new(origin())
    }

    #[test]
    fn assembles_delta_only_text() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "hello".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::EndTurn),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        assert_eq!(
            message.blocks,
            vec![Block::Text(TextBlock {
                text: "hello".to_string(),
                opaque: None,
            })]
        );
    }

    #[test]
    fn block_end_is_authoritative() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::BlockStart {
                index: 0,
                kind: BlockKind::Text,
            })
            .unwrap();
        assembler
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "first".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::BlockEnd {
                index: 0,
                opaque: None,
            })
            .unwrap();
        assembler
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "ignored".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::EndTurn),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        assert_eq!(
            message.blocks,
            vec![Block::Text(TextBlock {
                text: "first".to_string(),
                opaque: None,
            })]
        );
    }

    #[test]
    fn assembles_split_tool_arguments() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::ToolCallStart {
                index: 0,
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                provider_item_id: None,
            })
            .unwrap();
        assembler
            .push(StreamEvent::ToolArgsDelta {
                index: 0,
                fragment: "{\"city\":".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::ToolArgsDelta {
                index: 0,
                fragment: " \"Paris\"}".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::ToolCall),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        let Block::ToolCall(call) = &message.blocks[0] else {
            panic!("expected tool call");
        };
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.args, json!({ "city": "Paris" }));
        assert_eq!(call.args_error, None);
    }

    #[test]
    fn invalid_tool_arguments_set_args_error() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::ToolCallStart {
                index: 0,
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                provider_item_id: None,
            })
            .unwrap();
        assembler
            .push(StreamEvent::ToolArgsDelta {
                index: 0,
                fragment: "{".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::ToolCall),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        let Block::ToolCall(call) = &message.blocks[0] else {
            panic!("expected tool call");
        };
        assert_eq!(call.args, Value::Null);
        assert!(call.args_error.is_some());
    }

    #[test]
    fn max_tokens_marks_tool_calls() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::ToolCallStart {
                index: 0,
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                provider_item_id: None,
            })
            .unwrap();
        assembler
            .push(StreamEvent::ToolArgsDelta {
                index: 0,
                fragment: "{\"city\":\"Paris\"}".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::MaxTokens),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        let Block::ToolCall(call) = &message.blocks[0] else {
            panic!("expected tool call");
        };
        assert!(call.args_error.is_some());
    }

    #[test]
    fn interrupted_keeps_only_text_and_reasoning() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "hello".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::ReasoningDelta {
                index: 1,
                text: "thinking".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::ToolCallStart {
                index: 2,
                id: "call-1".to_string(),
                name: "get_weather".to_string(),
                provider_item_id: None,
            })
            .unwrap();
        assembler
            .push(StreamEvent::ToolArgsDelta {
                index: 2,
                fragment: "{\"city\":\"Paris\"}".to_string(),
            })
            .unwrap();

        let message = assembler.interrupted();
        assert_eq!(message.blocks.len(), 2);
        assert!(matches!(message.blocks[0], Block::Text(_)));
        assert!(matches!(message.blocks[1], Block::Reasoning(_)));
        assert_eq!(
            message.meta.expect("metadata").stop_reason,
            Some(StopReason::Aborted)
        );
    }

    #[test]
    fn empty_response_is_an_error() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::EndTurn),
            })
            .unwrap();

        let error = assembler.finish().unwrap_err();
        assert_eq!(error.code(), crate::error::ErrorCode::EmptyResponse);
    }

    #[test]
    fn missing_finish_is_an_error() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "hello".to_string(),
            })
            .unwrap();

        let error = assembler.finish().unwrap_err();
        assert_eq!(error.code(), crate::error::ErrorCode::StreamClosed);
    }

    #[test]
    fn usage_fields_merge() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::TextDelta {
                index: 0,
                text: "hello".to_string(),
            })
            .unwrap();
        assembler
            .push(StreamEvent::Usage(Usage {
                input_tokens: Some(1),
                output_tokens: Some(1),
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
                total_tokens: None,
                cost: None,
            }))
            .unwrap();
        assembler
            .push(StreamEvent::Usage(Usage {
                input_tokens: None,
                output_tokens: Some(2),
                cache_read_tokens: Some(1),
                cache_write_tokens: None,
                reasoning_tokens: Some(1),
                total_tokens: Some(3),
                cost: None,
            }))
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::EndTurn),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        assert_eq!(
            message.meta.expect("metadata").usage,
            Some(Usage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                cache_read_tokens: Some(1),
                cache_write_tokens: None,
                reasoning_tokens: Some(1),
                total_tokens: Some(3),
                cost: None,
            })
        );
    }

    #[test]
    fn missing_tool_id_uses_deterministic_fallback() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::ToolCallStart {
                index: 0,
                id: String::new(),
                name: "get_weather".to_string(),
                provider_item_id: None,
            })
            .unwrap();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::ToolCall),
            })
            .unwrap();

        let message = assembler.finish().unwrap();
        let Block::ToolCall(call) = &message.blocks[0] else {
            panic!("expected tool call");
        };
        assert_eq!(call.id, "call-0");
    }

    #[test]
    fn content_filter_empty_response_is_not_retryable() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::ContentFilter),
            })
            .unwrap();

        let error = assembler.finish().unwrap_err();
        assert_eq!(error.code(), crate::error::ErrorCode::ContentFilter);
        assert!(!error.is_retryable());
    }

    #[test]
    fn refusal_empty_response_is_not_retryable() {
        let mut assembler = assembler();
        assembler
            .push(StreamEvent::Finish {
                stop_reason: Some(StopReason::Refusal),
            })
            .unwrap();

        let error = assembler.finish().unwrap_err();
        assert_eq!(error.code(), crate::error::ErrorCode::Refusal);
        assert!(!error.is_retryable());
    }
}

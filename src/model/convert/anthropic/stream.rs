use std::collections::{HashMap, HashSet};

use crate::error::HarnessError;
use crate::model::convert::StreamDecoder;
use crate::model::convert::anthropic::{decode_usage, stop_reason};
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::message::{Block, Message, Opaque, Origin, StopReason};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, BlockKind, StreamEvent};
use crate::model::wire::anthropic as wire;

const THINKING_OPAQUE_KIND: &str = "anthropic.thinking";
const REDACTED_THINKING_OPAQUE_KIND: &str = "anthropic.redacted_thinking";

/// Per-stream Anthropic Messages decoder.
pub(super) struct AnthropicStream {
    origin: Origin,
    assembler: Assembler,
    stream_started: bool,
    stream_finished: bool,
    stop_reason: Option<StopReason>,
    signatures: HashMap<usize, String>,
    ignored_indices: HashSet<usize>,
}

impl AnthropicStream {
    pub(super) fn new(origin: Origin) -> Self {
        Self {
            assembler: Assembler::new(origin.clone()),
            origin,
            stream_started: false,
            stream_finished: false,
            stop_reason: None,
            signatures: HashMap::new(),
            ignored_indices: HashSet::new(),
        }
    }

    fn parse_event(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        let event: wire::StreamEvent = serde_json::from_str(payload)
            .map_err(|error| decode_error("Anthropic stream event", Protocol::Anthropic, error))?;

        let mut events = Vec::new();

        match event {
            wire::StreamEvent::MessageStart { message } => {
                if !self.stream_started {
                    self.stream_started = true;
                    events.push(StreamEvent::MessageStart {
                        response_id: message.id,
                    });
                }
                if let Some(usage) = decode_usage(message.usage.as_ref()) {
                    events.push(StreamEvent::Usage(usage));
                }
            }
            wire::StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                wire::StreamContentBlock::Text { text } => {
                    events.push(StreamEvent::BlockStart {
                        index,
                        kind: BlockKind::Text,
                    });
                    if let Some(text) = text
                        && !text.is_empty()
                    {
                        events.push(StreamEvent::TextDelta { index, text });
                    }
                }
                wire::StreamContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    events.push(StreamEvent::BlockStart {
                        index,
                        kind: BlockKind::Reasoning,
                    });
                    if let Some(thinking) = thinking
                        && !thinking.is_empty()
                    {
                        events.push(StreamEvent::ReasoningDelta {
                            index,
                            text: thinking,
                        });
                    }
                    if let Some(signature) = signature
                        && !signature.is_empty()
                    {
                        self.signatures.insert(index, signature);
                    }
                }
                wire::StreamContentBlock::RedactedThinking { data } => {
                    self.ignored_indices.insert(index);
                    events.push(StreamEvent::CompleteBlock {
                        index,
                        block: Block::Opaque(Opaque {
                            origin: self.origin.clone(),
                            kind: REDACTED_THINKING_OPAQUE_KIND.to_string(),
                            data,
                            item_id: None,
                        }),
                    });
                }
                wire::StreamContentBlock::ToolUse { id, name } => {
                    events.push(StreamEvent::ToolCallStart {
                        index,
                        id: id.unwrap_or_default(),
                        name: name.unwrap_or_default(),
                        provider_item_id: None,
                    });
                }
                wire::StreamContentBlock::Unknown => {
                    self.ignored_indices.insert(index);
                }
            },
            wire::StreamEvent::ContentBlockDelta { index, delta } => {
                if self.ignored_indices.contains(&index) {
                    return Ok(events);
                }
                match delta {
                    wire::StreamDelta::TextDelta { text } => {
                        if let Some(text) = text
                            && !text.is_empty()
                        {
                            events.push(StreamEvent::TextDelta { index, text });
                        }
                    }
                    wire::StreamDelta::ThinkingDelta { thinking } => {
                        if let Some(thinking) = thinking
                            && !thinking.is_empty()
                        {
                            events.push(StreamEvent::ReasoningDelta {
                                index,
                                text: thinking,
                            });
                        }
                    }
                    wire::StreamDelta::InputJsonDelta { partial_json } => {
                        if let Some(fragment) = partial_json
                            && !fragment.is_empty()
                        {
                            events.push(StreamEvent::ToolArgsDelta { index, fragment });
                        }
                    }
                    wire::StreamDelta::SignatureDelta { signature } => {
                        if let Some(signature) = signature
                            && !signature.is_empty()
                        {
                            self.signatures
                                .entry(index)
                                .or_default()
                                .push_str(&signature);
                        }
                    }
                    wire::StreamDelta::Unknown => {}
                }
            }
            wire::StreamEvent::ContentBlockStop { index } => {
                if !self.ignored_indices.remove(&index) {
                    let opaque = self
                        .signatures
                        .remove(&index)
                        .filter(|sig| !sig.is_empty())
                        .map(|data| Opaque {
                            origin: self.origin.clone(),
                            kind: THINKING_OPAQUE_KIND.to_string(),
                            data,
                            item_id: None,
                        });
                    events.push(StreamEvent::BlockEnd { index, opaque });
                }
            }
            wire::StreamEvent::MessageDelta { delta, usage } => {
                if let Some(usage) = decode_usage(usage.as_ref()) {
                    events.push(StreamEvent::Usage(usage));
                }
                if let Some(raw) = delta.stop_reason.as_deref() {
                    self.stop_reason = Some(stop_reason(raw));
                }
            }
            wire::StreamEvent::MessageStop => {
                self.stream_finished = true;
                events.push(StreamEvent::Finish {
                    stop_reason: self.stop_reason.take(),
                });
            }
            wire::StreamEvent::Ping | wire::StreamEvent::Unknown => {}
            wire::StreamEvent::Error { error } => {
                return Err(provider_error(
                    Protocol::Anthropic,
                    error.message.as_deref(),
                    error.kind.as_deref(),
                    error.code.as_ref(),
                    error.status,
                ));
            }
        }

        Ok(events)
    }
}

impl StreamDecoder for AnthropicStream {
    fn push(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        if crate::model::convert::util::is_done_payload(payload) {
            return Ok(Vec::new());
        }
        let events = self.parse_event(payload)?;
        for event in &events {
            self.assembler.push(event.clone())?;
        }
        Ok(events)
    }

    fn finish(self: Box<Self>) -> Result<Message, HarnessError> {
        let this = *self;
        if !this.stream_finished {
            return Err(HarnessError::stream_closed(Protocol::Anthropic));
        }
        this.assembler.finish()
    }

    fn interrupted(self: Box<Self>) -> Message {
        let this = *self;
        this.assembler.interrupted()
    }
}

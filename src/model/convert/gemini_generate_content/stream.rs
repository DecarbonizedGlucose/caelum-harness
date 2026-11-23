use crate::error::HarnessError;
use crate::model::convert::StreamDecoder;
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::convert::gemini_generate_content::response::signature_opaque;
use crate::model::convert::gemini_generate_content::{
    UsageAccumulator, is_protocol_failure_reason, map_finish_reason,
};
use crate::model::convert::util::{encode_tool_arguments, is_done_payload};
use crate::model::message::{
    Block, Message, Opaque, Origin, ReasoningBlock, StopReason, TextBlock,
};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, BlockKind, StreamEvent};
use crate::model::wire::gemini_generate_content as wire;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenKind {
    Text,
    Reasoning,
}

/// Per-stream Gemini GenerateContent decoder.
pub(super) struct GeminiGenerateContentStream {
    origin: Origin,
    assembler: Assembler,
    usage: UsageAccumulator,
    stream_started: bool,
    finish_seen: bool,
    stop_reason: Option<StopReason>,
    protocol_failure: Option<String>,
    blocked_prompt: bool,
    has_content: bool,
    saw_tool_call: bool,
    next_index: usize,
    open_block: Option<(usize, OpenKind)>,
}

impl GeminiGenerateContentStream {
    pub(super) fn new(origin: Origin) -> Self {
        Self {
            assembler: Assembler::new(origin.clone()),
            origin,
            usage: UsageAccumulator::default(),
            stream_started: false,
            finish_seen: false,
            stop_reason: None,
            protocol_failure: None,
            blocked_prompt: false,
            has_content: false,
            saw_tool_call: false,
            next_index: 0,
            open_block: None,
        }
    }

    fn parse_event(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        let response: wire::GenerateContentResponse =
            serde_json::from_str(payload).map_err(|error| {
                decode_error(
                    "Gemini stream event",
                    Protocol::GeminiGenerateContent,
                    error,
                )
            })?;

        if let Some(error) = &response.error {
            return Err(provider_error(
                Protocol::GeminiGenerateContent,
                error.message.as_deref(),
                error.status.as_deref(),
                None,
                error.code,
            ));
        }

        let mut events = Vec::new();

        if !self.stream_started {
            self.stream_started = true;
            events.push(StreamEvent::MessageStart {
                response_id: response.response_id.clone(),
            });
        }

        if let Some(usage) = self.usage.update(response.usage_metadata.as_ref()) {
            events.push(StreamEvent::Usage(usage));
        }

        if response
            .prompt_feedback
            .as_ref()
            .and_then(|feedback| feedback.block_reason.as_deref())
            .is_some()
        {
            self.close_open(None, &mut events);
            self.finish_seen = true;
            self.blocked_prompt = true;
            self.stop_reason = Some(StopReason::ContentFilter);
            events.push(StreamEvent::Finish {
                stop_reason: Some(StopReason::ContentFilter),
            });
            return Ok(events);
        }

        if self.finish_seen {
            return Ok(events);
        }

        let Some(candidate) = response.candidates.first() else {
            return Ok(events);
        };

        if let Some(content) = &candidate.content {
            for part in &content.parts {
                self.push_part(part, &mut events)?;
            }
        }

        if let Some(raw_reason) = candidate.finish_reason.as_deref() {
            self.close_open(None, &mut events);
            self.finish_seen = true;
            if is_protocol_failure_reason(raw_reason) {
                self.protocol_failure = Some(raw_reason.to_string());
            }
            let mapped = map_finish_reason(raw_reason, self.saw_tool_call);
            self.stop_reason = Some(mapped.clone());
            events.push(StreamEvent::Finish {
                stop_reason: Some(mapped),
            });
        }

        Ok(events)
    }

    fn push_part(
        &mut self,
        part: &wire::Part,
        events: &mut Vec<StreamEvent>,
    ) -> Result<(), HarnessError> {
        let opaque = signature_opaque(&self.origin, part.thought_signature.as_deref());

        if let Some(call) = &part.function_call {
            self.close_open(None, events);
            self.has_content = true;
            self.saw_tool_call = true;

            let index = self.next_index;
            self.next_index += 1;
            let id = call
                .id
                .as_deref()
                .filter(|id| !id.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("call-{index}"));
            let args_json = encode_tool_arguments(&call.args, None, None)?;

            events.push(StreamEvent::ToolCallStart {
                index,
                id,
                name: call.name.clone(),
                provider_item_id: None,
            });
            events.push(StreamEvent::ToolArgsDelta {
                index,
                fragment: args_json,
            });
            events.push(StreamEvent::BlockEnd { index, opaque });
            return Ok(());
        }

        if let Some(text) = part.text.as_deref() {
            let is_thought = part.thought.unwrap_or(false);
            let kind = if is_thought {
                OpenKind::Reasoning
            } else {
                OpenKind::Text
            };

            if text.is_empty() {
                let Some(opaque) = opaque else {
                    return Ok(());
                };

                if let Some((_, open_kind)) = self.open_block
                    && open_kind == kind
                {
                    self.close_open(Some(opaque), events);
                } else {
                    self.close_open(None, events);
                    let index = self.next_index;
                    self.next_index += 1;
                    let block = match kind {
                        OpenKind::Text => Block::Text(TextBlock {
                            text: String::new(),
                            opaque: Some(opaque),
                        }),
                        OpenKind::Reasoning => Block::Reasoning(ReasoningBlock {
                            text: String::new(),
                            opaque: Some(opaque),
                        }),
                    };
                    self.has_content = true;
                    events.push(StreamEvent::CompleteBlock { index, block });
                }
                return Ok(());
            }

            self.has_content = true;

            if opaque.is_some() {
                self.close_open(None, events);
                let index = self.next_index;
                self.next_index += 1;
                events.push(StreamEvent::BlockStart {
                    index,
                    kind: block_kind(kind),
                });
                events.push(delta_event(kind, index, text.to_string()));
                events.push(StreamEvent::BlockEnd { index, opaque });
                return Ok(());
            }

            let index = self.ensure_open(kind, events);
            events.push(delta_event(kind, index, text.to_string()));
        }

        Ok(())
    }

    fn ensure_open(&mut self, kind: OpenKind, events: &mut Vec<StreamEvent>) -> usize {
        if let Some((index, current)) = self.open_block {
            if current == kind {
                return index;
            }
            self.close_open(None, events);
        }

        let index = self.next_index;
        self.next_index += 1;
        self.open_block = Some((index, kind));
        events.push(StreamEvent::BlockStart {
            index,
            kind: block_kind(kind),
        });
        index
    }

    fn close_open(&mut self, opaque: Option<Opaque>, events: &mut Vec<StreamEvent>) {
        if let Some((index, _)) = self.open_block.take() {
            events.push(StreamEvent::BlockEnd { index, opaque });
        }
    }
}

impl StreamDecoder for GeminiGenerateContentStream {
    fn push(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        if is_done_payload(payload) {
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
        if !this.finish_seen {
            return Err(HarnessError::stream_closed(Protocol::GeminiGenerateContent));
        }
        if this.blocked_prompt {
            return Err(HarnessError::content_filter(
                Protocol::GeminiGenerateContent,
            ));
        }
        if !this.has_content
            && let Some(reason) = this.protocol_failure
        {
            return Err(HarnessError::protocol(
                Protocol::GeminiGenerateContent,
                "candidates[0].finishReason",
                format!("generation failed with {reason}"),
            ));
        }
        this.assembler.finish()
    }

    fn interrupted(self: Box<Self>) -> Message {
        let this = *self;
        this.assembler.interrupted()
    }
}

fn block_kind(kind: OpenKind) -> BlockKind {
    match kind {
        OpenKind::Text => BlockKind::Text,
        OpenKind::Reasoning => BlockKind::Reasoning,
    }
}

fn delta_event(kind: OpenKind, index: usize, text: String) -> StreamEvent {
    match kind {
        OpenKind::Text => StreamEvent::TextDelta { index, text },
        OpenKind::Reasoning => StreamEvent::ReasoningDelta { index, text },
    }
}

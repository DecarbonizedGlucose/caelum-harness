use std::collections::{HashMap, HashSet};

use crate::error::HarnessError;
use crate::model::convert::StreamDecoder;
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::convert::openai_chat::{OpenAiChatCompat, decode_usage, stop_reason};
use crate::model::message::{Message, Origin};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, BlockKind, StreamEvent};
use crate::model::wire::openai_chat as wire;

pub(super) struct OpenAiChatStream {
    assembler: Assembler,
    compat: OpenAiChatCompat,
    stream_started: bool,
    stream_finished: bool,
    next_index: usize,
    text_index: Option<usize>,
    reasoning_index: Option<usize>,
    tool_indices: HashMap<usize, usize>,
    tool_ids: HashMap<String, usize>,
    started_tools: HashSet<usize>,
}

impl OpenAiChatStream {
    pub(super) fn new(origin: Origin, compat: OpenAiChatCompat) -> Self {
        Self {
            assembler: Assembler::new(origin),
            compat,
            stream_started: false,
            stream_finished: false,
            next_index: 0,
            text_index: None,
            reasoning_index: None,
            tool_indices: HashMap::new(),
            tool_ids: HashMap::new(),
            started_tools: HashSet::new(),
        }
    }

    fn parse_event(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        let chunk: wire::ChatChunk = serde_json::from_str(payload).map_err(|error| {
            decode_error("OpenAI Chat stream event", Protocol::OpenaiChat, error)
        })?;

        if let Some(error) = &chunk.error {
            return Err(provider_error(
                Protocol::OpenaiChat,
                error.message.as_deref(),
                error.kind.as_deref(),
                error.code.as_ref(),
                error.status,
            ));
        }

        let mut events = Vec::new();

        if !self.stream_started {
            self.stream_started = true;
            events.push(StreamEvent::MessageStart {
                response_id: chunk.id.clone(),
            });
        }

        if let Some(usage) = decode_usage(chunk.usage.as_ref()) {
            events.push(StreamEvent::Usage(usage));
        }

        if self.stream_finished {
            return Ok(events);
        }

        let Some(choice) = chunk.choices.first() else {
            return Ok(events);
        };

        if let Some(error) = &choice.error {
            return Err(provider_error(
                Protocol::OpenaiChat,
                error.message.as_deref(),
                error.kind.as_deref(),
                error.code.as_ref(),
                error.status,
            ));
        }

        let delta = &choice.delta;

        if let Some(reasoning) = self.compat.reasoning_text(
            delta.reasoning_content.as_deref(),
            delta.reasoning.as_deref(),
            delta.reasoning_text.as_deref(),
        ) {
            let index = match self.reasoning_index {
                Some(index) => index,
                None => {
                    let index = self.next_index;
                    self.next_index += 1;
                    self.reasoning_index = Some(index);
                    events.push(StreamEvent::BlockStart {
                        index,
                        kind: BlockKind::Reasoning,
                    });
                    index
                }
            };
            events.push(StreamEvent::ReasoningDelta {
                index,
                text: reasoning.to_string(),
            });
        }

        if let Some(content) = delta.content.as_deref()
            && !content.is_empty()
        {
            let index = self.text_index();
            events.push(StreamEvent::TextDelta {
                index,
                text: content.to_string(),
            });
        }

        if let Some(refusal) = delta.refusal.as_deref()
            && !refusal.is_empty()
        {
            let index = self.text_index();
            events.push(StreamEvent::TextDelta {
                index,
                text: refusal.to_string(),
            });
        }

        if let Some(tool_calls) = &delta.tool_calls {
            for (position, tool_call) in tool_calls.iter().enumerate() {
                let index = self.tool_index(position, tool_call);
                let function = tool_call.function.as_ref();
                let id = tool_call.id.as_deref().unwrap_or("");
                let name = function
                    .and_then(|function| function.name.as_deref())
                    .unwrap_or("");

                if !self.started_tools.contains(&index) && (!id.is_empty() || !name.is_empty()) {
                    self.started_tools.insert(index);
                    events.push(StreamEvent::ToolCallStart {
                        index,
                        id: id.to_string(),
                        name: name.to_string(),
                        provider_item_id: None,
                    });
                }

                if let Some(arguments) = function.and_then(|function| function.arguments.as_deref())
                    && !arguments.is_empty()
                {
                    events.push(StreamEvent::ToolArgsDelta {
                        index,
                        fragment: arguments.to_string(),
                    });
                }
            }
        }

        if let Some(raw) = choice.finish_reason.as_deref() {
            self.stream_finished = true;
            events.push(StreamEvent::Finish {
                stop_reason: Some(stop_reason(raw)),
            });
        }

        Ok(events)
    }

    fn text_index(&mut self) -> usize {
        match self.text_index {
            Some(index) => index,
            None => {
                let index = self.next_index;
                self.next_index += 1;
                self.text_index = Some(index);
                index
            }
        }
    }

    fn tool_index(&mut self, position: usize, tool_call: &wire::ToolCallDelta) -> usize {
        let id = tool_call.id.as_deref().filter(|id| !id.is_empty());

        if let Some(provider_index) = tool_call.index {
            let provider_index = provider_index as usize;
            if let Some(index) = self.tool_indices.get(&provider_index) {
                return *index;
            }
            if let Some(index) = id.and_then(|id| self.tool_ids.get(id)) {
                self.tool_indices.insert(provider_index, *index);
                return *index;
            }
            let index = self.next_index;
            self.next_index += 1;
            self.tool_indices.insert(provider_index, index);
            if let Some(id) = id {
                self.tool_ids.insert(id.to_string(), index);
            }
            return index;
        }

        if let Some(id) = id {
            if let Some(index) = self.tool_ids.get(id) {
                return *index;
            }
            let index = self.next_index;
            self.next_index += 1;
            self.tool_ids.insert(id.to_string(), index);
            return index;
        }

        let fallback = usize::MAX - position;
        if let Some(index) = self.tool_indices.get(&fallback) {
            return *index;
        }
        let index = self.next_index;
        self.next_index += 1;
        self.tool_indices.insert(fallback, index);
        index
    }
}

impl StreamDecoder for OpenAiChatStream {
    fn push(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        let events = self.parse_event(payload)?;
        for event in &events {
            self.assembler.push(event.clone())?;
        }
        Ok(events)
    }

    fn finish(self: Box<Self>) -> Result<Message, HarnessError> {
        let this = *self;
        if !this.stream_finished {
            return Err(HarnessError::stream_closed(Protocol::OpenaiChat));
        }
        this.assembler.finish()
    }

    fn interrupted(self: Box<Self>) -> Message {
        let this = *self;
        this.assembler.interrupted()
    }
}

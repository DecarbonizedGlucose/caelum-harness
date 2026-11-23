use std::collections::{HashMap, HashSet};

use crate::error::HarnessError;
use crate::model::convert::StreamDecoder;
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::convert::openai_responses::{
    decode_usage, response::decode_output_item, terminal_stop_reason,
};
use crate::model::message::{Block, Message, Origin};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, BlockKind, StreamEvent};
use crate::model::wire::openai_responses as wire;

/// Per-stream OpenAI Responses decoder.
pub(super) struct OpenAiResponsesStream {
    origin: Origin,
    assembler: Assembler,
    stream_started: bool,
    stream_finished: bool,
    next_index: usize,
    item_indices: HashMap<String, usize>,
    text_started: HashSet<usize>,
    reasoning_started: HashSet<usize>,
    reasoning_text_seen: HashSet<usize>,
    reasoning_summary_seen: HashSet<usize>,
    args_seen: HashSet<usize>,
    ignored_indices: HashSet<usize>,
    completed_items: HashSet<usize>,
    saw_tool_call: bool,
}

impl OpenAiResponsesStream {
    pub(super) fn new(origin: Origin) -> Self {
        Self {
            assembler: Assembler::new(origin.clone()),
            origin,
            stream_started: false,
            stream_finished: false,
            next_index: 0,
            item_indices: HashMap::new(),
            text_started: HashSet::new(),
            reasoning_started: HashSet::new(),
            reasoning_text_seen: HashSet::new(),
            reasoning_summary_seen: HashSet::new(),
            args_seen: HashSet::new(),
            ignored_indices: HashSet::new(),
            completed_items: HashSet::new(),
            saw_tool_call: false,
        }
    }

    fn parse_event(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        let event: wire::StreamEvent = serde_json::from_str(payload).map_err(|error| {
            decode_error("OpenAI Responses stream event", Protocol::OpenaiResp, error)
        })?;

        let mut events = Vec::new();

        match event {
            wire::StreamEvent::Created { response }
            | wire::StreamEvent::InProgress { response } => {
                self.start(&response, &mut events);
            }
            wire::StreamEvent::OutputItemAdded { item, output_index } => {
                let index = self.resolve_index(output_index, item_id(&item));
                match item {
                    wire::OutputItem::Message { .. } => {}
                    wire::OutputItem::Reasoning { .. } => {}
                    wire::OutputItem::FunctionCall {
                        id, call_id, name, ..
                    } => {
                        self.saw_tool_call = true;
                        events.push(StreamEvent::ToolCallStart {
                            index,
                            id: call_id.unwrap_or_default(),
                            name: name.unwrap_or_default(),
                            provider_item_id: id,
                        });
                    }
                    wire::OutputItem::Unknown => {
                        self.ignored_indices.insert(index);
                    }
                }
            }
            wire::StreamEvent::ContentPartAdded { .. } => {}
            wire::StreamEvent::ContentPartDone {
                item_id,
                output_index,
                part,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if self.ignored_indices.contains(&index) {
                    return Ok(events);
                }
                match part {
                    wire::ContentPart::OutputText { text } => {
                        self.fallback_text(index, text, &mut events);
                    }
                    wire::ContentPart::Refusal { refusal } => {
                        self.fallback_text(index, refusal, &mut events);
                    }
                    wire::ContentPart::Unknown => {}
                }
            }
            wire::StreamEvent::OutputTextDelta {
                item_id,
                output_index,
                delta,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) && !delta.is_empty() {
                    self.start_text(index, &mut events);
                    events.push(StreamEvent::TextDelta { index, text: delta });
                }
            }
            wire::StreamEvent::OutputTextDone {
                item_id,
                output_index,
                text,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) {
                    self.fallback_text(index, text, &mut events);
                }
            }
            wire::StreamEvent::RefusalDelta {
                item_id,
                output_index,
                delta,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) && !delta.is_empty() {
                    self.start_text(index, &mut events);
                    events.push(StreamEvent::TextDelta { index, text: delta });
                }
            }
            wire::StreamEvent::RefusalDone {
                item_id,
                output_index,
                refusal,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) {
                    self.fallback_text(index, refusal, &mut events);
                }
            }
            wire::StreamEvent::FunctionCallArgumentsDelta {
                item_id,
                output_index,
                delta,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) && !delta.is_empty() {
                    self.saw_tool_call = true;
                    self.args_seen.insert(index);
                    events.push(StreamEvent::ToolArgsDelta {
                        index,
                        fragment: delta,
                    });
                }
            }
            wire::StreamEvent::FunctionCallArgumentsDone {
                item_id,
                output_index,
                arguments,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index)
                    && !arguments.is_empty()
                    && !self.args_seen.contains(&index)
                {
                    self.args_seen.insert(index);
                    events.push(StreamEvent::ToolArgsDelta {
                        index,
                        fragment: arguments,
                    });
                }
            }
            wire::StreamEvent::ReasoningTextDelta {
                item_id,
                output_index,
                delta,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if self.ignored_indices.contains(&index)
                    || self.reasoning_summary_seen.contains(&index)
                    || delta.is_empty()
                {
                    return Ok(events);
                }
                self.start_reasoning(index, &mut events);
                self.reasoning_text_seen.insert(index);
                events.push(StreamEvent::ReasoningDelta { index, text: delta });
            }
            wire::StreamEvent::ReasoningSummaryTextDelta {
                item_id,
                output_index,
                delta,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if self.ignored_indices.contains(&index)
                    || self.reasoning_text_seen.contains(&index)
                    || delta.is_empty()
                {
                    return Ok(events);
                }
                self.start_reasoning(index, &mut events);
                self.reasoning_summary_seen.insert(index);
                events.push(StreamEvent::ReasoningDelta { index, text: delta });
            }
            wire::StreamEvent::ReasoningTextDone {
                item_id,
                output_index,
                text,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) {
                    self.fallback_reasoning(index, text, &mut events);
                }
            }
            wire::StreamEvent::ReasoningSummaryTextDone {
                item_id,
                output_index,
                text,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index) {
                    self.fallback_reasoning(index, text, &mut events);
                }
            }
            wire::StreamEvent::ReasoningSummaryPartDone {
                item_id,
                output_index,
                part,
            } => {
                let index = self.resolve_index(output_index, item_id.as_deref());
                if !self.ignored_indices.contains(&index)
                    && let wire::SummaryPart::SummaryText { text } = part
                {
                    self.fallback_reasoning(index, text, &mut events);
                }
            }
            wire::StreamEvent::OutputItemDone { item, output_index } => {
                let index = self.resolve_index(output_index, item_id(&item));
                if self.ignored_indices.contains(&index) {
                    return Ok(events);
                }
                if matches!(item, wire::OutputItem::FunctionCall { .. }) {
                    self.saw_tool_call = true;
                }
                if let wire::OutputItem::FunctionCall { arguments, .. } = &item
                    && self.args_seen.contains(&index)
                    && is_empty_arguments(arguments)
                {
                    self.completed_items.insert(index);
                    return Ok(events);
                }
                if let Some(block) = decode_output_item(&item, &self.origin, index)? {
                    self.completed_items.insert(index);
                    events.push(StreamEvent::CompleteBlock { index, block });
                }
            }
            wire::StreamEvent::Completed { response }
            | wire::StreamEvent::Done { response }
            | wire::StreamEvent::Incomplete { response } => {
                events.extend(self.finish_terminal(&response)?);
            }
            wire::StreamEvent::Failed { response } => {
                let Some(error) = response.error.as_ref() else {
                    return Err(HarnessError::new(
                        crate::error::ErrorCode::Server,
                        format!("{}: response failed", Protocol::OpenaiResp),
                    ));
                };
                return Err(provider_error(
                    Protocol::OpenaiResp,
                    error.message.as_deref(),
                    error.kind.as_deref(),
                    error.code.as_ref(),
                    error.status,
                ));
            }
            wire::StreamEvent::Error {
                message,
                code,
                status,
                error,
            } => {
                let (message, kind, code, status) = match error {
                    Some(error) => (
                        error.message,
                        error.kind,
                        error.code,
                        error.status.or(status),
                    ),
                    None => (message, None, code, status),
                };
                return Err(provider_error(
                    Protocol::OpenaiResp,
                    message.as_deref(),
                    kind.as_deref(),
                    code.as_ref(),
                    status,
                ));
            }
            wire::StreamEvent::Unknown => {}
        }

        Ok(events)
    }

    fn start(&mut self, response: &wire::Response, events: &mut Vec<StreamEvent>) {
        if !self.stream_started {
            self.stream_started = true;
            events.push(StreamEvent::MessageStart {
                response_id: response.id.clone(),
            });
        }
        if let Some(usage) = decode_usage(response.usage.as_ref()) {
            events.push(StreamEvent::Usage(usage));
        }
    }

    fn finish_terminal(
        &mut self,
        response: &wire::Response,
    ) -> Result<Vec<StreamEvent>, HarnessError> {
        if self.stream_finished {
            return Ok(Vec::new());
        }

        let mut events = Vec::new();
        self.start(response, &mut events);

        let mut has_tool_calls = self.saw_tool_call;
        for (index, item) in response.output.iter().enumerate() {
            if self.completed_items.contains(&index) || self.ignored_indices.contains(&index) {
                if matches!(item, wire::OutputItem::FunctionCall { .. }) {
                    has_tool_calls = true;
                }
                continue;
            }
            if let Some(block) = decode_output_item(item, &self.origin, index)? {
                if matches!(block, Block::ToolCall(_)) {
                    has_tool_calls = true;
                }
                self.completed_items.insert(index);
                events.push(StreamEvent::CompleteBlock { index, block });
            }
        }

        self.stream_finished = true;
        events.push(StreamEvent::Finish {
            stop_reason: terminal_stop_reason(response, has_tool_calls),
        });
        Ok(events)
    }

    fn fallback_text(&mut self, index: usize, text: String, events: &mut Vec<StreamEvent>) {
        if text.is_empty() || self.text_started.contains(&index) {
            return;
        }
        self.start_text(index, events);
        events.push(StreamEvent::TextDelta { index, text });
    }

    fn fallback_reasoning(&mut self, index: usize, text: String, events: &mut Vec<StreamEvent>) {
        if text.is_empty()
            || self.reasoning_text_seen.contains(&index)
            || self.reasoning_summary_seen.contains(&index)
        {
            return;
        }
        self.start_reasoning(index, events);
        self.reasoning_text_seen.insert(index);
        events.push(StreamEvent::ReasoningDelta { index, text });
    }

    fn start_text(&mut self, index: usize, events: &mut Vec<StreamEvent>) {
        if self.text_started.insert(index) {
            events.push(StreamEvent::BlockStart {
                index,
                kind: BlockKind::Text,
            });
        }
    }

    fn start_reasoning(&mut self, index: usize, events: &mut Vec<StreamEvent>) {
        if self.reasoning_started.insert(index) {
            events.push(StreamEvent::BlockStart {
                index,
                kind: BlockKind::Reasoning,
            });
        }
    }

    fn resolve_index(&mut self, output_index: Option<usize>, item_id: Option<&str>) -> usize {
        if let Some(index) = output_index {
            self.next_index = self.next_index.max(index + 1);
            if let Some(id) = item_id {
                self.item_indices.insert(id.to_string(), index);
            }
            return index;
        }
        if let Some(index) = item_id.and_then(|id| self.item_indices.get(id)) {
            return *index;
        }

        let index = self.next_index;
        self.next_index += 1;
        if let Some(id) = item_id {
            self.item_indices.insert(id.to_string(), index);
        }
        index
    }
}

impl StreamDecoder for OpenAiResponsesStream {
    fn push(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError> {
        if self.stream_finished || crate::model::convert::util::is_done_payload(payload) {
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
            return Err(HarnessError::stream_closed(Protocol::OpenaiResp));
        }
        this.assembler.finish()
    }

    fn interrupted(self: Box<Self>) -> Message {
        let this = *self;
        this.assembler.interrupted()
    }
}

fn item_id(item: &wire::OutputItem) -> Option<&str> {
    match item {
        wire::OutputItem::Message { id, .. }
        | wire::OutputItem::Reasoning { id, .. }
        | wire::OutputItem::FunctionCall { id, .. } => id.as_deref(),
        wire::OutputItem::Unknown => None,
    }
}

fn is_empty_arguments(arguments: &serde_json::Value) -> bool {
    matches!(arguments, serde_json::Value::Null)
        || matches!(arguments, serde_json::Value::String(text) if text.trim().is_empty())
}

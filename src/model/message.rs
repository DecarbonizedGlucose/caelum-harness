use serde_json::Value;

use crate::model::protocol::Protocol;

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    System,
    User,
    Assistant,
    Tool,
}

/// A provider-neutral conversation message.
#[derive(Debug, Clone)]
pub struct Message {
    pub mtype: MessageType,
    pub blocks: Vec<Block>,
    pub meta: Option<AssistantMeta>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(MessageType::System, text)
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::text(MessageType::User, text)
    }

    pub fn assistant(blocks: Vec<Block>) -> Self {
        Self {
            mtype: MessageType::Assistant,
            blocks,
            meta: None,
        }
    }

    pub fn tool_result(
        call_id: impl Into<String>,
        name: impl Into<String>,
        text: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            mtype: MessageType::Tool,
            blocks: vec![Block::ToolResult(ToolResultBlock {
                call_id: call_id.into(),
                name: name.into(),
                content: vec![Block::Text(TextBlock {
                    text: text.into(),
                    opaque: None,
                })],
                is_error,
            })],
            meta: None,
        }
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCallBlock> {
        self.blocks.iter().filter_map(|block| match block {
            Block::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    fn text(mtype: MessageType, text: impl Into<String>) -> Self {
        Self {
            mtype,
            blocks: vec![Block::Text(TextBlock {
                text: text.into(),
                opaque: None,
            })],
            meta: None,
        }
    }
}

/// One content block inside a message.
#[derive(Debug, Clone)]
pub enum Block {
    Text(TextBlock),
    Image(ImageBlock),
    Reasoning(ReasoningBlock),
    ToolCall(ToolCallBlock),
    ToolResult(ToolResultBlock),
}

/// Visible text.
#[derive(Debug, Clone)]
pub struct TextBlock {
    pub text: String,
    pub opaque: Option<Opaque>,
}

/// An image input.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBlock {
    pub source: ImageSource,
    pub opaque: Option<Opaque>,
}

/// Where image bytes come from.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    Base64 { mime: String, data: Vec<u8> },
    Url(String),
}

/// Reasoning or thinking output.
#[derive(Debug, Clone)]
pub struct ReasoningBlock {
    pub text: String,
    pub opaque: Option<Opaque>,
}

/// A tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCallBlock {
    pub id: String,
    pub name: String,
    pub args: Value,
    pub raw_args: Option<String>,
    pub provider_item_id: Option<String>,
    pub opaque: Option<Opaque>,
}

/// The result of one tool call.
#[derive(Debug, Clone)]
pub struct ToolResultBlock {
    pub call_id: String,
    pub name: String,
    pub content: Vec<Block>,
    pub is_error: bool,
}

/// Provider and model that produced a response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub protocol: Protocol,
    pub vendor: String,
    pub model: String,
}

/// Provider data that must be replayed unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct Opaque {
    pub origin: Origin,
    pub kind: String,
    pub data: String,
    pub item_id: Option<String>,
}

/// Token counts for one model call. Input and cache counts are disjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

/// Provider-neutral finish reason.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolCall,
    StopSequence,
    ContentFilter,
    Refusal,
    Pause,
    Other(String),
}

/// Response metadata for an assistant message.
#[derive(Debug, Clone)]
pub struct AssistantMeta {
    pub origin: Origin,
    pub response_id: Option<String>,
    pub stop_reason: StopReason,
    pub usage: Option<Usage>,
    pub opaque: Option<Opaque>,
}

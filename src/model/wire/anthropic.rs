use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Anthropic Messages non-streaming response envelope.
#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub id: Option<String>,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Option<Usage>,
    pub error: Option<ProviderError>,
}

/// One content block in an Anthropic response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        signature: Option<String>,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Unknown,
}

/// Anthropic token usage envelope.
#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Thinking token breakdown in Anthropic usage.
#[derive(Debug, Deserialize)]
pub struct OutputTokensDetails {
    pub thinking_tokens: Option<u64>,
}

/// One streamed Anthropic SSE event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: StreamMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: StreamContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: StreamDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: StreamMessageDelta,
        usage: Option<Usage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ProviderError,
    },
    #[serde(other)]
    Unknown,
}

/// Initial message envelope in `message_start`.
#[derive(Debug, Deserialize)]
pub struct StreamMessageStart {
    pub id: Option<String>,
    pub usage: Option<Usage>,
}

/// Content block header in `content_block_start`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamContentBlock {
    Text {
        text: Option<String>,
    },
    Thinking {
        thinking: Option<String>,
        signature: Option<String>,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    ToolUse {
        id: Option<String>,
        name: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

/// Incremental content delta in `content_block_delta`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamDelta {
    TextDelta {
        text: Option<String>,
    },
    ThinkingDelta {
        thinking: Option<String>,
    },
    InputJsonDelta {
        partial_json: Option<String>,
    },
    SignatureDelta {
        signature: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

/// Message-level delta in `message_delta`.
#[derive(Debug, Default, Deserialize)]
pub struct StreamMessageDelta {
    pub stop_reason: Option<String>,
}

/// Provider error payload.
#[derive(Debug, Deserialize)]
pub struct ProviderError {
    pub message: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub code: Option<Value>,
    pub status: Option<u16>,
}

/// Anthropic Messages request body.
#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Extended thinking request configuration.
#[derive(Debug, Serialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub budget_tokens: u32,
}

/// One request message.
#[derive(Debug, Serialize)]
pub struct RequestMessage {
    pub role: &'static str,
    pub content: Vec<RequestContent>,
}

/// One outgoing content block.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestContent {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<RequestToolResultContent>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

/// One content block inside `tool_result`.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestToolResultContent {
    Text { text: String },
    Image { source: ImageSource },
}

/// Image source payload.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// One tool declaration.
#[derive(Debug, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

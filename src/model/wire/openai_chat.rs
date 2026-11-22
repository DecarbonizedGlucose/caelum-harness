use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// OpenAI Chat Completions response envelope.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
    pub error: Option<ProviderError>,
}

/// One response choice.
#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Assistant message envelope. Content stays dynamic.
#[derive(Debug, Deserialize)]
pub struct Message {
    pub content: Option<Value>,
    pub refusal: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_text: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Tool call envelope.
#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub function: Function,
}

/// Function call payload. Arguments can be a JSON string or an object.
#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

/// Token usage envelope.
#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    pub prompt_cache_hit_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cost: Option<f64>,
}

/// Cached prompt token details.
#[derive(Debug, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: Option<u64>,
}

/// Reasoning token details.
#[derive(Debug, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u64>,
}

/// One streamed chat completion chunk.
#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    pub usage: Option<Usage>,
    pub error: Option<ProviderError>,
}

/// One streamed choice.
#[derive(Debug, Default, Deserialize)]
pub struct ChunkChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
    pub error: Option<ProviderError>,
}

/// Streamed delta. Missing fields stay `None`.
#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
    pub refusal: Option<String>,
    pub reasoning_content: Option<String>,
    pub reasoning: Option<String>,
    pub reasoning_text: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One streamed tool call.
#[derive(Debug, Deserialize)]
pub struct ToolCallDelta {
    pub index: Option<u64>,
    pub id: Option<String>,
    pub function: Option<FunctionDelta>,
}

/// Streamed function fields.
#[derive(Debug, Default, Deserialize)]
pub struct FunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
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

/// OpenAI Chat Completions request body.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
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
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

/// One request message.
#[derive(Debug, Serialize)]
pub struct RequestMessage {
    pub role: String,
    pub content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<RequestToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One outgoing tool call.
#[derive(Debug, Serialize)]
pub struct RequestToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: RequestFunction,
}

/// One outgoing function call payload.
#[derive(Debug, Serialize)]
pub struct RequestFunction {
    pub name: String,
    pub arguments: String,
}

/// One tool declaration.
#[derive(Debug, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolFunction,
}

/// Tool function declaration payload.
#[derive(Debug, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Streaming request options.
#[derive(Debug, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

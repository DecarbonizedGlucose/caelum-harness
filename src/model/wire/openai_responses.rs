use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAI Responses non-streaming response envelope.
#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: Option<String>,
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    pub usage: Option<Usage>,
    pub error: Option<ProviderError>,
    pub incomplete_details: Option<IncompleteDetails>,
}

/// One item in a response output array.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message {
        id: Option<String>,
        status: Option<String>,
        phase: Option<String>,
        #[serde(default)]
        content: Vec<ContentPart>,
    },
    Reasoning {
        id: Option<String>,
        status: Option<String>,
        #[serde(default)]
        content: Vec<ReasoningPart>,
        #[serde(default)]
        summary: Vec<SummaryPart>,
        encrypted_content: Option<String>,
    },
    FunctionCall {
        id: Option<String>,
        status: Option<String>,
        call_id: Option<String>,
        name: Option<String>,
        #[serde(default = "empty_arguments")]
        arguments: Value,
    },
    #[serde(other)]
    Unknown,
}

fn empty_arguments() -> Value {
    Value::String(String::new())
}

/// One part inside a message item.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    OutputText {
        #[serde(default)]
        text: String,
    },
    Refusal {
        #[serde(default)]
        refusal: String,
    },
    #[serde(other)]
    Unknown,
}

/// One part inside a reasoning item.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningPart {
    ReasoningText {
        #[serde(default)]
        text: String,
    },
    #[serde(other)]
    Unknown,
}

/// One summary part inside a reasoning item.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SummaryPart {
    SummaryText {
        #[serde(default)]
        text: String,
    },
    #[serde(other)]
    Unknown,
}

/// Why a response is incomplete.
#[derive(Debug, Deserialize)]
pub struct IncompleteDetails {
    pub reason: Option<String>,
}

/// OpenAI Responses token usage envelope.
#[derive(Debug, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens_details: Option<OutputTokensDetails>,
    pub cost: Option<f64>,
}

/// Cached input token breakdown.
#[derive(Debug, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}

/// Output token breakdown.
#[derive(Debug, Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: Option<u64>,
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

/// One streamed Responses SSE event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "response.created")]
    Created { response: Response },
    #[serde(rename = "response.in_progress")]
    InProgress { response: Response },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        item: OutputItem,
        output_index: Option<usize>,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        item: OutputItem,
        output_index: Option<usize>,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: Option<String>,
        output_index: Option<usize>,
        part: ContentPart,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        part: ContentPart,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        refusal: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        arguments: String,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        item_id: Option<String>,
        output_index: Option<usize>,
        part: SummaryPart,
    },
    #[serde(rename = "response.completed")]
    Completed { response: Response },
    #[serde(rename = "response.done")]
    Done { response: Response },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: Response },
    #[serde(rename = "response.failed")]
    Failed { response: Response },
    #[serde(rename = "error")]
    Error {
        message: Option<String>,
        code: Option<Value>,
        status: Option<u16>,
        error: Option<ProviderError>,
    },
    #[serde(other)]
    Unknown,
}

/// OpenAI Responses request body.
#[derive(Debug, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    pub store: bool,
}

/// One item in the request input array.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum InputItem {
    Message {
        role: &'static str,
        content: Vec<InputContent>,
    },
    AssistantMessage {
        #[serde(rename = "type")]
        kind: &'static str,
        role: &'static str,
        id: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<String>,
        content: Vec<InputContent>,
    },
    FunctionCall {
        #[serde(rename = "type")]
        kind: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        #[serde(rename = "type")]
        kind: &'static str,
        call_id: String,
        output: FunctionOutput,
    },
    RawItem(Value),
}

/// One request content part.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText { text: String },
    InputImage { image_url: String },
    OutputText { text: String },
}

/// Tool result output. Images require the structured form.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum FunctionOutput {
    Text(String),
    Content(Vec<InputContent>),
}

/// One tool declaration.
#[derive(Debug, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Reasoning request configuration.
#[derive(Debug, Serialize)]
pub struct Reasoning {
    pub effort: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<&'static str>,
}

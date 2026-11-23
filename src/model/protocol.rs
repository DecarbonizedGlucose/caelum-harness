use std::fmt;

use serde_json::Value;

use crate::model::message::Message;

/// Wire protocol spoken by a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Protocol {
    OpenaiChat,
    OpenaiResp,
    GeminiGenerateContent,
    Anthropic,
    Unknown,
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OpenaiChat => "OpenAI Chat",
            Self::OpenaiResp => "OpenAI Responses",
            Self::GeminiGenerateContent => "Gemini GenerateContent",
            Self::Anthropic => "Anthropic",
            Self::Unknown => "Unknown protocol",
        })
    }
}

/// A tool declaration sent to the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// How the model may choose from the declared tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
}

/// Requested reasoning effort. Adapters map it to the closest provider setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    Max,
}

/// One provider-neutral model request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: ToolChoice,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: Vec::new(),
            reasoning_effort: None,
        }
    }
}

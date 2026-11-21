use async_trait::async_trait;
use serde_json::Value;

use crate::model::protocol::ToolDef;

/// A tool that the runtime can execute.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str {
        ""
    }

    /// JSON Schema for the tool arguments.
    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn execute(&self, arguments: &Value) -> Result<String, String>;
}

/// Converts tools into declarations sent to the model.
pub fn declarations(tools: &[Box<dyn Tool>]) -> Vec<ToolDef> {
    tools
        .iter()
        .map(|tool| ToolDef {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.parameters(),
        })
        .collect()
}

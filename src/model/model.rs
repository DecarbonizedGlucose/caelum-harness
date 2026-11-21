use async_trait::async_trait;

use crate::error::HarnessError;
use crate::model::message::Message;
use crate::model::protocol::ChatRequest;

/// A model provider.
#[async_trait]
pub trait Model: Send + Sync {
    /// Sends one request and returns one assistant message.
    async fn call_one_shot(&self, request: ChatRequest) -> Result<Message, HarnessError>;
}

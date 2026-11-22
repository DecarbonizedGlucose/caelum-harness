use serde_json::Value;

use crate::error::HarnessError;
use crate::model::message::{Message, Origin};
use crate::model::protocol::ChatRequest;
use crate::model::stream::StreamEvent;

pub mod error;
pub mod openai_chat;
pub(crate) mod util;

/// Provider-neutral codec for one wire protocol.
pub trait ProtocolCodec: Send + Sync {
    fn origin(&self) -> &Origin;

    fn encode_request(&self, request: &ChatRequest, stream: bool) -> Result<Value, HarnessError>;

    fn decode_response(&self, raw: &str) -> Result<Message, HarnessError>;

    fn stream_decoder(&self) -> Box<dyn StreamDecoder>;
}

/// State for one model stream.
pub trait StreamDecoder: Send {
    fn push(&mut self, payload: &str) -> Result<Vec<StreamEvent>, HarnessError>;

    fn finish(self: Box<Self>) -> Result<Message, HarnessError>;

    fn interrupted(self: Box<Self>) -> Message;
}

use crate::model::message::Message;
use crate::model::model::Model;

/// Conversation history and the active model.
pub struct Agent {
    model: Box<dyn Model>,
    messages: Vec<Message>,
}

impl Agent {
    pub(crate) fn new(model: impl Model + 'static) -> Self {
        Self {
            model: Box::new(model),
            messages: Vec::new(),
        }
    }

    pub(crate) fn set_model(&mut self, model: Box<dyn Model>) {
        self.model = model;
    }

    pub(crate) fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub(crate) fn push(&mut self, message: Message) {
        self.messages.push(message);
    }
}

use async_trait::async_trait;

use crate::agent::agent::Agent;
use crate::agent::tool::Tool;
use crate::error::HarnessError;
use crate::model::message::Message;
use crate::model::model::Model;

/// Execution mode selected when a runner is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerMode {
    Normal,
    Game,
    Human,
    Free,
}

impl RunnerMode {
    /// Creates the runner for this mode.
    pub fn create(self, model: impl Model + 'static) -> Box<dyn Runner> {
        let agent = Agent::new(model);
        match self {
            Self::Normal => Box::new(NormalRunner { agent }),
            Self::Game => Box::new(GameRunner { agent }),
            Self::Human => Box::new(HumanRunner { agent }),
            Self::Free => Box::new(FreeRunner { agent }),
        }
    }
}

/// Execution strategy for an agent.
#[async_trait]
pub trait Runner: Send + Sync {
    fn agent(&self) -> &Agent;
    fn agent_mut(&mut self) -> &mut Agent;

    fn messages(&self) -> &[Message] {
        self.agent().messages()
    }

    fn push(&mut self, message: Message) {
        self.agent_mut().push(message);
    }

    fn set_model(&mut self, model: Box<dyn Model>) {
        self.agent_mut().set_model(model);
    }

    /// Runs one model turn.
    async fn run(
        &mut self,
        task: &str,
        tools: &[Box<dyn Tool>],
    ) -> Result<Message, HarnessError>;
}

/// Full-history, fixed-flow runner.
pub(crate) struct NormalRunner {
    agent: Agent,
}

/// Fixed-schema, fixed-flow runner.
pub(crate) struct GameRunner {
    agent: Agent,
}

/// Mutable-schema, fixed-flow runner.
pub(crate) struct HumanRunner {
    agent: Agent,
}

/// Mutable-flow runner.
pub(crate) struct FreeRunner {
    agent: Agent,
}

#[async_trait]
impl Runner for NormalRunner {
    fn agent(&self) -> &Agent {
        &self.agent
    }

    fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    async fn run(
        &mut self,
        _task: &str,
        _tools: &[Box<dyn Tool>],
    ) -> Result<Message, HarnessError> {
        todo!()
    }
}

#[async_trait]
impl Runner for GameRunner {
    fn agent(&self) -> &Agent {
        &self.agent
    }

    fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    async fn run(
        &mut self,
        _task: &str,
        _tools: &[Box<dyn Tool>],
    ) -> Result<Message, HarnessError> {
        todo!()
    }
}

#[async_trait]
impl Runner for HumanRunner {
    fn agent(&self) -> &Agent {
        &self.agent
    }

    fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    async fn run(
        &mut self,
        _task: &str,
        _tools: &[Box<dyn Tool>],
    ) -> Result<Message, HarnessError> {
        todo!()
    }
}

#[async_trait]
impl Runner for FreeRunner {
    fn agent(&self) -> &Agent {
        &self.agent
    }

    fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    async fn run(
        &mut self,
        _task: &str,
        _tools: &[Box<dyn Tool>],
    ) -> Result<Message, HarnessError> {
        todo!()
    }
}

"""Agent orchestration layer.

This module provides the generic agent loop that ties together:
- Behaviors: What the agent can do
- Personality: Who the agent is
- Cognition: How the agent thinks (learning, beliefs, etc.)
- Tools: External capabilities

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                          Agent                                   │
    │                                                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
    │  │  Personality │  │   Behavior   │  │  Cognitive Services  │  │
    │  │   (Persona)  │  │  (Actions)   │  │  (Learning, Beliefs) │  │
    │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
    │                          │                                       │
    │                          ▼                                       │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │                  Decision Engine                         │   │
    │  │   Query → Activation → Decision → Execution → Response  │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                          │                                       │
    │                          ▼                                       │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │                    Tool Registry                         │   │
    │  │         (Handler implementations)                        │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.orchestration import Agent, AgentConfig
    from draagon_ai.behaviors import VOICE_ASSISTANT_TEMPLATE
    from draagon_ai.persona import Persona

    # Create an agent
    agent = Agent(
        agent_id="assistant",
        persona=my_persona,
        behavior=VOICE_ASSISTANT_TEMPLATE,
        tools=my_tool_registry,
        llm=my_llm_provider,
    )

    # Process a query
    response = await agent.process("What time is it?", context)
"""

from .agent import Agent, AgentConfig, MultiAgent
from .decision import DecisionEngine, DecisionResult
from .execution import ActionExecutor, ActionResult
from .loop import AgentLoop, AgentContext, AgentResponse
from .protocols import LLMProvider, MemoryProvider, ToolProvider
from .architect_agent import (
    ArchitectAgent,
    ArchitectResult,
    create_architect_agent,
)
from .learning_channel import (
    LearningChannel,
    Learning,
    LearningType,
    LearningScope,
    StubLearningChannel,
    InMemoryLearningChannel,
    create_learning_channel,
    get_learning_channel,
    set_learning_channel,
    reset_learning_channel,
)
from .multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    OrchestrationMode,
    AgentSpec,
    AgentRole,
    TaskContext,
    AgentResult,
    OrchestratorResult,
    execute_single_agent,
)

__all__ = [
    # Core Agent
    "Agent",
    "AgentConfig",
    "MultiAgent",
    # Decision & Execution
    "DecisionEngine",
    "DecisionResult",
    "ActionExecutor",
    "ActionResult",
    # Loop
    "AgentLoop",
    "AgentContext",
    "AgentResponse",
    # Protocols
    "LLMProvider",
    "MemoryProvider",
    "ToolProvider",
    # Behavior Architect
    "ArchitectAgent",
    "ArchitectResult",
    "create_architect_agent",
    # Learning Channel (C.1)
    "LearningChannel",
    "Learning",
    "LearningType",
    "LearningScope",
    "StubLearningChannel",
    "InMemoryLearningChannel",
    "create_learning_channel",
    "get_learning_channel",
    "set_learning_channel",
    "reset_learning_channel",
    # Multi-Agent Orchestrator (C.1)
    "MultiAgentOrchestrator",
    "OrchestrationMode",
    "AgentSpec",
    "AgentRole",
    "TaskContext",
    "AgentResult",
    "OrchestratorResult",
    "execute_single_agent",
]

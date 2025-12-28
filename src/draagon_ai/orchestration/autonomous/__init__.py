"""Autonomous background agent for draagon-ai.

This module provides autonomous background cognitive capabilities:
- Background decision-making during idle time
- Multi-layer guardrail system for safe operation
- Self-monitoring and transparency logging
- Configurable action budgets and timing

Example:
    from draagon_ai.orchestration.autonomous import (
        AutonomousAgentService,
        AutonomousConfig,
        LLMProvider,
        SearchProvider,
        MemoryStoreProvider,
    )

    # Create service with your providers
    service = AutonomousAgentService(
        llm=my_llm_provider,
        config=AutonomousConfig(
            enabled=True,
            daily_action_budget=20,
        ),
        search=my_search_provider,
        memory_store=my_memory_store,
    )

    # Start the agent
    await service.start()
"""

from .service import AutonomousAgentService
from .types import (
    # Enums
    ActionType,
    ActionTier,
    # Config
    AutonomousConfig,
    # Action Models
    ProposedAction,
    ApprovedAction,
    ActionResult,
    ActionLog,
    HarmCheck,
    SafetyCheck,
    # Context
    AutonomousContext,
    SelfMonitoringFinding,
    SelfMonitoringResult,
    # Protocols (for implementing providers)
    LLMProvider,
    SearchProvider,
    MemoryStoreProvider,
    ContextProvider,
    NotificationProvider,
)
from .prompts import (
    AUTONOMOUS_AGENT_SYSTEM_PROMPT,
    HARM_CHECK_PROMPT,
    SEMANTIC_SAFETY_PROMPT,
    REFLECTION_PROMPT,
    SELF_MONITORING_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
    VERIFY_ASSESSMENT_PROMPT,
)

__all__ = [
    # Service
    "AutonomousAgentService",
    # Config
    "AutonomousConfig",
    # Enums
    "ActionType",
    "ActionTier",
    # Types
    "ProposedAction",
    "ApprovedAction",
    "ActionResult",
    "ActionLog",
    "HarmCheck",
    "SafetyCheck",
    "AutonomousContext",
    "SelfMonitoringFinding",
    "SelfMonitoringResult",
    # Protocols
    "LLMProvider",
    "SearchProvider",
    "MemoryStoreProvider",
    "ContextProvider",
    "NotificationProvider",
    # Prompts
    "AUTONOMOUS_AGENT_SYSTEM_PROMPT",
    "HARM_CHECK_PROMPT",
    "SEMANTIC_SAFETY_PROMPT",
    "REFLECTION_PROMPT",
    "SELF_MONITORING_PROMPT",
    "RESEARCH_SYNTHESIS_PROMPT",
    "VERIFY_ASSESSMENT_PROMPT",
]

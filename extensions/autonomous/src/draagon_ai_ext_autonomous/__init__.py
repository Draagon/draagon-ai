"""Autonomous background agent extension for draagon-ai.

This extension provides autonomous background cognitive capabilities,
including:
- Background decision-making during idle time
- Multi-layer guardrail system for safe operation
- Self-monitoring and transparency logging
- Configurable action budgets and timing

Example:
    from draagon_ai.extensions import get_extension_manager

    manager = get_extension_manager()
    ext = manager.get_extension("autonomous")

    # Create service with your providers
    service = ext.create_service(
        llm=my_llm_provider,
        search=my_search_provider,
        memory_store=my_memory_store,
    )

    # Start the agent
    await service.start()
"""

from .extension import AutonomousExtension
from .service import AutonomousAgentService
from .types import (
    ActionLog,
    ActionResult,
    ActionTier,
    ActionType,
    ApprovedAction,
    AutonomousConfig,
    AutonomousContext,
    ContextProvider,
    HarmCheck,
    LLMProvider,
    MemoryStoreProvider,
    NotificationProvider,
    ProposedAction,
    SafetyCheck,
    SearchProvider,
    SelfMonitoringFinding,
    SelfMonitoringResult,
)

__version__ = "0.1.0"

__all__ = [
    # Extension
    "AutonomousExtension",
    # Service
    "AutonomousAgentService",
    # Config
    "AutonomousConfig",
    # Types
    "ActionType",
    "ActionTier",
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
    # Version
    "__version__",
]

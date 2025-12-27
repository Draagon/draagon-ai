"""Autonomous agent extension implementation.

This module contains the AutonomousExtension class that registers
the autonomous agent service with draagon-ai.
"""

from __future__ import annotations

from typing import Any

from draagon_ai.extensions import Extension, ExtensionInfo

from .types import AutonomousConfig
from .service import AutonomousAgentService


class AutonomousExtension(Extension):
    """Autonomous background agent extension for draagon-ai.

    This extension provides:
    - Autonomous agent service for background cognitive processes
    - Multi-layer guardrail system for safe autonomous operation
    - Self-monitoring and transparency logging
    - Configurable action budgets and timing

    Configuration options:
    - enabled: Enable/disable the agent
    - cycle_minutes: Minutes between autonomous cycles
    - daily_budget: Maximum actions per day
    - shadow_mode: Log only, don't execute actions
    - self_monitoring: Enable self-review of actions
    - active_hours_start: Don't run before this hour (0-23)
    - active_hours_end: Don't run after this hour (0-23)
    """

    def __init__(self) -> None:
        self._config = AutonomousConfig()
        self._service: AutonomousAgentService | None = None
        self._initialized: bool = False

    @property
    def info(self) -> ExtensionInfo:
        """Return extension metadata."""
        return ExtensionInfo(
            name="autonomous",
            version="0.1.0",
            description="Autonomous background agent with guardrails and self-monitoring",
            author="draagon-ai",
            requires_core=">=0.1.0",
            provides_services=[
                "autonomous_agent",
            ],
            provides_prompt_domains=[
                "autonomous",
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable the autonomous agent",
                    },
                    "cycle_minutes": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 120,
                        "default": 30,
                        "description": "Minutes between autonomous cycles",
                    },
                    "daily_budget": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "Maximum autonomous actions per day",
                    },
                    "shadow_mode": {
                        "type": "boolean",
                        "default": False,
                        "description": "Log actions without executing them",
                    },
                    "self_monitoring": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable self-review of autonomous actions",
                    },
                    "active_hours_start": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 23,
                        "default": 8,
                        "description": "Don't run before this hour",
                    },
                    "active_hours_end": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 23,
                        "default": 22,
                        "description": "Don't run after this hour",
                    },
                },
            },
            homepage="https://github.com/draagon-ai/draagon-ai",
            license="AGPL-3.0-or-later",
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize extension with configuration.

        Note: The service is created here but NOT started. The application
        must call start() on the service and provide the required providers
        (LLM, search, memory, etc.).

        Args:
            config: Extension configuration from draagon.yaml.
        """
        self._config = AutonomousConfig(
            enabled=config.get("enabled", True),
            cycle_interval_minutes=config.get("cycle_minutes", 30),
            daily_action_budget=config.get("daily_budget", 20),
            shadow_mode=config.get("shadow_mode", False),
            enable_self_monitoring=config.get("self_monitoring", True),
            active_hours_start=config.get("active_hours_start", 8),
            active_hours_end=config.get("active_hours_end", 22),
            persist_logs=config.get("persist_logs", True),
        )
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up on shutdown."""
        # Note: The application is responsible for stopping the service
        # before shutdown since it manages the async lifecycle
        self._service = None
        self._initialized = False

    def get_services(self) -> dict[str, Any]:
        """Return services provided by this extension.

        Note: Returns the config, not an instantiated service, because
        the service requires async providers (LLM, search, etc.) that
        the application must provide.

        Returns:
            Dict with autonomous agent config for service creation.
        """
        return {
            "autonomous_agent_config": self._config,
        }

    def create_service(
        self,
        llm: Any,
        search: Any = None,
        memory_store: Any = None,
        context_provider: Any = None,
        notification_provider: Any = None,
    ) -> AutonomousAgentService:
        """Create the autonomous agent service with providers.

        This factory method allows applications to provide their own
        implementations of the required protocols.

        Args:
            llm: LLM provider implementing LLMProvider protocol.
            search: Optional search provider implementing SearchProvider.
            memory_store: Optional storage implementing MemoryStoreProvider.
            context_provider: Optional context provider implementing ContextProvider.
            notification_provider: Optional notification provider.

        Returns:
            Configured AutonomousAgentService instance.
        """
        self._service = AutonomousAgentService(
            llm=llm,
            config=self._config,
            search=search,
            memory_store=memory_store,
            context_provider=context_provider,
            notification_provider=notification_provider,
        )
        return self._service

    def get_prompt_domains(self) -> dict[str, dict[str, str]]:
        """Return autonomous agent prompt domains.

        Returns:
            Dict with autonomous domain prompts.
        """
        from .prompts import (
            AUTONOMOUS_AGENT_SYSTEM_PROMPT,
            HARM_CHECK_PROMPT,
            SEMANTIC_SAFETY_PROMPT,
            REFLECTION_PROMPT,
            SELF_MONITORING_PROMPT,
        )

        return {
            "autonomous": {
                "AUTONOMOUS_AGENT_SYSTEM_PROMPT": AUTONOMOUS_AGENT_SYSTEM_PROMPT,
                "HARM_CHECK_PROMPT": HARM_CHECK_PROMPT,
                "SEMANTIC_SAFETY_PROMPT": SEMANTIC_SAFETY_PROMPT,
                "REFLECTION_PROMPT": REFLECTION_PROMPT,
                "SELF_MONITORING_PROMPT": SELF_MONITORING_PROMPT,
            },
        }

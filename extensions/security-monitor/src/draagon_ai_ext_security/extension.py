"""Security Monitor Extension implementation."""

from __future__ import annotations

import logging
from typing import Any

from draagon_ai.extensions import Extension, ExtensionInfo

logger = logging.getLogger(__name__)


class SecurityMonitorExtension(Extension):
    """Autonomous security monitoring with AI-powered analysis.

    This extension provides:
    - Multi-source monitoring (Suricata, syslogs, system health)
    - Groq 70B analysis for intelligent threat classification
    - Agentic investigation loop for deep analysis
    - Memory integration for learning from past issues
    - Voice announcements through Home Assistant

    Configuration options:
    - check_interval_seconds: How often to check for alerts
    - llm.provider: LLM provider (groq, openai, anthropic)
    - llm.model: Model to use for analysis
    - monitors.*: Enable/configure specific monitors
    - notifications.*: Notification channel settings

    Voice-configurable settings (via progressive discovery):
    - network.known_services: Add/remove known devices
    - notifications.voice.quiet_hours: Set quiet hours
    - notifications.voice.min_severity: Set minimum alert severity
    - check_interval_seconds: Set check interval
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._initialized: bool = False
        self._service: Any = None  # SecurityMonitorService
        self._config_service: Any = None  # ExtensionConfigService

    @property
    def info(self) -> ExtensionInfo:
        """Return extension metadata."""
        return ExtensionInfo(
            name="security-monitor",
            version="0.1.0",
            description="Autonomous security monitoring with agentic investigation",
            author="Doug",
            requires_core=">=0.1.0",
            provides_behaviors=[
                "security_analyst",
            ],
            provides_services=[
                "security_monitor",
            ],
            provides_tools=[
                "check_suricata_alerts",
                "search_threat_intel",
                "check_ip_reputation",
                "recall_security_memory",
                "security_status",
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "check_interval_seconds": {
                        "type": "integer",
                        "minimum": 60,
                        "maximum": 3600,
                        "default": 300,
                        "description": "How often to check for new alerts",
                    },
                    "llm": {
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "enum": ["groq", "openai", "anthropic", "ollama"],
                                "default": "groq",
                            },
                            "model": {
                                "type": "string",
                                "default": "llama-3.3-70b-versatile",
                            },
                            "api_key": {
                                "type": "string",
                                "description": "API key (or use env var)",
                            },
                        },
                    },
                    "network": {
                        "type": "object",
                        "properties": {
                            "home_net": {
                                "type": "string",
                                "default": "192.168.168.0/24",
                            },
                            "known_services": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "ip": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                    "monitors": {
                        "type": "object",
                        "properties": {
                            "suricata": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean", "default": True},
                                    "eve_log": {
                                        "type": "string",
                                        "default": "/var/log/suricata/eve.json",
                                    },
                                },
                            },
                            "syslog": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean", "default": True},
                                    "paths": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "default": ["/var/log/auth.log"],
                                    },
                                },
                            },
                            "system_health": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean", "default": True},
                                    "nvme_warning_celsius": {
                                        "type": "integer",
                                        "default": 60,
                                    },
                                },
                            },
                        },
                    },
                    "notifications": {
                        "type": "object",
                        "properties": {
                            "voice": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean", "default": True},
                                    "entity_id": {"type": "string"},
                                    "min_severity": {
                                        "type": "string",
                                        "enum": ["critical", "high", "medium"],
                                        "default": "high",
                                    },
                                },
                            },
                        },
                    },
                },
            },
            homepage="https://github.com/yourusername/draagon-ai",
            license="MIT",
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize extension with configuration.

        Args:
            config: Extension configuration from draagon.yaml.
        """
        self._config = config
        self._initialized = True

        # Set up the config service for hybrid YAML + memory config
        try:
            from draagon_ai.extensions import get_extension_config_service
            self._config_service = get_extension_config_service()
            logger.info("Extension config service initialized")
        except ImportError:
            logger.warning("Extension config service not available")

        # Initialize will be called by the framework
        # Actual service startup happens in get_services()

    def shutdown(self) -> None:
        """Clean up on shutdown."""
        if self._service:
            # Stop background monitoring
            self._service.stop()
        self._initialized = False

    def get_behaviors(self) -> list:
        """Return security-related behaviors.

        Returns:
            List of behavior templates for security analysis.
        """
        from draagon_ai_ext_security.behavior import SECURITY_ANALYST_TEMPLATE

        return [SECURITY_ANALYST_TEMPLATE]

    def get_services(self) -> dict[str, Any]:
        """Return security monitor service.

        Returns:
            Dict with the security monitor service instance.
        """
        from draagon_ai_ext_security.service import SecurityMonitorService

        if not self._service:
            self._service = SecurityMonitorService(self._config)
            # Wire up the config service
            if self._config_service:
                self._service.set_config_service(self._config_service)

        return {
            "security_monitor": self._service,
        }

    def get_tools(self) -> list:
        """Return security-specific tools.

        Returns:
            List of tools for security analysis.
        """
        from draagon_ai_ext_security.tools import get_security_tools, set_config_service

        # Wire up config service for tools
        if self._config_service:
            set_config_service(self._config_service)

        return get_security_tools(self._config)

    def get_prompt_domains(self) -> dict[str, dict[str, str]]:
        """Return security prompt domains.

        Returns:
            Dict with security-related prompts.
        """
        from draagon_ai_ext_security.prompts import (
            CLASSIFICATION_PROMPT,
            INVESTIGATION_PROMPT,
            VOICE_ANNOUNCEMENT_PROMPT,
            UNKNOWN_DEVICE_DISCOVERY_PROMPT,
            LEARN_SERVICE_FROM_RESPONSE_PROMPT,
            CONFIG_CHANGE_INTENT_PROMPT,
        )

        return {
            "security": {
                "CLASSIFICATION_PROMPT": CLASSIFICATION_PROMPT,
                "INVESTIGATION_PROMPT": INVESTIGATION_PROMPT,
                "VOICE_ANNOUNCEMENT_PROMPT": VOICE_ANNOUNCEMENT_PROMPT,
                # Progressive discovery prompts
                "UNKNOWN_DEVICE_DISCOVERY_PROMPT": UNKNOWN_DEVICE_DISCOVERY_PROMPT,
                "LEARN_SERVICE_FROM_RESPONSE_PROMPT": LEARN_SERVICE_FROM_RESPONSE_PROMPT,
                "CONFIG_CHANGE_INTENT_PROMPT": CONFIG_CHANGE_INTENT_PROMPT,
            },
        }

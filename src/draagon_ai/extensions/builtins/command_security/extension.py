"""Command Security Extension for draagon-ai.

This extension provides secure shell command execution with multi-level
security classification. It wraps command execution behind a security
layer that classifies commands before execution.

The extension uses pluggable backends for actual execution, defaulting
to HTTP API backend for security isolation.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from draagon_ai.extensions import Extension, ExtensionInfo
from draagon_ai.orchestration.registry import Tool, ToolParameter

from .backends import (
    CommandBackend,
    HTTPCommandBackend,
    LocalCommandBackend,
    SSHCommandBackend,
    ExecutionResult,
)
from .security import (
    CommandClassifier,
    SecurityLevel,
    SecurityClassification,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CommandSecurityExtension(Extension):
    """Secure command execution extension.

    Provides tools for executing shell commands with security classification.
    Commands are classified as safe, confirm, passcode, or blocked based on
    their potential impact.

    Features:
    - Multi-level security (safe/confirm/passcode/blocked)
    - LLM-based classification for unknown commands
    - Regex blocklist for dangerous patterns
    - Pluggable execution backends (http/local/ssh)
    - Audit logging of all command attempts

    Configuration (draagon.yaml):
        extensions:
          command_security:
            enabled: true
            config:
              passcode: "7734"
              backend: "http"
              http:
                url: "http://192.168.168.200:5555"
                token: "${COMMAND_API_TOKEN}"
              allowed_hosts:
                - local
                - beelink
              additional_blocked_patterns:
                - "my_dangerous_pattern"

    Example:
        ext = CommandSecurityExtension()
        ext.initialize({
            "passcode": "1234",
            "backend": "local",
        })
        tools = ext.get_tools()
    """

    def __init__(self) -> None:
        """Initialize the extension."""
        self._classifier = CommandClassifier()
        self._backend: CommandBackend | None = None
        self._passcode: str = "7734"
        self._allowed_hosts: list[str] = ["local"]
        self._enable_audit: bool = True
        self._audit_log: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}
        self._config: dict[str, Any] = {}
        self._initialized: bool = False

    @property
    def info(self) -> ExtensionInfo:
        """Return extension metadata."""
        return ExtensionInfo(
            name="command_security",
            version="1.0.0",
            description="Secure shell command execution with classification",
            author="draagon-ai",
            requires_core=">=0.1.0",
            provides_behaviors=[],
            provides_tools=[
                "execute_command",
                "confirm_command",
                "verify_passcode",
                "get_audit_log",
            ],
            provides_prompt_domains=["command_security"],
            provides_mcp_servers=[],
            config_schema={
                "type": "object",
                "properties": {
                    "passcode": {
                        "type": "string",
                        "description": "Security passcode for sensitive operations",
                        "default": "7734",
                    },
                    "backend": {
                        "type": "string",
                        "description": "Execution backend: http, local, or ssh",
                        "enum": ["http", "local", "ssh"],
                        "default": "http",
                    },
                    "enable_audit": {
                        "type": "boolean",
                        "description": "Enable audit logging",
                        "default": True,
                    },
                    "allowed_hosts": {
                        "type": "array",
                        "description": "Allowed execution hosts",
                        "items": {"type": "string"},
                        "default": ["local"],
                    },
                    "http": {
                        "type": "object",
                        "description": "HTTP backend configuration",
                        "properties": {
                            "url": {"type": "string"},
                            "token": {"type": "string"},
                        },
                    },
                    "ssh": {
                        "type": "object",
                        "description": "SSH backend host mappings",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Extension configuration from draagon.yaml
        """
        self._config = config
        self._passcode = config.get("passcode", "7734")
        self._enable_audit = config.get("enable_audit", True)
        self._allowed_hosts = config.get("allowed_hosts", ["local"])

        # Add additional blocked patterns if configured
        additional_patterns = config.get("additional_blocked_patterns", [])
        if additional_patterns:
            self._classifier = CommandClassifier(
                additional_blocked_patterns=additional_patterns
            )

        # Initialize backend
        backend_type = config.get("backend", "http")

        if backend_type == "http":
            http_config = config.get("http", {})
            self._backend = HTTPCommandBackend(
                base_url=http_config.get("url", "http://localhost:5555"),
                token=http_config.get("token"),
            )
        elif backend_type == "local":
            self._backend = LocalCommandBackend(
                allowed_hosts=self._allowed_hosts,
            )
        elif backend_type == "ssh":
            ssh_hosts = config.get("ssh", {})
            self._backend = SSHCommandBackend(hosts=ssh_hosts)
        else:
            logger.warning(f"Unknown backend type: {backend_type}, using local")
            self._backend = LocalCommandBackend(allowed_hosts=self._allowed_hosts)

        self._initialized = True
        logger.info(f"CommandSecurityExtension initialized with {backend_type} backend")

    def set_llm_classifier(self, classifier_fn: Any) -> None:
        """Set the LLM function for command classification.

        This should be called after initialization to enable LLM-based
        classification of unknown commands.

        Args:
            classifier_fn: Async function that takes a command string and
                          returns dict with level, reason, concerns
        """
        self._classifier.set_llm_classifier(classifier_fn)

    def _log_audit(
        self,
        command: str,
        host: str,
        classification: SecurityClassification,
        result: str,
        user_id: str,
    ) -> None:
        """Log a command attempt to the audit log."""
        if not self._enable_audit:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "host": host,
            "classification": classification.level.value,
            "reason": classification.reason,
            "llm_classified": classification.llm_classified,
            "result": result,
            "user_id": user_id,
        }
        self._audit_log.append(entry)

        # Keep last 100 entries
        if len(self._audit_log) > 100:
            self._audit_log = self._audit_log[-100:]

        logger.info(
            f"Audit: {classification.level.value.upper()} - "
            f"{command[:50]}{'...' if len(command) > 50 else ''} -> {result}"
        )

    async def _execute_command(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a command with security classification.

        Args:
            args: Tool arguments (command, host)
            context: Execution context (user_id, conversation_id)

        Returns:
            Result dict with success, output, or security prompts
        """
        command = args.get("command", "")
        host = args.get("host", "local")
        context = context or {}
        user_id = context.get("user_id", "unknown")
        conversation_id = context.get("conversation_id", "default")

        if not command:
            return {"error": "Command is required"}

        if host not in self._allowed_hosts:
            return {
                "error": f"Host '{host}' not allowed. Available: {self._allowed_hosts}"
            }

        # Classify the command
        classification = await self._classifier.classify(command)

        # Handle based on classification
        if classification.level == SecurityLevel.BLOCKED:
            self._log_audit(command, host, classification, "blocked", user_id)
            return {
                "success": False,
                "error": classification.reason,
                "blocked": True,
                "concerns": classification.concerns,
            }

        if classification.level == SecurityLevel.PASSCODE:
            pending_id = str(uuid.uuid4())
            self._pending[pending_id] = {
                "command": command,
                "host": host,
                "classification": classification,
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "needs_passcode": True,
            }
            self._log_audit(command, host, classification, "pending_passcode", user_id)
            reason_part = (
                f"({classification.reason}) " if classification.llm_classified else ""
            )
            return {
                "success": False,
                "needs_passcode": True,
                "pending_id": pending_id,
                "command": command,
                "host": host,
                "message": (
                    f"This command requires passcode verification. {reason_part}"
                    'Say your 4-digit passcode to proceed, or "cancel" to abort.'
                ),
            }

        if classification.level == SecurityLevel.CONFIRM:
            pending_id = str(uuid.uuid4())
            self._pending[pending_id] = {
                "command": command,
                "host": host,
                "classification": classification,
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "user_id": user_id,
            }
            self._log_audit(command, host, classification, "pending_confirm", user_id)
            return {
                "success": False,
                "needs_confirmation": True,
                "pending_id": pending_id,
                "command": command,
                "host": host,
                "message": (
                    f'I\'ll run "{command}" on {host}. '
                    'Say "yes" to confirm or "cancel" to abort.'
                ),
            }

        # Safe command - execute immediately
        return await self._do_execute(command, host, classification, user_id)

    async def _do_execute(
        self,
        command: str,
        host: str,
        classification: SecurityClassification,
        user_id: str,
    ) -> dict[str, Any]:
        """Actually execute a command via the backend."""
        if not self._backend:
            return {"error": "No execution backend configured"}

        try:
            result: ExecutionResult = await self._backend.execute(command, host)
            self._log_audit(command, host, classification, "executed", user_id)

            return {
                "success": result.success,
                "output": result.output,
                "exit_code": result.exit_code,
                "host": result.host,
                "elapsed_ms": result.elapsed_ms,
            }
        except Exception as e:
            self._log_audit(command, host, classification, f"error: {e}", user_id)
            return {"success": False, "error": str(e)}

    async def _confirm_command(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Confirm or cancel a pending command.

        Args:
            args: Tool arguments (pending_id, confirmed)
            context: Execution context

        Returns:
            Execution result or cancellation message
        """
        pending_id = args.get("pending_id", "")
        confirmed = args.get("confirmed", False)
        context = context or {}
        user_id = context.get("user_id", "unknown")
        conversation_id = context.get("conversation_id", "default")

        # Clean up expired pending commands (60 second timeout)
        self._cleanup_expired_pending()

        # Find pending command
        pending = None
        if pending_id and pending_id in self._pending:
            pending = self._pending.pop(pending_id)
        else:
            # Try to find by conversation_id
            for pid, cmd in list(self._pending.items()):
                if cmd["conversation_id"] == conversation_id:
                    pending = self._pending.pop(pid)
                    break

        if not pending:
            return {"error": "No pending command to confirm."}

        if pending.get("needs_passcode"):
            return {"error": "This command requires a passcode, not just confirmation."}

        if not confirmed:
            self._log_audit(
                pending["command"],
                pending["host"],
                pending["classification"],
                "cancelled",
                user_id,
            )
            return {"cancelled": True, "message": "Command cancelled."}

        return await self._do_execute(
            pending["command"],
            pending["host"],
            pending["classification"],
            user_id,
        )

    async def _verify_passcode(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify passcode and execute pending command.

        Args:
            args: Tool arguments (passcode, pending_id)
            context: Execution context

        Returns:
            Execution result or error message
        """
        passcode = args.get("passcode", "")
        pending_id = args.get("pending_id", "")
        context = context or {}
        user_id = context.get("user_id", "unknown")
        conversation_id = context.get("conversation_id", "default")

        # Clean up expired pending commands
        self._cleanup_expired_pending()

        # Find pending command
        pending = None
        if pending_id and pending_id in self._pending:
            pending = self._pending.pop(pending_id)
        else:
            # Try to find by conversation_id
            for pid, cmd in list(self._pending.items()):
                if cmd["conversation_id"] == conversation_id:
                    pending = self._pending.pop(pid)
                    break

        if not pending:
            return {"error": "No pending command to verify."}

        if passcode != self._passcode:
            self._log_audit(
                pending["command"],
                pending["host"],
                pending["classification"],
                "wrong_passcode",
                user_id,
            )
            return {"error": "Incorrect passcode. Command not executed."}

        return await self._do_execute(
            pending["command"],
            pending["host"],
            pending["classification"],
            user_id,
        )

    async def _get_audit_log(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get recent audit log entries.

        Args:
            args: Tool arguments (limit)
            context: Execution context

        Returns:
            Audit log entries
        """
        limit = args.get("limit", 20)
        return {
            "entries": self._audit_log[-limit:],
            "total": len(self._audit_log),
        }

    def _cleanup_expired_pending(self) -> None:
        """Remove pending commands older than 60 seconds."""
        now = time.time()
        expired = [
            pid for pid, cmd in self._pending.items() if now - cmd["timestamp"] > 60
        ]
        for pid in expired:
            del self._pending[pid]

    def get_tools(self) -> list[Tool]:
        """Return tools provided by this extension."""
        return [
            Tool(
                name="execute_command",
                description=(
                    "Execute a shell command on local or remote host. "
                    "Commands are security-classified and may require "
                    "confirmation or passcode."
                ),
                handler=self._execute_command,
                parameters=[
                    ToolParameter(
                        name="command",
                        type="string",
                        description="The shell command to execute",
                        required=True,
                    ),
                    ToolParameter(
                        name="host",
                        type="string",
                        description="Target host (default: local)",
                        required=False,
                        default="local",
                    ),
                ],
                requires_confirmation=True,
            ),
            Tool(
                name="confirm_command",
                description="Confirm or cancel a pending command",
                handler=self._confirm_command,
                parameters=[
                    ToolParameter(
                        name="confirmed",
                        type="boolean",
                        description="Whether to proceed with the command",
                        required=True,
                    ),
                    ToolParameter(
                        name="pending_id",
                        type="string",
                        description="ID of pending command (optional)",
                        required=False,
                    ),
                ],
            ),
            Tool(
                name="verify_passcode",
                description="Verify security passcode for a pending command",
                handler=self._verify_passcode,
                parameters=[
                    ToolParameter(
                        name="passcode",
                        type="string",
                        description="The 4-digit security PIN",
                        required=True,
                    ),
                    ToolParameter(
                        name="pending_id",
                        type="string",
                        description="ID of pending command (optional)",
                        required=False,
                    ),
                ],
            ),
            Tool(
                name="get_audit_log",
                description="Get recent command security audit log",
                handler=self._get_audit_log,
                parameters=[
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum entries to return",
                        required=False,
                        default=20,
                    ),
                ],
            ),
        ]

    def get_prompt_domains(self) -> dict[str, dict[str, str]]:
        """Return prompt domains for this extension."""
        return {
            "command_security": {
                "COMMAND_CLASSIFIER_PROMPT": CommandClassifier.get_classifier_prompt(),
            }
        }

    def shutdown(self) -> None:
        """Clean up resources."""
        self._pending.clear()
        self._initialized = False

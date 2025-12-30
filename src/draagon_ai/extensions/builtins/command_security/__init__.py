"""Command Security Extension for draagon-ai.

Provides secure shell command execution with multi-level security
classification. Commands are classified as safe, confirm, passcode,
or blocked based on their potential impact.

Security Levels:
- SAFE: Execute immediately (read-only commands like ls, df, docker ps)
- CONFIRM: Verbal confirmation required (service restarts, config changes)
- PASSCODE: PIN required (file deletion, package installation)
- BLOCKED: Never execute (rm -rf /, privilege escalation, crypto mining)

Backends:
- http: Execute via HTTP API (default)
- local: Execute locally via subprocess
- ssh: Execute via SSH on remote hosts

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

Usage:
    from draagon_ai.extensions.builtins.command_security import (
        CommandSecurityExtension,
    )

    ext = CommandSecurityExtension()
    ext.initialize({"passcode": "1234", "backend": "local"})
    tools = ext.get_tools()
"""

from .extension import CommandSecurityExtension
from .backends import (
    CommandBackend,
    HTTPCommandBackend,
    LocalCommandBackend,
    SSHCommandBackend,
    ExecutionResult,
)
from .security import (
    SecurityLevel,
    SecurityClassification,
    CommandClassifier,
    BLOCKED_PATTERNS,
)

__all__ = [
    # Extension
    "CommandSecurityExtension",
    # Backends
    "CommandBackend",
    "HTTPCommandBackend",
    "LocalCommandBackend",
    "SSHCommandBackend",
    "ExecutionResult",
    # Security
    "SecurityLevel",
    "SecurityClassification",
    "CommandClassifier",
    "BLOCKED_PATTERNS",
]

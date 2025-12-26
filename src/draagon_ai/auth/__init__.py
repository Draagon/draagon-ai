"""Authentication and credential management for Draagon AI.

Provides abstract interfaces and implementations for credential storage
with multi-level scoping (global, org, user, session).
"""

from draagon_ai.auth.scopes import CredentialScope
from draagon_ai.auth.store import (
    Credential,
    CredentialStore,
)
from draagon_ai.auth.adapters import (
    EnvCredentialStore,
    InMemoryCredentialStore,
)

__all__ = [
    # Scopes
    "CredentialScope",
    # Core types
    "Credential",
    "CredentialStore",
    # Implementations
    "EnvCredentialStore",
    "InMemoryCredentialStore",
]

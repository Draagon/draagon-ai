"""Reference implementations of CredentialStore.

These adapters provide simple credential storage for common use cases:
- EnvCredentialStore: Read credentials from environment variables
- InMemoryCredentialStore: Store credentials in memory (for testing)
"""

import os
from typing import Any

from draagon_ai.auth.scopes import CredentialScope
from draagon_ai.auth.store import Credential, CredentialStore


class EnvCredentialStore(CredentialStore):
    """Simple credential store that reads from environment variables.

    Credentials are looked up as:
    - DRAAGON_CRED_{NAME} for the value
    - DRAAGON_CRED_{NAME}_TYPE for the type (default: api_key)

    This store only supports GLOBAL scope and is read-only.

    Example:
        # In environment:
        # DRAAGON_CRED_GROQ=gsk_xxx
        # DRAAGON_CRED_GOOGLE_CALENDAR=oauth:token:refresh

        store = EnvCredentialStore()
        cred = await store.get_credential(
            "groq",
            CredentialScope.GLOBAL,
            {},
        )
    """

    def __init__(self, prefix: str = "DRAAGON_CRED_"):
        """Initialize with custom environment variable prefix.

        Args:
            prefix: Prefix for environment variable names.
        """
        self._prefix = prefix

    async def get_credential(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> Credential | None:
        """Get credential from environment variable."""
        # Only global scope is supported
        if scope != CredentialScope.GLOBAL:
            return None

        env_key = f"{self._prefix}{name.upper()}"
        value = os.environ.get(env_key)

        if not value:
            return None

        # Check for type override
        type_key = f"{env_key}_TYPE"
        cred_type = os.environ.get(type_key, "api_key")

        return Credential(
            type=cred_type,
            value=value,
            name=name,
            scope=CredentialScope.GLOBAL,
        )

    async def store_credential(
        self,
        credential: Credential,
        context: dict[str, Any],
    ) -> None:
        """Not supported - environment variables are read-only."""
        raise NotImplementedError(
            "EnvCredentialStore is read-only. "
            "Set credentials via environment variables."
        )

    async def delete_credential(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> bool:
        """Not supported - environment variables are read-only."""
        raise NotImplementedError(
            "EnvCredentialStore is read-only. "
            "Remove credentials from environment variables."
        )

    async def list_credentials(
        self,
        scope: CredentialScope | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Credential]:
        """List all credentials found in environment."""
        if scope is not None and scope != CredentialScope.GLOBAL:
            return []

        credentials = []
        for key, value in os.environ.items():
            if key.startswith(self._prefix) and not key.endswith("_TYPE"):
                name = key[len(self._prefix):].lower()
                type_key = f"{key}_TYPE"
                cred_type = os.environ.get(type_key, "api_key")
                credentials.append(Credential(
                    type=cred_type,
                    value=value,
                    name=name,
                    scope=CredentialScope.GLOBAL,
                ))

        return credentials


class InMemoryCredentialStore(CredentialStore):
    """In-memory credential store for testing.

    All credentials are stored in memory and lost when the process exits.
    Supports all scopes and operations.

    Example:
        store = InMemoryCredentialStore()
        await store.store_credential(
            Credential(
                type="oauth",
                value="access_token",
                name="google_calendar",
                scope=CredentialScope.USER,
                refresh_token="refresh_token",
            ),
            {"user_id": "doug"},
        )
    """

    def __init__(self) -> None:
        """Initialize empty credential store."""
        self._credentials: dict[str, Credential] = {}

    def _make_key(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> str:
        """Create storage key from name, scope, and context."""
        parts = [name, scope.value]

        if scope == CredentialScope.ORG:
            parts.append(context.get("org_id", ""))
        elif scope == CredentialScope.USER:
            parts.append(context.get("user_id", ""))
        elif scope == CredentialScope.SESSION:
            parts.append(context.get("session_id", ""))

        return ":".join(parts)

    async def get_credential(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> Credential | None:
        """Get credential from memory."""
        key = self._make_key(name, scope, context)
        return self._credentials.get(key)

    async def store_credential(
        self,
        credential: Credential,
        context: dict[str, Any],
    ) -> None:
        """Store credential in memory."""
        key = self._make_key(credential.name, credential.scope, context)
        self._credentials[key] = credential

    async def delete_credential(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> bool:
        """Delete credential from memory."""
        key = self._make_key(name, scope, context)
        if key in self._credentials:
            del self._credentials[key]
            return True
        return False

    async def list_credentials(
        self,
        scope: CredentialScope | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Credential]:
        """List credentials, optionally filtered by scope."""
        credentials = list(self._credentials.values())

        if scope is not None:
            credentials = [c for c in credentials if c.scope == scope]

        return credentials

    def clear(self) -> None:
        """Clear all stored credentials."""
        self._credentials.clear()

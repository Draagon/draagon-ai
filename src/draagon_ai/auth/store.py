"""Abstract credential storage interface.

CredentialStore provides a protocol for storing and retrieving
credentials at different scopes. Applications provide their own
implementation (file-based, database, vault, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from draagon_ai.auth.scopes import CredentialScope


@dataclass
class Credential:
    """A stored credential.

    Credentials can be of various types:
    - api_key: Simple API key
    - oauth: OAuth2 token with refresh capability
    - bearer: Bearer token
    - basic: Username/password pair (value = "username:password")

    Attributes:
        type: Type of credential (api_key, oauth, bearer, basic)
        value: The credential value (should be encrypted at rest)
        name: Identifier for this credential (e.g., "google_calendar")
        scope: At what level this credential is stored
        expires_at: When this credential expires (if applicable)
        refresh_token: OAuth refresh token (if applicable)
        metadata: Additional info (e.g., scopes granted)
    """

    type: str
    value: str
    name: str
    scope: CredentialScope
    expires_at: datetime | None = None
    refresh_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if this credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_oauth(self) -> bool:
        """Check if this is an OAuth credential."""
        return self.type == "oauth"

    @property
    def can_refresh(self) -> bool:
        """Check if this credential can be refreshed."""
        return self.is_oauth and self.refresh_token is not None


class CredentialStore(ABC):
    """Abstract interface for credential storage.

    Applications provide their own implementation based on their
    storage needs (environment variables, file, database, vault, etc.).

    Context dict typically contains:
    - org_id: Organization identifier (for ORG scope)
    - user_id: User identifier (for USER scope)
    - session_id: Session identifier (for SESSION scope)

    Example:
        class MyCredentialStore(CredentialStore):
            async def get_credential(self, name, scope, context):
                # Look up in your storage
                return Credential(...)

        store = MyCredentialStore()
        cred = await store.get_credential(
            "google_calendar",
            CredentialScope.USER,
            {"user_id": "doug"},
        )
    """

    @abstractmethod
    async def get_credential(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> Credential | None:
        """Get a credential by name and scope.

        Args:
            name: Credential identifier (e.g., "google_calendar")
            scope: The scope to look up
            context: Context with org_id, user_id, session_id as needed

        Returns:
            The credential if found, None otherwise.
        """
        ...

    @abstractmethod
    async def store_credential(
        self,
        credential: Credential,
        context: dict[str, Any],
    ) -> None:
        """Store a credential.

        Args:
            credential: The credential to store
            context: Context with org_id, user_id, session_id as needed
        """
        ...

    @abstractmethod
    async def delete_credential(
        self,
        name: str,
        scope: CredentialScope,
        context: dict[str, Any],
    ) -> bool:
        """Delete a credential.

        Args:
            name: Credential identifier
            scope: The scope to delete from
            context: Context with org_id, user_id, session_id as needed

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def list_credentials(
        self,
        scope: CredentialScope | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Credential]:
        """List credentials, optionally filtered by scope.

        Args:
            scope: Filter to this scope (or all if None)
            context: Context for scoped lookups

        Returns:
            List of credentials matching the filter.
        """
        ...

    async def get_credential_cascading(
        self,
        name: str,
        context: dict[str, Any],
    ) -> Credential | None:
        """Get a credential, cascading through scopes.

        Looks up from most specific to most general:
        session -> user -> org -> global

        Returns the first match found.
        """
        # Session scope (most specific)
        if "session_id" in context:
            cred = await self.get_credential(name, CredentialScope.SESSION, context)
            if cred:
                return cred

        # User scope
        if "user_id" in context:
            cred = await self.get_credential(name, CredentialScope.USER, context)
            if cred:
                return cred

        # Org scope
        if "org_id" in context:
            cred = await self.get_credential(name, CredentialScope.ORG, context)
            if cred:
                return cred

        # Global scope (most general)
        return await self.get_credential(name, CredentialScope.GLOBAL, context)

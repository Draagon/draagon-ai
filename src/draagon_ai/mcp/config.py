"""Configuration for MCP Memory Server.

This module defines configuration dataclasses for the MCP server,
including Qdrant connection, authentication, and scope settings.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPScope(str, Enum):
    """Scopes available for MCP clients.

    Maps to draagon-ai MemoryScope:
    - PRIVATE -> USER (per-user within agent)
    - SHARED -> CONTEXT (shared within context/household)
    - SYSTEM -> WORLD (global facts)

    Scope Hierarchy (for access control):
    - SYSTEM (highest) - can read all, can only write SYSTEM
    - SHARED (middle) - can read SHARED/SYSTEM, can write SHARED/PRIVATE
    - PRIVATE (lowest) - can only read/write PRIVATE
    """

    PRIVATE = "private"
    SHARED = "shared"
    SYSTEM = "system"


# Scope hierarchy level (higher number = broader scope)
SCOPE_HIERARCHY: dict[MCPScope, int] = {
    MCPScope.PRIVATE: 0,  # Most restrictive
    MCPScope.SHARED: 1,   # Household level
    MCPScope.SYSTEM: 2,   # Global level (most permissive for read)
}


def get_scope_level(scope: MCPScope | str) -> int:
    """Get hierarchy level for a scope.

    Args:
        scope: Scope to get level for.

    Returns:
        Hierarchy level (0=PRIVATE, 1=SHARED, 2=SYSTEM).
    """
    if isinstance(scope, str):
        scope = MCPScope(scope.lower())
    return SCOPE_HIERARCHY.get(scope, 0)


def can_read_scope(client_scopes: list[MCPScope], target_scope: MCPScope | str) -> bool:
    """Check if client can read from target scope.

    Read access rules:
    - Client can read from scopes at or above their highest allowed scope
    - E.g., SHARED client can read SHARED and SYSTEM, not PRIVATE (of others)
    - PRIVATE scope is always readable by the owning user
    - SYSTEM scope is always readable (global facts)

    Args:
        client_scopes: Scopes the client is allowed to access.
        target_scope: Scope to read from.

    Returns:
        True if read is allowed.
    """
    if isinstance(target_scope, str):
        target_scope = MCPScope(target_scope.lower())

    # Special case: Everyone can read SYSTEM (global facts)
    if target_scope == MCPScope.SYSTEM:
        return True

    # No scopes means no access (except SYSTEM above)
    if not client_scopes:
        return False

    # Client can read from their allowed scopes
    return target_scope in client_scopes


def can_write_scope(client_scopes: list[MCPScope], target_scope: MCPScope | str) -> bool:
    """Check if client can write to target scope.

    Write access rules:
    - Client can only write to scopes they are explicitly allowed
    - SYSTEM scope requires explicit SYSTEM permission
    - This is more restrictive than read

    Args:
        client_scopes: Scopes the client is allowed to access.
        target_scope: Scope to write to.

    Returns:
        True if write is allowed.
    """
    if isinstance(target_scope, str):
        target_scope = MCPScope(target_scope.lower())

    # Write requires explicit permission
    return target_scope in client_scopes


def get_readable_scopes(client_scopes: list[MCPScope]) -> list[MCPScope]:
    """Get all scopes a client can read from.

    Args:
        client_scopes: Scopes the client is allowed to access.

    Returns:
        List of scopes the client can read.
    """
    # Always include explicitly allowed scopes
    readable = set(client_scopes)

    # SYSTEM is always readable (public knowledge)
    readable.add(MCPScope.SYSTEM)

    return list(readable)


@dataclass
class ClientConfig:
    """Configuration for an authenticated MCP client."""

    client_id: str
    """Unique identifier for this client."""

    name: str
    """Human-readable name for the client."""

    allowed_scopes: list[MCPScope] = field(default_factory=lambda: [MCPScope.PRIVATE])
    """Scopes this client can access."""

    default_user_id: str | None = None
    """Default user ID for this client (used if not specified in requests)."""

    default_agent_id: str = "claude-code"
    """Default agent ID for this client."""

    max_requests_per_minute: int = 60
    """Rate limit for this client."""


@dataclass
class MCPConfig:
    """Configuration for the Memory MCP Server."""

    # Qdrant connection
    qdrant_url: str = field(
        default_factory=lambda: os.environ.get(
            "QDRANT_URL", "http://192.168.168.216:6333"
        )
    )
    """Qdrant server URL."""

    qdrant_api_key: str | None = field(
        default_factory=lambda: os.environ.get("QDRANT_API_KEY")
    )
    """Qdrant API key (optional)."""

    qdrant_collection: str = field(
        default_factory=lambda: os.environ.get(
            "QDRANT_COLLECTION", "draagon_memories"
        )
    )
    """Qdrant collection name."""

    # Embedding provider
    ollama_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_URL", "http://192.168.168.200:11434"
        )
    )
    """Ollama server URL for embeddings."""

    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    )
    """Embedding model name."""

    embedding_dimension: int = 768
    """Embedding dimension (must match model)."""

    # Server settings
    server_name: str = "draagon-memory"
    """MCP server name."""

    server_version: str = "1.0.0"
    """MCP server version."""

    # Authentication
    api_keys: dict[str, ClientConfig] = field(default_factory=dict)
    """Map of API keys to client configurations."""

    require_auth: bool = field(
        default_factory=lambda: os.environ.get("MCP_REQUIRE_AUTH", "false").lower()
        == "true"
    )
    """Whether to require authentication."""

    # Default client (when auth not required)
    default_client_id: str = "default"
    """Client ID used when auth is not required."""

    default_user_id: str = field(
        default_factory=lambda: os.environ.get("MCP_DEFAULT_USER", "claude-code-user")
    )
    """Default user ID when not specified."""

    default_agent_id: str = field(
        default_factory=lambda: os.environ.get("MCP_DEFAULT_AGENT", "claude-code")
    )
    """Default agent ID."""

    # Scope settings
    default_scope: MCPScope = MCPScope.PRIVATE
    """Default scope for memory operations."""

    allowed_scopes: list[MCPScope] = field(
        default_factory=lambda: [MCPScope.PRIVATE, MCPScope.SHARED, MCPScope.SYSTEM]
    )
    """Scopes available to unauthenticated clients."""

    # Performance
    search_limit_default: int = 5
    """Default limit for search results."""

    search_limit_max: int = 50
    """Maximum allowed search limit."""

    list_limit_default: int = 10
    """Default limit for list results."""

    list_limit_max: int = 100
    """Maximum allowed list limit."""

    # Logging
    log_level: str = field(
        default_factory=lambda: os.environ.get("MCP_LOG_LEVEL", "INFO")
    )
    """Logging level."""

    log_requests: bool = True
    """Whether to log all requests."""

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """Create config from environment variables.

        Environment variables:
        - QDRANT_URL: Qdrant server URL
        - QDRANT_API_KEY: Qdrant API key
        - QDRANT_COLLECTION: Qdrant collection name
        - OLLAMA_URL: Ollama server URL
        - EMBEDDING_MODEL: Embedding model name
        - MCP_REQUIRE_AUTH: Whether to require authentication
        - MCP_DEFAULT_USER: Default user ID
        - MCP_DEFAULT_AGENT: Default agent ID
        - MCP_LOG_LEVEL: Logging level
        """
        return cls()

    def get_client(self, api_key: str | None) -> ClientConfig | None:
        """Get client config for an API key.

        Args:
            api_key: The API key to look up.

        Returns:
            Client config if found and valid, None otherwise.
        """
        if not api_key:
            return None
        return self.api_keys.get(api_key)

    def add_client(
        self,
        api_key: str,
        client_id: str,
        name: str,
        scopes: list[MCPScope] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Add a new authenticated client.

        Args:
            api_key: API key for authentication.
            client_id: Unique client identifier.
            name: Human-readable client name.
            scopes: Allowed scopes (defaults to [PRIVATE]).
            user_id: Default user ID for this client.
            agent_id: Default agent ID for this client.
        """
        self.api_keys[api_key] = ClientConfig(
            client_id=client_id,
            name=name,
            allowed_scopes=scopes or [MCPScope.PRIVATE],
            default_user_id=user_id,
            default_agent_id=agent_id or self.default_agent_id,
        )

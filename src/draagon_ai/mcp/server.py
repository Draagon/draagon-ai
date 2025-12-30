"""Memory MCP Server for Draagon AI.

This module implements an MCP server that exposes draagon-ai memory operations
as tools. It allows Claude Code and other MCP clients to store and search
a shared knowledge base.

Usage:
    # Run as module
    python -m draagon_ai.mcp.server

    # Or import and run
    from draagon_ai.mcp import create_memory_mcp_server
    server = create_memory_mcp_server()
    server.run()

Tools provided:
    - memory.store: Store a memory in the knowledge base
    - memory.search: Search for memories
    - memory.list: List recent memories
    - memory.get: Get a specific memory by ID
    - memory.delete: Delete a memory
    - beliefs.reconcile: Add an observation to reconcile with beliefs
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from mcp.server.fastmcp import FastMCP


# =============================================================================
# Authentication
# =============================================================================


@dataclass
class AuthResult:
    """Result of authentication attempt."""

    authenticated: bool
    client_config: Any | None = None
    error: str | None = None


@dataclass
class AuthAuditEntry:
    """Audit log entry for authentication attempts."""

    timestamp: datetime
    api_key_prefix: str  # Only first/last 4 chars for security
    success: bool
    client_id: str | None
    error: str | None = None


class MCPAuthenticator:
    """Handles API key authentication for MCP server.

    Attributes:
        config: MCP server configuration.
        audit_log: List of recent auth attempts.
    """

    def __init__(self, config: Any):
        """Initialize authenticator with config.

        Args:
            config: MCPConfig instance.
        """
        self.config = config
        self.audit_log: list[AuthAuditEntry] = []
        self._max_audit_entries = 1000

    def authenticate(self, api_key: str | None) -> AuthResult:
        """Authenticate a request using API key.

        Args:
            api_key: The API key from request headers.

        Returns:
            AuthResult with authentication status and client config.
        """
        # If auth not required, return success with default context
        if not self.config.require_auth:
            return AuthResult(authenticated=True, client_config=None)

        # Auth required but no key provided
        if not api_key:
            self._log_auth_attempt(None, False, None, "No API key provided")
            return AuthResult(
                authenticated=False,
                error="Authentication required. Provide API key in X-API-Key header.",
            )

        # Look up client config
        client_config = self.config.get_client(api_key)

        if client_config is None:
            self._log_auth_attempt(api_key, False, None, "Invalid API key")
            return AuthResult(
                authenticated=False,
                error="Invalid API key.",
            )

        # Success
        self._log_auth_attempt(api_key, True, client_config.client_id)
        return AuthResult(
            authenticated=True,
            client_config=client_config,
        )

    def _log_auth_attempt(
        self,
        api_key: str | None,
        success: bool,
        client_id: str | None,
        error: str | None = None,
    ) -> None:
        """Log an authentication attempt.

        Args:
            api_key: The API key used (will be masked).
            success: Whether auth succeeded.
            client_id: Client ID if authenticated.
            error: Error message if failed.
        """
        # Mask API key for security
        if api_key and len(api_key) >= 8:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        elif api_key:
            masked_key = f"{api_key[:2]}..."
        else:
            masked_key = "(none)"

        entry = AuthAuditEntry(
            timestamp=datetime.utcnow(),
            api_key_prefix=masked_key,
            success=success,
            client_id=client_id,
            error=error,
        )

        self.audit_log.append(entry)

        # Trim log if too large
        if len(self.audit_log) > self._max_audit_entries:
            self.audit_log = self.audit_log[-self._max_audit_entries:]

        # Log to standard logger
        if success:
            logger.info(f"Auth success: client={client_id}, key={masked_key}")
        else:
            logger.warning(f"Auth failed: key={masked_key}, error={error}")

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of audit entries as dicts.
        """
        entries = self.audit_log[-limit:]
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "api_key_prefix": e.api_key_prefix,
                "success": e.success,
                "client_id": e.client_id,
                "error": e.error,
            }
            for e in entries
        ]

from draagon_ai.mcp.config import (
    ClientConfig,
    MCPConfig,
    MCPScope,
    can_read_scope,
    can_write_scope,
    get_readable_scopes,
)

# Optional dependencies - memory provider
try:
    from draagon_ai.memory.providers.layered import (
        LayeredMemoryConfig,
        LayeredMemoryProvider,
    )
    from draagon_ai.memory.base import MemoryScope, MemoryType

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    LayeredMemoryProvider = None  # type: ignore
    LayeredMemoryConfig = None  # type: ignore
    MemoryScope = None  # type: ignore
    MemoryType = None  # type: ignore

# Optional - embedding provider
try:
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    OllamaEmbeddingProvider = None  # type: ignore

# Optional - belief reconciliation
try:
    from draagon_ai.cognition.beliefs import BeliefReconciliationService

    BELIEFS_AVAILABLE = True
except ImportError:
    BELIEFS_AVAILABLE = False
    BeliefReconciliationService = None  # type: ignore


logger = logging.getLogger(__name__)


# =============================================================================
# Scope Mapping
# =============================================================================

# Map MCP scopes to draagon-ai MemoryScope (lowercase values to match enum)
SCOPE_MAPPING = {
    "private": "user",
    "shared": "context",
    "system": "world",
}

# Map MCP memory types to draagon-ai MemoryType (lowercase values to match enum)
TYPE_MAPPING = {
    "fact": "fact",
    "skill": "skill",
    "insight": "insight",
    "preference": "preference",
    "episodic": "episodic",
    "instruction": "instruction",
}


def map_scope_to_draagon(scope: str) -> str:
    """Map MCP scope to draagon-ai MemoryScope."""
    return SCOPE_MAPPING.get(scope.lower(), "user")


def map_type_to_draagon(memory_type: str) -> str:
    """Map MCP memory type to draagon-ai MemoryType."""
    return TYPE_MAPPING.get(memory_type.lower(), "fact")


# =============================================================================
# Memory MCP Server
# =============================================================================


class MemoryMCPServer:
    """MCP server exposing memory operations.

    This server provides tools for storing, searching, and managing
    memories in the draagon-ai knowledge base.

    Attributes:
        config: Server configuration.
        mcp: FastMCP server instance.
        memory: LayeredMemoryProvider for memory operations.
    """

    def __init__(
        self,
        config: MCPConfig | None = None,
        memory_provider: Any | None = None,
    ):
        """Initialize the Memory MCP Server.

        Args:
            config: Server configuration (defaults to MCPConfig.from_env()).
            memory_provider: Optional pre-configured memory provider.
        """
        self.config = config or MCPConfig.from_env()
        self._memory_provider = memory_provider
        self._memory_initialized = False
        self._client_context: ClientConfig | None = None

        # Create authenticator
        self.authenticator = MCPAuthenticator(self.config)

        # Create FastMCP server
        self.mcp = FastMCP(
            name=self.config.server_name,
        )

        # Register tools
        self._register_tools()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )

    async def _get_memory(self) -> Any:
        """Get or create memory provider.

        Returns:
            Initialized LayeredMemoryProvider.

        Raises:
            RuntimeError: If memory system not available.
        """
        if not MEMORY_AVAILABLE:
            raise RuntimeError(
                "Memory system not available. "
                "Install draagon-ai with memory support."
            )

        if self._memory_provider is not None:
            return self._memory_provider

        if not self._memory_initialized:
            # Create embedding provider
            if not EMBEDDING_AVAILABLE:
                raise RuntimeError(
                    "Embedding provider not available. "
                    "Install draagon-ai with embedding support."
                )

            embedding = OllamaEmbeddingProvider(
                base_url=self.config.ollama_url,
                model=self.config.embedding_model,
            )

            # Create memory config
            memory_config = LayeredMemoryConfig(
                qdrant_url=self.config.qdrant_url,
                qdrant_api_key=self.config.qdrant_api_key,
                qdrant_nodes_collection=f"{self.config.qdrant_collection}_nodes",
                qdrant_edges_collection=f"{self.config.qdrant_collection}_edges",
                embedding_dimension=self.config.embedding_dimension,
            )

            # Create and initialize memory provider
            self._memory_provider = LayeredMemoryProvider(
                config=memory_config,
                embedding_provider=embedding,
            )
            await self._memory_provider.initialize()
            self._memory_initialized = True

        return self._memory_provider

    def _get_user_id(self, user_id: str | None = None) -> str:
        """Get effective user ID.

        Args:
            user_id: Explicit user ID or None.

        Returns:
            User ID to use for operation.
        """
        if user_id:
            return user_id
        if self._client_context and self._client_context.default_user_id:
            return self._client_context.default_user_id
        return self.config.default_user_id

    def _get_agent_id(self, agent_id: str | None = None) -> str:
        """Get effective agent ID.

        Args:
            agent_id: Explicit agent ID or None.

        Returns:
            Agent ID to use for operation.
        """
        if agent_id:
            return agent_id
        if self._client_context and self._client_context.default_agent_id:
            return self._client_context.default_agent_id
        return self.config.default_agent_id

    def _get_allowed_scopes(self) -> list[MCPScope]:
        """Get allowed scopes for current client.

        Returns:
            List of scopes the client can access.
        """
        if self._client_context:
            return self._client_context.allowed_scopes
        return self.config.allowed_scopes

    def _check_write_permission(self, scope: str) -> tuple[bool, str | None]:
        """Check if client has permission to write to scope.

        Args:
            scope: Target scope for write operation.

        Returns:
            Tuple of (allowed, error_message).
        """
        try:
            target_scope = MCPScope(scope.lower())
        except ValueError:
            return False, f"Invalid scope: {scope}"

        allowed_scopes = self._get_allowed_scopes()
        if can_write_scope(allowed_scopes, target_scope):
            return True, None

        # Log the violation
        client_id = (
            self._client_context.client_id
            if self._client_context
            else self.config.default_client_id
        )
        logger.warning(
            f"Scope violation: client={client_id} attempted write to "
            f"scope={scope}, allowed_scopes={[s.value for s in allowed_scopes]}"
        )
        return False, (
            f"Permission denied: cannot write to scope '{scope}'. "
            f"Allowed scopes: {[s.value for s in allowed_scopes]}"
        )

    def _check_read_permission(self, scope: str) -> tuple[bool, str | None]:
        """Check if client has permission to read from scope.

        Args:
            scope: Target scope for read operation.

        Returns:
            Tuple of (allowed, error_message).
        """
        try:
            target_scope = MCPScope(scope.lower())
        except ValueError:
            return False, f"Invalid scope: {scope}"

        allowed_scopes = self._get_allowed_scopes()
        if can_read_scope(allowed_scopes, target_scope):
            return True, None

        # Log the violation
        client_id = (
            self._client_context.client_id
            if self._client_context
            else self.config.default_client_id
        )
        logger.warning(
            f"Scope violation: client={client_id} attempted read from "
            f"scope={scope}, allowed_scopes={[s.value for s in allowed_scopes]}"
        )
        return False, (
            f"Permission denied: cannot read from scope '{scope}'. "
            f"Allowed scopes: {[s.value for s in allowed_scopes]}"
        )

    def _get_search_scopes(self, requested_scope: str | None) -> list[str]:
        """Get scopes to search based on client permissions.

        Args:
            requested_scope: Specific scope requested (optional).

        Returns:
            List of draagon-ai scope strings to search.
        """
        allowed_scopes = self._get_allowed_scopes()
        readable_scopes = get_readable_scopes(allowed_scopes)

        if requested_scope:
            # Validate the requested scope
            try:
                target = MCPScope(requested_scope.lower())
                if target in readable_scopes:
                    return [map_scope_to_draagon(requested_scope)]
                else:
                    # Not allowed to read this scope - return only allowed ones
                    logger.warning(
                        f"Client requested scope {requested_scope} but only "
                        f"allowed to read {[s.value for s in readable_scopes]}"
                    )
            except ValueError:
                pass  # Invalid scope, fall through to default

        # Return all readable scopes mapped to draagon-ai format
        return [map_scope_to_draagon(s.value) for s in readable_scopes]

    def set_client_context(self, client_config: ClientConfig) -> None:
        """Set the client context for scope enforcement.

        Args:
            client_config: Client configuration with allowed scopes.
        """
        self._client_context = client_config
        logger.info(
            f"Client context set: id={client_config.client_id}, "
            f"scopes={[s.value for s in client_config.allowed_scopes]}"
        )

    def authenticate(self, api_key: str | None) -> AuthResult:
        """Authenticate a request and set client context if successful.

        Args:
            api_key: API key from request headers.

        Returns:
            AuthResult with authentication status.
        """
        result = self.authenticator.authenticate(api_key)

        if result.authenticated and result.client_config:
            self.set_client_context(result.client_config)

        return result

    def get_auth_audit_log(self, limit: int = 100) -> list[dict]:
        """Get authentication audit log.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of audit log entries.
        """
        return self.authenticator.get_audit_log(limit)

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        # =====================================================================
        # memory.store
        # =====================================================================
        @self.mcp.tool(name="memory_store")
        async def memory_store(
            content: str,
            memory_type: str = "fact",
            scope: str = "private",
            entities: list[str] | None = None,
            importance: float = 0.5,
            user_id: str | None = None,
            agent_id: str | None = None,
        ) -> dict[str, Any]:
            """Store a memory in the shared knowledge base.

            Use this to remember important information about the user,
            their projects, preferences, or learned facts.

            Args:
                content: The content to remember.
                memory_type: Type of memory (fact, skill, insight, preference).
                scope: Visibility scope (private, shared, system).
                entities: Related entities/keywords (optional, auto-extracted if not provided).
                importance: Importance score 0.0-1.0 (default 0.5).
                user_id: User ID (optional, uses default).
                agent_id: Agent ID (optional, uses default).

            Returns:
                Dict with memory_id and success status.
            """
            try:
                # Check write permission for scope
                allowed, error = self._check_write_permission(scope)
                if not allowed:
                    return {
                        "success": False,
                        "error": error,
                    }

                memory = await self._get_memory()

                # Map types
                draagon_type = map_type_to_draagon(memory_type)
                draagon_scope = map_scope_to_draagon(scope)

                # Get effective IDs
                effective_user = self._get_user_id(user_id)
                effective_agent = self._get_agent_id(agent_id)

                # Store memory
                result = await memory.store(
                    content=content,
                    memory_type=draagon_type,
                    scope=draagon_scope,
                    user_id=effective_user,
                    agent_id=effective_agent,
                    importance=importance,
                    entities=entities or [],
                    metadata={
                        "source": "mcp",
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )

                logger.info(
                    f"Stored memory: type={memory_type}, scope={scope}, "
                    f"user={effective_user}"
                )

                return {
                    "success": True,
                    "memory_id": result.id if result else None,
                    "message": f"Memory stored successfully as {memory_type}",
                }

            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        # =====================================================================
        # memory.search
        # =====================================================================
        @self.mcp.tool(name="memory_search")
        async def memory_search(
            query: str,
            limit: int = 5,
            memory_types: list[str] | None = None,
            scope: str | None = None,
            user_id: str | None = None,
            agent_id: str | None = None,
        ) -> dict[str, Any]:
            """Search the shared knowledge base for relevant memories.

            Use this to recall information about the user, their projects,
            preferences, or any previously stored facts.

            Args:
                query: Search query (semantic search).
                limit: Maximum results to return (default 5, max 50).
                memory_types: Filter by memory types (optional).
                scope: Filter by scope (optional).
                user_id: User ID (optional, uses default).
                agent_id: Agent ID (optional, uses default).

            Returns:
                Dict with list of matching memories and their scores.
            """
            try:
                # Check read permission for scope if specified
                if scope:
                    allowed, error = self._check_read_permission(scope)
                    if not allowed:
                        return {
                            "success": False,
                            "error": error,
                            "results": [],
                        }

                memory = await self._get_memory()

                # Clamp limit
                limit = min(limit, self.config.search_limit_max)

                # Get effective IDs
                effective_user = self._get_user_id(user_id)
                effective_agent = self._get_agent_id(agent_id)

                # Map types if provided
                draagon_types = None
                if memory_types:
                    draagon_types = [map_type_to_draagon(t) for t in memory_types]

                # Get scopes to search (filtered by client permissions)
                search_scopes = self._get_search_scopes(scope)

                # Search
                results = await memory.search(
                    query=query,
                    limit=limit,
                    user_id=effective_user,
                    agent_id=effective_agent,
                    memory_types=draagon_types,
                    scopes=search_scopes if search_scopes else None,
                )

                # Format results
                # SearchResult has .memory (Memory object) and .score
                # Memory has .id, .content, .memory_type, .scope, .importance, etc.
                formatted_results = []
                for r in results:
                    mem = r.memory
                    formatted_results.append(
                        {
                            "id": mem.id,
                            "content": mem.content,
                            "memory_type": mem.memory_type.value
                            if hasattr(mem.memory_type, "value")
                            else str(mem.memory_type),
                            "scope": mem.scope.value
                            if hasattr(mem.scope, "value")
                            else str(mem.scope),
                            "score": r.score,
                            "importance": getattr(mem, "importance", 0.5),
                            "entities": getattr(mem, "entities", []),
                            "created_at": getattr(mem, "created_at", None),
                        }
                    )

                logger.info(
                    f"Search query='{query[:50]}...' returned {len(formatted_results)} results"
                )

                return {
                    "success": True,
                    "count": len(formatted_results),
                    "results": formatted_results,
                }

            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "results": [],
                }

        # =====================================================================
        # memory.list
        # =====================================================================
        @self.mcp.tool(name="memory_list")
        async def memory_list(
            memory_type: str | None = None,
            limit: int = 10,
            user_id: str | None = None,
            agent_id: str | None = None,
        ) -> dict[str, Any]:
            """List recent memories, optionally filtered by type.

            Use this to see what has been remembered recently.

            Args:
                memory_type: Filter by type (optional).
                limit: Maximum results (default 10, max 100).
                user_id: User ID (optional, uses default).
                agent_id: Agent ID (optional, uses default).

            Returns:
                Dict with list of recent memories.
            """
            try:
                memory = await self._get_memory()

                # Clamp limit
                limit = min(limit, self.config.list_limit_max)

                # Get effective IDs
                effective_user = self._get_user_id(user_id)
                effective_agent = self._get_agent_id(agent_id)

                # Map type if provided
                draagon_type = None
                if memory_type:
                    draagon_type = map_type_to_draagon(memory_type)

                # List memories (use search with empty query for now)
                # TODO: Add proper list method to LayeredMemoryProvider
                results = await memory.search(
                    query="",  # Empty query for listing
                    limit=limit,
                    user_id=effective_user,
                    agent_id=effective_agent,
                    memory_types=[draagon_type] if draagon_type else None,
                )

                # Format results
                formatted_results = []
                for r in results:
                    formatted_results.append(
                        {
                            "id": r.id,
                            "content": r.content[:200]
                            + ("..." if len(r.content) > 200 else ""),
                            "memory_type": r.memory_type.value
                            if hasattr(r.memory_type, "value")
                            else str(r.memory_type),
                            "importance": getattr(r, "importance", 0.5),
                            "created_at": getattr(r, "created_at", None),
                        }
                    )

                return {
                    "success": True,
                    "count": len(formatted_results),
                    "memories": formatted_results,
                }

            except Exception as e:
                logger.error(f"Failed to list memories: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "memories": [],
                }

        # =====================================================================
        # memory.get
        # =====================================================================
        @self.mcp.tool(name="memory_get")
        async def memory_get(
            memory_id: str,
            user_id: str | None = None,
            agent_id: str | None = None,
        ) -> dict[str, Any]:
            """Get a specific memory by ID.

            Args:
                memory_id: The memory ID to retrieve.
                user_id: User ID (optional, uses default).
                agent_id: Agent ID (optional, uses default).

            Returns:
                Dict with memory details or error.
            """
            try:
                memory = await self._get_memory()

                # Get effective IDs
                effective_user = self._get_user_id(user_id)
                effective_agent = self._get_agent_id(agent_id)

                # Get memory (get() only takes memory_id)
                result = await memory.get(memory_id=memory_id)

                if result is None:
                    return {
                        "success": False,
                        "error": f"Memory not found: {memory_id}",
                    }

                return {
                    "success": True,
                    "memory": {
                        "id": result.id,
                        "content": result.content,
                        "memory_type": result.memory_type.value
                        if hasattr(result.memory_type, "value")
                        else str(result.memory_type),
                        "scope": result.scope.value
                        if hasattr(result.scope, "value")
                        else str(result.scope),
                        "importance": getattr(result, "importance", 0.5),
                        "entities": getattr(result, "entities", []),
                        "created_at": getattr(result, "created_at", None),
                        "metadata": getattr(result, "metadata", {}),
                    },
                }

            except Exception as e:
                logger.error(f"Failed to get memory: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        # =====================================================================
        # memory.delete
        # =====================================================================
        @self.mcp.tool(name="memory_delete")
        async def memory_delete(
            memory_id: str,
            user_id: str | None = None,
            agent_id: str | None = None,
        ) -> dict[str, Any]:
            """Delete a memory by ID.

            Args:
                memory_id: The memory ID to delete.
                user_id: User ID (optional, uses default).
                agent_id: Agent ID (optional, uses default).

            Returns:
                Dict with success status.
            """
            try:
                memory = await self._get_memory()

                # Get effective IDs
                effective_user = self._get_user_id(user_id)
                effective_agent = self._get_agent_id(agent_id)

                # Delete memory (delete() only takes memory_id)
                success = await memory.delete(memory_id=memory_id)

                if success:
                    logger.info(f"Deleted memory: {memory_id}")
                    return {
                        "success": True,
                        "message": f"Memory {memory_id} deleted",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Memory not found or could not be deleted: {memory_id}",
                    }

            except Exception as e:
                logger.error(f"Failed to delete memory: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        # =====================================================================
        # beliefs.reconcile
        # =====================================================================
        @self.mcp.tool(name="beliefs_reconcile")
        async def beliefs_reconcile(
            observation: str,
            source: str = "mcp",
            confidence: float = 0.8,
            user_id: str | None = None,
            agent_id: str | None = None,
        ) -> dict[str, Any]:
            """Add an observation that will be reconciled with existing beliefs.

            Use this when you learn something that might conflict with
            existing knowledge. The system will detect conflicts and
            reconcile them.

            Args:
                observation: The observation to reconcile.
                source: Source of observation (e.g., 'user', 'web', 'code').
                confidence: Confidence in observation 0.0-1.0 (default 0.8).
                user_id: User ID (optional, uses default).
                agent_id: Agent ID (optional, uses default).

            Returns:
                Dict with reconciliation result.
            """
            if not BELIEFS_AVAILABLE:
                return {
                    "success": False,
                    "error": "Belief reconciliation not available. "
                    "Install draagon-ai with cognition support.",
                }

            try:
                memory = await self._get_memory()

                # Get effective IDs
                effective_user = self._get_user_id(user_id)
                effective_agent = self._get_agent_id(agent_id)

                # For now, store as a fact with metadata indicating it's an observation
                # Full belief reconciliation would require the BeliefReconciliationService
                result = await memory.store(
                    content=observation,
                    memory_type="fact",  # lowercase to match MemoryType enum
                    scope="user",  # lowercase to match MemoryScope enum
                    user_id=effective_user,
                    agent_id=effective_agent,
                    importance=confidence,
                    metadata={
                        "source": source,
                        "is_observation": True,
                        "confidence": confidence,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )

                logger.info(
                    f"Added observation: source={source}, confidence={confidence}"
                )

                return {
                    "success": True,
                    "memory_id": result.id if result else None,
                    "message": "Observation recorded for reconciliation",
                    "conflicts_detected": False,  # TODO: Implement conflict detection
                }

            except Exception as e:
                logger.error(f"Failed to reconcile belief: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server.

        Args:
            transport: Transport to use ('stdio' or 'sse').
        """
        logger.info(f"Starting Memory MCP Server: {self.config.server_name}")
        self.mcp.run(transport=transport)

    async def close(self) -> None:
        """Close the server and cleanup resources."""
        if self._memory_provider and self._memory_initialized:
            await self._memory_provider.close()
            self._memory_initialized = False


# =============================================================================
# Factory Functions
# =============================================================================


def create_memory_mcp_server(
    config: MCPConfig | None = None,
    memory_provider: Any | None = None,
) -> MemoryMCPServer:
    """Create a Memory MCP Server instance.

    Args:
        config: Server configuration (defaults to MCPConfig.from_env()).
        memory_provider: Optional pre-configured memory provider.

    Returns:
        Configured MemoryMCPServer instance.
    """
    return MemoryMCPServer(config=config, memory_provider=memory_provider)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Draagon AI Memory MCP Server")
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://192.168.168.216:6333"),
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://192.168.168.200:11434"),
        help="Ollama server URL",
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "draagon_memories"),
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Create config from args
    config = MCPConfig(
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        qdrant_collection=args.collection,
        log_level=args.log_level,
    )

    # Create and run server
    server = create_memory_mcp_server(config=config)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()

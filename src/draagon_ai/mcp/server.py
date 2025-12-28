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
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from mcp.server.fastmcp import FastMCP

from draagon_ai.mcp.config import ClientConfig, MCPConfig, MCPScope

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

# Map MCP scopes to draagon-ai MemoryScope
SCOPE_MAPPING = {
    "private": "USER",
    "shared": "CONTEXT",
    "system": "WORLD",
}

# Map MCP memory types to draagon-ai MemoryType
TYPE_MAPPING = {
    "fact": "FACT",
    "skill": "SKILL",
    "insight": "INSIGHT",
    "preference": "PREFERENCE",
    "episodic": "EPISODIC",
    "instruction": "INSTRUCTION",
}


def map_scope_to_draagon(scope: str) -> str:
    """Map MCP scope to draagon-ai MemoryScope."""
    return SCOPE_MAPPING.get(scope.lower(), "USER")


def map_type_to_draagon(memory_type: str) -> str:
    """Map MCP memory type to draagon-ai MemoryType."""
    return TYPE_MAPPING.get(memory_type.lower(), "FACT")


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
                node_collection=f"{self.config.qdrant_collection}_nodes",
                edge_collection=f"{self.config.qdrant_collection}_edges",
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
                    "memory_id": result.get("memory_id") if result else None,
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

                # Map scope if provided
                draagon_scope = None
                if scope:
                    draagon_scope = map_scope_to_draagon(scope)

                # Search
                results = await memory.search(
                    query=query,
                    limit=limit,
                    user_id=effective_user,
                    agent_id=effective_agent,
                    memory_types=draagon_types,
                    scopes=[draagon_scope] if draagon_scope else None,
                )

                # Format results
                formatted_results = []
                for r in results:
                    formatted_results.append(
                        {
                            "id": r.id,
                            "content": r.content,
                            "memory_type": r.memory_type.value
                            if hasattr(r.memory_type, "value")
                            else str(r.memory_type),
                            "scope": r.scope.value
                            if hasattr(r.scope, "value")
                            else str(r.scope),
                            "score": r.score,
                            "importance": getattr(r, "importance", 0.5),
                            "entities": getattr(r, "entities", []),
                            "created_at": getattr(r, "created_at", None),
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

                # Get memory
                result = await memory.get(
                    memory_id=memory_id,
                    user_id=effective_user,
                    agent_id=effective_agent,
                )

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

                # Delete memory
                success = await memory.delete(
                    memory_id=memory_id,
                    user_id=effective_user,
                    agent_id=effective_agent,
                )

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
                    memory_type="FACT",
                    scope="USER",
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
                    "memory_id": result.get("memory_id") if result else None,
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

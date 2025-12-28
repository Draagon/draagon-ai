"""Adapter for integrating Roxy voice assistant with draagon-ai memory system.

This module provides an adapter that allows Roxy to use draagon-ai's
LayeredMemoryProvider instead of direct Qdrant access. This enables Roxy
to leverage the 4-layer cognitive memory architecture (working, episodic,
semantic, metacognitive).

The adapter implements the same interface as Roxy's MemoryService, making
it a drop-in replacement.

Usage in Roxy:
    from draagon_ai.adapters.roxy import RoxyLayeredAdapter
    from draagon_ai.memory.providers import LayeredMemoryProvider, LayeredMemoryConfig

    # Configure the provider
    config = LayeredMemoryConfig(
        qdrant_url="http://192.168.168.216:6333",
        qdrant_nodes_collection="roxy_memory_nodes",
        qdrant_edges_collection="roxy_memory_edges",
    )
    provider = LayeredMemoryProvider(config=config, embedding_provider=embedder)
    await provider.initialize()

    # Create the adapter (drop-in replacement for MemoryService)
    memory_adapter = RoxyLayeredAdapter(provider)

    # Use it like Roxy's MemoryService
    result = await memory_adapter.store(
        content="Doug's birthday is March 15",
        user_id="doug",
        scope="private",
        memory_type="fact",
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from draagon_ai.memory.base import (
    Memory,
    MemoryScope,
    MemoryType,
    SearchResult,
)
from draagon_ai.memory.providers.layered import LayeredMemoryProvider
from draagon_ai.memory.retrieval import (
    RetrievalAugmenter,
    RetrievalConfig,
    SelfRAGResult,
    CRAGResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RoxyMemoryType(str, Enum):
    """Roxy's memory types (mirrors Roxy's MemoryType enum)."""

    FACT = "fact"
    PREFERENCE = "preference"
    EPISODIC = "episodic"
    INSTRUCTION = "instruction"
    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    INSIGHT = "insight"
    SELF_KNOWLEDGE = "self_knowledge"
    RELATIONSHIP = "relationship"


class RoxyMemoryScope(str, Enum):
    """Roxy's memory scopes (mirrors Roxy's MemoryScope enum)."""

    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"
    SYSTEM = "system"


# Mapping from Roxy types to draagon-ai types
ROXY_TYPE_MAPPING = {
    RoxyMemoryType.FACT: MemoryType.FACT,
    RoxyMemoryType.PREFERENCE: MemoryType.PREFERENCE,
    RoxyMemoryType.EPISODIC: MemoryType.EPISODIC,
    RoxyMemoryType.INSTRUCTION: MemoryType.INSTRUCTION,
    RoxyMemoryType.KNOWLEDGE: MemoryType.KNOWLEDGE,
    RoxyMemoryType.SKILL: MemoryType.SKILL,
    RoxyMemoryType.INSIGHT: MemoryType.INSIGHT,
    RoxyMemoryType.SELF_KNOWLEDGE: MemoryType.SELF_KNOWLEDGE,
    RoxyMemoryType.RELATIONSHIP: MemoryType.RELATIONSHIP,
    # String fallbacks
    "fact": MemoryType.FACT,
    "preference": MemoryType.PREFERENCE,
    "episodic": MemoryType.EPISODIC,
    "instruction": MemoryType.INSTRUCTION,
    "knowledge": MemoryType.KNOWLEDGE,
    "skill": MemoryType.SKILL,
    "insight": MemoryType.INSIGHT,
    "self_knowledge": MemoryType.SELF_KNOWLEDGE,
    "relationship": MemoryType.RELATIONSHIP,
}

DRAAGON_TYPE_MAPPING = {
    MemoryType.FACT: RoxyMemoryType.FACT,
    MemoryType.PREFERENCE: RoxyMemoryType.PREFERENCE,
    MemoryType.EPISODIC: RoxyMemoryType.EPISODIC,
    MemoryType.INSTRUCTION: RoxyMemoryType.INSTRUCTION,
    MemoryType.KNOWLEDGE: RoxyMemoryType.KNOWLEDGE,
    MemoryType.SKILL: RoxyMemoryType.SKILL,
    MemoryType.INSIGHT: RoxyMemoryType.INSIGHT,
    MemoryType.SELF_KNOWLEDGE: RoxyMemoryType.SELF_KNOWLEDGE,
    MemoryType.RELATIONSHIP: RoxyMemoryType.RELATIONSHIP,
    MemoryType.OBSERVATION: RoxyMemoryType.FACT,
    MemoryType.BELIEF: RoxyMemoryType.FACT,
}

# Mapping from Roxy scopes to draagon-ai scopes
ROXY_SCOPE_MAPPING = {
    RoxyMemoryScope.PRIVATE: MemoryScope.USER,
    RoxyMemoryScope.SHARED: MemoryScope.CONTEXT,
    RoxyMemoryScope.PUBLIC: MemoryScope.WORLD,
    RoxyMemoryScope.SYSTEM: MemoryScope.WORLD,
    # String fallbacks
    "private": MemoryScope.USER,
    "shared": MemoryScope.CONTEXT,
    "public": MemoryScope.WORLD,
    "system": MemoryScope.WORLD,
}

DRAAGON_SCOPE_MAPPING = {
    MemoryScope.WORLD: RoxyMemoryScope.SYSTEM,
    MemoryScope.CONTEXT: RoxyMemoryScope.SHARED,
    MemoryScope.AGENT: RoxyMemoryScope.PRIVATE,
    MemoryScope.USER: RoxyMemoryScope.PRIVATE,
    MemoryScope.SESSION: RoxyMemoryScope.PRIVATE,
}


class RoxyLayeredAdapter:
    """Adapter that allows Roxy to use LayeredMemoryProvider.

    This adapter presents the same interface as Roxy's MemoryService,
    but internally uses draagon-ai's LayeredMemoryProvider for storage.

    Benefits:
    - 4-layer cognitive memory (working, episodic, semantic, metacognitive)
    - Automatic memory promotion between layers
    - Scope-based access control
    - Temporal graph for relationships and provenance

    Example:
        from draagon_ai.memory.providers import LayeredMemoryProvider, LayeredMemoryConfig

        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config, embedding_provider=embedder)
        await provider.initialize()

        adapter = RoxyLayeredAdapter(provider)

        # Use exactly like Roxy's MemoryService
        result = await adapter.store(
            content="Doug's birthday is March 15",
            user_id="doug",
            scope="private",
            memory_type="fact",
        )
    """

    def __init__(
        self,
        provider: LayeredMemoryProvider,
        llm_provider: Any | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            provider: A configured and initialized LayeredMemoryProvider instance.
            llm_provider: Optional LLM provider for Self-RAG/CRAG (must have chat() method)
            retrieval_config: Configuration for retrieval augmentation
        """
        self._provider = provider
        self._llm = llm_provider
        self._retrieval_config = retrieval_config or RetrievalConfig()

        # Create retrieval augmenter if LLM is provided
        self._augmenter: RetrievalAugmenter | None = None
        if llm_provider is not None:
            self._augmenter = RetrievalAugmenter(
                llm=llm_provider,
                memory=provider,
                config=self._retrieval_config,
            )

    @property
    def provider(self) -> LayeredMemoryProvider:
        """Access the underlying LayeredMemoryProvider."""
        return self._provider

    # =========================================================================
    # Roxy MemoryService Interface
    # =========================================================================

    async def store(
        self,
        content: str,
        user_id: str,
        scope: str | RoxyMemoryScope = "private",
        memory_type: str | RoxyMemoryType = "fact",
        importance: float | None = None,
        entities: list[str] | None = None,
        household_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Store a memory using LayeredMemoryProvider.

        This method matches Roxy's MemoryService.store() signature.

        Args:
            content: The memory content
            user_id: User ID who owns this memory
            scope: Visibility scope ("private", "shared", "public", "system")
            memory_type: Type of memory ("fact", "skill", "preference", etc.)
            importance: Optional importance score (0-1)
            entities: Extracted entities
            household_id: Household/context ID for shared memories
            metadata: Additional metadata
            source: Source of the memory

        Returns:
            Dict with "success", "memory_id", and optionally "error"
        """
        try:
            # Map Roxy types to draagon-ai types
            draagon_type = ROXY_TYPE_MAPPING.get(memory_type, MemoryType.FACT)
            draagon_scope = ROXY_SCOPE_MAPPING.get(scope, MemoryScope.USER)

            # Calculate importance if not provided
            if importance is None:
                importance = 0.5  # Let LayeredMemoryProvider use its weights

            # Store via LayeredMemoryProvider
            memory = await self._provider.store(
                content=content,
                memory_type=draagon_type,
                scope=draagon_scope,
                user_id=user_id,
                context_id=household_id,
                importance=importance,
                entities=entities,
                source=source,
                metadata=metadata,
            )

            logger.debug(
                f"Stored memory via LayeredMemoryProvider: {memory.id} "
                f"(type={memory_type}, scope={scope})"
            )

            return {
                "success": True,
                "memory_id": memory.id,
            }

        except PermissionError as e:
            logger.warning(f"Permission denied storing memory: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        memory_types: list[str | RoxyMemoryType] | None = None,
        min_score: float | None = None,
        household_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories using LayeredMemoryProvider.

        This method matches Roxy's MemoryService.search() signature.

        Args:
            query: Search query text
            user_id: User ID for scoping results
            limit: Maximum number of results
            memory_types: Optional filter by memory types
            min_score: Minimum similarity score
            household_id: Household/context ID for shared memories

        Returns:
            List of memory dicts with "id", "payload", "score"
        """
        try:
            # Map Roxy memory types to draagon-ai types
            draagon_types = None
            if memory_types:
                draagon_types = [
                    ROXY_TYPE_MAPPING.get(t, MemoryType.FACT) for t in memory_types
                ]

            # Search via LayeredMemoryProvider
            results = await self._provider.search(
                query=query,
                user_id=user_id,
                context_id=household_id,
                memory_types=draagon_types,
                limit=limit,
                min_score=min_score,
            )

            # Convert to Roxy's expected format
            return [self._result_to_roxy_format(r) for r in results]

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID.

        This method matches Roxy's MemoryService.get_by_id() signature.

        Args:
            memory_id: The memory ID

        Returns:
            Memory dict or None if not found
        """
        try:
            memory = await self._provider.get(memory_id)
            if memory is None:
                return None

            return self._memory_to_roxy_format(memory)

        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None

    async def update_memory(
        self,
        memory_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a memory.

        This method matches Roxy's MemoryService.update_memory() signature.

        Args:
            memory_id: ID of memory to update
            updates: Dict of fields to update

        Returns:
            Dict with "success" and optionally "error"
        """
        try:
            memory = await self._provider.update(
                memory_id,
                content=updates.get("content"),
                importance=updates.get("importance"),
                confidence=updates.get("confidence"),
                metadata=updates,
            )

            if memory is None:
                return {"success": False, "error": "Memory not found"}

            return {"success": True}

        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return {"success": False, "error": str(e)}

    async def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory.

        This method matches Roxy's MemoryService.delete() signature.

        Args:
            memory_id: ID of memory to delete

        Returns:
            Dict with "success" and optionally "error"
        """
        try:
            success = await self._provider.delete(memory_id)
            return {"success": success}

        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Additional Roxy MemoryService Methods
    # =========================================================================

    async def reinforce_memory(
        self,
        memory_id: str,
        amount: float = 0.1,
    ) -> dict[str, Any]:
        """Reinforce a memory by boosting its importance.

        Args:
            memory_id: ID of memory to reinforce
            amount: Amount to boost importance (default 0.1)

        Returns:
            Dict with "success" and optionally "error"
        """
        try:
            memory = await self._provider.get(memory_id)
            if memory is None:
                return {"success": False, "error": "Memory not found"}

            new_importance = min(1.0, memory.importance + amount)
            await self._provider.update(memory_id, importance=new_importance)

            return {"success": True, "new_importance": new_importance}

        except Exception as e:
            logger.error(f"Error reinforcing memory {memory_id}: {e}")
            return {"success": False, "error": str(e)}

    async def search_with_self_rag(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        detect_contradictions: bool = True,
        auto_invalidate: bool = True,
    ) -> dict[str, Any]:
        """Search with Self-RAG quality assessment and contradiction detection.

        If an LLM provider was passed to the adapter, this uses full Self-RAG
        with relevance assessment, query refinement, and contradiction detection.
        Otherwise, falls back to basic search.

        Args:
            query: Search query
            user_id: User ID
            limit: Maximum results
            detect_contradictions: Whether to detect contradictions
            auto_invalidate: Whether to auto-invalidate contradicting memories

        Returns:
            Dict with "results", "assessment", "contradictions", etc.
        """
        # Use full Self-RAG if augmenter is available
        if self._augmenter is not None:
            result: SelfRAGResult = await self._augmenter.search_with_self_rag(
                query=query,
                user_id=user_id,
                limit=limit,
                detect_contradictions=detect_contradictions,
                auto_invalidate=auto_invalidate,
            )

            return {
                "results": [self._result_to_roxy_format(r) for r in result.results],
                "assessment": {
                    "score": result.assessment.score,
                    "action": result.assessment.action,
                    "reason": result.assessment.reason,
                },
                "refined_query": result.refined_query,
                "retrieval_attempts": result.retrieval_attempts,
                "timings": result.timings,
                "contradictions": [
                    {
                        "memory_id_1": c.memory_id_1,
                        "memory_id_2": c.memory_id_2,
                        "content_1": c.content_1,
                        "content_2": c.content_2,
                        "conflict_type": c.conflict_type,
                        "to_invalidate": c.to_invalidate,
                        "reason": c.reason,
                    }
                    for c in result.contradictions
                ],
                "has_contradictions": result.has_contradictions,
            }

        # Fallback to basic search
        results = await self.search(query, user_id, limit=limit)

        return {
            "results": results,
            "assessment": {
                "score": 0.5,
                "action": "use",
                "reason": "Basic search (no LLM for Self-RAG)",
            },
            "refined_query": None,
            "retrieval_attempts": 1,
            "timings": {},
            "contradictions": [],
            "has_contradictions": False,
        }

    async def search_with_crag(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Search with Corrective RAG chunk grading.

        If an LLM provider was passed to the adapter, this uses full CRAG
        with chunk grading and knowledge strip extraction.
        Otherwise, falls back to basic search.

        Args:
            query: Search query
            user_id: User ID
            limit: Maximum results

        Returns:
            Dict with "results", "knowledge_strips", "grading", etc.
        """
        # Use full CRAG if augmenter is available
        if self._augmenter is not None:
            result: CRAGResult = await self._augmenter.search_with_crag(
                query=query,
                user_id=user_id,
                limit=limit,
            )

            return {
                "results": [
                    {
                        **self._result_to_roxy_format(c.result),
                        "grade": c.grade,
                        "knowledge_strip": c.knowledge_strip,
                    }
                    for c in result.results
                ],
                "knowledge_strips": result.knowledge_strips,
                "grading": result.grading,
                "needs_web_search": result.needs_web_search,
                "timings": result.timings,
            }

        # Fallback to basic search
        results = await self.search(query, user_id, limit=limit)

        return {
            "results": results,
            "knowledge_strips": [],
            "grading": {"relevant": len(results), "irrelevant": 0, "ambiguous": 0},
            "needs_web_search": len(results) == 0,
            "timings": {},
        }

    # =========================================================================
    # Promotion and Consolidation (Exposed from LayeredMemoryProvider)
    # =========================================================================

    async def promote_all(self) -> dict[str, Any]:
        """Run a full promotion cycle across all memory layers.

        Returns:
            Promotion statistics from LayeredMemoryProvider.
        """
        stats = await self._provider.promote_all()
        return {
            "working_to_episodic": stats.working_to_episodic,
            "episodic_to_semantic": stats.episodic_to_semantic,
            "semantic_to_metacognitive": stats.semantic_to_metacognitive,
            "total_promoted": stats.total_promoted,
            "duration_ms": stats.duration_ms,
        }

    async def consolidate(self) -> dict[str, Any]:
        """Run a full consolidation cycle (decay + cleanup + promotion).

        Returns:
            Consolidation statistics.
        """
        return await self._provider.consolidate()

    def get_promotion_stats(self) -> dict[str, Any]:
        """Get current promotion service statistics."""
        return self._provider.get_promotion_stats()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _memory_to_roxy_format(self, memory: Memory) -> dict[str, Any]:
        """Convert draagon-ai Memory to Roxy's expected format."""
        roxy_type = DRAAGON_TYPE_MAPPING.get(memory.memory_type, RoxyMemoryType.FACT)
        roxy_scope = DRAAGON_SCOPE_MAPPING.get(memory.scope, RoxyMemoryScope.PRIVATE)

        return {
            "id": memory.id,
            "payload": {
                "content": memory.content,
                "memory_type": roxy_type.value,
                "scope": roxy_scope.value,
                "user_id": memory.user_id,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "entities": memory.entities,
                "source": memory.source,
                "stated_count": memory.stated_count,
                "household_id": memory.context_id,
                "created_at": (
                    memory.created_at.isoformat() if memory.created_at else None
                ),
            },
        }

    def _result_to_roxy_format(self, result: SearchResult) -> dict[str, Any]:
        """Convert draagon-ai SearchResult to Roxy's expected format."""
        roxy_memory = self._memory_to_roxy_format(result.memory)
        roxy_memory["score"] = result.score
        roxy_memory["relevance_grade"] = result.relevance_grade
        return roxy_memory


__all__ = [
    "RoxyLayeredAdapter",
    "RoxyMemoryType",
    "RoxyMemoryScope",
    "ROXY_TYPE_MAPPING",
    "ROXY_SCOPE_MAPPING",
    # Re-exported for convenience
    "RetrievalConfig",
]

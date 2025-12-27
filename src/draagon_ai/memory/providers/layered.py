"""4-Layer Memory Provider.

This module provides a MemoryProvider implementation that uses draagon-ai's
4-layer cognitive memory architecture (working, episodic, semantic, metacognitive).

The provider implements the MemoryProvider protocol while leveraging the full
power of the 4-layer system, including:
- Layer-appropriate storage based on memory type
- Cross-layer search aggregation
- Automatic memory promotion
- Entity extraction and relationship tracking

Usage:
    from draagon_ai.memory.providers import LayeredMemoryProvider

    memory = LayeredMemoryProvider()

    # Store a fact (goes to semantic layer)
    await memory.store(
        content="User's favorite color is blue",
        user_id="user123",
        memory_type="fact",
        entities=["user", "blue"],
    )

    # Search across all layers
    results = await memory.search("favorite color", user_id="user123")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from draagon_ai.memory.base import (
    Memory,
    MemoryProvider,
    MemoryScope,
    MemoryType,
    SearchResult,
)
from draagon_ai.memory.layers import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    MetacognitiveMemory,
    LayerConfig,
)
from draagon_ai.memory.temporal_graph import TemporalCognitiveGraph

if TYPE_CHECKING:
    pass


# Mapping from MemoryType to appropriate layer
LAYER_MAPPING = {
    MemoryType.FACT: "semantic",
    MemoryType.PREFERENCE: "semantic",
    MemoryType.KNOWLEDGE: "semantic",
    MemoryType.BELIEF: "semantic",
    MemoryType.RELATIONSHIP: "semantic",
    MemoryType.EPISODIC: "episodic",
    MemoryType.OBSERVATION: "episodic",
    MemoryType.SKILL: "metacognitive",
    MemoryType.INSIGHT: "metacognitive",
    MemoryType.SELF_KNOWLEDGE: "metacognitive",
    MemoryType.INSTRUCTION: "semantic",
}

# Importance weights for different memory types
IMPORTANCE_WEIGHTS = {
    MemoryType.INSTRUCTION: 0.9,
    MemoryType.FACT: 0.7,
    MemoryType.PREFERENCE: 0.6,
    MemoryType.SKILL: 0.8,
    MemoryType.INSIGHT: 0.7,
    MemoryType.EPISODIC: 0.5,
    MemoryType.OBSERVATION: 0.4,
    MemoryType.KNOWLEDGE: 0.6,
    MemoryType.BELIEF: 0.6,
    MemoryType.SELF_KNOWLEDGE: 0.8,
    MemoryType.RELATIONSHIP: 0.7,
}


@dataclass
class LayeredMemoryConfig:
    """Configuration for the layered memory provider."""

    # Working memory config
    working_memory_capacity: int = 7
    working_memory_decay_rate: float = 0.1

    # Session management
    auto_create_session: bool = True
    session_timeout_minutes: int = 30

    # Entity extraction
    enable_entity_extraction: bool = True

    # Search config
    default_search_limit: int = 10
    search_threshold: float = 0.3


class LayeredMemoryProvider(MemoryProvider):
    """MemoryProvider that uses the 4-layer cognitive memory architecture.

    Layers:
    - Working: Session-scoped, limited capacity, high attention
    - Episodic: Episode/event sequences, chronologically linked
    - Semantic: Facts, entities, relationships, knowledge
    - Metacognitive: Skills, strategies, insights, behaviors

    This provider maps incoming memories to the appropriate layer based on
    their MemoryType, and aggregates search results from all layers.
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph | None = None,
        config: LayeredMemoryConfig | None = None,
    ):
        """Initialize the layered memory provider.

        Args:
            graph: The temporal cognitive graph to use. If None, creates one.
            config: Configuration options. If None, uses defaults.
        """
        self._graph = graph or TemporalCognitiveGraph()
        self._config = config or LayeredMemoryConfig()

        # Session tracking
        self._session_id = str(uuid.uuid4())
        self._session_start = datetime.now()

        # Initialize the 4 layers
        layer_config = LayerConfig()
        self._working = WorkingMemory(
            self._graph,
            session_id=self._session_id,
            config=layer_config,
        )
        self._episodic = EpisodicMemory(self._graph, config=layer_config)
        self._semantic = SemanticMemory(self._graph, config=layer_config)
        self._metacognitive = MetacognitiveMemory(self._graph, config=layer_config)

    # --- Properties ---

    @property
    def graph(self) -> TemporalCognitiveGraph:
        """The underlying temporal cognitive graph."""
        return self._graph

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    @property
    def working(self) -> WorkingMemory:
        """Working memory layer."""
        return self._working

    @property
    def episodic(self) -> EpisodicMemory:
        """Episodic memory layer."""
        return self._episodic

    @property
    def semantic(self) -> SemanticMemory:
        """Semantic memory layer."""
        return self._semantic

    @property
    def metacognitive(self) -> MetacognitiveMemory:
        """Metacognitive memory layer."""
        return self._metacognitive

    # --- Session Management ---

    def set_session(self, session_id: str) -> None:
        """Set a new session ID and reset working memory.

        Args:
            session_id: New session identifier.
        """
        self._session_id = session_id
        self._session_start = datetime.now()

        # Recreate working memory for new session
        layer_config = LayerConfig()
        self._working = WorkingMemory(
            self._graph,
            session_id=self._session_id,
            config=layer_config,
        )

    # --- MemoryProvider Interface ---

    async def store(
        self,
        content: str,
        memory_type: MemoryType | str,
        scope: MemoryScope | str = MemoryScope.USER,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        importance: float | None = None,
        confidence: float = 1.0,
        entities: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a memory in the appropriate layer.

        Args:
            content: The memory content.
            memory_type: Type of memory.
            scope: Visibility scope.
            agent_id: Owning agent.
            user_id: Associated user.
            context_id: Associated context.
            importance: Override default importance.
            confidence: Certainty level.
            entities: Extracted entities.
            source: Source of the memory.
            metadata: Additional metadata.

        Returns:
            The stored Memory object.
        """
        # Normalize types
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        if isinstance(scope, str):
            scope = MemoryScope(scope)

        # Calculate importance
        if importance is None:
            importance = IMPORTANCE_WEIGHTS.get(memory_type, 0.5)

        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            scope=scope,
            agent_id=agent_id,
            user_id=user_id,
            context_id=context_id,
            importance=importance,
            confidence=confidence,
            entities=entities or [],
            source=source,
        )

        # Route to appropriate layer
        layer = LAYER_MAPPING.get(memory_type, "semantic")

        if layer == "working":
            await self._store_working(memory, metadata)
        elif layer == "episodic":
            await self._store_episodic(memory, metadata)
        elif layer == "metacognitive":
            await self._store_metacognitive(memory, metadata)
        else:  # semantic is default
            await self._store_semantic(memory, metadata)

        return memory

    async def _store_working(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Store in working memory."""
        await self._working.add_item(
            content=memory.content,
            attention_weight=memory.importance,
            source=memory.source or "store",
            user_id=memory.user_id,
        )

    async def _store_episodic(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Store in episodic memory."""
        # Add as event to current episode or create new one
        episode = await self._episodic.get_current_episode(
            user_id=memory.user_id,
            agent_id=memory.agent_id,
        )

        if episode is None:
            episode = await self._episodic.start_episode(
                title="Auto-created episode",
                description=memory.content[:100],
                user_id=memory.user_id,
                agent_id=memory.agent_id,
            )

        await self._episodic.add_event(
            episode_id=episode.episode_id,
            content=memory.content,
            event_type=str(memory.memory_type.value),
            importance=memory.importance,
            entities=memory.entities,
        )

    async def _store_semantic(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Store in semantic memory as a fact."""
        await self._semantic.store_fact(
            content=memory.content,
            entities=memory.entities,
            source=memory.source or "store",
            confidence=memory.confidence,
            user_id=memory.user_id,
            agent_id=memory.agent_id,
        )

    async def _store_metacognitive(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Store in metacognitive memory."""
        if memory.memory_type == MemoryType.SKILL:
            await self._metacognitive.store_skill(
                name=memory.content[:50],
                description=memory.content,
                domain=metadata.get("domain", "general") if metadata else "general",
            )
        elif memory.memory_type == MemoryType.INSIGHT:
            await self._metacognitive.store_insight(
                insight=memory.content,
                source=memory.source or "store",
                confidence=memory.confidence,
            )
        else:
            # Store as insight by default
            await self._metacognitive.store_insight(
                insight=memory.content,
                source=memory.source or "store",
                confidence=memory.confidence,
            )

    async def search(
        self,
        query: str,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        scope: MemoryScope | str | None = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[SearchResult]:
        """Search across all memory layers.

        Args:
            query: Search query.
            agent_id: Filter by agent.
            user_id: Filter by user.
            context_id: Filter by context.
            memory_type: Filter by type.
            scope: Filter by scope.
            limit: Max results.
            min_relevance: Minimum relevance score.

        Returns:
            List of SearchResult objects.
        """
        results: list[SearchResult] = []

        # Search working memory
        working_results = await self._working.search(query, limit=limit)
        for item in working_results:
            results.append(
                SearchResult(
                    memory=Memory(
                        id=item.item_id,
                        content=item.content,
                        memory_type=MemoryType.EPISODIC,
                        scope=MemoryScope.SESSION,
                        user_id=user_id,
                        importance=item.attention_weight,
                    ),
                    relevance=item.attention_weight,
                )
            )

        # Search semantic memory (for facts)
        semantic_results = await self._semantic.search_facts(
            query=query,
            user_id=user_id,
            limit=limit,
        )
        for fact in semantic_results:
            results.append(
                SearchResult(
                    memory=Memory(
                        id=fact.fact_id,
                        content=fact.content,
                        memory_type=MemoryType.FACT,
                        scope=MemoryScope.USER,
                        user_id=user_id,
                        confidence=fact.confidence,
                        entities=fact.entity_ids,
                    ),
                    relevance=fact.confidence,
                )
            )

        # Sort by relevance and limit
        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    async def get(
        self,
        memory_id: str,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID.
            agent_id: Optional agent filter.
            user_id: Optional user filter.

        Returns:
            The Memory if found, None otherwise.
        """
        # Try to find in graph
        node = await self._graph.get_node(memory_id)
        if node is None:
            return None

        return Memory(
            id=node.node_id,
            content=node.content,
            memory_type=MemoryType.FACT,  # Default
            scope=MemoryScope.USER,
            created_at=node.created_at,
        )

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a memory.

        Args:
            memory_id: The memory to update.
            content: New content.
            importance: New importance.
            metadata: New metadata.

        Returns:
            True if updated, False if not found.
        """
        node = await self._graph.get_node(memory_id)
        if node is None:
            return False

        if content is not None:
            node.content = content
        if importance is not None:
            node.importance = importance
        if metadata is not None:
            node.metadata.update(metadata)

        return True

    async def delete(
        self,
        memory_id: str,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Delete a memory.

        Args:
            memory_id: The memory to delete.
            agent_id: Optional agent filter.
            user_id: Optional user filter.

        Returns:
            True if deleted, False if not found.
        """
        # Remove from graph
        try:
            await self._graph.remove_node(memory_id)
            return True
        except Exception:
            return False


__all__ = [
    "LayeredMemoryProvider",
    "LayeredMemoryConfig",
    "LAYER_MAPPING",
    "IMPORTANCE_WEIGHTS",
]

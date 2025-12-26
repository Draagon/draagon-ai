"""Base classes for memory layers.

Each memory layer has different temporal characteristics:
- Working: Seconds to minutes, session-bound, limited capacity
- Episodic: Hours to days, experience-based, chronological
- Semantic: Days to months, factual knowledge, entity-centric
- Metacognitive: Weeks to permanent, skills and strategies

Based on research from:
- Zep/Graphiti: Bi-temporal knowledge graphs
- Mem0: Hybrid vector-graph architecture
- Cognitive science: Multi-store memory model
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar
import logging

from ..temporal_nodes import TemporalNode, NodeType, MemoryLayer
from ..temporal_graph import TemporalCognitiveGraph, GraphSearchResult

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=TemporalNode)


@dataclass
class LayerConfig:
    """Configuration for a memory layer."""

    # Capacity settings
    max_items: int | None = None  # None = unlimited

    # Decay settings
    default_ttl: timedelta | None = None
    decay_factor: float = 0.95  # Applied per decay cycle
    decay_interval: timedelta = timedelta(hours=1)

    # Promotion settings
    importance_threshold: float = 0.7  # Promote if importance >= threshold
    access_threshold: int = 5  # Promote if access_count >= threshold
    auto_promote: bool = True

    # Layer-specific
    node_types: list[NodeType] = field(default_factory=list)


class MemoryLayerBase(ABC, Generic[T]):
    """Abstract base class for memory layers.

    Each layer wraps the TemporalCognitiveGraph with layer-specific behavior:
    - Filtering by node types appropriate to the layer
    - Capacity management (for Working layer)
    - Promotion logic to parent layers
    - Layer-specific decay and cleanup
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        config: LayerConfig,
        layer: MemoryLayer,
    ):
        """Initialize the memory layer.

        Args:
            graph: The underlying temporal cognitive graph
            config: Layer configuration
            layer: The memory layer type
        """
        self._graph = graph
        self._config = config
        self._layer = layer
        self._last_decay = datetime.now()

    @property
    def layer(self) -> MemoryLayer:
        """Get the memory layer type."""
        return self._layer

    @property
    def config(self) -> LayerConfig:
        """Get the layer configuration."""
        return self._config

    @abstractmethod
    async def add(self, content: str, **kwargs: Any) -> T:
        """Add an item to this layer.

        Args:
            content: The content to store
            **kwargs: Additional properties

        Returns:
            The created node
        """
        ...

    @abstractmethod
    async def get(self, node_id: str) -> T | None:
        """Get an item by ID.

        Args:
            node_id: The node identifier

        Returns:
            The node or None if not found
        """
        ...

    async def search(
        self,
        query: str,
        scope_ids: list[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[GraphSearchResult]:
        """Search for items in this layer.

        Args:
            query: Search query
            scope_ids: Filter by scopes
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        return await self._graph.search(
            query=query,
            scope_ids=scope_ids,
            layers=[self._layer],
            node_types=self._config.node_types or None,
            limit=limit,
            min_score=min_score,
        )

    async def delete(self, node_id: str) -> bool:
        """Delete an item from this layer.

        Args:
            node_id: The node identifier

        Returns:
            True if deleted
        """
        node = await self._graph.get_node(node_id)
        if not node or node.layer != self._layer:
            return False
        return await self._graph.delete_node(node_id)

    async def count(self, scope_id: str | None = None) -> int:
        """Count items in this layer.

        Args:
            scope_id: Optional scope filter

        Returns:
            Number of items
        """
        stats = self._graph.stats()
        layer_count = stats.get("nodes_by_layer", {}).get(self._layer.value, 0)
        return layer_count

    async def apply_decay(self) -> int:
        """Apply temporal decay to items in this layer.

        Returns:
            Number of items decayed
        """
        now = datetime.now()
        if now - self._last_decay < self._config.decay_interval:
            return 0

        self._last_decay = now
        decayed = 0

        # Get all nodes in this layer
        for node_id, node in list(self._graph._nodes.items()):
            if node.layer != self._layer:
                continue

            # Apply decay
            node.decay(self._config.decay_factor)
            decayed += 1

            # Check TTL expiration
            if self._config.default_ttl:
                age = now - node.ingestion_time
                if age > self._config.default_ttl:
                    await self._graph.delete_node(node_id)

        logger.debug(f"Applied decay to {decayed} nodes in {self._layer.value} layer")
        return decayed

    async def get_promotion_candidates(self) -> list[T]:
        """Get items ready for promotion to the next layer.

        Returns:
            List of nodes that meet promotion criteria
        """
        if not self._config.auto_promote:
            return []

        candidates = []
        for node_id, node in self._graph._nodes.items():
            if node.layer != self._layer:
                continue

            # Check promotion criteria
            meets_importance = node.importance >= self._config.importance_threshold
            meets_access = node.access_count >= self._config.access_threshold

            if meets_importance or meets_access:
                candidates.append(node)

        return candidates

    async def cleanup_expired(self) -> int:
        """Remove expired items from this layer.

        Returns:
            Number of items removed
        """
        if self._config.default_ttl is None:
            return 0

        now = datetime.now()
        removed = 0

        for node_id, node in list(self._graph._nodes.items()):
            if node.layer != self._layer:
                continue

            age = now - node.ingestion_time
            if age > self._config.default_ttl and node.importance < 0.3:
                await self._graph.delete_node(node_id)
                removed += 1

        return removed


@dataclass
class PromotionResult:
    """Result of promoting items between layers."""

    source_layer: MemoryLayer
    target_layer: MemoryLayer
    promoted_count: int
    promoted_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)

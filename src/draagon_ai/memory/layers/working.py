"""Working Memory Layer - Active session context.

Working memory holds the current conversational context with:
- Limited capacity (Miller's Law: 7±2 items)
- Attention weighting (focus on what matters most)
- Auto-decay (items expire if not accessed)
- Promotion to Episodic layer on significance

Based on research from:
- Cognitive psychology: Working memory models (Baddeley, Miller)
- ACE Framework: Active context management
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import logging
import heapq

from ..temporal_nodes import TemporalNode, NodeType, MemoryLayer
from ..temporal_graph import TemporalCognitiveGraph
from .base import MemoryLayerBase, LayerConfig

logger = logging.getLogger(__name__)


# Default working memory capacity (Miller's Law: 7±2)
DEFAULT_CAPACITY = 7

# Default TTL for working memory items
DEFAULT_TTL = timedelta(minutes=5)


@dataclass
class WorkingMemoryItem(TemporalNode):
    """Extended node for working memory with attention tracking.

    Working memory items have additional properties:
    - attention_weight: How important this is right now (0-1)
    - activation_level: How recently accessed (decays over time)
    - session_id: Which session this belongs to
    """

    attention_weight: float = 0.5
    activation_level: float = 1.0
    session_id: str = ""
    source: str = ""  # Where this context came from

    @property
    def effective_priority(self) -> float:
        """Calculate effective priority for capacity management.

        Combines importance, attention, and activation.
        Higher = more likely to be retained.
        """
        return (
            self.importance * 0.4 +
            self.attention_weight * 0.35 +
            self.activation_level * 0.25
        )

    def decay_activation(self, factor: float = 0.9) -> None:
        """Decay activation level over time.

        Args:
            factor: Decay multiplier (0.9 = 10% decay)
        """
        self.activation_level = max(0.0, self.activation_level * factor)

    def boost_attention(self, boost: float = 0.2) -> None:
        """Boost attention weight when accessed.

        Args:
            boost: Amount to increase attention
        """
        self.attention_weight = min(1.0, self.attention_weight + boost)
        self.activation_level = 1.0  # Reset activation on access


class WorkingMemory(MemoryLayerBase[WorkingMemoryItem]):
    """Working Memory Layer - Active session context.

    Key features:
    - Limited capacity with LRU-like eviction based on priority
    - Attention weighting to focus on important context
    - Auto-decay of activation levels
    - Session isolation (each session has its own working memory)
    - Automatic promotion to Episodic layer

    Example:
        working = WorkingMemory(graph, session_id="session_123")

        # Add context
        item = await working.add(
            content="User asked about Paris vacation",
            attention_weight=0.8,
            source="conversation",
        )

        # Get current context
        context = await working.get_active_context()

        # Focus attention on specific item
        await working.focus(item.node_id)

        # Get items ready for promotion
        promotable = await working.get_promotion_candidates()
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        session_id: str,
        capacity: int = DEFAULT_CAPACITY,
        ttl: timedelta = DEFAULT_TTL,
    ):
        """Initialize working memory.

        Args:
            graph: The underlying temporal cognitive graph
            session_id: Session identifier for isolation
            capacity: Maximum items (default 7, Miller's Law)
            ttl: Time-to-live for items (default 5 minutes)
        """
        config = LayerConfig(
            max_items=capacity,
            default_ttl=ttl,
            decay_factor=0.95,
            decay_interval=timedelta(seconds=30),  # Frequent decay for working memory
            importance_threshold=0.8,  # High bar for promotion
            access_threshold=3,  # Accessed 3+ times = important
            auto_promote=True,
            node_types=[NodeType.CONTEXT, NodeType.GOAL],
        )
        super().__init__(graph, config, MemoryLayer.WORKING)

        self._session_id = session_id
        self._attention_weights: dict[str, float] = {}  # node_id -> attention
        self._last_access: dict[str, datetime] = {}  # node_id -> last access time

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def capacity(self) -> int:
        """Get the capacity limit."""
        return self._config.max_items or DEFAULT_CAPACITY

    async def add(
        self,
        content: str,
        *,
        node_type: NodeType = NodeType.CONTEXT,
        scope_id: str = "session:default",
        attention_weight: float = 0.5,
        source: str = "unknown",
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        """Add an item to working memory.

        If capacity is exceeded, lowest priority items are evicted
        (and potentially promoted to episodic memory).

        Args:
            content: The content to store
            node_type: Type of node (CONTEXT or GOAL)
            scope_id: Hierarchical scope
            attention_weight: Initial attention (0-1)
            source: Where this context came from
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The created WorkingMemoryItem
        """
        # Check capacity and evict if needed
        await self._ensure_capacity()

        # Create the node in the graph
        node = await self._graph.add_node(
            content=content,
            node_type=node_type,
            scope_id=scope_id,
            entities=entities,
            metadata={
                **(metadata or {}),
                "session_id": self._session_id,
                "attention_weight": attention_weight,
                "activation_level": 1.0,
                "source": source,
            },
        )

        # Wrap in WorkingMemoryItem
        item = WorkingMemoryItem(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            attention_weight=attention_weight,
            activation_level=1.0,
            session_id=self._session_id,
            source=source,
        )

        # Track attention and access
        self._attention_weights[node.node_id] = attention_weight
        self._last_access[node.node_id] = datetime.now()

        logger.debug(
            f"Added to working memory: {node.node_id[:8]}... "
            f"(attention={attention_weight:.2f})"
        )

        return item

    async def get(self, node_id: str) -> WorkingMemoryItem | None:
        """Get an item and boost its activation.

        Args:
            node_id: The node identifier

        Returns:
            The item or None if not found
        """
        node = await self._graph.get_node(node_id)
        if not node or node.layer != MemoryLayer.WORKING:
            return None

        # Check session
        if node.metadata.get("session_id") != self._session_id:
            return None

        # Boost activation on access
        attention = self._attention_weights.get(node_id, 0.5)
        self._last_access[node_id] = datetime.now()

        # Reinforce in graph
        node.reinforce()

        return WorkingMemoryItem(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            attention_weight=attention,
            activation_level=node.metadata.get("activation_level", 1.0),
            session_id=self._session_id,
            source=node.metadata.get("source", ""),
        )

    async def focus(self, node_id: str, boost: float = 0.3) -> bool:
        """Focus attention on a specific item.

        Args:
            node_id: The node to focus on
            boost: Attention boost amount

        Returns:
            True if item exists and was focused
        """
        node = await self._graph.get_node(node_id)
        if not node or node.layer != MemoryLayer.WORKING:
            return False

        # Boost attention
        current = self._attention_weights.get(node_id, 0.5)
        self._attention_weights[node_id] = min(1.0, current + boost)

        # Reset activation
        node.metadata["activation_level"] = 1.0
        self._last_access[node_id] = datetime.now()

        # Reinforce
        node.reinforce()

        logger.debug(f"Focused attention on {node_id[:8]}...: {current:.2f} -> {self._attention_weights[node_id]:.2f}")
        return True

    async def get_active_context(
        self,
        min_attention: float = 0.3,
        limit: int | None = None,
    ) -> list[WorkingMemoryItem]:
        """Get current active context items.

        Args:
            min_attention: Minimum attention threshold
            limit: Maximum items to return

        Returns:
            List of active items sorted by effective priority
        """
        items = []

        for node_id, node in self._graph._nodes.items():
            if node.layer != MemoryLayer.WORKING:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            attention = self._attention_weights.get(node_id, 0.5)
            if attention < min_attention:
                continue

            item = WorkingMemoryItem(
                node_id=node.node_id,
                content=node.content,
                node_type=node.node_type,
                scope_id=node.scope_id,
                embedding=node.embedding,
                event_time=node.event_time,
                ingestion_time=node.ingestion_time,
                valid_from=node.valid_from,
                valid_until=node.valid_until,
                confidence=node.confidence,
                importance=node.importance,
                stated_count=node.stated_count,
                access_count=node.access_count,
                entities=node.entities,
                metadata=node.metadata,
                created_at=node.created_at,
                updated_at=node.updated_at,
                attention_weight=attention,
                activation_level=node.metadata.get("activation_level", 1.0),
                session_id=self._session_id,
                source=node.metadata.get("source", ""),
            )
            items.append(item)

        # Sort by effective priority (highest first)
        items.sort(key=lambda x: x.effective_priority, reverse=True)

        if limit:
            items = items[:limit]

        return items

    async def get_goals(self) -> list[WorkingMemoryItem]:
        """Get current active goals.

        Returns:
            List of GOAL type items
        """
        goals = []
        for node_id, node in self._graph._nodes.items():
            if node.layer != MemoryLayer.WORKING:
                continue
            if node.node_type != NodeType.GOAL:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            attention = self._attention_weights.get(node_id, 0.5)
            item = WorkingMemoryItem(
                node_id=node.node_id,
                content=node.content,
                node_type=node.node_type,
                scope_id=node.scope_id,
                embedding=node.embedding,
                event_time=node.event_time,
                ingestion_time=node.ingestion_time,
                valid_from=node.valid_from,
                valid_until=node.valid_until,
                confidence=node.confidence,
                importance=node.importance,
                stated_count=node.stated_count,
                access_count=node.access_count,
                entities=node.entities,
                metadata=node.metadata,
                created_at=node.created_at,
                updated_at=node.updated_at,
                attention_weight=attention,
                activation_level=node.metadata.get("activation_level", 1.0),
                session_id=self._session_id,
                source=node.metadata.get("source", ""),
            )
            goals.append(item)

        return goals

    async def set_goal(
        self,
        goal: str,
        attention_weight: float = 0.9,
        **kwargs: Any,
    ) -> WorkingMemoryItem:
        """Set a new goal in working memory.

        Goals have higher default attention.

        Args:
            goal: The goal description
            attention_weight: Attention weight (default 0.9)
            **kwargs: Additional properties

        Returns:
            The created goal item
        """
        return await self.add(
            content=goal,
            node_type=NodeType.GOAL,
            attention_weight=attention_weight,
            source="goal",
            **kwargs,
        )

    async def complete_goal(self, node_id: str) -> bool:
        """Mark a goal as completed.

        Args:
            node_id: The goal node ID

        Returns:
            True if goal was completed
        """
        node = await self._graph.get_node(node_id)
        if not node or node.node_type != NodeType.GOAL:
            return False

        node.metadata["completed"] = True
        node.metadata["completed_at"] = datetime.now().isoformat()
        node.valid_until = datetime.now()

        return True

    async def _ensure_capacity(self) -> None:
        """Ensure we don't exceed capacity by evicting lowest priority items."""
        session_items = []

        for node_id, node in self._graph._nodes.items():
            if node.layer != MemoryLayer.WORKING:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            attention = self._attention_weights.get(node_id, 0.5)
            activation = node.metadata.get("activation_level", 1.0)
            priority = (
                node.importance * 0.4 +
                attention * 0.35 +
                activation * 0.25
            )
            session_items.append((priority, node_id, node))

        # If under capacity, nothing to do
        if len(session_items) < self.capacity:
            return

        # Sort by priority (lowest first)
        session_items.sort(key=lambda x: x[0])

        # Evict lowest priority items
        items_to_evict = len(session_items) - self.capacity + 1  # +1 for the new item

        for i in range(items_to_evict):
            _, node_id, node = session_items[i]

            # Check if should promote to episodic before deleting
            if node.importance >= 0.6 or node.access_count >= 2:
                logger.debug(f"Promoting {node_id[:8]}... to episodic before eviction")
                # Note: Actual promotion handled by MemoryPromotion service
                node.metadata["promoted_from_working"] = True

            await self._graph.delete_node(node_id)
            self._attention_weights.pop(node_id, None)
            self._last_access.pop(node_id, None)

            logger.debug(f"Evicted from working memory: {node_id[:8]}... (priority={session_items[i][0]:.2f})")

    async def apply_decay(self) -> int:
        """Apply activation decay to all items in this session.

        Returns:
            Number of items decayed
        """
        now = datetime.now()
        if now - self._last_decay < self._config.decay_interval:
            return 0

        self._last_decay = now
        decayed = 0

        for node_id, node in list(self._graph._nodes.items()):
            if node.layer != MemoryLayer.WORKING:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            # Decay activation based on time since last access
            last_access = self._last_access.get(node_id, node.ingestion_time)
            time_since_access = (now - last_access).total_seconds()

            # Faster decay for items not accessed recently
            decay_factor = 0.9 if time_since_access < 60 else 0.8

            current_activation = node.metadata.get("activation_level", 1.0)
            new_activation = max(0.0, current_activation * decay_factor)
            node.metadata["activation_level"] = new_activation

            decayed += 1

            # Auto-expire if activation is too low and TTL exceeded
            if new_activation < 0.1 and self._config.default_ttl:
                age = now - node.ingestion_time
                if age > self._config.default_ttl:
                    await self._graph.delete_node(node_id)
                    self._attention_weights.pop(node_id, None)
                    self._last_access.pop(node_id, None)
                    logger.debug(f"Expired from working memory: {node_id[:8]}...")

        return decayed

    async def get_promotion_candidates(self) -> list[WorkingMemoryItem]:
        """Get items ready for promotion to episodic layer.

        Returns:
            List of WorkingMemoryItem objects that meet promotion criteria
        """
        if not self._config.auto_promote:
            return []

        candidates = []
        for node_id, node in self._graph._nodes.items():
            if node.layer != self._layer:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            # Check promotion criteria
            meets_importance = node.importance >= self._config.importance_threshold
            meets_access = node.access_count >= self._config.access_threshold

            if meets_importance or meets_access:
                attention = self._attention_weights.get(node_id, 0.5)
                item = WorkingMemoryItem(
                    node_id=node.node_id,
                    content=node.content,
                    node_type=node.node_type,
                    scope_id=node.scope_id,
                    embedding=node.embedding,
                    event_time=node.event_time,
                    ingestion_time=node.ingestion_time,
                    valid_from=node.valid_from,
                    valid_until=node.valid_until,
                    confidence=node.confidence,
                    importance=node.importance,
                    stated_count=node.stated_count,
                    access_count=node.access_count,
                    entities=node.entities,
                    metadata=node.metadata,
                    created_at=node.created_at,
                    updated_at=node.updated_at,
                    attention_weight=attention,
                    activation_level=node.metadata.get("activation_level", 1.0),
                    session_id=self._session_id,
                    source=node.metadata.get("source", ""),
                )
                candidates.append(item)

        return candidates

    async def clear_session(self) -> int:
        """Clear all items for this session.

        Returns:
            Number of items cleared
        """
        cleared = 0

        for node_id, node in list(self._graph._nodes.items()):
            if node.layer != MemoryLayer.WORKING:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            await self._graph.delete_node(node_id)
            cleared += 1

        self._attention_weights.clear()
        self._last_access.clear()

        logger.info(f"Cleared {cleared} items from working memory session {self._session_id}")
        return cleared

    def stats(self) -> dict[str, Any]:
        """Get working memory statistics.

        Returns:
            Dictionary with stats
        """
        session_items = []
        total_attention = 0.0
        total_activation = 0.0

        for node_id, node in self._graph._nodes.items():
            if node.layer != MemoryLayer.WORKING:
                continue
            if node.metadata.get("session_id") != self._session_id:
                continue

            session_items.append(node)
            total_attention += self._attention_weights.get(node_id, 0.5)
            total_activation += node.metadata.get("activation_level", 1.0)

        count = len(session_items)
        return {
            "session_id": self._session_id,
            "item_count": count,
            "capacity": self.capacity,
            "capacity_used": count / self.capacity if self.capacity else 0,
            "avg_attention": total_attention / count if count else 0,
            "avg_activation": total_activation / count if count else 0,
            "goal_count": sum(1 for n in session_items if n.node_type == NodeType.GOAL),
            "context_count": sum(1 for n in session_items if n.node_type == NodeType.CONTEXT),
        }

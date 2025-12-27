"""Episodic Memory Layer - Raw experiences with temporal context.

Episodic memory stores:
- Conversation episodes (summaries of interactions)
- Events (discrete occurrences within episodes)
- Chronological ordering (temporal chains)
- Entity extraction for semantic linking

Based on research from:
- Zep/Graphiti: Episode → Semantic Entity flow
- Cognitive psychology: Autobiographical memory
- ACE Framework: Experience capture patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import logging

from ..temporal_nodes import TemporalNode, TemporalEdge, NodeType, EdgeType, MemoryLayer
from ..temporal_graph import TemporalCognitiveGraph, GraphSearchResult
from .base import MemoryLayerBase, LayerConfig

logger = logging.getLogger(__name__)


# Default TTL for episodic memories
DEFAULT_TTL = timedelta(days=7)


@dataclass
class Episode(TemporalNode):
    """An episode representing a conversation or interaction.

    Episodes are the primary unit of episodic memory, capturing:
    - Summary of what happened
    - Participants involved
    - Entities mentioned (for semantic linking)
    - Emotional valence (positive/negative experience)
    - Duration and temporal context
    """

    episode_type: str = "conversation"  # conversation, task, observation
    summary: str = ""
    participants: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)

    # Temporal linking
    duration_seconds: int = 0
    preceding_episode_id: str | None = None
    following_episode_id: str | None = None

    # Event tracking
    event_count: int = 0

    @property
    def closed(self) -> bool:
        """Check if this episode is closed.

        Returns:
            True if the episode is closed (not open)
        """
        return not self.metadata.get("is_open", True)


@dataclass
class Event(TemporalNode):
    """A discrete event within an episode.

    Events are more granular than episodes:
    - Specific user utterance
    - Tool call result
    - System action
    """

    event_type: str = "utterance"  # utterance, action, observation, tool_call
    episode_id: str = ""
    actor: str = ""  # Who performed the action
    sequence_number: int = 0  # Order within episode


class EpisodicMemory(MemoryLayerBase[Episode]):
    """Episodic Memory Layer - Experience storage and linking.

    Key features:
    - Episode creation and summarization
    - Chronological linking (preceding/following)
    - Event tracking within episodes
    - Entity extraction for semantic layer
    - Emotional valence tracking

    Example:
        episodic = EpisodicMemory(graph)

        # Start a new episode
        episode = await episodic.start_episode(
            episode_type="conversation",
            participants=["user", "assistant"],
        )

        # Add events to the episode
        await episodic.add_event(
            episode_id=episode.node_id,
            content="User asked about vacation plans",
            event_type="utterance",
            actor="doug",
        )

        # Close and summarize the episode
        await episodic.close_episode(episode.node_id)

        # Search episodes
        results = await episodic.search("vacation planning")
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        ttl: timedelta = DEFAULT_TTL,
    ):
        """Initialize episodic memory.

        Args:
            graph: The underlying temporal cognitive graph
            ttl: Time-to-live for episodes (default 7 days)
        """
        config = LayerConfig(
            max_items=None,  # Unlimited episodes
            default_ttl=ttl,
            decay_factor=0.95,
            decay_interval=timedelta(hours=6),
            importance_threshold=0.7,  # Promote important episodes to semantic
            access_threshold=3,
            auto_promote=True,
            node_types=[NodeType.EPISODE, NodeType.EVENT],
        )
        super().__init__(graph, config, MemoryLayer.EPISODIC)

        # Track open episodes
        self._open_episodes: dict[str, datetime] = {}  # episode_id -> start_time
        self._last_episode_id: str | None = None  # For chronological linking

    async def add(
        self,
        content: str,
        *,
        episode_type: str = "conversation",
        scope_id: str = "user:default",
        participants: list[str] | None = None,
        entities: list[str] | None = None,
        emotional_valence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Add a new episode to episodic memory.

        This is the simplified add method. For full episode lifecycle,
        use start_episode() -> add_event() -> close_episode().

        Args:
            content: Episode summary
            episode_type: Type of episode
            scope_id: Hierarchical scope
            participants: Who was involved
            entities: Extracted entities
            emotional_valence: Emotional tone (-1 to 1)
            metadata: Additional metadata

        Returns:
            The created Episode
        """
        return await self.start_episode(
            content=content,
            episode_type=episode_type,
            scope_id=scope_id,
            participants=participants,
            entities=entities,
            emotional_valence=emotional_valence,
            auto_close=True,
            metadata=metadata,
        )

    async def start_episode(
        self,
        content: str = "",
        *,
        episode_type: str = "conversation",
        scope_id: str = "user:default",
        participants: list[str] | None = None,
        entities: list[str] | None = None,
        emotional_valence: float = 0.0,
        auto_close: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Start a new episode.

        Open episodes can have events added to them until closed.

        Args:
            content: Initial summary (can be updated on close)
            episode_type: Type of episode
            scope_id: Hierarchical scope
            participants: Who is involved
            entities: Extracted entities
            emotional_valence: Initial emotional tone
            auto_close: If True, episode is immediately closed
            metadata: Additional metadata

        Returns:
            The created Episode
        """
        # Create the node
        node = await self._graph.add_node(
            content=content or f"Started {episode_type} episode",
            node_type=NodeType.EPISODE,
            scope_id=scope_id,
            entities=entities,
            importance=0.5,  # Episodes start at medium importance
            metadata={
                **(metadata or {}),
                "episode_type": episode_type,
                "participants": participants or [],
                "emotional_valence": emotional_valence,
                "duration_seconds": 0,
                "event_count": 0,
                "is_open": not auto_close,
            },
        )

        # Create chronological link to previous episode
        if self._last_episode_id:
            await self._graph.add_edge(
                source_id=self._last_episode_id,
                target_id=node.node_id,
                edge_type=EdgeType.BEFORE,
                label="followed_by",
            )
            # Update previous episode's following reference
            prev_node = await self._graph.get_node(self._last_episode_id)
            if prev_node:
                prev_node.metadata["following_episode_id"] = node.node_id

        # Track open episode
        if not auto_close:
            self._open_episodes[node.node_id] = datetime.now()

        self._last_episode_id = node.node_id

        episode = Episode(
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
            episode_type=episode_type,
            summary=content,
            participants=participants or [],
            emotional_valence=emotional_valence,
            preceding_episode_id=self._last_episode_id if self._last_episode_id != node.node_id else None,
        )

        logger.debug(f"Started episode: {node.node_id[:8]}... type={episode_type}")
        return episode

    async def add_event(
        self,
        episode_id: str,
        content: str,
        *,
        event_type: str = "utterance",
        actor: str = "",
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Add an event to an open episode.

        Args:
            episode_id: The episode to add to
            content: Event content
            event_type: Type of event
            actor: Who performed the action
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The created Event

        Raises:
            ValueError: If episode not found or closed
        """
        episode_node = await self._graph.get_node(episode_id)
        if not episode_node:
            raise ValueError(f"Episode not found: {episode_id}")

        if not episode_node.metadata.get("is_open", False):
            raise ValueError(f"Episode is closed: {episode_id}")

        # Get current event count
        event_count = episode_node.metadata.get("event_count", 0)

        # Create event node
        node = await self._graph.add_node(
            content=content,
            node_type=NodeType.EVENT,
            scope_id=episode_node.scope_id,
            entities=entities,
            importance=0.4,  # Events start lower importance
            metadata={
                **(metadata or {}),
                "event_type": event_type,
                "episode_id": episode_id,
                "actor": actor,
                "sequence_number": event_count,
            },
        )

        # Link event to episode
        await self._graph.add_edge(
            source_id=episode_id,
            target_id=node.node_id,
            edge_type=EdgeType.HAS,
            label="contains_event",
        )

        # Update episode event count
        episode_node.metadata["event_count"] = event_count + 1

        # Update episode entities (union of all event entities)
        if entities:
            existing_entities = set(episode_node.entities)
            existing_entities.update(entities)
            episode_node.entities = list(existing_entities)

        event = Event(
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
            event_type=event_type,
            episode_id=episode_id,
            actor=actor,
            sequence_number=event_count,
        )

        logger.debug(f"Added event {event_count} to episode {episode_id[:8]}...")
        return event

    async def close_episode(
        self,
        episode_id: str,
        summary: str | None = None,
        final_valence: float | None = None,
    ) -> Episode | None:
        """Close an open episode.

        Args:
            episode_id: The episode to close
            summary: Final summary (updates content)
            final_valence: Final emotional valence

        Returns:
            The closed Episode or None if not found
        """
        episode_node = await self._graph.get_node(episode_id)
        if not episode_node:
            return None

        # Calculate duration
        start_time = self._open_episodes.pop(episode_id, episode_node.ingestion_time)
        duration = int((datetime.now() - start_time).total_seconds())

        # Update episode
        episode_node.metadata["is_open"] = False
        episode_node.metadata["duration_seconds"] = duration
        episode_node.valid_until = None  # Keep valid indefinitely

        if summary:
            episode_node.content = summary
            # Regenerate embedding for new summary
            if self._graph._embedder:
                episode_node.embedding = await self._graph._embedder.embed(summary)

        if final_valence is not None:
            episode_node.metadata["emotional_valence"] = final_valence

        episode_node.updated_at = datetime.now()

        # Calculate importance based on events, duration, entities
        event_count = episode_node.metadata.get("event_count", 0)
        entity_count = len(episode_node.entities)
        duration_minutes = duration / 60

        importance_boost = (
            min(0.2, event_count * 0.02) +  # More events = more important
            min(0.1, entity_count * 0.02) +  # More entities = more important
            min(0.1, duration_minutes * 0.01)  # Longer = more important
        )
        episode_node.importance = min(1.0, episode_node.importance + importance_boost)

        logger.debug(
            f"Closed episode {episode_id[:8]}... "
            f"duration={duration}s, events={event_count}, importance={episode_node.importance:.2f}"
        )

        return Episode(
            node_id=episode_node.node_id,
            content=episode_node.content,
            node_type=episode_node.node_type,
            scope_id=episode_node.scope_id,
            embedding=episode_node.embedding,
            event_time=episode_node.event_time,
            ingestion_time=episode_node.ingestion_time,
            valid_from=episode_node.valid_from,
            valid_until=episode_node.valid_until,
            confidence=episode_node.confidence,
            importance=episode_node.importance,
            stated_count=episode_node.stated_count,
            access_count=episode_node.access_count,
            entities=episode_node.entities,
            metadata=episode_node.metadata,
            created_at=episode_node.created_at,
            updated_at=episode_node.updated_at,
            episode_type=episode_node.metadata.get("episode_type", "conversation"),
            summary=episode_node.content,
            participants=episode_node.metadata.get("participants", []),
            emotional_valence=episode_node.metadata.get("emotional_valence", 0.0),
            duration_seconds=duration,
            event_count=event_count,
        )

    def get_current_episode(self) -> Episode | None:
        """Get the current (most recently started) open episode.

        Returns:
            The current open Episode, or None if no episodes are open
        """
        if not self._open_episodes:
            return None

        # Find most recently started open episode
        latest_id = max(self._open_episodes, key=lambda k: self._open_episodes[k])

        # Get the node synchronously (it's in-memory)
        node = self._graph._nodes.get(latest_id)
        if not node or node.node_type != NodeType.EPISODE:
            # Clean up stale entry
            del self._open_episodes[latest_id]
            return None

        return Episode(
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
            episode_type=node.metadata.get("episode_type", "conversation"),
            summary=node.content,
            participants=node.metadata.get("participants", []),
            emotional_valence=node.metadata.get("emotional_valence", 0.0),
            duration_seconds=node.metadata.get("duration_seconds", 0),
            preceding_episode_id=node.metadata.get("preceding_episode_id"),
            following_episode_id=node.metadata.get("following_episode_id"),
            event_count=node.metadata.get("event_count", 0),
        )

    async def get(self, node_id: str) -> Episode | None:
        """Get an episode by ID.

        Args:
            node_id: The episode ID

        Returns:
            The Episode or None
        """
        node = await self._graph.get_node(node_id)
        if not node or node.node_type != NodeType.EPISODE:
            return None

        return Episode(
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
            episode_type=node.metadata.get("episode_type", "conversation"),
            summary=node.content,
            participants=node.metadata.get("participants", []),
            emotional_valence=node.metadata.get("emotional_valence", 0.0),
            duration_seconds=node.metadata.get("duration_seconds", 0),
            preceding_episode_id=node.metadata.get("preceding_episode_id"),
            following_episode_id=node.metadata.get("following_episode_id"),
            event_count=node.metadata.get("event_count", 0),
        )

    async def get_events(self, episode_id: str) -> list[Event]:
        """Get all events for an episode.

        Args:
            episode_id: The episode ID

        Returns:
            List of events in sequence order
        """
        events = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.EVENT:
                continue
            if node.metadata.get("episode_id") != episode_id:
                continue

            event = Event(
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
                event_type=node.metadata.get("event_type", "utterance"),
                episode_id=episode_id,
                actor=node.metadata.get("actor", ""),
                sequence_number=node.metadata.get("sequence_number", 0),
            )
            events.append(event)

        # Sort by sequence number
        events.sort(key=lambda e: e.sequence_number)
        return events

    async def get_recent_episodes(
        self,
        scope_id: str | None = None,
        limit: int = 10,
        include_open: bool = False,
    ) -> list[Episode]:
        """Get recent episodes in chronological order.

        Args:
            scope_id: Optional scope filter
            limit: Maximum episodes
            include_open: Include open (ongoing) episodes

        Returns:
            List of episodes, most recent first
        """
        episodes = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.EPISODE:
                continue
            if node.layer != MemoryLayer.EPISODIC:
                continue
            if scope_id and node.scope_id != scope_id:
                continue
            if not include_open and node.metadata.get("is_open", False):
                continue

            episode = Episode(
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
                episode_type=node.metadata.get("episode_type", "conversation"),
                summary=node.content,
                participants=node.metadata.get("participants", []),
                emotional_valence=node.metadata.get("emotional_valence", 0.0),
                duration_seconds=node.metadata.get("duration_seconds", 0),
                event_count=node.metadata.get("event_count", 0),
            )
            episodes.append(episode)

        # Sort by ingestion time, most recent first
        episodes.sort(key=lambda e: e.ingestion_time, reverse=True)
        return episodes[:limit]

    async def get_episodes_with_entity(
        self,
        entity: str,
        scope_id: str | None = None,
        limit: int = 10,
    ) -> list[Episode]:
        """Get episodes mentioning a specific entity.

        Args:
            entity: Entity to search for
            scope_id: Optional scope filter
            limit: Maximum episodes

        Returns:
            List of episodes mentioning the entity
        """
        episodes = []
        entity_lower = entity.lower()

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.EPISODE:
                continue
            if scope_id and node.scope_id != scope_id:
                continue

            # Check if entity is mentioned
            if any(e.lower() == entity_lower for e in node.entities):
                episode = await self.get(node_id)
                if episode:
                    episodes.append(episode)

        # Sort by importance
        episodes.sort(key=lambda e: e.importance, reverse=True)
        return episodes[:limit]

    async def get_episode_chain(
        self,
        episode_id: str,
        direction: str = "both",  # "before", "after", "both"
        max_hops: int = 5,
    ) -> list[Episode]:
        """Get a chain of related episodes.

        Args:
            episode_id: Starting episode
            direction: Which direction to traverse
            max_hops: Maximum chain length

        Returns:
            List of episodes in the chain
        """
        chain = []
        visited = {episode_id}

        # Get starting episode
        start = await self.get(episode_id)
        if not start:
            return []

        chain.append(start)

        # Traverse before
        if direction in ("before", "both"):
            current = start
            for _ in range(max_hops):
                if not current.preceding_episode_id:
                    break
                if current.preceding_episode_id in visited:
                    break

                visited.add(current.preceding_episode_id)
                prev_ep = await self.get(current.preceding_episode_id)
                if prev_ep:
                    chain.insert(0, prev_ep)
                    current = prev_ep
                else:
                    break

        # Traverse after
        if direction in ("after", "both"):
            current = start
            for _ in range(max_hops):
                if not current.following_episode_id:
                    break
                if current.following_episode_id in visited:
                    break

                visited.add(current.following_episode_id)
                next_ep = await self.get(current.following_episode_id)
                if next_ep:
                    chain.append(next_ep)
                    current = next_ep
                else:
                    break

        return chain

    async def get_promotion_candidates(self) -> list[Episode]:
        """Get episodes ready for promotion to semantic layer.

        Episodes are promoted when they:
        - Have high importance
        - Have many entities (good for semantic extraction)
        - Have been accessed multiple times

        Returns:
            List of episodes ready for promotion
        """
        candidates = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.EPISODE:
                continue
            if node.layer != MemoryLayer.EPISODIC:
                continue
            if node.metadata.get("is_open", False):
                continue  # Don't promote open episodes

            # Check promotion criteria
            meets_importance = node.importance >= self._config.importance_threshold
            meets_access = node.access_count >= self._config.access_threshold
            has_entities = len(node.entities) >= 2  # Needs entities for semantic

            if (meets_importance or meets_access) and has_entities:
                episode = await self.get(node_id)
                if episode:
                    candidates.append(episode)

        return candidates

    def stats(self) -> dict[str, Any]:
        """Get episodic memory statistics.

        Returns:
            Dictionary with stats
        """
        episode_count = 0
        event_count = 0
        open_count = len(self._open_episodes)
        total_events = 0
        total_entities = 0

        for node in self._graph._nodes.values():
            if node.node_type == NodeType.EPISODE and node.layer == MemoryLayer.EPISODIC:
                episode_count += 1
                total_events += node.metadata.get("event_count", 0)
                total_entities += len(node.entities)
            elif node.node_type == NodeType.EVENT:
                event_count += 1

        return {
            "episode_count": episode_count,
            "event_count": event_count,
            "open_episodes": open_count,
            "avg_events_per_episode": total_events / episode_count if episode_count else 0,
            "avg_entities_per_episode": total_entities / episode_count if episode_count else 0,
        }

"""Unified Temporal Memory Model for Probabilistic Graph Reasoning.

This module implements a 4-layer memory system directly in Neo4j with:
- TTL/expiration encoded on nodes and edges
- Differential decay rates by content type
- Volatile working memory for ReAct swarms
- Reinforcement-based persistence

Memory Layers (encoded as properties, not separate storage):
    Layer 1: Working (5 min TTL) - Current context, recent extractions
    Layer 2: Episodic (2 weeks TTL) - Conversation patterns, event sequences
    Layer 3: Semantic (6 months TTL) - Facts, skills, preferences
    Layer 4: Metacognitive (Permanent) - Core identity, learned patterns

Decay Philosophy:
    - Phase 1 extractions (10 attributes) start volatile
    - Core concepts (entities, relationships) have slower decay
    - Peripheral info (modality, temporal markers) decays fast
    - Reinforcement from usage slows/reverses decay
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ..decomposition.graph import (
    SemanticGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    Neo4jGraphStoreSync,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Memory Layer Definitions
# =============================================================================


class MemoryLayer(Enum):
    """The 4 cognitive memory layers with TTL."""

    WORKING = "working"           # 5 minutes
    EPISODIC = "episodic"         # 2 weeks
    SEMANTIC = "semantic"         # 6 months
    METACOGNITIVE = "metacognitive"  # Permanent


# Default TTLs for each layer
LAYER_TTL: dict[MemoryLayer, timedelta | None] = {
    MemoryLayer.WORKING: timedelta(minutes=5),
    MemoryLayer.EPISODIC: timedelta(weeks=2),
    MemoryLayer.SEMANTIC: timedelta(days=180),
    MemoryLayer.METACOGNITIVE: None,  # Permanent
}


# =============================================================================
# Content Type Decay Rates
# =============================================================================


class ContentType(Enum):
    """Types of content with different decay rates.

    Based on Phase 1 extraction outputs - some should persist,
    others should fade quickly unless reinforced.
    """

    # Core concepts (slow decay, high value)
    ENTITY = "entity"           # Named entities (Doug, Whiskers)
    RELATIONSHIP = "relationship"  # Core relationships (owns, knows)
    FACT = "fact"               # Declarative facts
    SKILL = "skill"             # Procedural knowledge
    PREFERENCE = "preference"   # User preferences
    INSTRUCTION = "instruction"  # User directives

    # Contextual info (medium decay)
    EVENT = "event"             # Events and actions
    TEMPORAL = "temporal"       # Time references
    LOCATION = "location"       # Place references

    # Peripheral info (fast decay, unless reinforced)
    MODALITY = "modality"       # Certainty markers
    PRESUPPOSITION = "presupposition"  # Implied assumptions
    COMMONSENSE = "commonsense"  # Inferred common knowledge
    SENTIMENT = "sentiment"     # Emotional tone
    NEGATION = "negation"       # Polarity markers


# Decay multipliers (1.0 = normal, <1.0 = slower, >1.0 = faster)
DECAY_RATES: dict[ContentType, float] = {
    # Slow decay (core)
    ContentType.INSTRUCTION: 0.1,   # Almost never decays
    ContentType.PREFERENCE: 0.2,
    ContentType.SKILL: 0.3,
    ContentType.FACT: 0.4,
    ContentType.ENTITY: 0.5,
    ContentType.RELATIONSHIP: 0.5,

    # Medium decay (contextual)
    ContentType.EVENT: 1.0,
    ContentType.TEMPORAL: 1.5,
    ContentType.LOCATION: 0.8,

    # Fast decay (peripheral)
    ContentType.MODALITY: 3.0,
    ContentType.PRESUPPOSITION: 2.5,
    ContentType.COMMONSENSE: 2.0,
    ContentType.SENTIMENT: 2.5,
    ContentType.NEGATION: 3.0,
}


# =============================================================================
# Memory Properties for Nodes/Edges
# =============================================================================


@dataclass
class MemoryProperties:
    """Properties that encode memory layer and decay on nodes/edges.

    These are stored as node/edge properties in Neo4j:
    - memory_layer: Current layer (working/episodic/semantic/metacognitive)
    - memory_created_at: When the memory was created
    - memory_expires_at: When it should be garbage collected (None = permanent)
    - memory_last_accessed: Last time this was retrieved/used
    - memory_access_count: How many times accessed (for reinforcement)
    - memory_importance: Base importance score (0.0-1.0)
    - memory_confidence: How confident we are in this (0.0-1.0)
    - memory_content_type: Type for decay rate lookup
    - memory_reinforcement_score: Accumulated reinforcement (can promote layers)
    """

    layer: MemoryLayer = MemoryLayer.WORKING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5
    confidence: float = 1.0
    content_type: ContentType = ContentType.ENTITY
    reinforcement_score: float = 0.0

    def __post_init__(self):
        """Calculate expiration if not set."""
        if self.expires_at is None and self.layer != MemoryLayer.METACOGNITIVE:
            ttl = LAYER_TTL[self.layer]
            if ttl:
                self.expires_at = self.created_at + ttl

    @property
    def is_expired(self) -> bool:
        """Check if this memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def time_to_expiry(self) -> timedelta | None:
        """Get time until expiration."""
        if self.expires_at is None:
            return None
        return self.expires_at - datetime.now(timezone.utc)

    @property
    def decay_rate(self) -> float:
        """Get decay rate based on content type."""
        return DECAY_RATES.get(self.content_type, 1.0)

    def record_access(self) -> None:
        """Record that this memory was accessed (slows decay)."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

        # Each access extends expiration slightly
        if self.expires_at is not None:
            extension = timedelta(minutes=1) / self.decay_rate
            self.expires_at += extension

    def reinforce(self, amount: float = 0.1) -> None:
        """Reinforce this memory (can promote to higher layer)."""
        self.reinforcement_score += amount
        self.record_access()

        # Check for layer promotion
        self._check_promotion()

    def _check_promotion(self) -> None:
        """Check if reinforcement warrants layer promotion."""
        promotion_thresholds = {
            MemoryLayer.WORKING: 0.3,      # Promote to episodic
            MemoryLayer.EPISODIC: 0.6,     # Promote to semantic
            MemoryLayer.SEMANTIC: 0.9,     # Promote to metacognitive
        }

        if self.layer in promotion_thresholds:
            threshold = promotion_thresholds[self.layer]
            if self.reinforcement_score >= threshold:
                self._promote()

    def _promote(self) -> None:
        """Promote to next memory layer."""
        promotions = {
            MemoryLayer.WORKING: MemoryLayer.EPISODIC,
            MemoryLayer.EPISODIC: MemoryLayer.SEMANTIC,
            MemoryLayer.SEMANTIC: MemoryLayer.METACOGNITIVE,
        }

        if self.layer in promotions:
            new_layer = promotions[self.layer]
            self.layer = new_layer

            # Reset reinforcement and update TTL
            self.reinforcement_score = 0.0
            new_ttl = LAYER_TTL[new_layer]
            if new_ttl:
                self.expires_at = datetime.now(timezone.utc) + new_ttl
            else:
                self.expires_at = None  # Permanent

            logger.info(f"Memory promoted to {new_layer.value}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for Neo4j storage."""
        return {
            "memory_layer": self.layer.value,
            "memory_created_at": self.created_at.isoformat(),
            "memory_expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "memory_last_accessed": self.last_accessed.isoformat(),
            "memory_access_count": self.access_count,
            "memory_importance": self.importance,
            "memory_confidence": self.confidence,
            "memory_content_type": self.content_type.value,
            "memory_reinforcement_score": self.reinforcement_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryProperties:
        """Create from Neo4j node/edge properties."""
        return cls(
            layer=MemoryLayer(data.get("memory_layer", "working")),
            created_at=datetime.fromisoformat(data["memory_created_at"]) if data.get("memory_created_at") else datetime.now(timezone.utc),
            expires_at=datetime.fromisoformat(data["memory_expires_at"]) if data.get("memory_expires_at") else None,
            last_accessed=datetime.fromisoformat(data["memory_last_accessed"]) if data.get("memory_last_accessed") else datetime.now(timezone.utc),
            access_count=data.get("memory_access_count", 0),
            importance=data.get("memory_importance", 0.5),
            confidence=data.get("memory_confidence", 1.0),
            content_type=ContentType(data.get("memory_content_type", "entity")),
            reinforcement_score=data.get("memory_reinforcement_score", 0.0),
        )


# =============================================================================
# Volatile Working Memory for ReAct Swarms
# =============================================================================


@dataclass
class VolatileObservation:
    """A volatile observation in working memory.

    These are short-lived, shared across parallel ReAct agents,
    and may become permanent through reinforcement.
    """

    observation_id: str = field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    content: str = ""
    source_agent_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))

    # Graph fragment this observation represents
    graph_fragment: SemanticGraph | None = None

    # Attention weight (for relevance scoring)
    attention_weight: float = 1.0

    # Whether this should be considered for persistence
    persist_candidate: bool = False
    content_type: ContentType = ContentType.EVENT

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def boost_attention(self, amount: float = 0.2) -> None:
        """Boost attention weight (used when accessed/referenced)."""
        self.attention_weight = min(1.0, self.attention_weight + amount)

    def decay_attention(self, factor: float = 0.9) -> None:
        """Decay attention weight over time."""
        self.attention_weight *= factor


class VolatileWorkingMemory:
    """Volatile working memory shared across ReAct swarm.

    This is the "scratchpad" used during active processing:
    - Agents add observations as they work
    - Other agents can read observations
    - Observations decay rapidly unless boosted
    - Good observations can be promoted to Neo4j

    Based on Miller's Law (7±2 items) and attention mechanisms.
    """

    def __init__(
        self,
        task_id: str,
        max_items: int = 9,  # 7+2 from Miller's Law
        decay_interval_seconds: float = 30.0,
    ):
        self.task_id = task_id
        self.max_items = max_items
        self.decay_interval = decay_interval_seconds

        self._observations: dict[str, VolatileObservation] = {}
        self._lock = asyncio.Lock()
        self._last_decay = time.time()

    async def add(
        self,
        content: str,
        source_agent_id: str,
        graph_fragment: SemanticGraph | None = None,
        content_type: ContentType = ContentType.EVENT,
        attention_weight: float = 1.0,
    ) -> VolatileObservation:
        """Add an observation to working memory."""
        async with self._lock:
            # Apply decay before adding
            self._apply_decay()

            obs = VolatileObservation(
                content=content,
                source_agent_id=source_agent_id,
                graph_fragment=graph_fragment,
                content_type=content_type,
                attention_weight=attention_weight,
            )

            self._observations[obs.observation_id] = obs

            # Enforce capacity limit (remove lowest attention)
            self._enforce_capacity()

            return obs

    async def get_context(
        self,
        agent_id: str,
        max_items: int = 7,
    ) -> list[VolatileObservation]:
        """Get relevant context for an agent.

        Returns observations sorted by attention weight,
        excluding expired ones.
        """
        async with self._lock:
            self._apply_decay()
            self._remove_expired()

            # Get non-expired, sorted by attention
            active = [
                obs for obs in self._observations.values()
                if not obs.is_expired
            ]
            active.sort(key=lambda o: o.attention_weight, reverse=True)

            # Boost accessed observations
            for obs in active[:max_items]:
                obs.boost_attention(0.05)

            return active[:max_items]

    async def mark_for_persistence(
        self,
        observation_id: str,
    ) -> bool:
        """Mark an observation as a candidate for Neo4j persistence."""
        async with self._lock:
            if observation_id in self._observations:
                self._observations[observation_id].persist_candidate = True
                return True
            return False

    async def get_persistence_candidates(self) -> list[VolatileObservation]:
        """Get observations marked for persistence."""
        async with self._lock:
            return [
                obs for obs in self._observations.values()
                if obs.persist_candidate and not obs.is_expired
            ]

    def _apply_decay(self) -> None:
        """Apply attention decay to all observations."""
        now = time.time()
        if now - self._last_decay < self.decay_interval:
            return

        for obs in self._observations.values():
            # Decay based on content type
            decay_rate = DECAY_RATES.get(obs.content_type, 1.0)
            decay_factor = 0.9 ** decay_rate  # Faster decay for peripheral content
            obs.decay_attention(decay_factor)

        self._last_decay = now

    def _remove_expired(self) -> None:
        """Remove expired observations."""
        expired_ids = [
            oid for oid, obs in self._observations.items()
            if obs.is_expired
        ]
        for oid in expired_ids:
            del self._observations[oid]

    def _enforce_capacity(self) -> None:
        """Remove lowest attention items if over capacity."""
        while len(self._observations) > self.max_items:
            # Find lowest attention (that's not a persist candidate)
            candidates = [
                (oid, obs) for oid, obs in self._observations.items()
                if not obs.persist_candidate
            ]
            if not candidates:
                # All are persist candidates, remove oldest
                candidates = list(self._observations.items())

            candidates.sort(key=lambda x: x[1].attention_weight)
            if candidates:
                del self._observations[candidates[0][0]]


# =============================================================================
# Memory-Aware Graph Store
# =============================================================================


class MemoryAwareGraphStore:
    """Neo4j graph store with memory layer awareness.

    Extends basic storage with:
    - Automatic TTL/expiration on nodes and edges
    - Memory property management
    - Garbage collection of expired memories
    - Reinforcement tracking
    - Layer-based querying
    """

    def __init__(
        self,
        store: Neo4jGraphStoreSync,
        instance_id: str,
    ):
        self.store = store
        self.instance_id = instance_id

    def save_with_memory(
        self,
        graph: SemanticGraph,
        default_layer: MemoryLayer = MemoryLayer.WORKING,
        content_type_map: dict[str, ContentType] | None = None,
    ) -> dict[str, int]:
        """Save graph with memory properties on all nodes/edges.

        Args:
            graph: The graph to save
            default_layer: Default memory layer for new items
            content_type_map: Map node_id -> ContentType for decay rates

        Returns:
            Dict with counts
        """
        content_type_map = content_type_map or {}

        # First save the base graph
        result = self.store.save(graph, self.instance_id)

        # Now add memory properties directly to Neo4j nodes
        # (bypassing the prop_ prefix that the base store uses)
        with self.store.driver.session() as session:
            for node in graph.iter_nodes():
                content_type = content_type_map.get(
                    node.node_id,
                    self._infer_content_type(node)
                )

                memory_props = MemoryProperties(
                    layer=default_layer,
                    content_type=content_type,
                    importance=self._calculate_importance(node, content_type),
                )

                # Set memory properties directly on Neo4j node
                props_dict = memory_props.to_dict()
                session.run("""
                    MATCH (n:Entity {node_id: $node_id, instance_id: $instance_id})
                    SET n += $props
                """, node_id=node.node_id, instance_id=self.instance_id, props=props_dict)

            # Add memory properties to edges
            for edge in graph.iter_edges(current_only=False):
                memory_props = MemoryProperties(
                    layer=default_layer,
                    content_type=ContentType.RELATIONSHIP,
                    importance=0.6,  # Relationships are moderately important
                )
                props_dict = memory_props.to_dict()
                session.run("""
                    MATCH (s:Entity {node_id: $source_id, instance_id: $instance_id})
                          -[r]->(t:Entity {node_id: $target_id, instance_id: $instance_id})
                    WHERE r.edge_id = $edge_id
                    SET r += $props
                """, source_id=edge.source_node_id, target_id=edge.target_node_id,
                    edge_id=edge.edge_id, instance_id=self.instance_id, props=props_dict)

        return result

    def load_active(
        self,
        include_layers: list[MemoryLayer] | None = None,
    ) -> SemanticGraph:
        """Load only non-expired memories.

        Args:
            include_layers: Optional filter by layer

        Returns:
            Graph with only active (non-expired) nodes/edges
        """
        now = datetime.now(timezone.utc).isoformat()
        active_graph = SemanticGraph()

        # Build layer filter
        layer_filter = ""
        if include_layers:
            layer_values = [layer.value for layer in include_layers]
            layer_filter = f"AND n.memory_layer IN {layer_values}"

        # Query only active (non-expired) nodes directly from Neo4j
        with self.store.driver.session() as session:
            # Load active nodes
            result = session.run(f"""
                MATCH (n:Entity {{instance_id: $instance_id}})
                WHERE (n.memory_expires_at IS NULL OR n.memory_expires_at > $now)
                {layer_filter}
                RETURN n
            """, instance_id=self.instance_id, now=now)

            for record in result:
                node = self._record_to_node(record["n"])
                active_graph.add_node(node)

            # Load edges between active nodes
            result = session.run(f"""
                MATCH (s:Entity {{instance_id: $instance_id}})
                      -[r]->(t:Entity {{instance_id: $instance_id}})
                WHERE (s.memory_expires_at IS NULL OR s.memory_expires_at > $now)
                  AND (t.memory_expires_at IS NULL OR t.memory_expires_at > $now)
                  AND (r.memory_expires_at IS NULL OR r.memory_expires_at > $now)
                  {layer_filter.replace('n.', 's.').replace('AND ', 'AND ')}
                RETURN s.node_id AS source_id, t.node_id AS target_id,
                       type(r) AS rel_type, properties(r) AS props
            """, instance_id=self.instance_id, now=now)

            for record in result:
                edge = self._record_to_edge(
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                    record["props"],
                )
                active_graph.add_edge(edge)

        return active_graph

    def _record_to_node(self, neo4j_node) -> GraphNode:
        """Convert a Neo4j node record to GraphNode."""
        props = dict(neo4j_node)

        # Determine node type from label or property
        node_type_str = props.pop("node_type", "instance")
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.INSTANCE

        # Parse datetime fields
        created_at = props.pop("created_at", None)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        updated_at = props.pop("updated_at", None)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        # Extract standard fields
        node = GraphNode(
            node_id=props.pop("node_id", ""),
            node_type=node_type,
            canonical_name=props.pop("canonical_name", ""),
            synset_id=props.pop("synset_id", None),
            wikidata_qid=props.pop("wikidata_qid", None),
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=updated_at or datetime.now(timezone.utc),
            confidence=props.pop("confidence", 1.0),
            source_ids=props.pop("source_ids", []),
            embedding=props.pop("embedding", None),
        )

        # Store remaining properties (including memory properties)
        props.pop("instance_id", None)  # Don't store instance_id in properties
        props.pop("entity_type", None)

        # Extract prop_ prefixed properties
        for key in list(props.keys()):
            if key.startswith("prop_"):
                node.properties[key[5:]] = props.pop(key)
            elif key.startswith("memory_"):
                node.properties[key] = props.pop(key)

        return node

    def _record_to_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        props: dict,
    ) -> GraphEdge:
        """Convert Neo4j edge data to GraphEdge."""
        from ..decomposition.graph.models import GraphEdge

        # Parse datetime fields
        valid_from = props.get("valid_from")
        if isinstance(valid_from, str):
            valid_from = datetime.fromisoformat(valid_from)

        valid_to = props.get("valid_to")
        if isinstance(valid_to, str) and valid_to:
            valid_to = datetime.fromisoformat(valid_to)
        else:
            valid_to = None

        return GraphEdge(
            edge_id=props.get("edge_id", ""),
            source_node_id=source_id,
            target_node_id=target_id,
            relation_type=props.get("relation_type", rel_type.lower()),
            valid_from=valid_from or datetime.now(timezone.utc),
            valid_to=valid_to,
            confidence=props.get("confidence", 1.0),
            source_ids=props.get("source_ids", []),
            properties={k: v for k, v in props.items()
                       if k.startswith("memory_") or k.startswith("prop_")},
        )

    def reinforce_nodes(
        self,
        node_ids: list[str],
        amount: float = 0.1,
    ) -> int:
        """Reinforce nodes (extends TTL, can promote layer)."""
        reinforced = 0

        with self.store.driver.session() as session:
            for node_id in node_ids:
                result = session.run("""
                    MATCH (n:Entity {node_id: $node_id, instance_id: $instance_id})
                    SET n.memory_reinforcement_score = COALESCE(n.memory_reinforcement_score, 0) + $amount,
                        n.memory_access_count = COALESCE(n.memory_access_count, 0) + 1,
                        n.memory_last_accessed = $now
                    RETURN n.memory_reinforcement_score AS score, n.memory_layer AS layer
                """, node_id=node_id, instance_id=self.instance_id,
                    amount=amount, now=datetime.now(timezone.utc).isoformat())

                record = result.single()
                if record:
                    reinforced += 1
                    # Check for promotion
                    self._check_and_promote(session, node_id, record["score"], record["layer"])

        return reinforced

    def garbage_collect(self) -> int:
        """Remove expired memories from Neo4j."""
        now = datetime.now(timezone.utc).isoformat()

        with self.store.driver.session() as session:
            # Delete expired nodes (edges auto-deleted via DETACH)
            result = session.run("""
                MATCH (n:Entity {instance_id: $instance_id})
                WHERE n.memory_expires_at IS NOT NULL
                  AND n.memory_expires_at < $now
                WITH n, count(*) as cnt
                DETACH DELETE n
                RETURN sum(cnt) as deleted
            """, instance_id=self.instance_id, now=now)

            record = result.single()
            deleted = record["deleted"] if record else 0

            logger.info(f"Garbage collected {deleted} expired memories")
            return deleted

    def get_layer_statistics(self) -> dict[str, Any]:
        """Get statistics by memory layer."""
        with self.store.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity {instance_id: $instance_id})
                RETURN n.memory_layer AS layer, count(*) AS count
            """, instance_id=self.instance_id)

            stats = {layer.value: 0 for layer in MemoryLayer}
            for record in result:
                if record["layer"]:
                    stats[record["layer"]] = record["count"]

            return stats

    def _infer_content_type(self, node: GraphNode) -> ContentType:
        """Infer content type from node properties."""
        # Check explicit type
        if "content_type" in node.properties:
            try:
                return ContentType(node.properties["content_type"])
            except ValueError:
                pass

        # Infer from node type
        type_mapping = {
            NodeType.INSTANCE: ContentType.ENTITY,
            NodeType.CLASS: ContentType.FACT,
            NodeType.EVENT: ContentType.EVENT,
            NodeType.ATTRIBUTE: ContentType.MODALITY,
            NodeType.COLLECTION: ContentType.ENTITY,
        }

        return type_mapping.get(node.node_type, ContentType.ENTITY)

    def _calculate_importance(
        self,
        node: GraphNode,
        content_type: ContentType,
    ) -> float:
        """Calculate importance score for a node."""
        # Base importance by content type
        importance_base = {
            ContentType.INSTRUCTION: 1.0,
            ContentType.PREFERENCE: 0.9,
            ContentType.SKILL: 0.85,
            ContentType.FACT: 0.8,
            ContentType.ENTITY: 0.7,
            ContentType.RELATIONSHIP: 0.7,
            ContentType.EVENT: 0.6,
            ContentType.LOCATION: 0.6,
            ContentType.TEMPORAL: 0.5,
            ContentType.MODALITY: 0.3,
            ContentType.PRESUPPOSITION: 0.4,
            ContentType.COMMONSENSE: 0.4,
            ContentType.SENTIMENT: 0.3,
            ContentType.NEGATION: 0.3,
        }

        return importance_base.get(content_type, 0.5)

    def _check_and_promote(
        self,
        session,
        node_id: str,
        score: float,
        current_layer: str,
    ) -> None:
        """Check if node should be promoted to higher layer."""
        promotion_thresholds = {
            "working": ("episodic", 0.3),
            "episodic": ("semantic", 0.6),
            "semantic": ("metacognitive", 0.9),
        }

        if current_layer in promotion_thresholds:
            new_layer, threshold = promotion_thresholds[current_layer]
            if score >= threshold:
                # Calculate new TTL
                new_ttl = LAYER_TTL[MemoryLayer(new_layer)]
                new_expires = None
                if new_ttl:
                    new_expires = (datetime.now(timezone.utc) + new_ttl).isoformat()

                session.run("""
                    MATCH (n:Entity {node_id: $node_id, instance_id: $instance_id})
                    SET n.memory_layer = $new_layer,
                        n.memory_expires_at = $new_expires,
                        n.memory_reinforcement_score = 0
                """, node_id=node_id, instance_id=self.instance_id,
                    new_layer=new_layer, new_expires=new_expires)

                logger.info(f"Promoted node {node_id} to {new_layer}")


# =============================================================================
# Content Type Inference for Phase 1 Outputs
# =============================================================================


def classify_phase1_content(
    graph: SemanticGraph,
) -> dict[str, ContentType]:
    """Classify nodes from Phase 1 extraction into content types.

    Maps the 10 Phase 1 attribute types to content types for decay:
    1. Semantic Roles → RELATIONSHIP
    2. Presuppositions → PRESUPPOSITION
    3. Commonsense → COMMONSENSE
    4. Temporal → TEMPORAL
    5. Modality → MODALITY
    6. Negation → NEGATION
    7. Coreference → ENTITY
    8. Sentiment → SENTIMENT
    9. Discourse → EVENT
    10. Named Entities → ENTITY/FACT

    Returns:
        Map of node_id -> ContentType
    """
    classification: dict[str, ContentType] = {}

    for node in graph.iter_nodes():
        # Check node properties for Phase 1 markers
        props = node.properties

        if props.get("is_presupposition"):
            classification[node.node_id] = ContentType.PRESUPPOSITION
        elif props.get("is_commonsense"):
            classification[node.node_id] = ContentType.COMMONSENSE
        elif props.get("is_temporal"):
            classification[node.node_id] = ContentType.TEMPORAL
        elif props.get("is_modality"):
            classification[node.node_id] = ContentType.MODALITY
        elif props.get("is_negation"):
            classification[node.node_id] = ContentType.NEGATION
        elif props.get("is_sentiment"):
            classification[node.node_id] = ContentType.SENTIMENT
        elif node.node_type == NodeType.EVENT:
            classification[node.node_id] = ContentType.EVENT
        elif node.node_type == NodeType.INSTANCE:
            # Named entities are valuable
            if node.synset_id or node.wikidata_qid:
                classification[node.node_id] = ContentType.FACT
            else:
                classification[node.node_id] = ContentType.ENTITY
        elif node.node_type == NodeType.CLASS:
            classification[node.node_id] = ContentType.FACT
        else:
            classification[node.node_id] = ContentType.ENTITY

    return classification

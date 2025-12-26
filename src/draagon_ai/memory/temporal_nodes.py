"""Temporal node classes for the Temporal Cognitive Graph.

These nodes form the foundation of the AGI-Lite memory architecture,
implementing bi-temporal tracking (event time + ingestion time) and
hierarchical scoping.

Based on research from:
- Zep/Graphiti: Bi-temporal knowledge graphs
- Mem0: Hybrid vector-graph architecture
- ACE Framework: Context evolution patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
import uuid


class NodeType(str, Enum):
    """Types of nodes in the temporal cognitive graph.

    Organized by memory layer:
    - Working: CONTEXT, GOAL
    - Episodic: EPISODE, EVENT
    - Semantic: ENTITY, RELATIONSHIP, FACT, BELIEF
    - Metacognitive: SKILL, STRATEGY, INSIGHT, BEHAVIOR
    """

    # Working memory (seconds to minutes)
    CONTEXT = "context"      # Active conversation context
    GOAL = "goal"            # Current task/goal

    # Episodic memory (hours to days)
    EPISODE = "episode"      # Conversation episode
    EVENT = "event"          # Discrete event within episode

    # Semantic memory (days to months)
    ENTITY = "entity"        # Person, place, thing
    RELATIONSHIP = "relationship"  # Connection between entities
    FACT = "fact"            # Declarative knowledge
    BELIEF = "belief"        # Agent's reconciled understanding

    # Metacognitive memory (weeks to permanent)
    SKILL = "skill"          # Procedural knowledge
    STRATEGY = "strategy"    # Problem-solving approach
    INSIGHT = "insight"      # Meta-learning about patterns
    BEHAVIOR = "behavior"    # Learned behavior pattern


class MemoryLayer(str, Enum):
    """The four layers of the temporal cognitive graph.

    Each layer has different temporal characteristics:
    - WORKING: Seconds to minutes, session-bound
    - EPISODIC: Hours to days, experience-based
    - SEMANTIC: Days to months, factual knowledge
    - METACOGNITIVE: Weeks to permanent, skills and strategies
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    METACOGNITIVE = "metacognitive"


# Mapping from node type to memory layer
NODE_TYPE_TO_LAYER: dict[NodeType, MemoryLayer] = {
    NodeType.CONTEXT: MemoryLayer.WORKING,
    NodeType.GOAL: MemoryLayer.WORKING,
    NodeType.EPISODE: MemoryLayer.EPISODIC,
    NodeType.EVENT: MemoryLayer.EPISODIC,
    NodeType.ENTITY: MemoryLayer.SEMANTIC,
    NodeType.RELATIONSHIP: MemoryLayer.SEMANTIC,
    NodeType.FACT: MemoryLayer.SEMANTIC,
    NodeType.BELIEF: MemoryLayer.SEMANTIC,
    NodeType.SKILL: MemoryLayer.METACOGNITIVE,
    NodeType.STRATEGY: MemoryLayer.METACOGNITIVE,
    NodeType.INSIGHT: MemoryLayer.METACOGNITIVE,
    NodeType.BEHAVIOR: MemoryLayer.METACOGNITIVE,
}


# Default TTLs by layer (for decay/cleanup)
LAYER_DEFAULT_TTL: dict[MemoryLayer, timedelta | None] = {
    MemoryLayer.WORKING: timedelta(minutes=30),
    MemoryLayer.EPISODIC: timedelta(days=7),
    MemoryLayer.SEMANTIC: timedelta(days=90),
    MemoryLayer.METACOGNITIVE: None,  # Permanent by default
}


@dataclass
class TemporalNode:
    """Base class for all nodes in the temporal cognitive graph.

    Implements bi-temporal tracking as described in Zep/Graphiti research:
    - event_time: When the event actually occurred
    - ingestion_time: When the agent learned about it

    Also tracks validity intervals for temporal reasoning:
    - valid_from: When this node became true/relevant
    - valid_until: When it stopped being true (None = still current)

    Example:
        # User tells agent about their birthday
        node = TemporalNode(
            content="Doug's birthday is March 15",
            node_type=NodeType.FACT,
            event_time=datetime(1985, 3, 15),  # When birthday actually is
            ingestion_time=datetime.now(),       # When we learned this
        )
    """

    # Required fields
    content: str
    node_type: NodeType

    # Auto-generated
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Bi-temporal tracking (Zep/Graphiti pattern)
    event_time: datetime = field(default_factory=datetime.now)
    ingestion_time: datetime = field(default_factory=datetime.now)

    # Validity interval for temporal reasoning
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None  # None = still current/valid

    # Vector embedding for similarity search
    embedding: list[float] | None = None

    # Confidence and evolution
    confidence: float = 1.0  # 0.0-1.0, certainty level
    importance: float = 0.5  # 0.0-1.0, retrieval priority
    stated_count: int = 1    # Times this was stated/reinforced
    access_count: int = 0    # Times this was retrieved

    # Hierarchical scope (see HierarchicalScope)
    scope_id: str = "agent:default"

    # Memory layer (derived from node_type)
    layer: MemoryLayer = field(init=False)

    # Lineage tracking for provenance
    derived_from: list[str] = field(default_factory=list)  # Parent node IDs
    supersedes: list[str] = field(default_factory=list)    # Nodes this replaces
    superseded_by: str | None = None  # Node that replaced this

    # Entity extraction (for graph connections)
    entities: list[str] = field(default_factory=list)

    # Flexible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime | None = None

    def __post_init__(self):
        """Set derived fields after initialization."""
        self.layer = NODE_TYPE_TO_LAYER.get(self.node_type, MemoryLayer.SEMANTIC)

    @property
    def is_current(self) -> bool:
        """Check if this node is currently valid (not superseded)."""
        return self.valid_until is None and self.superseded_by is None

    @property
    def is_expired(self) -> bool:
        """Check if this node has passed its validity period."""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until

    @property
    def age_seconds(self) -> float:
        """Get age of this node in seconds since ingestion."""
        return (datetime.now() - self.ingestion_time).total_seconds()

    @property
    def default_ttl(self) -> timedelta | None:
        """Get default TTL based on memory layer."""
        return LAYER_DEFAULT_TTL.get(self.layer)

    def reinforce(self, boost: float = 0.1) -> None:
        """Reinforce this node (called when accessed/used).

        Increments access_count and boosts importance. Does NOT increment
        stated_count - use restate() when user re-states a fact.

        Args:
            boost: Amount to boost importance (max 1.0)
        """
        self.access_count += 1
        self.importance = min(1.0, self.importance + boost)
        self.last_accessed = datetime.now()
        self.updated_at = datetime.now()

    def restate(self, boost: float = 0.05) -> None:
        """Record that user re-stated this fact.

        Called when user says the same fact again (e.g., "My birthday is March 15"
        stated multiple times). This is different from reinforce() which is called
        when the fact is retrieved/used.

        Args:
            boost: Amount to boost importance (default smaller than reinforce)
        """
        self.stated_count += 1
        self.importance = min(1.0, self.importance + boost)
        self.updated_at = datetime.now()

    def decay(self, factor: float = 0.95) -> None:
        """Apply temporal decay to importance.

        Args:
            factor: Decay multiplier (0.95 = 5% decay)
        """
        self.importance = max(0.0, self.importance * factor)
        self.updated_at = datetime.now()

    def supersede(self, new_node_id: str) -> None:
        """Mark this node as superseded by another.

        Args:
            new_node_id: ID of the node that replaces this one
        """
        self.superseded_by = new_node_id
        self.valid_until = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type.value,
            "event_time": self.event_time.isoformat(),
            "ingestion_time": self.ingestion_time.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "embedding": self.embedding,
            "confidence": self.confidence,
            "importance": self.importance,
            "stated_count": self.stated_count,
            "access_count": self.access_count,
            "scope_id": self.scope_id,
            "layer": self.layer.value,
            "derived_from": self.derived_from,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "entities": self.entities,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalNode":
        """Create from dictionary."""
        # Parse datetime fields
        def parse_dt(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val)

        return cls(
            node_id=data["node_id"],
            content=data["content"],
            node_type=NodeType(data["node_type"]),
            event_time=parse_dt(data["event_time"]) or datetime.now(),
            ingestion_time=parse_dt(data["ingestion_time"]) or datetime.now(),
            valid_from=parse_dt(data["valid_from"]) or datetime.now(),
            valid_until=parse_dt(data.get("valid_until")),
            embedding=data.get("embedding"),
            confidence=data.get("confidence", 1.0),
            importance=data.get("importance", 0.5),
            stated_count=data.get("stated_count", 1),
            access_count=data.get("access_count", 0),
            scope_id=data.get("scope_id", "agent:default"),
            derived_from=data.get("derived_from", []),
            supersedes=data.get("supersedes", []),
            superseded_by=data.get("superseded_by"),
            entities=data.get("entities", []),
            metadata=data.get("metadata", {}),
            created_at=parse_dt(data.get("created_at")) or datetime.now(),
            updated_at=parse_dt(data.get("updated_at")) or datetime.now(),
            last_accessed=parse_dt(data.get("last_accessed")),
        )


class EdgeType(str, Enum):
    """Types of edges connecting nodes in the graph.

    Edges capture semantic, temporal, and provenance relationships.
    """

    # Semantic relationships
    IS_A = "is_a"              # Category membership
    HAS = "has"                # Possession/attribute
    RELATED_TO = "related_to"  # General association
    PART_OF = "part_of"        # Component relationship

    # Temporal relationships
    BEFORE = "before"          # Temporal ordering
    AFTER = "after"            # Temporal ordering
    DURING = "during"          # Temporal overlap
    CAUSES = "causes"          # Causal relationship

    # Provenance relationships
    DERIVED_FROM = "derived_from"  # Lineage
    SUPERSEDES = "supersedes"      # Replacement
    SUPPORTS = "supports"          # Evidence
    CONTRADICTS = "contradicts"    # Conflict

    # Agent relationships
    TRIGGERS = "triggers"      # Behavior activation
    ENABLES = "enables"        # Capability enablement
    DELEGATES_TO = "delegates_to"  # Agent handoff
    LEARNS_FROM = "learns_from"    # Learning source

    # Story/narrative relationships (for StoryTeller)
    KNOWS = "knows"            # Character knowledge
    INTERACTS_WITH = "interacts_with"  # Character interaction
    LOCATED_IN = "located_in"  # Spatial relationship


@dataclass
class TemporalEdge:
    """An edge connecting two temporal nodes.

    Edges also have bi-temporal tracking, allowing us to reason about
    when relationships were true and when we learned about them.

    Example:
        # Doug lives in Philadelphia (relationship edge)
        edge = TemporalEdge(
            source_id=doug_entity.node_id,
            target_id=philadelphia_entity.node_id,
            edge_type=EdgeType.LOCATED_IN,
            event_time=datetime(2020, 1, 1),  # When he moved there
            ingestion_time=datetime.now(),     # When we learned this
        )
    """

    # Required fields
    source_id: str
    target_id: str
    edge_type: EdgeType

    # Auto-generated
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Bi-temporal tracking
    event_time: datetime = field(default_factory=datetime.now)
    ingestion_time: datetime = field(default_factory=datetime.now)

    # Validity interval
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None

    # Edge properties
    weight: float = 1.0       # Strength of connection
    confidence: float = 1.0   # Certainty of relationship

    # Optional label for more specific relationships
    label: str | None = None  # e.g., "brother of", "works at"

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_current(self) -> bool:
        """Check if this edge is currently valid."""
        return self.valid_until is None

    def invalidate(self) -> None:
        """Mark this edge as no longer valid."""
        self.valid_until = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "event_time": self.event_time.isoformat(),
            "ingestion_time": self.ingestion_time.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "weight": self.weight,
            "confidence": self.confidence,
            "label": self.label,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalEdge":
        """Create from dictionary."""
        def parse_dt(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val)

        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            event_time=parse_dt(data["event_time"]) or datetime.now(),
            ingestion_time=parse_dt(data["ingestion_time"]) or datetime.now(),
            valid_from=parse_dt(data["valid_from"]) or datetime.now(),
            valid_until=parse_dt(data.get("valid_until")),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            label=data.get("label"),
            metadata=data.get("metadata", {}),
            created_at=parse_dt(data.get("created_at")) or datetime.now(),
            updated_at=parse_dt(data.get("updated_at")) or datetime.now(),
        )


# Convenience factory functions

def create_fact_node(
    content: str,
    scope_id: str = "agent:default",
    entities: list[str] | None = None,
    confidence: float = 1.0,
    event_time: datetime | None = None,
) -> TemporalNode:
    """Create a fact node (semantic layer).

    Args:
        content: The factual statement
        scope_id: Hierarchical scope
        entities: Extracted entities
        confidence: Certainty level
        event_time: When this fact became true

    Returns:
        TemporalNode of type FACT
    """
    return TemporalNode(
        content=content,
        node_type=NodeType.FACT,
        scope_id=scope_id,
        entities=entities or [],
        confidence=confidence,
        importance=0.7,  # Facts are generally important
        event_time=event_time or datetime.now(),
    )


def create_episode_node(
    content: str,
    scope_id: str = "agent:default",
    conversation_id: str | None = None,
) -> TemporalNode:
    """Create an episode node (episodic layer).

    Args:
        content: Episode summary/description
        scope_id: Hierarchical scope
        conversation_id: Associated conversation

    Returns:
        TemporalNode of type EPISODE
    """
    metadata = {}
    if conversation_id:
        metadata["conversation_id"] = conversation_id

    return TemporalNode(
        content=content,
        node_type=NodeType.EPISODE,
        scope_id=scope_id,
        importance=0.5,  # Episodic memories start at medium importance
        metadata=metadata,
    )


def create_skill_node(
    content: str,
    scope_id: str = "agent:default",
    skill_name: str | None = None,
) -> TemporalNode:
    """Create a skill node (metacognitive layer).

    Args:
        content: Skill description/procedure
        scope_id: Hierarchical scope
        skill_name: Name of the skill

    Returns:
        TemporalNode of type SKILL
    """
    metadata = {}
    if skill_name:
        metadata["skill_name"] = skill_name

    return TemporalNode(
        content=content,
        node_type=NodeType.SKILL,
        scope_id=scope_id,
        importance=0.85,  # Skills are highly important
        metadata=metadata,
    )


def create_entity_node(
    name: str,
    entity_type: str,
    scope_id: str = "agent:default",
    attributes: dict[str, Any] | None = None,
) -> TemporalNode:
    """Create an entity node (semantic layer).

    Args:
        name: Entity name
        entity_type: Type of entity (person, place, thing)
        scope_id: Hierarchical scope
        attributes: Entity attributes

    Returns:
        TemporalNode of type ENTITY
    """
    metadata = {
        "entity_name": name,
        "entity_type": entity_type,
    }
    if attributes:
        metadata["attributes"] = attributes

    return TemporalNode(
        content=f"{entity_type}: {name}",
        node_type=NodeType.ENTITY,
        scope_id=scope_id,
        entities=[name],
        importance=0.6,
        metadata=metadata,
    )


def create_behavior_node(
    behavior_id: str,
    behavior_name: str,
    description: str,
    scope_id: str = "agent:default",
    triggers: list[str] | None = None,
    actions: list[str] | None = None,
) -> TemporalNode:
    """Create a behavior node (metacognitive layer).

    This represents a learned behavior pattern that can be triggered
    and evolved over time.

    Args:
        behavior_id: Unique behavior identifier
        behavior_name: Human-readable name
        description: What the behavior does
        scope_id: Hierarchical scope
        triggers: Activation triggers
        actions: Available actions

    Returns:
        TemporalNode of type BEHAVIOR
    """
    metadata = {
        "behavior_id": behavior_id,
        "behavior_name": behavior_name,
        "triggers": triggers or [],
        "actions": actions or [],
        "execution_count": 0,
        "success_rate": 1.0,
    }

    return TemporalNode(
        content=description,
        node_type=NodeType.BEHAVIOR,
        scope_id=scope_id,
        importance=0.9,  # Behaviors are very important
        metadata=metadata,
    )

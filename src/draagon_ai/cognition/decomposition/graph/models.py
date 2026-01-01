"""Semantic Graph Data Models for Phase 2 Memory Integration.

This module defines the entity-centric graph structure for storing decomposed
knowledge. The graph model enables:

- Natural representation of entity relationships (Doug -> [has] -> Cat)
- Efficient traversal queries ("How many cats?")
- Incremental knowledge building (add properties to existing entities)
- Context compression (entity summary vs all source texts)
- Bi-temporal edge tracking (valid_from, valid_to for historical queries)

Key Design Decisions (from PHASE_2_MEMORY_INTEGRATION.md):

DD-2.1: Graph-First Storage
    Store decomposed knowledge in a semantic graph structure rather than
    document-oriented storage for natural relationship representation.

DD-2.3: Bi-Temporal Edge Model
    Track both valid_from and valid_to for all relationships to enable
    historical queries and temporal conflict resolution.

Example:
    >>> from draagon_ai.cognition.decomposition.graph import (
    ...     GraphNode, GraphEdge, NodeType, SemanticGraph
    ... )
    >>>
    >>> # Create instance nodes (specific individuals)
    >>> doug = GraphNode(
    ...     node_id="doug-001",
    ...     node_type=NodeType.INSTANCE,
    ...     canonical_name="Doug",
    ... )
    >>> whiskers = GraphNode(
    ...     node_id="whiskers-001",
    ...     node_type=NodeType.INSTANCE,
    ...     canonical_name="Whiskers",
    ...     properties={"breed": "tabby", "color": "orange"},
    ... )
    >>>
    >>> # Create class node (abstract type)
    >>> cat_class = GraphNode(
    ...     node_id="cat-class",
    ...     node_type=NodeType.CLASS,
    ...     canonical_name="cat.n.01",
    ...     synset_id="cat.n.01",
    ... )
    >>>
    >>> # Create relationship edges
    >>> owns_edge = GraphEdge(
    ...     source_node_id=doug.node_id,
    ...     target_node_id=whiskers.node_id,
    ...     relation_type="owns",
    ... )
    >>> instance_of_edge = GraphEdge(
    ...     source_node_id=whiskers.node_id,
    ...     target_node_id=cat_class.node_id,
    ...     relation_type="instance_of",
    ... )
    >>>
    >>> # Build graph
    >>> graph = SemanticGraph()
    >>> graph.add_node(doug)
    >>> graph.add_node(whiskers)
    >>> graph.add_node(cat_class)
    >>> graph.add_edge(owns_edge)
    >>> graph.add_edge(instance_of_edge)
    >>>
    >>> # Query graph
    >>> count = graph.count_relations(doug.node_id, "owns")
    >>> print(f"Doug owns {count} pet(s)")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

from ..identifiers import EntityType


# =============================================================================
# Enumerations
# =============================================================================


class NodeType(str, Enum):
    """Types of nodes in the semantic graph.

    Following RDF/OWL ontology patterns (see docs/architecture/GRAPH_ONTOLOGY_DESIGN.md):

    INSTANCE: A specific, identifiable individual thing.
        - Named: "Doug", "Whiskers", "Apple Inc.", "Paris"
        - Anonymous: "[Doug's cat]" (we know it exists but don't have a name)
        These are targets of INSTANCE_OF edges, linking to CLASS nodes.
        Analogous to rdf:type subject in RDF.

    CLASS: An abstract category/type/concept - the ontology layer.
        - WordNet synsets: cat.n.01, person.n.01, city.n.01
        - Taxonomic types: Mammal, Animal, Pet
        These are deduplicated globally - one node per synset.
        Analogous to rdfs:Class in RDF Schema.

    EVENT: A specific occurrence at a point or period in time.
        Events are instances that are inherently temporal.
        Examples: "Doug's birthday party on March 15", "The 3pm meeting"

    ATTRIBUTE: A reified property value with metadata.
        Used when we need to attach confidence/provenance to property values.
        Example: "orange" color with certainty 0.8 and source "user stated"

    COLLECTION: A group/set of instances with cardinality.
        Examples: "Doug's cats" (count: 3), "Team Alpha members"
        Enables efficient cardinality queries ("How many cats does Doug have?")
    """

    INSTANCE = "instance"    # Specific individual (was ENTITY)
    CLASS = "class"          # Abstract type/concept (was CONCEPT)
    EVENT = "event"          # Temporal instance
    ATTRIBUTE = "attribute"  # Reified property value
    COLLECTION = "collection"  # Group with cardinality

    # Backwards compatibility aliases
    @classmethod
    def _missing_(cls, value: object) -> "NodeType | None":
        """Handle legacy values for backwards compatibility."""
        if value == "entity":
            return cls.INSTANCE
        if value == "concept":
            return cls.CLASS
        return None


class EdgeRelationType(str, Enum):
    """Common relationship types in the semantic graph.

    This is not exhaustive - the graph supports arbitrary relation_type strings.
    These are common patterns for standardization.

    See also: SemanticEdgeType in builder.py for Phase 1 extraction edges.
    """

    # === Ontology-Level Relations (RDF/OWL patterns) ===
    # Instance-to-Class: links specific things to their types
    INSTANCE_OF = "instance_of"  # Whiskers INSTANCE_OF cat.n.01

    # Class-to-Class: taxonomic hierarchy
    SUBCLASS_OF = "subclass_of"  # cat.n.01 SUBCLASS_OF mammal.n.01
    EQUIVALENT_CLASS = "equivalent_class"  # Synonymous classes
    RELATED_CLASS = "related_class"  # Loose conceptual relation

    # WSD-specific
    HAS_SENSE = "has_sense"  # Entity to its disambiguated synset
    ALTERNATIVE_SENSE = "alternative_sense"  # Less likely senses

    # === Instance-Level Relations ===
    # Ownership/Possession
    HAS = "has"
    OWNS = "owns"
    BELONGS_TO = "belongs_to"

    # Identity/Naming
    IS_A = "is_a"  # Legacy, prefer INSTANCE_OF
    NAME = "name"
    ALIAS = "alias"

    # Properties
    HAS_PROPERTY = "has_property"
    HAS_ATTRIBUTE = "has_attribute"

    # Composition/Membership
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    MEMBER_OF = "member_of"  # Instance is member of a COLLECTION
    CONTAINS = "contains"

    # Social
    KNOWS = "knows"
    WORKS_WITH = "works_with"
    MARRIED_TO = "married_to"

    # Location
    LOCATED_IN = "located_in"
    LIVES_IN = "lives_in"

    # Temporal
    HAPPENED_AT = "happened_at"
    HAPPENED_BEFORE = "happened_before"
    HAPPENED_AFTER = "happened_after"

    # Causation
    CAUSED_BY = "caused_by"
    CAUSES = "causes"

    # Custom (for user-defined relations)
    CUSTOM = "custom"


class ConflictType(str, Enum):
    """Types of conflicts that can occur during graph merging.

    CONTRADICTS: New information directly contradicts existing
        Example: "Doug has 5 cats" vs existing "Doug has 6 cats"

    SUPERSEDES: New information updates/replaces old (temporal update)
        Example: "Doug moved to Seattle" supersedes "Doug lives in Portland"

    AMBIGUOUS: Cannot determine relationship between old and new
        Example: Unclear whether new info is update or error
    """

    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    AMBIGUOUS = "ambiguous"


# =============================================================================
# Graph Node
# =============================================================================


@dataclass
class GraphNode:
    """A node in the semantic graph representing an entity, concept, or event.

    Nodes are the primary elements of the knowledge graph. Each node has:
    - A unique identifier
    - A type classification
    - A canonical name for display
    - Arbitrary properties (metadata)
    - Optional WordNet/external sense information
    - Provenance tracking (what decompositions contributed)

    Attributes:
        node_id: Unique identifier (UUID format)
        node_type: Classification of this node (ENTITY, CONCEPT, etc.)
        canonical_name: Human-readable name for display
        entity_type: EntityType from Phase 0 classification (if applicable)
        properties: Key-value properties (breed, color, age, etc.)
        synset_id: WordNet synset ID if this is a disambiguated concept
        wikidata_qid: Wikidata entity ID if known
        created_at: When this node was first created
        updated_at: When this node was last modified
        source_ids: List of decomposition IDs that contributed to this node
        confidence: Overall confidence in this node (0.0-1.0)
        embedding: Optional vector embedding for similarity search
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.INSTANCE  # Default to INSTANCE (specific entity)
    canonical_name: str = ""

    # Entity classification from Phase 0
    entity_type: EntityType | None = None

    # Arbitrary properties
    properties: dict[str, Any] = field(default_factory=dict)

    # External identifiers
    synset_id: str | None = None
    wikidata_qid: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Provenance
    source_ids: list[str] = field(default_factory=list)

    # Confidence
    confidence: float = 1.0

    # Vector embedding (optional, for similarity search)
    embedding: list[float] | None = None

    def __hash__(self) -> int:
        """Hash based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on node_id."""
        if isinstance(other, GraphNode):
            return self.node_id == other.node_id
        return False

    def add_property(self, key: str, value: Any) -> None:
        """Add or update a property.

        Args:
            key: Property name
            value: Property value
        """
        self.properties[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value.

        Args:
            key: Property name
            default: Default value if not found

        Returns:
            Property value or default
        """
        return self.properties.get(key, default)

    def has_property(self, key: str) -> bool:
        """Check if a property exists.

        Args:
            key: Property name

        Returns:
            True if property exists
        """
        return key in self.properties

    def add_source(self, source_id: str) -> None:
        """Add a source decomposition ID.

        Args:
            source_id: ID of the decomposition that contributed to this node
        """
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)
            self.updated_at = datetime.now(timezone.utc)

    def matches_name(self, name: str, case_sensitive: bool = False) -> bool:
        """Check if this node matches a name.

        Args:
            name: Name to match
            case_sensitive: Whether to match case-sensitively

        Returns:
            True if canonical_name or any alias matches
        """
        if case_sensitive:
            if self.canonical_name == name:
                return True
            aliases = self.properties.get("aliases", [])
            return name in aliases
        else:
            name_lower = name.lower()
            if self.canonical_name.lower() == name_lower:
                return True
            aliases = self.properties.get("aliases", [])
            return any(alias.lower() == name_lower for alias in aliases)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type.value if self.entity_type else None,
            "properties": self.properties,
            "synset_id": self.synset_id,
            "wikidata_qid": self.wikidata_qid,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_ids": self.source_ids,
            "confidence": self.confidence,
            "embedding": self.embedding,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed GraphNode
        """
        return cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            node_type=NodeType(data.get("node_type", "entity")),
            canonical_name=data.get("canonical_name", ""),
            entity_type=EntityType(data["entity_type"]) if data.get("entity_type") else None,
            properties=data.get("properties", {}),
            synset_id=data.get("synset_id"),
            wikidata_qid=data.get("wikidata_qid"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            source_ids=data.get("source_ids", []),
            confidence=data.get("confidence", 1.0),
            embedding=data.get("embedding"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> GraphNode:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"GraphNode({self.node_type.value}:{self.canonical_name!r}, "
            f"id={self.node_id[:8]}...)"
        )


# =============================================================================
# Graph Edge
# =============================================================================


@dataclass
class GraphEdge:
    """A directed relationship edge in the semantic graph.

    Edges connect nodes and represent relationships. Key features:
    - Bi-temporal tracking: valid_from/valid_to for historical queries
    - Properties: Arbitrary metadata (count, confidence, etc.)
    - Provenance: Which decomposition created this edge

    Bi-Temporal Model (DD-2.3):
        - valid_from: When this relationship became true
        - valid_to: When this relationship stopped being true (None = still true)
        - Enables queries like "What cats did Doug have in 2024?"
        - Supports conflict resolution via temporal superseding

    Attributes:
        edge_id: Unique identifier (UUID format)
        source_node_id: ID of the source (from) node
        target_node_id: ID of the target (to) node
        relation_type: Type of relationship (has, is_a, etc.)
        properties: Key-value properties (count, confidence, etc.)
        valid_from: When this relationship became true
        valid_to: When this stopped being true (None = current)
        source_decomposition_id: Which decomposition created this edge
        confidence: Confidence in this relationship (0.0-1.0)
        created_at: When this edge was created in the graph
    """

    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    relation_type: str = "related_to"

    # Edge properties
    properties: dict[str, Any] = field(default_factory=dict)

    # Bi-temporal validity (DD-2.3)
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: datetime | None = None  # None means currently valid

    # Provenance
    source_decomposition_id: str | None = None

    # Confidence
    confidence: float = 1.0

    # Graph metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __hash__(self) -> int:
        """Hash based on edge_id."""
        return hash(self.edge_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on edge_id."""
        if isinstance(other, GraphEdge):
            return self.edge_id == other.edge_id
        return False

    @property
    def is_current(self) -> bool:
        """Check if this edge is currently valid.

        Returns:
            True if valid_to is None (still active)
        """
        return self.valid_to is None

    def invalidate(self, when: datetime | None = None) -> None:
        """Mark this edge as no longer valid.

        Args:
            when: When the edge became invalid (default: now)
        """
        self.valid_to = when or datetime.now(timezone.utc)

    def is_valid_at(self, when: datetime) -> bool:
        """Check if this edge was valid at a specific time.

        Args:
            when: The time to check

        Returns:
            True if edge was valid at that time
        """
        if when < self.valid_from:
            return False
        if self.valid_to is not None and when >= self.valid_to:
            return False
        return True

    def add_property(self, key: str, value: Any) -> None:
        """Add or update a property.

        Args:
            key: Property name
            value: Property value
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value.

        Args:
            key: Property name
            default: Default value if not found

        Returns:
            Property value or default
        """
        return self.properties.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation
        """
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "source_decomposition_id": self.source_decomposition_id,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed GraphEdge
        """
        return cls(
            edge_id=data.get("edge_id", str(uuid.uuid4())),
            source_node_id=data.get("source_node_id", ""),
            target_node_id=data.get("target_node_id", ""),
            relation_type=data.get("relation_type", "related_to"),
            properties=data.get("properties", {}),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else datetime.now(timezone.utc),
            valid_to=datetime.fromisoformat(data["valid_to"]) if data.get("valid_to") else None,
            source_decomposition_id=data.get("source_decomposition_id"),
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
        )

    @classmethod
    def from_json(cls, json_str: str) -> GraphEdge:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Concise string representation."""
        status = "current" if self.is_current else "expired"
        return (
            f"GraphEdge({self.source_node_id[:8]}... "
            f"-[{self.relation_type}]-> "
            f"{self.target_node_id[:8]}..., {status})"
        )


# =============================================================================
# Merge Result Types
# =============================================================================


@dataclass
class MergeConflict:
    """A conflict detected during graph merge operations.

    When new knowledge contradicts or updates existing knowledge,
    a MergeConflict is created to track the issue.

    Attributes:
        conflict_id: Unique identifier for this conflict
        conflict_type: Type of conflict (CONTRADICTS, SUPERSEDES, AMBIGUOUS)
        existing_edge_id: ID of the existing edge in conflict
        new_content: The new content that conflicts
        existing_content: The existing content that's in conflict
        resolution: How the conflict was resolved (if at all)
        confidence: Confidence in the conflict detection
        detected_at: When the conflict was detected
    """

    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.AMBIGUOUS
    existing_edge_id: str = ""
    new_content: str = ""
    existing_content: str = ""
    resolution: str | None = None
    confidence: float = 1.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "existing_edge_id": self.existing_edge_id,
            "new_content": self.new_content,
            "existing_content": self.existing_content,
            "resolution": self.resolution,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeConflict:
        """Deserialize from dictionary."""
        return cls(
            conflict_id=data.get("conflict_id", str(uuid.uuid4())),
            conflict_type=ConflictType(data.get("conflict_type", "ambiguous")),
            existing_edge_id=data.get("existing_edge_id", ""),
            new_content=data.get("new_content", ""),
            existing_content=data.get("existing_content", ""),
            resolution=data.get("resolution"),
            confidence=data.get("confidence", 1.0),
            detected_at=datetime.fromisoformat(data["detected_at"]) if data.get("detected_at") else datetime.now(timezone.utc),
        )


@dataclass
class MergeResult:
    """Result of merging new knowledge into the graph.

    Tracks all changes made during a merge operation.

    Attributes:
        nodes_created: IDs of newly created nodes
        nodes_updated: IDs of nodes that were updated
        edges_created: IDs of newly created edges
        edges_invalidated: IDs of edges that were marked as expired
        conflicts: List of conflicts detected during merge
        success: Whether the merge completed successfully
        error_message: Error message if merge failed
    """

    nodes_created: list[str] = field(default_factory=list)
    nodes_updated: list[str] = field(default_factory=list)
    edges_created: list[str] = field(default_factory=list)
    edges_invalidated: list[str] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None

    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0

    @property
    def total_changes(self) -> int:
        """Count total number of changes made."""
        return (
            len(self.nodes_created)
            + len(self.nodes_updated)
            + len(self.edges_created)
            + len(self.edges_invalidated)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nodes_created": self.nodes_created,
            "nodes_updated": self.nodes_updated,
            "edges_created": self.edges_created,
            "edges_invalidated": self.edges_invalidated,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeResult:
        """Deserialize from dictionary."""
        return cls(
            nodes_created=data.get("nodes_created", []),
            nodes_updated=data.get("nodes_updated", []),
            edges_created=data.get("edges_created", []),
            edges_invalidated=data.get("edges_invalidated", []),
            conflicts=[MergeConflict.from_dict(c) for c in data.get("conflicts", [])],
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )

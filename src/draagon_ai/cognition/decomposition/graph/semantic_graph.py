"""Semantic Graph Container and Query Operations.

This module provides the SemanticGraph class, which is the primary container
for storing and querying the knowledge graph.

The graph provides:
- Node and edge storage with efficient lookup
- Query methods (find_node, get_edges, traverse, count_relations)
- Temporal filtering (get only current or historical edges)
- Serialization to/from JSON for persistence

Example:
    >>> from draagon_ai.cognition.decomposition.graph import (
    ...     SemanticGraph, GraphNode, GraphEdge, NodeType
    ... )
    >>>
    >>> # Build a graph
    >>> graph = SemanticGraph()
    >>>
    >>> # Create instance nodes (specific individuals)
    >>> doug = graph.create_node(canonical_name="Doug", node_type=NodeType.INSTANCE)
    >>> whiskers = graph.create_node(canonical_name="Whiskers", node_type=NodeType.INSTANCE)
    >>>
    >>> # Create class node and link
    >>> cat_class = graph.create_node("cat.n.01", NodeType.CLASS, synset_id="cat.n.01")
    >>> graph.create_edge(whiskers.node_id, cat_class.node_id, "instance_of")
    >>>
    >>> graph.create_edge(doug.node_id, whiskers.node_id, "owns")
    >>>
    >>> # Query the graph
    >>> pets = graph.get_outgoing_edges(doug.node_id, relation_type="owns")
    >>> print(f"Doug owns {len(pets)} pet(s)")
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from .models import (
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeRelationType,
    ConflictType,
    MergeConflict,
    MergeResult,
)
from ..identifiers import EntityType


@dataclass
class TraversalResult:
    """Result of a graph traversal operation.

    Attributes:
        nodes: Nodes found during traversal
        edges: Edges traversed
        depth: Maximum depth reached
        path: The path taken (list of node IDs)
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    depth: int = 0
    path: list[str] = field(default_factory=list)


class SemanticGraph:
    """The complete semantic memory graph.

    A graph container that stores nodes (entities, concepts, events) and
    edges (relationships) with support for:

    - Efficient node/edge lookup by ID
    - Name-based node search (case-insensitive)
    - Relation type filtering
    - Bi-temporal edge queries (current vs historical)
    - Graph traversal with depth limits
    - Cardinality counting for "how many" queries

    Thread Safety:
        This class is NOT thread-safe. External synchronization is required
        for concurrent access. For async contexts, consider using asyncio.Lock.

    Attributes:
        nodes: Dictionary of node_id -> GraphNode
        edges: Dictionary of edge_id -> GraphEdge
        _outgoing_index: Index of node_id -> list of outgoing edge IDs
        _incoming_index: Index of node_id -> list of incoming edge IDs
        _name_index: Index of lowercase name -> list of node IDs
    """

    def __init__(self) -> None:
        """Initialize an empty semantic graph."""
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

        # Indexes for efficient lookup
        self._outgoing_index: dict[str, list[str]] = defaultdict(list)
        self._incoming_index: dict[str, list[str]] = defaultdict(list)
        self._name_index: dict[str, list[str]] = defaultdict(list)

    # =========================================================================
    # Node Operations
    # =========================================================================

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph.

        If a node with the same ID already exists, it will be replaced.

        Args:
            node: The node to add
        """
        # Remove from name index if replacing
        if node.node_id in self.nodes:
            old_name = self.nodes[node.node_id].canonical_name.lower()
            if node.node_id in self._name_index.get(old_name, []):
                self._name_index[old_name].remove(node.node_id)

        self.nodes[node.node_id] = node

        # Update name index
        name_key = node.canonical_name.lower()
        if node.node_id not in self._name_index[name_key]:
            self._name_index[name_key].append(node.node_id)

    def create_node(
        self,
        canonical_name: str,
        node_type: NodeType = NodeType.INSTANCE,
        entity_type: EntityType | None = None,
        properties: dict[str, Any] | None = None,
        synset_id: str | None = None,
        wikidata_qid: str | None = None,
        confidence: float = 1.0,
        source_id: str | None = None,
    ) -> GraphNode:
        """Create and add a new node to the graph.

        Args:
            canonical_name: Human-readable name
            node_type: Type of node (INSTANCE, CLASS, etc.)
            entity_type: EntityType from Phase 0 classification
            properties: Key-value properties
            synset_id: WordNet synset ID
            wikidata_qid: Wikidata entity ID
            confidence: Confidence in this node
            source_id: Decomposition ID that created this node

        Returns:
            The newly created node
        """
        node = GraphNode(
            node_type=node_type,
            canonical_name=canonical_name,
            entity_type=entity_type,
            properties=properties or {},
            synset_id=synset_id,
            wikidata_qid=wikidata_qid,
            confidence=confidence,
            source_ids=[source_id] if source_id else [],
        )
        self.add_node(node)
        return node

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID.

        Args:
            node_id: The node's unique identifier

        Returns:
            The node, or None if not found
        """
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges from the graph.

        Args:
            node_id: The node's unique identifier

        Returns:
            True if node was removed, False if not found
        """
        if node_id not in self.nodes:
            return False

        # Remove from name index
        node = self.nodes[node_id]
        name_key = node.canonical_name.lower()
        if node_id in self._name_index.get(name_key, []):
            self._name_index[name_key].remove(node_id)

        # Remove all connected edges
        edges_to_remove = []
        for edge_id in self._outgoing_index.get(node_id, []):
            edges_to_remove.append(edge_id)
        for edge_id in self._incoming_index.get(node_id, []):
            edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove the node
        del self.nodes[node_id]

        # Clean up indexes
        if node_id in self._outgoing_index:
            del self._outgoing_index[node_id]
        if node_id in self._incoming_index:
            del self._incoming_index[node_id]

        return True

    def find_node(
        self,
        name: str,
        node_type: NodeType | None = None,
        case_sensitive: bool = False,
    ) -> GraphNode | None:
        """Find a node by canonical name.

        If multiple nodes match, returns the first one found.
        Use find_nodes() to get all matches.

        Args:
            name: The canonical name to search for
            node_type: Optional type filter
            case_sensitive: Whether to match case-sensitively

        Returns:
            The first matching node, or None if not found
        """
        nodes = self.find_nodes(name, node_type, case_sensitive)
        return nodes[0] if nodes else None

    def find_nodes(
        self,
        name: str,
        node_type: NodeType | None = None,
        case_sensitive: bool = False,
    ) -> list[GraphNode]:
        """Find all nodes matching a name.

        Args:
            name: The canonical name to search for
            node_type: Optional type filter
            case_sensitive: Whether to match case-sensitively

        Returns:
            List of matching nodes (may be empty)
        """
        results = []

        if case_sensitive:
            # Linear search for case-sensitive matching
            for node in self.nodes.values():
                if node.matches_name(name, case_sensitive=True):
                    if node_type is None or node.node_type == node_type:
                        results.append(node)
        else:
            # Use index for case-insensitive search
            name_key = name.lower()
            for node_id in self._name_index.get(name_key, []):
                node = self.nodes.get(node_id)
                if node:
                    if node_type is None or node.node_type == node_type:
                        results.append(node)

            # Also check aliases (not indexed)
            for node in self.nodes.values():
                if node.node_id not in [r.node_id for r in results]:
                    aliases = node.properties.get("aliases", [])
                    if any(alias.lower() == name_key for alias in aliases):
                        if node_type is None or node.node_type == node_type:
                            results.append(node)

        return results

    def find_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Find all nodes of a specific type.

        Args:
            node_type: The type to filter by

        Returns:
            List of matching nodes
        """
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def find_nodes_by_property(
        self,
        property_name: str,
        property_value: Any,
    ) -> list[GraphNode]:
        """Find nodes with a specific property value.

        Args:
            property_name: The property key
            property_value: The value to match

        Returns:
            List of matching nodes
        """
        return [
            n for n in self.nodes.values()
            if n.properties.get(property_name) == property_value
        ]

    def find_nodes_by_synset(
        self, synset_id: str, class_only: bool = True
    ) -> list[GraphNode]:
        """Find nodes with a specific WordNet synset ID.

        By default, returns only CLASS nodes (synset nodes), not instance nodes
        that merely reference a synset. This enables synset deduplication.

        The CLASS node represents the abstract concept (e.g., cat.n.01), while
        INSTANCE nodes are specific individuals that are INSTANCE_OF that class.

        Args:
            synset_id: The synset ID to match (e.g., "cat.n.01")
            class_only: If True (default), only return CLASS type nodes.
                       If False, return all nodes with this synset_id.

        Returns:
            List of matching nodes
        """
        if class_only:
            return [
                n for n in self.nodes.values()
                if n.synset_id == synset_id and n.node_type == NodeType.CLASS
            ]
        return [n for n in self.nodes.values() if n.synset_id == synset_id]

    # =========================================================================
    # Edge Operations
    # =========================================================================

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph.

        Args:
            edge: The edge to add

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_node_id} not found")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_node_id} not found")

        self.edges[edge.edge_id] = edge

        # Update indexes
        if edge.edge_id not in self._outgoing_index[edge.source_node_id]:
            self._outgoing_index[edge.source_node_id].append(edge.edge_id)
        if edge.edge_id not in self._incoming_index[edge.target_node_id]:
            self._incoming_index[edge.target_node_id].append(edge.edge_id)

    def create_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
        valid_from: datetime | None = None,
        confidence: float = 1.0,
        source_decomposition_id: str | None = None,
    ) -> GraphEdge:
        """Create and add a new edge to the graph.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            relation_type: Type of relationship
            properties: Edge properties
            valid_from: When this relationship became true
            confidence: Confidence in this relationship
            source_decomposition_id: Decomposition that created this

        Returns:
            The newly created edge

        Raises:
            ValueError: If source or target node doesn't exist
        """
        edge = GraphEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            properties=properties or {},
            valid_from=valid_from or datetime.now(timezone.utc),
            confidence=confidence,
            source_decomposition_id=source_decomposition_id,
        )
        self.add_edge(edge)
        return edge

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        """Get an edge by ID.

        Args:
            edge_id: The edge's unique identifier

        Returns:
            The edge, or None if not found
        """
        return self.edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph.

        Args:
            edge_id: The edge's unique identifier

        Returns:
            True if edge was removed, False if not found
        """
        if edge_id not in self.edges:
            return False

        edge = self.edges[edge_id]

        # Remove from indexes
        if edge_id in self._outgoing_index.get(edge.source_node_id, []):
            self._outgoing_index[edge.source_node_id].remove(edge_id)
        if edge_id in self._incoming_index.get(edge.target_node_id, []):
            self._incoming_index[edge.target_node_id].remove(edge_id)

        # Remove the edge
        del self.edges[edge_id]
        return True

    def get_outgoing_edges(
        self,
        node_id: str,
        relation_type: str | None = None,
        current_only: bool = True,
    ) -> list[GraphEdge]:
        """Get edges going out from a node.

        Args:
            node_id: The source node ID
            relation_type: Optional filter by relation type
            current_only: If True, only return currently valid edges

        Returns:
            List of outgoing edges
        """
        edge_ids = self._outgoing_index.get(node_id, [])
        edges = []

        for edge_id in edge_ids:
            edge = self.edges.get(edge_id)
            if edge:
                if current_only and not edge.is_current:
                    continue
                if relation_type is not None and edge.relation_type != relation_type:
                    continue
                edges.append(edge)

        return edges

    def get_incoming_edges(
        self,
        node_id: str,
        relation_type: str | None = None,
        current_only: bool = True,
    ) -> list[GraphEdge]:
        """Get edges coming into a node.

        Args:
            node_id: The target node ID
            relation_type: Optional filter by relation type
            current_only: If True, only return currently valid edges

        Returns:
            List of incoming edges
        """
        edge_ids = self._incoming_index.get(node_id, [])
        edges = []

        for edge_id in edge_ids:
            edge = self.edges.get(edge_id)
            if edge:
                if current_only and not edge.is_current:
                    continue
                if relation_type is not None and edge.relation_type != relation_type:
                    continue
                edges.append(edge)

        return edges

    def get_edges_between(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: str | None = None,
        current_only: bool = True,
    ) -> list[GraphEdge]:
        """Get edges between two specific nodes.

        Args:
            source_node_id: The source node ID
            target_node_id: The target node ID
            relation_type: Optional filter by relation type
            current_only: If True, only return currently valid edges

        Returns:
            List of edges between the nodes
        """
        outgoing = self.get_outgoing_edges(
            source_node_id, relation_type, current_only
        )
        return [e for e in outgoing if e.target_node_id == target_node_id]

    def get_edges_at_time(
        self,
        when: datetime,
        node_id: str | None = None,
    ) -> list[GraphEdge]:
        """Get edges that were valid at a specific time.

        Useful for historical queries like "What did Doug have in 2024?"

        Args:
            when: The time to query
            node_id: Optional node to filter by

        Returns:
            List of edges valid at that time
        """
        edges = []

        if node_id:
            # Get edges connected to this node
            edge_ids = set(
                self._outgoing_index.get(node_id, []) +
                self._incoming_index.get(node_id, [])
            )
            for edge_id in edge_ids:
                edge = self.edges.get(edge_id)
                if edge and edge.is_valid_at(when):
                    edges.append(edge)
        else:
            # Get all edges valid at that time
            for edge in self.edges.values():
                if edge.is_valid_at(when):
                    edges.append(edge)

        return edges

    def invalidate_edge(
        self,
        edge_id: str,
        when: datetime | None = None,
    ) -> bool:
        """Mark an edge as no longer valid.

        This implements temporal versioning - the edge remains in the graph
        but is marked with a valid_to timestamp.

        Args:
            edge_id: The edge to invalidate
            when: When it became invalid (default: now)

        Returns:
            True if edge was invalidated, False if not found
        """
        edge = self.edges.get(edge_id)
        if not edge:
            return False

        edge.invalidate(when)
        return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    def count_relations(
        self,
        node_id: str,
        relation_type: str,
        target_type: NodeType | None = None,
        current_only: bool = True,
    ) -> int:
        """Count relations of a specific type from a node.

        Useful for "how many" queries like "How many cats does Doug have?"

        Args:
            node_id: The source node ID
            relation_type: The relation type to count
            target_type: Optional filter by target node type
            current_only: If True, only count currently valid edges

        Returns:
            Number of matching relations
        """
        edges = self.get_outgoing_edges(node_id, relation_type, current_only)

        if target_type is None:
            return len(edges)

        # Filter by target node type
        count = 0
        for edge in edges:
            target_node = self.nodes.get(edge.target_node_id)
            if target_node and target_node.node_type == target_type:
                count += 1

        return count

    def get_related_nodes(
        self,
        node_id: str,
        relation_type: str | None = None,
        direction: str = "outgoing",
        current_only: bool = True,
    ) -> list[GraphNode]:
        """Get nodes related to a given node.

        Args:
            node_id: The starting node ID
            relation_type: Optional filter by relation type
            direction: "outgoing", "incoming", or "both"
            current_only: If True, only follow currently valid edges

        Returns:
            List of related nodes
        """
        related_ids = set()

        if direction in ("outgoing", "both"):
            edges = self.get_outgoing_edges(node_id, relation_type, current_only)
            for edge in edges:
                related_ids.add(edge.target_node_id)

        if direction in ("incoming", "both"):
            edges = self.get_incoming_edges(node_id, relation_type, current_only)
            for edge in edges:
                related_ids.add(edge.source_node_id)

        return [self.nodes[nid] for nid in related_ids if nid in self.nodes]

    def traverse(
        self,
        start_node_id: str,
        relation_path: list[str] | None = None,
        max_depth: int = 3,
        current_only: bool = True,
        node_filter: Callable[[GraphNode], bool] | None = None,
    ) -> TraversalResult:
        """Traverse the graph following a relation path.

        Args:
            start_node_id: Where to start traversal
            relation_path: List of relation types to follow (e.g., ["has", "name"])
                          If None, follows any relation
            max_depth: Maximum depth to traverse
            current_only: If True, only follow currently valid edges
            node_filter: Optional filter function for nodes to include

        Returns:
            TraversalResult with found nodes, edges, and path
        """
        result = TraversalResult()
        result.path.append(start_node_id)

        if start_node_id not in self.nodes:
            return result

        visited = set([start_node_id])
        current_nodes = [start_node_id]

        for depth in range(max_depth):
            if relation_path and depth >= len(relation_path):
                break

            next_nodes = []
            relation_filter = relation_path[depth] if relation_path else None

            for node_id in current_nodes:
                edges = self.get_outgoing_edges(node_id, relation_filter, current_only)

                for edge in edges:
                    target_id = edge.target_node_id
                    if target_id in visited:
                        continue

                    target_node = self.nodes.get(target_id)
                    if target_node is None:
                        continue

                    if node_filter and not node_filter(target_node):
                        continue

                    visited.add(target_id)
                    result.nodes.append(target_node)
                    result.edges.append(edge)
                    next_nodes.append(target_id)

            if not next_nodes:
                break

            current_nodes = next_nodes
            result.depth = depth + 1

        result.path.extend([n.node_id for n in result.nodes])
        return result

    def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        current_only: bool = True,
    ) -> list[GraphEdge] | None:
        """Find a path between two nodes.

        Uses breadth-first search to find the shortest path.

        Args:
            start_node_id: Starting node ID
            end_node_id: Target node ID
            max_depth: Maximum path length
            current_only: If True, only follow currently valid edges

        Returns:
            List of edges forming the path, or None if no path exists
        """
        if start_node_id == end_node_id:
            return []

        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return None

        # BFS
        visited = set([start_node_id])
        queue: list[tuple[str, list[GraphEdge]]] = [(start_node_id, [])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            edges = self.get_outgoing_edges(current_id, None, current_only)

            for edge in edges:
                target_id = edge.target_node_id
                new_path = path + [edge]

                if target_id == end_node_id:
                    return new_path

                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, new_path))

        return None

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.edges)

    @property
    def current_edge_count(self) -> int:
        """Get the number of currently valid edges."""
        return sum(1 for e in self.edges.values() if e.is_current)

    def get_node_type_counts(self) -> dict[NodeType, int]:
        """Get counts of nodes by type.

        Returns:
            Dictionary of NodeType -> count
        """
        counts: dict[NodeType, int] = {}
        for node in self.nodes.values():
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts

    def get_relation_type_counts(self, current_only: bool = True) -> dict[str, int]:
        """Get counts of edges by relation type.

        Args:
            current_only: If True, only count currently valid edges

        Returns:
            Dictionary of relation_type -> count
        """
        counts: dict[str, int] = {}
        for edge in self.edges.values():
            if current_only and not edge.is_current:
                continue
            counts[edge.relation_type] = counts.get(edge.relation_type, 0) + 1
        return counts

    # =========================================================================
    # Iteration
    # =========================================================================

    def iter_nodes(self, node_type: NodeType | None = None) -> Iterator[GraphNode]:
        """Iterate over nodes.

        Args:
            node_type: Optional filter by type

        Yields:
            GraphNode instances
        """
        for node in self.nodes.values():
            if node_type is None or node.node_type == node_type:
                yield node

    def iter_edges(
        self,
        relation_type: str | None = None,
        current_only: bool = True,
    ) -> Iterator[GraphEdge]:
        """Iterate over edges.

        Args:
            relation_type: Optional filter by relation type
            current_only: If True, only yield currently valid edges

        Yields:
            GraphEdge instances
        """
        for edge in self.edges.values():
            if current_only and not edge.is_current:
                continue
            if relation_type is None or edge.relation_type == relation_type:
                yield edge

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dictionary.

        Returns:
            Dictionary representation of the entire graph
        """
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph to JSON.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticGraph:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed SemanticGraph
        """
        graph = cls()

        # Add nodes first
        for node_id, node_data in data.get("nodes", {}).items():
            node = GraphNode.from_dict(node_data)
            graph.add_node(node)

        # Then add edges
        for edge_id, edge_data in data.get("edges", {}).items():
            edge = GraphEdge.from_dict(edge_data)
            # Verify nodes exist before adding edge
            if edge.source_node_id in graph.nodes and edge.target_node_id in graph.nodes:
                graph.add_edge(edge)

        return graph

    @classmethod
    def from_json(cls, json_str: str) -> SemanticGraph:
        """Deserialize from JSON.

        Args:
            json_str: JSON string representation

        Returns:
            Reconstructed SemanticGraph
        """
        return cls.from_dict(json.loads(json_str))

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self._outgoing_index.clear()
        self._incoming_index.clear()
        self._name_index.clear()

    def copy(self) -> SemanticGraph:
        """Create a deep copy of the graph.

        Returns:
            A new SemanticGraph with copied nodes and edges
        """
        return SemanticGraph.from_dict(self.to_dict())

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SemanticGraph(nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"current_edges={self.current_edge_count})"
        )

    def __len__(self) -> int:
        """Return the number of nodes."""
        return self.node_count

    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID exists in the graph."""
        return node_id in self.nodes

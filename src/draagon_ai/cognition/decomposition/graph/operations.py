"""Graph update operations for incremental knowledge integration.

This module handles the evolution of knowledge graphs as new information arrives.
Key scenarios:

1. Single → Collection Promotion:
   "Doug has a cat" → "Doug got another cat"
   Direct OWNS edge becomes COLLECTION with MEMBER_OF edges

2. Anonymous → Named Resolution:
   "[Doug's cat]" → "Doug's cat is named Whiskers"
   Anonymous instance gets a proper name

3. Conflicting Information:
   "Doug has 2 cats" → "Doug has 3 cats"
   Temporal edge invalidation + new edge creation

4. Cardinality Updates:
   Collection count property updates as members change
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import GraphNode, GraphEdge, NodeType, EdgeRelationType
from .semantic_graph import SemanticGraph

if TYPE_CHECKING:
    pass


@dataclass
class UpdateResult:
    """Result of a graph update operation."""

    success: bool = True
    operation: str = ""
    nodes_created: list[str] = field(default_factory=list)
    nodes_modified: list[str] = field(default_factory=list)
    edges_created: list[str] = field(default_factory=list)
    edges_invalidated: list[str] = field(default_factory=list)
    collection_id: str | None = None  # If a collection was created/used
    message: str = ""


class GraphUpdater:
    """Handles incremental updates to the semantic graph.

    This class manages the evolution of knowledge as new information arrives,
    including:
    - Promoting single relations to collections
    - Resolving anonymous instances to named ones
    - Handling conflicting information with temporal edges
    """

    def __init__(self, graph: SemanticGraph):
        self.graph = graph

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        use_collections: bool = True,
        collection_threshold: int = 2,
        confidence: float = 1.0,
        source_decomposition_id: str | None = None,
    ) -> UpdateResult:
        """Add a relation, promoting to collection if needed.

        When use_collections=True and this would be the Nth relation of the same
        type from source (where N >= collection_threshold), we:
        1. Create a COLLECTION node if not exists
        2. Migrate existing targets into the collection
        3. Add new target to collection

        Args:
            source_id: Source node ID (e.g., Doug)
            target_id: Target node ID (e.g., new cat)
            relation_type: Relation type (e.g., "owns")
            use_collections: Whether to use collections for multiple relations
            collection_threshold: Number of relations before creating collection
            confidence: Confidence in this relation
            source_decomposition_id: Provenance tracking

        Returns:
            UpdateResult with operation details
        """
        result = UpdateResult(operation="add_relation")

        source_node = self.graph.get_node(source_id)
        target_node = self.graph.get_node(target_id)

        if not source_node or not target_node:
            result.success = False
            result.message = "Source or target node not found"
            return result

        # Get existing relations of this type from source
        existing_edges = self.graph.get_outgoing_edges(
            source_id, relation_type=relation_type
        )
        current_targets = [e.target_node_id for e in existing_edges if e.is_current]

        # Check if any existing target is already a collection
        existing_collection = None
        for tid in current_targets:
            node = self.graph.get_node(tid)
            if node and node.node_type == NodeType.COLLECTION:
                existing_collection = node
                break

        if existing_collection:
            # Already have a collection - add new target as member
            return self._add_to_collection(
                existing_collection.node_id,
                target_id,
                confidence=confidence,
                source_decomposition_id=source_decomposition_id,
            )

        if not use_collections or len(current_targets) < collection_threshold - 1:
            # Simple case: just add the edge
            edge = self.graph.create_edge(
                source_id,
                target_id,
                relation_type,
                confidence=confidence,
                source_decomposition_id=source_decomposition_id,
            )
            result.edges_created.append(edge.edge_id)
            result.message = f"Added direct {relation_type} edge"
            return result

        # Need to promote to collection
        return self._promote_to_collection(
            source_id=source_id,
            relation_type=relation_type,
            existing_targets=current_targets,
            new_target_id=target_id,
            confidence=confidence,
            source_decomposition_id=source_decomposition_id,
        )

    def _promote_to_collection(
        self,
        source_id: str,
        relation_type: str,
        existing_targets: list[str],
        new_target_id: str,
        confidence: float,
        source_decomposition_id: str | None,
    ) -> UpdateResult:
        """Promote direct relations to a collection.

        Steps:
        1. Create COLLECTION node
        2. Invalidate existing direct edges
        3. Create edge from source to collection
        4. Create MEMBER_OF edges from existing targets to collection
        5. Create MEMBER_OF edge from new target to collection
        """
        result = UpdateResult(operation="promote_to_collection")

        source_node = self.graph.get_node(source_id)
        if not source_node:
            result.success = False
            result.message = "Source node not found"
            return result

        # Determine collection name based on relation and target types
        # e.g., "Doug's cats" or "Doug's pets"
        target_sample = self.graph.get_node(existing_targets[0]) if existing_targets else None
        target_type = "items"
        if target_sample:
            # Try to get type from INSTANCE_OF edge
            instance_edges = self.graph.get_outgoing_edges(
                target_sample.node_id, relation_type="instance_of"
            )
            if instance_edges:
                class_node = self.graph.get_node(instance_edges[0].target_node_id)
                if class_node:
                    # Use synset name without version (e.g., "cat" from "cat.n.01")
                    target_type = class_node.canonical_name.split(".")[0] + "s"

        collection_name = f"{source_node.canonical_name}'s {target_type}"

        # 1. Create collection node
        collection = self.graph.create_node(
            canonical_name=collection_name,
            node_type=NodeType.COLLECTION,
            properties={
                "owner_id": source_id,
                "relation_type": relation_type,
                "count": len(existing_targets) + 1,
            },
            confidence=confidence,
            source_id=source_decomposition_id,
        )
        result.nodes_created.append(collection.node_id)
        result.collection_id = collection.node_id

        # 2. Invalidate existing direct edges
        for edge in self.graph.get_outgoing_edges(source_id, relation_type=relation_type):
            if edge.is_current and edge.target_node_id in existing_targets:
                edge.invalidate()
                result.edges_invalidated.append(edge.edge_id)

        # 3. Create edge from source to collection
        source_to_collection = self.graph.create_edge(
            source_id,
            collection.node_id,
            relation_type,
            confidence=confidence,
            source_decomposition_id=source_decomposition_id,
        )
        result.edges_created.append(source_to_collection.edge_id)

        # 4. Create MEMBER_OF edges from existing targets
        for target_id in existing_targets:
            member_edge = self.graph.create_edge(
                target_id,
                collection.node_id,
                EdgeRelationType.MEMBER_OF,
                confidence=confidence,
                source_decomposition_id=source_decomposition_id,
            )
            result.edges_created.append(member_edge.edge_id)

        # 5. Add new target to collection
        new_member_edge = self.graph.create_edge(
            new_target_id,
            collection.node_id,
            EdgeRelationType.MEMBER_OF,
            confidence=confidence,
            source_decomposition_id=source_decomposition_id,
        )
        result.edges_created.append(new_member_edge.edge_id)

        result.message = (
            f"Promoted to collection '{collection_name}' with "
            f"{len(existing_targets) + 1} members"
        )
        return result

    def _add_to_collection(
        self,
        collection_id: str,
        target_id: str,
        confidence: float,
        source_decomposition_id: str | None,
    ) -> UpdateResult:
        """Add a new member to an existing collection."""
        result = UpdateResult(operation="add_to_collection")

        collection = self.graph.get_node(collection_id)
        if not collection:
            result.success = False
            result.message = "Collection not found"
            return result

        # Check if target is already a member
        existing_member_edges = self.graph.get_incoming_edges(
            collection_id, relation_type=EdgeRelationType.MEMBER_OF
        )
        current_members = [e.source_node_id for e in existing_member_edges if e.is_current]

        if target_id in current_members:
            result.message = "Target is already a member of collection"
            return result

        # Add MEMBER_OF edge
        member_edge = self.graph.create_edge(
            target_id,
            collection_id,
            EdgeRelationType.MEMBER_OF,
            confidence=confidence,
            source_decomposition_id=source_decomposition_id,
        )
        result.edges_created.append(member_edge.edge_id)

        # Update collection count
        collection.properties["count"] = len(current_members) + 1
        collection.updated_at = datetime.now(timezone.utc)
        result.nodes_modified.append(collection_id)

        result.collection_id = collection_id
        result.message = f"Added member to collection (now {len(current_members) + 1} members)"
        return result

    def resolve_anonymous(
        self,
        anonymous_id: str,
        name: str,
        additional_properties: dict | None = None,
    ) -> UpdateResult:
        """Resolve an anonymous instance to a named one.

        When we learn "[Doug's cat]" is named "Whiskers", we update the node.

        Args:
            anonymous_id: ID of the anonymous node
            name: The resolved name
            additional_properties: Any additional properties to add

        Returns:
            UpdateResult with operation details
        """
        result = UpdateResult(operation="resolve_anonymous")

        node = self.graph.get_node(anonymous_id)
        if not node:
            result.success = False
            result.message = "Anonymous node not found"
            return result

        # Update the node
        old_name = node.canonical_name
        node.canonical_name = name
        node.properties["resolved_from"] = old_name
        node.properties["is_anonymous"] = False
        node.updated_at = datetime.now(timezone.utc)

        if additional_properties:
            for key, value in additional_properties.items():
                node.properties[key] = value

        result.nodes_modified.append(anonymous_id)
        result.message = f"Resolved '{old_name}' to '{name}'"
        return result

    def update_cardinality(
        self,
        source_id: str,
        relation_type: str,
        new_count: int,
        confidence: float = 1.0,
    ) -> UpdateResult:
        """Update the cardinality of a relation.

        When we learn "Doug has 3 cats" but the collection only has 2,
        we can create placeholder anonymous instances.

        Args:
            source_id: Source node ID
            relation_type: Relation type
            new_count: The new cardinality
            confidence: Confidence in this count

        Returns:
            UpdateResult with operation details
        """
        result = UpdateResult(operation="update_cardinality")

        # Find existing collection
        existing_edges = self.graph.get_outgoing_edges(
            source_id, relation_type=relation_type
        )

        collection = None
        for edge in existing_edges:
            if edge.is_current:
                node = self.graph.get_node(edge.target_node_id)
                if node and node.node_type == NodeType.COLLECTION:
                    collection = node
                    break

        if not collection:
            result.success = False
            result.message = "No collection found for this relation"
            return result

        current_count = collection.properties.get("count", 0)

        if new_count == current_count:
            result.message = "Count already matches"
            return result

        if new_count < current_count:
            # Cardinality decreased - just update count, don't remove members
            # (We don't know which ones to remove)
            collection.properties["count"] = new_count
            collection.properties["count_confidence"] = confidence
            collection.updated_at = datetime.now(timezone.utc)
            result.nodes_modified.append(collection.node_id)
            result.message = f"Updated count from {current_count} to {new_count}"
            return result

        # Cardinality increased - create anonymous placeholders
        source_node = self.graph.get_node(source_id)
        source_name = source_node.canonical_name if source_node else "unknown"

        # Determine type from existing members
        member_edges = self.graph.get_incoming_edges(
            collection.node_id, relation_type=EdgeRelationType.MEMBER_OF
        )
        instance_type = "item"
        for edge in member_edges:
            if edge.is_current:
                member = self.graph.get_node(edge.source_node_id)
                if member:
                    type_edges = self.graph.get_outgoing_edges(
                        member.node_id, relation_type="instance_of"
                    )
                    if type_edges:
                        class_node = self.graph.get_node(type_edges[0].target_node_id)
                        if class_node:
                            instance_type = class_node.canonical_name.split(".")[0]
                            break

        # Create anonymous instances for the difference
        for i in range(new_count - current_count):
            anon_node = self.graph.create_node(
                canonical_name=f"[{source_name}'s {instance_type} #{current_count + i + 1}]",
                node_type=NodeType.INSTANCE,
                properties={
                    "is_anonymous": True,
                    "owner_ref": source_id,
                    "placeholder_index": current_count + i + 1,
                },
                confidence=confidence * 0.8,  # Lower confidence for inferred instances
            )
            result.nodes_created.append(anon_node.node_id)

            # Add to collection
            member_edge = self.graph.create_edge(
                anon_node.node_id,
                collection.node_id,
                EdgeRelationType.MEMBER_OF,
                confidence=confidence * 0.8,
            )
            result.edges_created.append(member_edge.edge_id)

        # Update collection count
        collection.properties["count"] = new_count
        collection.properties["count_confidence"] = confidence
        collection.updated_at = datetime.now(timezone.utc)
        result.nodes_modified.append(collection.node_id)
        result.collection_id = collection.node_id

        result.message = (
            f"Updated count from {current_count} to {new_count}, "
            f"created {new_count - current_count} anonymous placeholders"
        )
        return result

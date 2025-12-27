"""Temporal Cognitive Graph - the core memory architecture for AGI-Lite.

The TCG is a hybrid vector-graph architecture that:
1. Uses Qdrant for vector similarity search (fast semantic retrieval)
2. Emulates graph relationships through metadata and edge storage
3. Provides bi-temporal tracking for temporal reasoning
4. Supports hierarchical scopes for multi-tenant memory isolation

This follows the Netflix pattern: "Simpler to emulate graph-like relationships
in existing data storage systems rather than adopting specialized graph infrastructure."

Based on research from:
- Zep/Graphiti: Bi-temporal knowledge graphs
- Mem0: Hybrid vector-graph architecture
- Netflix: Graph emulation at scale
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator
import logging

from .temporal_nodes import (
    TemporalNode,
    TemporalEdge,
    NodeType,
    EdgeType,
    MemoryLayer,
)
from .scopes import (
    HierarchicalScope,
    ScopeType,
    Permission,
    ScopeRegistry,
    get_scope_registry,
)


class PermissionDeniedError(Exception):
    """Raised when an operation is attempted without sufficient permissions."""

    def __init__(self, operation: str, scope_id: str, agent_id: str, user_id: str | None = None):
        self.operation = operation
        self.scope_id = scope_id
        self.agent_id = agent_id
        self.user_id = user_id
        user_part = f", user={user_id}" if user_id else ""
        super().__init__(
            f"Permission denied: {operation} on scope '{scope_id}' for agent '{agent_id}'{user_part}"
        )

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    """Result from a graph search operation."""

    node: TemporalNode
    score: float  # Similarity score (0-1)
    edges: list[TemporalEdge] = field(default_factory=list)  # Connected edges
    path_length: int = 0  # Hops from query (for traversal)


@dataclass
class GraphTraversalResult:
    """Result from a multi-hop graph traversal."""

    nodes: list[TemporalNode]
    edges: list[TemporalEdge]
    paths: list[list[str]]  # Paths as lists of node IDs


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation.

    The graph doesn't know how to generate embeddings - that's
    provided by the host application.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...


class TemporalCognitiveGraph:
    """The core temporal cognitive graph.

    This is an in-memory implementation for Phase C.1.
    Production would use Qdrant backend (see QdrantTemporalGraph).

    The graph maintains:
    - Nodes: Temporal nodes with bi-temporal tracking
    - Edges: Relationships between nodes
    - Scopes: Hierarchical access control

    Example:
        graph = TemporalCognitiveGraph(embedding_provider=my_embedder)

        # Store a fact
        node = await graph.add_node(
            content="Doug's birthday is March 15",
            node_type=NodeType.FACT,
            scope_id="user:assistant:alice",
            entities=["Doug", "birthday", "March 15"],
        )

        # Search semantically
        results = await graph.search("When is Doug's birthday?", limit=5)

        # Traverse relationships
        related = await graph.traverse(node.node_id, max_hops=2)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        scope_registry: ScopeRegistry | None = None,
    ):
        """Initialize the graph.

        Args:
            embedding_provider: Provider for generating embeddings
            scope_registry: Registry for scope management
        """
        self._embedder = embedding_provider
        self._scopes = scope_registry or get_scope_registry()

        # In-memory storage (Phase C.1)
        self._nodes: dict[str, TemporalNode] = {}
        self._edges: dict[str, TemporalEdge] = {}

        # Indexes for efficient lookup
        self._nodes_by_scope: dict[str, set[str]] = {}  # scope_id -> node_ids
        self._nodes_by_type: dict[NodeType, set[str]] = {}  # type -> node_ids
        self._nodes_by_layer: dict[MemoryLayer, set[str]] = {}  # layer -> node_ids
        self._edges_by_source: dict[str, set[str]] = {}  # source_id -> edge_ids
        self._edges_by_target: dict[str, set[str]] = {}  # target_id -> edge_ids

    def _check_permission(
        self,
        scope_id: str,
        agent_id: str,
        user_id: str | None,
        permission: Permission,
        operation: str,
        enforce: bool = True,
    ) -> bool:
        """Check if an agent/user has permission on a scope.

        Uses the scope registry's permission inheritance.

        Args:
            scope_id: Scope to check
            agent_id: Agent requesting access
            user_id: Optional user within agent
            permission: Required permission
            operation: Description for error message
            enforce: If True, raise PermissionDeniedError on failure

        Returns:
            True if permission granted

        Raises:
            PermissionDeniedError: If enforce=True and permission denied
        """
        has_permission = self._scopes.check_permission(
            scope_id, agent_id, user_id, permission
        )

        if not has_permission and enforce:
            raise PermissionDeniedError(operation, scope_id, agent_id, user_id)

        return has_permission

    # =========================================================================
    # Node Operations
    # =========================================================================

    async def add_node(
        self,
        content: str,
        node_type: NodeType,
        scope_id: str = "agent:default",
        *,
        entities: list[str] | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        enforce_permissions: bool = False,
    ) -> TemporalNode:
        """Add a new node to the graph.

        Args:
            content: Node content
            node_type: Type of node
            scope_id: Hierarchical scope
            entities: Extracted entities
            confidence: Certainty level
            importance: Retrieval priority
            event_time: When the event occurred
            metadata: Additional metadata
            embedding: Pre-computed embedding (computed if None)
            agent_id: Agent creating the node (for permission check)
            user_id: User within agent (for permission check)
            enforce_permissions: If True, check WRITE permission before adding

        Returns:
            The created TemporalNode

        Raises:
            PermissionDeniedError: If enforce_permissions=True and WRITE denied
        """
        # Check permission if enforcement is enabled
        if enforce_permissions and agent_id:
            self._check_permission(
                scope_id, agent_id, user_id, Permission.WRITE, "add_node"
            )

        # Generate embedding if not provided
        if embedding is None and self._embedder:
            embedding = await self._embedder.embed(content)

        # Create node
        node = TemporalNode(
            content=content,
            node_type=node_type,
            scope_id=scope_id,
            entities=entities or [],
            confidence=confidence,
            importance=importance,
            event_time=event_time or datetime.now(),
            metadata=metadata or {},
            embedding=embedding,
        )

        # Store node
        self._nodes[node.node_id] = node

        # Update indexes
        self._index_node(node)

        logger.debug(f"Added node {node.node_id}: {content[:50]}...")
        return node

    def _index_node(self, node: TemporalNode) -> None:
        """Update indexes for a node."""
        # Scope index
        if node.scope_id not in self._nodes_by_scope:
            self._nodes_by_scope[node.scope_id] = set()
        self._nodes_by_scope[node.scope_id].add(node.node_id)

        # Type index
        if node.node_type not in self._nodes_by_type:
            self._nodes_by_type[node.node_type] = set()
        self._nodes_by_type[node.node_type].add(node.node_id)

        # Layer index
        if node.layer not in self._nodes_by_layer:
            self._nodes_by_layer[node.layer] = set()
        self._nodes_by_layer[node.layer].add(node.node_id)

    def _unindex_node(self, node: TemporalNode) -> None:
        """Remove node from indexes."""
        if node.scope_id in self._nodes_by_scope:
            self._nodes_by_scope[node.scope_id].discard(node.node_id)

        if node.node_type in self._nodes_by_type:
            self._nodes_by_type[node.node_type].discard(node.node_id)

        if node.layer in self._nodes_by_layer:
            self._nodes_by_layer[node.layer].discard(node.node_id)

    async def get_node(self, node_id: str) -> TemporalNode | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Node or None if not found
        """
        return self._nodes.get(node_id)

    async def update_node(
        self,
        node_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalNode | None:
        """Update an existing node.

        Args:
            node_id: Node to update
            content: New content (regenerates embedding)
            confidence: New confidence
            importance: New importance
            metadata: Metadata to merge

        Returns:
            Updated node or None if not found
        """
        node = self._nodes.get(node_id)
        if not node:
            return None

        # Update content and regenerate embedding
        if content is not None:
            node.content = content
            if self._embedder:
                node.embedding = await self._embedder.embed(content)

        if confidence is not None:
            node.confidence = confidence

        if importance is not None:
            node.importance = importance

        if metadata is not None:
            node.metadata.update(metadata)

        node.updated_at = datetime.now()
        return node

    async def delete_node(
        self,
        node_id: str,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        enforce_permissions: bool = False,
    ) -> bool:
        """Delete a node and its edges.

        Args:
            node_id: Node to delete
            agent_id: Agent deleting the node (for permission check)
            user_id: User within agent (for permission check)
            enforce_permissions: If True, check DELETE permission before deleting

        Returns:
            True if deleted, False if not found

        Raises:
            PermissionDeniedError: If enforce_permissions=True and DELETE denied
        """
        node = self._nodes.get(node_id)
        if not node:
            return False

        # Check permission if enforcement is enabled
        if enforce_permissions and agent_id:
            self._check_permission(
                node.scope_id, agent_id, user_id, Permission.DELETE, "delete_node"
            )

        # Remove from indexes
        self._unindex_node(node)

        # Delete connected edges
        edge_ids_to_delete = set()
        for edge_id, edge in self._edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edge_ids_to_delete.add(edge_id)

        for edge_id in edge_ids_to_delete:
            await self.delete_edge(edge_id)

        # Delete node
        del self._nodes[node_id]
        return True

    async def supersede_node(
        self,
        old_node_id: str,
        new_content: str,
        **kwargs: Any,
    ) -> TemporalNode | None:
        """Create a new node that supersedes an existing one.

        This is the preferred way to handle updates to factual knowledge,
        as it preserves provenance.

        Args:
            old_node_id: Node being superseded
            new_content: Content for the new node
            **kwargs: Additional args for new node

        Returns:
            New node or None if old node not found
        """
        old_node = self._nodes.get(old_node_id)
        if not old_node:
            return None

        # Create new node with reference to old
        new_node = await self.add_node(
            content=new_content,
            node_type=old_node.node_type,
            scope_id=old_node.scope_id,
            entities=kwargs.get("entities", old_node.entities),
            confidence=kwargs.get("confidence", old_node.confidence),
            importance=kwargs.get("importance", old_node.importance),
            metadata=kwargs.get("metadata", {}),
        )

        # Link provenance
        new_node.supersedes.append(old_node_id)
        old_node.supersede(new_node.node_id)

        # Create supersedes edge
        await self.add_edge(
            source_id=new_node.node_id,
            target_id=old_node_id,
            edge_type=EdgeType.SUPERSEDES,
        )

        return new_node

    # =========================================================================
    # Edge Operations
    # =========================================================================

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        *,
        label: str | None = None,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalEdge | None:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            label: Optional specific label
            weight: Relationship strength
            confidence: Certainty of relationship
            metadata: Additional metadata

        Returns:
            Created edge or None if nodes not found
        """
        # Validate nodes exist
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning(f"Cannot create edge: nodes not found {source_id} -> {target_id}")
            return None

        edge = TemporalEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=label,
            weight=weight,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Store edge
        self._edges[edge.edge_id] = edge

        # Update indexes
        if source_id not in self._edges_by_source:
            self._edges_by_source[source_id] = set()
        self._edges_by_source[source_id].add(edge.edge_id)

        if target_id not in self._edges_by_target:
            self._edges_by_target[target_id] = set()
        self._edges_by_target[target_id].add(edge.edge_id)

        return edge

    async def get_edge(self, edge_id: str) -> TemporalEdge | None:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge.

        Args:
            edge_id: Edge to delete

        Returns:
            True if deleted
        """
        edge = self._edges.get(edge_id)
        if not edge:
            return False

        # Remove from indexes
        if edge.source_id in self._edges_by_source:
            self._edges_by_source[edge.source_id].discard(edge_id)
        if edge.target_id in self._edges_by_target:
            self._edges_by_target[edge.target_id].discard(edge_id)

        del self._edges[edge_id]
        return True

    async def get_edges_from(self, node_id: str) -> list[TemporalEdge]:
        """Get all edges originating from a node."""
        edge_ids = self._edges_by_source.get(node_id, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    async def get_edges_to(self, node_id: str) -> list[TemporalEdge]:
        """Get all edges pointing to a node."""
        edge_ids = self._edges_by_target.get(node_id, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        *,
        scope_ids: list[str] | None = None,
        node_types: list[NodeType] | None = None,
        layers: list[MemoryLayer] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
        include_superseded: bool = False,
        include_edges: bool = False,
        include_ancestor_scopes: bool = True,
    ) -> list[GraphSearchResult]:
        """Search for nodes by semantic similarity.

        Args:
            query: Search query
            scope_ids: Filter by scopes (None = all)
            node_types: Filter by node types
            layers: Filter by memory layers
            limit: Maximum results
            min_score: Minimum similarity score
            include_superseded: Include superseded nodes
            include_edges: Include connected edges in results
            include_ancestor_scopes: If True, also include nodes from ancestor scopes.
                For example, searching with scope_id="user:assistant:alice" will also
                return nodes from "agent:assistant", "context:home", and "world:global".
                This implements hierarchical scope inheritance. (default: True)

        Returns:
            List of search results sorted by score
        """
        if not self._embedder:
            logger.warning("No embedding provider - returning empty results")
            return []

        # Get query embedding
        query_embedding = await self._embedder.embed(query)

        # Build candidate set based on filters
        candidates = self._get_candidates(
            scope_ids=scope_ids,
            node_types=node_types,
            layers=layers,
            include_superseded=include_superseded,
            include_ancestor_scopes=include_ancestor_scopes,
        )

        # Score candidates
        results = []
        for node_id in candidates:
            node = self._nodes[node_id]
            if node.embedding is None:
                continue

            score = self._cosine_similarity(query_embedding, node.embedding)
            if score >= min_score:
                edges = []
                if include_edges:
                    edges = await self.get_edges_from(node_id)
                    edges.extend(await self.get_edges_to(node_id))

                results.append(GraphSearchResult(
                    node=node,
                    score=score,
                    edges=edges,
                ))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _get_candidates(
        self,
        scope_ids: list[str] | None,
        node_types: list[NodeType] | None,
        layers: list[MemoryLayer] | None,
        include_superseded: bool,
        include_ancestor_scopes: bool = True,
    ) -> set[str]:
        """Get candidate node IDs based on filters.

        Args:
            scope_ids: Filter by scopes (None = all)
            node_types: Filter by node types
            layers: Filter by memory layers
            include_superseded: Include superseded nodes
            include_ancestor_scopes: If True, also include nodes from ancestor scopes
                (e.g., searching user:assistant:alice also includes agent:assistant and world:global)

        Returns:
            Set of candidate node IDs
        """
        # Start with all nodes or scope-filtered
        if scope_ids:
            candidates = set()
            for scope_id in scope_ids:
                # Include nodes from this scope
                candidates.update(self._nodes_by_scope.get(scope_id, set()))

                # Include nodes from ancestor scopes (hierarchical inheritance)
                if include_ancestor_scopes:
                    ancestors = self._scopes.get_ancestors(scope_id)
                    for ancestor in ancestors:
                        candidates.update(self._nodes_by_scope.get(ancestor.scope_id, set()))
        else:
            candidates = set(self._nodes.keys())

        # Filter by type
        if node_types:
            type_nodes = set()
            for nt in node_types:
                type_nodes.update(self._nodes_by_type.get(nt, set()))
            candidates &= type_nodes

        # Filter by layer
        if layers:
            layer_nodes = set()
            for layer in layers:
                layer_nodes.update(self._nodes_by_layer.get(layer, set()))
            candidates &= layer_nodes

        # Filter superseded
        if not include_superseded:
            candidates = {
                nid for nid in candidates
                if self._nodes[nid].is_current
            }

        return candidates

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    async def traverse(
        self,
        start_node_id: str,
        *,
        max_hops: int = 2,
        edge_types: list[EdgeType] | None = None,
        direction: str = "both",  # "out", "in", "both"
    ) -> GraphTraversalResult:
        """Traverse the graph from a starting node.

        Args:
            start_node_id: Starting node
            max_hops: Maximum traversal depth
            edge_types: Filter by edge types
            direction: Traversal direction

        Returns:
            Traversal result with nodes, edges, and paths
        """
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        all_paths: list[list[str]] = []

        async def _traverse(node_id: str, path: list[str], depth: int):
            if depth > max_hops or node_id in visited_nodes:
                return

            visited_nodes.add(node_id)
            current_path = path + [node_id]

            # Get connected edges
            edges = []
            if direction in ("out", "both"):
                edges.extend(await self.get_edges_from(node_id))
            if direction in ("in", "both"):
                edges.extend(await self.get_edges_to(node_id))

            # Filter by edge type
            if edge_types:
                edges = [e for e in edges if e.edge_type in edge_types]

            # Recurse
            for edge in edges:
                if edge.edge_id in visited_edges:
                    continue

                visited_edges.add(edge.edge_id)

                # Get next node
                next_node_id = edge.target_id if edge.source_id == node_id else edge.source_id
                await _traverse(next_node_id, current_path, depth + 1)

            if depth > 0:  # Don't add the start node as a path
                all_paths.append(current_path)

        await _traverse(start_node_id, [], 0)

        # Build result
        nodes = [self._nodes[nid] for nid in visited_nodes if nid in self._nodes]
        edges = [self._edges[eid] for eid in visited_edges if eid in self._edges]

        return GraphTraversalResult(
            nodes=nodes,
            edges=edges,
            paths=all_paths,
        )

    async def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_hops: int = 5,
    ) -> list[str] | None:
        """Find a path between two nodes.

        Uses breadth-first search.

        Args:
            start_node_id: Starting node
            end_node_id: Target node
            max_hops: Maximum path length

        Returns:
            Path as list of node IDs, or None if no path
        """
        if start_node_id == end_node_id:
            return [start_node_id]

        visited = {start_node_id}
        queue = [(start_node_id, [start_node_id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            # Get neighbors
            edges = await self.get_edges_from(current_id)
            edges.extend(await self.get_edges_to(current_id))

            for edge in edges:
                next_id = edge.target_id if edge.source_id == current_id else edge.source_id

                if next_id == end_node_id:
                    return path + [next_id]

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))

        return None

    # =========================================================================
    # Scope-Aware Operations
    # =========================================================================

    async def get_nodes_in_scope(
        self,
        scope_id: str,
        include_children: bool = False,
    ) -> list[TemporalNode]:
        """Get all nodes in a scope.

        Args:
            scope_id: Scope to query
            include_children: Include nodes from child scopes

        Returns:
            List of nodes
        """
        node_ids = set(self._nodes_by_scope.get(scope_id, set()))

        if include_children:
            scope = self._scopes.get(scope_id)
            if scope:
                for descendant in self._scopes.get_descendants(scope_id):
                    node_ids.update(self._nodes_by_scope.get(descendant.scope_id, set()))

        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    async def promote_node(self, node_id: str) -> bool:
        """Promote a node to its parent scope.

        Args:
            node_id: Node to promote

        Returns:
            True if promoted
        """
        node = self._nodes.get(node_id)
        if not node:
            return False

        scope = self._scopes.get(node.scope_id)
        if not scope or not scope.parent_scope_id:
            return False

        # Update indexes
        self._unindex_node(node)
        node.scope_id = scope.parent_scope_id
        self._index_node(node)

        node.updated_at = datetime.now()
        logger.info(f"Promoted node {node_id} to scope {scope.parent_scope_id}")
        return True

    # =========================================================================
    # Temporal Queries
    # =========================================================================

    async def get_nodes_at_time(
        self,
        timestamp: datetime,
        scope_ids: list[str] | None = None,
    ) -> list[TemporalNode]:
        """Get nodes that were valid at a specific time.

        Uses the validity interval (valid_from, valid_until) for filtering.

        Args:
            timestamp: Point in time
            scope_ids: Optional scope filter

        Returns:
            Nodes that were valid at the timestamp
        """
        candidates = self._get_candidates(
            scope_ids=scope_ids,
            node_types=None,
            layers=None,
            include_superseded=True,
        )

        results = []
        for node_id in candidates:
            node = self._nodes[node_id]
            if node.valid_from <= timestamp:
                if node.valid_until is None or node.valid_until > timestamp:
                    results.append(node)

        return results

    async def get_node_history(self, node_id: str) -> list[TemporalNode]:
        """Get the history of a node (all versions).

        Follows the supersedes chain to get all versions.

        Args:
            node_id: Any version of the node

        Returns:
            All versions, newest first
        """
        # Find the current (newest) version
        node = self._nodes.get(node_id)
        if not node:
            return []

        # Follow superseded_by to get newest
        current = node
        while current.superseded_by:
            newer = self._nodes.get(current.superseded_by)
            if newer:
                current = newer
            else:
                break

        # Now follow supersedes chain backwards
        history = [current]
        seen = {current.node_id}

        def collect_predecessors(n: TemporalNode):
            for pred_id in n.supersedes:
                if pred_id not in seen:
                    pred = self._nodes.get(pred_id)
                    if pred:
                        seen.add(pred_id)
                        history.append(pred)
                        collect_predecessors(pred)

        collect_predecessors(current)

        # Sort by ingestion time, newest first
        history.sort(key=lambda n: n.ingestion_time, reverse=True)
        return history

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        type_counts = {nt.value: len(ids) for nt, ids in self._nodes_by_type.items()}
        layer_counts = {l.value: len(ids) for l, ids in self._nodes_by_layer.items()}
        scope_counts = {s: len(ids) for s, ids in self._nodes_by_scope.items()}

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_type": type_counts,
            "nodes_by_layer": layer_counts,
            "nodes_by_scope": scope_counts,
            "current_nodes": sum(1 for n in self._nodes.values() if n.is_current),
            "superseded_nodes": sum(1 for n in self._nodes.values() if n.superseded_by),
        }

    def clear(self) -> None:
        """Clear all nodes and edges (for testing)."""
        self._nodes.clear()
        self._edges.clear()
        self._nodes_by_scope.clear()
        self._nodes_by_type.clear()
        self._nodes_by_layer.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()

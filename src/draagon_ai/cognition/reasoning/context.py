"""Context retrieval for probabilistic reasoning.

Handles:
- Recency window: Recent message graphs weighted by time
- Graph traversal: Multi-hop context from Neo4j
- Context scoring: Relevance ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..decomposition.graph import (
    SemanticGraph,
    GraphNode,
    Neo4jGraphStore,
    Neo4jGraphStoreSync,
)


@dataclass
class RecencyWindow:
    """Sliding window of recent conversation graphs.

    Maintains recent message graphs weighted by recency for
    context during probabilistic expansion.
    """

    graphs: list[tuple[SemanticGraph, datetime]] = field(default_factory=list)
    window_size: int = 10
    time_decay: float = 0.9  # Older messages weighted less

    def add(self, graph: SemanticGraph) -> None:
        """Add a new graph to the window."""
        self.graphs.append((graph, datetime.now(timezone.utc)))

        # Trim to window size
        if len(self.graphs) > self.window_size:
            self.graphs = self.graphs[-self.window_size:]

    def get_weighted_nodes(self) -> list[tuple[GraphNode, float]]:
        """Get all nodes weighted by recency.

        Most recent graphs have weight 1.0, older ones decay
        by time_decay factor per position.
        """
        weighted = []
        for i, (graph, _) in enumerate(reversed(self.graphs)):
            weight = self.time_decay ** i
            for node in graph.iter_nodes():
                weighted.append((node, weight))
        return weighted

    def get_recent_entities(self, limit: int = 20) -> list[GraphNode]:
        """Get the most recent entity nodes for context."""
        entities = []
        for graph, _ in reversed(self.graphs):
            for node in graph.iter_nodes():
                # Skip class nodes, we want instances
                if node.node_type.value == "instance":
                    entities.append(node)
                    if len(entities) >= limit:
                        return entities
        return entities

    def to_summary(self) -> str:
        """Create a text summary of recent context for LLM."""
        if not self.graphs:
            return "No recent context."

        lines = []
        for i, (graph, timestamp) in enumerate(reversed(self.graphs)):
            age = datetime.now(timezone.utc) - timestamp
            age_str = f"{age.seconds}s ago" if age.seconds < 60 else f"{age.seconds // 60}m ago"

            # Get key entities
            entities = [n.canonical_name for n in graph.iter_nodes()
                       if n.node_type.value == "instance"][:5]

            if entities:
                lines.append(f"[{age_str}] Entities: {', '.join(entities)}")

        return "\n".join(lines[:10])  # Limit summary size


@dataclass
class RetrievedContext:
    """Context retrieved from knowledge graph."""

    subgraph: SemanticGraph
    anchor_nodes: list[GraphNode]
    traversal_depth: int
    retrieval_time_ms: float
    node_count: int
    edge_count: int

    def to_summary(self) -> str:
        """Create text summary for LLM context."""
        lines = []

        # Summarize by relationship
        relations = {}
        for edge in self.subgraph.iter_edges():
            source = self.subgraph.get_node(edge.source_node_id)
            target = self.subgraph.get_node(edge.target_node_id)
            if source and target:
                rel = edge.relation_type
                if rel not in relations:
                    relations[rel] = []
                relations[rel].append(f"{source.canonical_name} → {target.canonical_name}")

        for rel, pairs in relations.items():
            lines.append(f"{rel.upper()}:")
            for pair in pairs[:5]:  # Limit per relation
                lines.append(f"  {pair}")

        return "\n".join(lines)


class ContextRetriever:
    """Retrieves relevant context from Neo4j knowledge graph.

    Uses GraphRAG-style multi-hop traversal starting from
    anchor nodes (entities in current message).
    """

    def __init__(
        self,
        store: Neo4jGraphStoreSync | None = None,
        default_depth: int = 2,
        max_nodes: int = 100,
    ):
        self.store = store
        self.default_depth = default_depth
        self.max_nodes = max_nodes

    def set_store(self, store: Neo4jGraphStoreSync) -> None:
        """Set the Neo4j store (can be deferred)."""
        self.store = store

    def retrieve(
        self,
        instance_id: str,
        anchor_nodes: list[GraphNode],
        depth: int | None = None,
        relation_types: list[str] | None = None,
    ) -> RetrievedContext:
        """
        Retrieve context subgraph from Neo4j.

        Args:
            instance_id: Data instance to query
            anchor_nodes: Starting nodes (entities in current message)
            depth: Traversal depth (default: 2)
            relation_types: Optional filter for relation types

        Returns:
            RetrievedContext with subgraph and metadata
        """
        if not self.store:
            # Return empty context if no store configured
            return RetrievedContext(
                subgraph=SemanticGraph(),
                anchor_nodes=anchor_nodes,
                traversal_depth=0,
                retrieval_time_ms=0,
                node_count=0,
                edge_count=0,
            )

        import time
        start = time.perf_counter()

        depth = depth or self.default_depth
        anchor_ids = [n.node_id for n in anchor_nodes]

        # Load and traverse
        subgraph = self._traverse_sync(instance_id, anchor_ids, depth, relation_types)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RetrievedContext(
            subgraph=subgraph,
            anchor_nodes=anchor_nodes,
            traversal_depth=depth,
            retrieval_time_ms=elapsed_ms,
            node_count=subgraph.node_count,
            edge_count=subgraph.edge_count,
        )

    def _traverse_sync(
        self,
        instance_id: str,
        anchor_ids: list[str],
        depth: int,
        relation_types: list[str] | None,
    ) -> SemanticGraph:
        """Synchronous graph traversal (sync store version)."""
        # Load full graph and filter (simple approach for now)
        # TODO: Use Neo4j native traversal for efficiency
        full_graph = self.store.load(instance_id)

        # BFS from anchors
        visited = set(anchor_ids)
        frontier = list(anchor_ids)
        result = SemanticGraph()

        # Add anchor nodes
        for node_id in anchor_ids:
            node = full_graph.get_node(node_id)
            if node:
                result.add_node(node)

        for d in range(depth):
            next_frontier = []
            for node_id in frontier:
                # Get outgoing edges
                for edge in full_graph.get_outgoing_edges(node_id):
                    if relation_types and edge.relation_type not in relation_types:
                        continue

                    target_id = edge.target_node_id
                    if target_id not in visited:
                        visited.add(target_id)
                        next_frontier.append(target_id)

                        target = full_graph.get_node(target_id)
                        if target and target.node_id not in result.nodes:
                            result.add_node(target)

                    # Add edge if both nodes in result
                    if edge.source_node_id in result.nodes and edge.target_node_id in result.nodes:
                        if edge.edge_id not in result.edges:
                            result.add_edge(edge)

                # Get incoming edges too
                for edge in full_graph.get_incoming_edges(node_id):
                    if relation_types and edge.relation_type not in relation_types:
                        continue

                    source_id = edge.source_node_id
                    if source_id not in visited:
                        visited.add(source_id)
                        next_frontier.append(source_id)

                        source = full_graph.get_node(source_id)
                        if source and source.node_id not in result.nodes:
                            result.add_node(source)

                    if edge.source_node_id in result.nodes and edge.target_node_id in result.nodes:
                        if edge.edge_id not in result.edges:
                            result.add_edge(edge)

            frontier = next_frontier
            if not frontier:
                break

            # Safety limit
            if result.node_count >= self.max_nodes:
                break

        return result


class AsyncContextRetriever:
    """Async version of context retriever."""

    def __init__(
        self,
        store: Neo4jGraphStore | None = None,
        default_depth: int = 2,
        max_nodes: int = 100,
    ):
        self.store = store
        self.default_depth = default_depth
        self.max_nodes = max_nodes

    async def retrieve(
        self,
        instance_id: str,
        anchor_nodes: list[GraphNode],
        depth: int | None = None,
        relation_types: list[str] | None = None,
    ) -> RetrievedContext:
        """Async context retrieval."""
        if not self.store:
            return RetrievedContext(
                subgraph=SemanticGraph(),
                anchor_nodes=anchor_nodes,
                traversal_depth=0,
                retrieval_time_ms=0,
                node_count=0,
                edge_count=0,
            )

        import time
        start = time.perf_counter()

        depth = depth or self.default_depth
        anchor_ids = [n.node_id for n in anchor_nodes]

        # Use native Neo4j traversal
        subgraph = await self.store.traverse(
            instance_id=instance_id,
            start_node_ids=anchor_ids,
            max_depth=depth,
            relation_types=relation_types,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RetrievedContext(
            subgraph=subgraph,
            anchor_nodes=anchor_nodes,
            traversal_depth=depth,
            retrieval_time_ms=elapsed_ms,
            node_count=subgraph.node_count,
            edge_count=subgraph.edge_count,
        )

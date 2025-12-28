"""Qdrant-backed storage for TemporalCognitiveGraph.

This module provides persistent storage for the temporal cognitive graph
using Qdrant vector database. It stores both nodes and edges, enabling
full graph reconstruction and semantic search.

Key Features:
- Bi-temporal node storage (event_time + ingestion_time)
- Edge relationship persistence
- Incremental persistence (changes saved immediately)
- Graph loading from Qdrant on startup
- Semantic search via vector embeddings
- Scope-based filtering

Based on the in-memory TemporalCognitiveGraph with Qdrant persistence added.

Example:
    from draagon_ai.memory.providers.qdrant_graph import QdrantGraphStore, QdrantGraphConfig

    config = QdrantGraphConfig(
        url="http://192.168.168.216:6333",
        nodes_collection="draagon_nodes",
        edges_collection="draagon_edges",
    )

    store = QdrantGraphStore(config, embedding_provider)
    await store.initialize()

    # All TemporalCognitiveGraph operations work, but are persisted to Qdrant
    node = await store.add_node(
        content="Doug's birthday is March 15",
        node_type=NodeType.FACT,
        scope_id="user:doug",
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID
import logging

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        MatchAny,
        HasIdCondition,
        PayloadSchemaType,
        PointIdsList,
        FilterSelector,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    AsyncQdrantClient = None

from draagon_ai.memory.temporal_graph import (
    TemporalCognitiveGraph,
    EmbeddingProvider,
    GraphSearchResult,
    GraphTraversalResult,
)
from draagon_ai.memory.temporal_nodes import (
    TemporalNode,
    TemporalEdge,
    NodeType,
    EdgeType,
    MemoryLayer,
    NODE_TYPE_TO_LAYER,
)
from draagon_ai.memory.scopes import (
    ScopeRegistry,
    get_scope_registry,
)


logger = logging.getLogger(__name__)


@dataclass
class QdrantGraphConfig:
    """Configuration for Qdrant graph storage."""

    url: str = "http://localhost:6333"
    api_key: str | None = None

    # Collection names
    nodes_collection: str = "draagon_graph_nodes"
    edges_collection: str = "draagon_graph_edges"

    # Vector settings
    embedding_dimension: int = 768

    # Search defaults
    default_limit: int = 10
    similarity_threshold: float = 0.3

    # Performance settings
    on_disk_payload: bool = True
    timeout: float = 30.0

    # Load settings
    load_batch_size: int = 1000

    # Type importance weights for search ranking
    type_weights: dict[str, float] = field(default_factory=lambda: {
        NodeType.CONTEXT.value: 0.5,
        NodeType.GOAL.value: 0.6,
        NodeType.EPISODE.value: 0.5,
        NodeType.EVENT.value: 0.55,
        NodeType.ENTITY.value: 0.7,
        NodeType.RELATIONSHIP.value: 0.65,
        NodeType.FACT.value: 0.8,
        NodeType.BELIEF.value: 0.75,
        NodeType.SKILL.value: 0.85,
        NodeType.STRATEGY.value: 0.8,
        NodeType.INSIGHT.value: 0.75,
        NodeType.BEHAVIOR.value: 0.7,
    })


class QdrantGraphStore(TemporalCognitiveGraph):
    """Qdrant-backed implementation of TemporalCognitiveGraph.

    This extends the in-memory TemporalCognitiveGraph with Qdrant persistence.
    All node and edge operations are persisted incrementally to Qdrant.

    The graph can be loaded from Qdrant on startup, enabling persistence
    across restarts.

    Nodes Collection Schema:
        - Vector: embedding (768-dim by default)
        - Payload:
            - node_id: str (UUID)
            - content: str
            - node_type: str
            - layer: str
            - scope_id: str
            - confidence: float
            - importance: float
            - stated_count: int
            - access_count: int
            - event_time: str (ISO)
            - ingestion_time: str (ISO)
            - valid_from: str (ISO)
            - valid_until: str | None
            - entities: list[str]
            - derived_from: list[str]
            - supersedes: list[str]
            - superseded_by: str | None
            - is_current: bool
            - metadata: dict

    Edges Collection Schema:
        - Vector: None (no embedding, just payload)
        - Payload:
            - edge_id: str (UUID)
            - source_id: str
            - target_id: str
            - edge_type: str
            - label: str | None
            - weight: float
            - confidence: float
            - created_at: str (ISO)
            - metadata: dict
    """

    def __init__(
        self,
        config: QdrantGraphConfig,
        embedding_provider: EmbeddingProvider | None = None,
        scope_registry: ScopeRegistry | None = None,
    ):
        """Initialize the Qdrant-backed graph store.

        Args:
            config: Qdrant configuration
            embedding_provider: Provider for generating embeddings
            scope_registry: Registry for scope management
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for QdrantGraphStore. "
                "Install with: pip install qdrant-client"
            )

        # Initialize parent (in-memory graph)
        super().__init__(
            embedding_provider=embedding_provider,
            scope_registry=scope_registry,
        )

        self.config = config
        self._client: AsyncQdrantClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Qdrant connection and ensure collections exist."""
        if self._initialized:
            return

        self._client = AsyncQdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

        await self._ensure_collections()
        self._initialized = True
        logger.info(
            f"QdrantGraphStore initialized with collections: "
            f"{self.config.nodes_collection}, {self.config.edges_collection}"
        )

    async def _ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}

        # Create nodes collection (with vectors)
        if self.config.nodes_collection not in existing:
            await self._client.create_collection(
                collection_name=self.config.nodes_collection,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE,
                    on_disk=self.config.on_disk_payload,
                ),
                on_disk_payload=self.config.on_disk_payload,
            )
            logger.info(f"Created nodes collection: {self.config.nodes_collection}")

        # Create edges collection (no vectors, just payload)
        if self.config.edges_collection not in existing:
            # Edges don't need vectors - we look them up by source/target ID
            # But Qdrant requires vectors, so we use a minimal 1-dim placeholder
            await self._client.create_collection(
                collection_name=self.config.edges_collection,
                vectors_config=VectorParams(
                    size=1,  # Minimal placeholder
                    distance=Distance.COSINE,
                ),
                on_disk_payload=self.config.on_disk_payload,
            )
            logger.info(f"Created edges collection: {self.config.edges_collection}")

    async def close(self) -> None:
        """Close the Qdrant connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False

    # =========================================================================
    # Node Serialization
    # =========================================================================

    def _node_to_payload(self, node: TemporalNode) -> dict[str, Any]:
        """Convert a TemporalNode to Qdrant payload."""
        return {
            "node_id": node.node_id,
            "content": node.content,
            "node_type": node.node_type.value,
            "layer": node.layer.value,
            "scope_id": node.scope_id,
            "confidence": node.confidence,
            "importance": node.importance,
            "stated_count": node.stated_count,
            "access_count": node.access_count,
            "event_time": node.event_time.isoformat(),
            "ingestion_time": node.ingestion_time.isoformat(),
            "valid_from": node.valid_from.isoformat(),
            "valid_until": node.valid_until.isoformat() if node.valid_until else None,
            "entities": node.entities,
            "derived_from": node.derived_from,
            "supersedes": node.supersedes,
            "superseded_by": node.superseded_by,
            "is_current": node.is_current,
            "metadata": node.metadata,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None,
        }

    def _payload_to_node(self, payload: dict[str, Any], embedding: list[float] | None = None) -> TemporalNode:
        """Convert Qdrant payload back to a TemporalNode."""
        node = TemporalNode(
            content=payload["content"],
            node_type=NodeType(payload["node_type"]),
            node_id=payload["node_id"],
            event_time=datetime.fromisoformat(payload["event_time"]),
            ingestion_time=datetime.fromisoformat(payload["ingestion_time"]),
            valid_from=datetime.fromisoformat(payload["valid_from"]),
            valid_until=datetime.fromisoformat(payload["valid_until"]) if payload.get("valid_until") else None,
            embedding=embedding,
            confidence=payload.get("confidence", 1.0),
            importance=payload.get("importance", 0.5),
            stated_count=payload.get("stated_count", 1),
            access_count=payload.get("access_count", 0),
            scope_id=payload.get("scope_id", "agent:default"),
            derived_from=payload.get("derived_from", []),
            supersedes=payload.get("supersedes", []),
            superseded_by=payload.get("superseded_by"),
            entities=payload.get("entities", []),
            metadata=payload.get("metadata", {}),
        )

        # Restore updated_at if present
        if payload.get("updated_at"):
            node.updated_at = datetime.fromisoformat(payload["updated_at"])

        return node

    # =========================================================================
    # Edge Serialization
    # =========================================================================

    def _edge_to_payload(self, edge: TemporalEdge) -> dict[str, Any]:
        """Convert a TemporalEdge to Qdrant payload."""
        return {
            "edge_id": edge.edge_id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type.value,
            "label": edge.label,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "created_at": edge.created_at.isoformat(),
            "valid_from": edge.valid_from.isoformat(),
            "valid_until": edge.valid_until.isoformat() if edge.valid_until else None,
            "metadata": edge.metadata,
        }

    def _payload_to_edge(self, payload: dict[str, Any]) -> TemporalEdge:
        """Convert Qdrant payload back to a TemporalEdge."""
        edge = TemporalEdge(
            source_id=payload["source_id"],
            target_id=payload["target_id"],
            edge_type=EdgeType(payload["edge_type"]),
            edge_id=payload["edge_id"],
            label=payload.get("label"),
            weight=payload.get("weight", 1.0),
            confidence=payload.get("confidence", 1.0),
            metadata=payload.get("metadata", {}),
        )

        # Restore timestamps
        if payload.get("created_at"):
            edge.created_at = datetime.fromisoformat(payload["created_at"])
        if payload.get("valid_from"):
            edge.valid_from = datetime.fromisoformat(payload["valid_from"])
        if payload.get("valid_until"):
            edge.valid_until = datetime.fromisoformat(payload["valid_until"])

        return edge

    # =========================================================================
    # Node Operations (Override with persistence)
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
        """Add a node and persist to Qdrant."""
        # Use parent implementation to create node
        node = await super().add_node(
            content=content,
            node_type=node_type,
            scope_id=scope_id,
            entities=entities,
            confidence=confidence,
            importance=importance,
            event_time=event_time,
            metadata=metadata,
            embedding=embedding,
            agent_id=agent_id,
            user_id=user_id,
            enforce_permissions=enforce_permissions,
        )

        # Persist to Qdrant
        await self._persist_node(node)

        return node

    async def _persist_node(self, node: TemporalNode) -> None:
        """Persist a node to Qdrant.

        Raises:
            RuntimeError: If the store is not initialized.
            Exception: If Qdrant upsert fails (logged before re-raising).
        """
        if not self._client:
            raise RuntimeError("QdrantGraphStore not initialized. Call initialize() first.")

        # Use node_id as point ID (must be valid UUID)
        point_id = node.node_id

        # Create point with embedding (or zero vector if no embedding)
        vector = node.embedding if node.embedding else [0.0] * self.config.embedding_dimension

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=self._node_to_payload(node),
        )

        try:
            await self._client.upsert(
                collection_name=self.config.nodes_collection,
                points=[point],
            )
            logger.debug(f"Persisted node {node.node_id}: {node.content[:50]}...")
        except Exception as e:
            logger.error(f"Failed to persist node {node.node_id} to Qdrant: {e}")
            raise

    async def update_node(
        self,
        node_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalNode | None:
        """Update a node and persist to Qdrant."""
        node = await super().update_node(
            node_id=node_id,
            content=content,
            confidence=confidence,
            importance=importance,
            metadata=metadata,
        )

        if node:
            await self._persist_node(node)

        return node

    async def delete_node(
        self,
        node_id: str,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        enforce_permissions: bool = False,
    ) -> bool:
        """Delete a node and remove from Qdrant."""
        # Get connected edges before deletion (they'll be deleted by parent)
        # Use indexes for O(1) lookup instead of iterating all edges
        edge_ids_to_delete = (
            self._edges_by_source.get(node_id, set()) |
            self._edges_by_target.get(node_id, set())
        )

        # Use parent to delete from memory
        result = await super().delete_node(
            node_id=node_id,
            agent_id=agent_id,
            user_id=user_id,
            enforce_permissions=enforce_permissions,
        )

        if result and self._client:
            # Delete node from Qdrant using PointIdsList
            await self._client.delete(
                collection_name=self.config.nodes_collection,
                points_selector=PointIdsList(points=[node_id]),
            )

            # Delete connected edges from Qdrant
            if edge_ids_to_delete:
                await self._client.delete(
                    collection_name=self.config.edges_collection,
                    points_selector=PointIdsList(points=list(edge_ids_to_delete)),
                )

            logger.debug(f"Deleted node {node_id} and {len(edge_ids_to_delete)} edges from Qdrant")

        return result

    async def supersede_node(
        self,
        old_node_id: str,
        new_content: str,
        **kwargs: Any,
    ) -> TemporalNode | None:
        """Supersede a node and persist both old and new to Qdrant."""
        new_node = await super().supersede_node(
            old_node_id=old_node_id,
            new_content=new_content,
            **kwargs,
        )

        if new_node:
            # Persist both old (now superseded) and new node
            old_node = self._nodes.get(old_node_id)
            if old_node:
                await self._persist_node(old_node)
            await self._persist_node(new_node)

        return new_node

    # =========================================================================
    # Edge Operations (Override with persistence)
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
        """Add an edge and persist to Qdrant."""
        edge = await super().add_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=label,
            weight=weight,
            confidence=confidence,
            metadata=metadata,
        )

        if edge:
            await self._persist_edge(edge)

        return edge

    async def _persist_edge(self, edge: TemporalEdge) -> None:
        """Persist an edge to Qdrant.

        Raises:
            RuntimeError: If the store is not initialized.
            Exception: If Qdrant upsert fails (logged before re-raising).
        """
        if not self._client:
            raise RuntimeError("QdrantGraphStore not initialized. Call initialize() first.")

        # Edges use a placeholder vector since Qdrant requires one
        point = PointStruct(
            id=edge.edge_id,
            vector=[0.0],  # Placeholder - edges are looked up by payload, not vector
            payload=self._edge_to_payload(edge),
        )

        try:
            await self._client.upsert(
                collection_name=self.config.edges_collection,
                points=[point],
            )
            logger.debug(f"Persisted edge {edge.edge_id}: {edge.source_id} -> {edge.target_id}")
        except Exception as e:
            logger.error(f"Failed to persist edge {edge.edge_id} to Qdrant: {e}")
            raise

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge and remove from Qdrant."""
        result = await super().delete_edge(edge_id)

        if result and self._client:
            await self._client.delete(
                collection_name=self.config.edges_collection,
                points_selector=PointIdsList(points=[edge_id]),
            )
            logger.debug(f"Deleted edge {edge_id} from Qdrant")

        return result

    # =========================================================================
    # Search Operations (Use Qdrant for vector search)
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
        """Search for nodes using Qdrant vector search.

        This uses Qdrant's native vector search for better performance
        than the in-memory implementation.
        """
        if not self._client:
            raise RuntimeError("QdrantGraphStore not initialized. Call initialize() first.")

        if not self._embedder:
            logger.warning("No embedding provider - returning empty results")
            return []

        # Get query embedding
        query_embedding = await self._embedder.embed(query)

        # Build filter conditions
        filter_conditions = self._build_search_filter(
            scope_ids=scope_ids,
            node_types=node_types,
            layers=layers,
            include_superseded=include_superseded,
            include_ancestor_scopes=include_ancestor_scopes,
        )

        # Search Qdrant using query_points (async API)
        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        query_result = await self._client.query_points(
            collection_name=self.config.nodes_collection,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score,
            with_payload=True,
            with_vectors=True,
        )
        results = query_result.points

        # Convert to GraphSearchResult
        search_results = []
        for result in results:
            node = self._payload_to_node(
                result.payload,
                embedding=result.vector if isinstance(result.vector, list) else None,
            )

            # Update in-memory cache
            self._nodes[node.node_id] = node
            self._index_node(node)

            # Get edges if requested
            edges = []
            if include_edges:
                edges = await self.get_edges_from(node.node_id)
                edges.extend(await self.get_edges_to(node.node_id))

            # Apply type-based importance weighting
            type_weight = self.config.type_weights.get(node.node_type.value, 0.5)
            adjusted_score = result.score * (0.7 + 0.3 * type_weight) * (0.8 + 0.2 * node.importance)

            search_results.append(GraphSearchResult(
                node=node,
                score=adjusted_score,
                edges=edges,
            ))

        # Re-sort by adjusted score
        search_results.sort(key=lambda r: r.score, reverse=True)

        return search_results

    def _build_search_filter(
        self,
        scope_ids: list[str] | None,
        node_types: list[NodeType] | None,
        layers: list[MemoryLayer] | None,
        include_superseded: bool,
        include_ancestor_scopes: bool,
    ) -> list[FieldCondition]:
        """Build Qdrant filter conditions for search."""
        conditions = []

        # Scope filter
        if scope_ids:
            all_scopes = set(scope_ids)

            # Include ancestor scopes for hierarchical inheritance
            if include_ancestor_scopes:
                for scope_id in scope_ids:
                    ancestors = self._scopes.get_ancestors(scope_id)
                    for ancestor in ancestors:
                        all_scopes.add(ancestor.scope_id)

            conditions.append(
                FieldCondition(
                    key="scope_id",
                    match=MatchAny(any=list(all_scopes)),
                )
            )

        # Node type filter
        if node_types:
            conditions.append(
                FieldCondition(
                    key="node_type",
                    match=MatchAny(any=[nt.value for nt in node_types]),
                )
            )

        # Layer filter
        if layers:
            conditions.append(
                FieldCondition(
                    key="layer",
                    match=MatchAny(any=[l.value for l in layers]),
                )
            )

        # Superseded filter
        if not include_superseded:
            conditions.append(
                FieldCondition(
                    key="is_current",
                    match=MatchValue(value=True),
                )
            )

        return conditions

    # =========================================================================
    # Graph Loading
    # =========================================================================

    async def load_from_qdrant(
        self,
        scope_ids: list[str] | None = None,
    ) -> tuple[int, int]:
        """Load graph data from Qdrant into memory.

        This loads nodes and edges from Qdrant into the in-memory
        indexes for fast traversal and operations.

        Args:
            scope_ids: Optional list of scopes to load (None = all)

        Returns:
            Tuple of (nodes_loaded, edges_loaded)
        """
        if not self._client:
            raise RuntimeError("QdrantGraphStore not initialized. Call initialize() first.")

        nodes_loaded = 0
        edges_loaded = 0

        # Load nodes
        offset = None
        while True:
            # Build filter for scopes if specified
            scroll_filter = None
            if scope_ids:
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="scope_id",
                            match=MatchAny(any=scope_ids),
                        )
                    ]
                )

            records, next_offset = await self._client.scroll(
                collection_name=self.config.nodes_collection,
                scroll_filter=scroll_filter,
                limit=self.config.load_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for record in records:
                embedding = record.vector if isinstance(record.vector, list) else None
                node = self._payload_to_node(record.payload, embedding=embedding)

                # Add to in-memory storage
                self._nodes[node.node_id] = node
                self._index_node(node)
                nodes_loaded += 1

            if next_offset is None:
                break
            offset = next_offset

        # Load edges
        offset = None
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self.config.edges_collection,
                limit=self.config.load_batch_size,
                offset=offset,
                with_payload=True,
            )

            for record in records:
                edge = self._payload_to_edge(record.payload)

                # Only load edges where both nodes exist
                if edge.source_id in self._nodes and edge.target_id in self._nodes:
                    self._edges[edge.edge_id] = edge

                    # Update indexes
                    if edge.source_id not in self._edges_by_source:
                        self._edges_by_source[edge.source_id] = set()
                    self._edges_by_source[edge.source_id].add(edge.edge_id)

                    if edge.target_id not in self._edges_by_target:
                        self._edges_by_target[edge.target_id] = set()
                    self._edges_by_target[edge.target_id].add(edge.edge_id)

                    edges_loaded += 1

            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Loaded {nodes_loaded} nodes and {edges_loaded} edges from Qdrant")
        return nodes_loaded, edges_loaded

    # =========================================================================
    # Scope-Aware Operations (Load from Qdrant if needed)
    # =========================================================================

    async def get_node(self, node_id: str) -> TemporalNode | None:
        """Get a node by ID, loading from Qdrant if not in memory."""
        # Check memory first
        node = self._nodes.get(node_id)
        if node:
            return node

        # Try to load from Qdrant
        if self._client:
            try:
                results = await self._client.retrieve(
                    collection_name=self.config.nodes_collection,
                    ids=[node_id],
                    with_payload=True,
                    with_vectors=True,
                )

                if results:
                    record = results[0]
                    embedding = record.vector if isinstance(record.vector, list) else None
                    node = self._payload_to_node(record.payload, embedding=embedding)

                    # Cache in memory
                    self._nodes[node.node_id] = node
                    self._index_node(node)

                    return node
            except Exception as e:
                logger.warning(f"Failed to retrieve node {node_id} from Qdrant: {e}")

        return None

    async def get_edge(self, edge_id: str) -> TemporalEdge | None:
        """Get an edge by ID, loading from Qdrant if not in memory."""
        # Check memory first
        edge = self._edges.get(edge_id)
        if edge:
            return edge

        # Try to load from Qdrant
        if self._client:
            try:
                results = await self._client.retrieve(
                    collection_name=self.config.edges_collection,
                    ids=[edge_id],
                    with_payload=True,
                )

                if results:
                    record = results[0]
                    edge = self._payload_to_edge(record.payload)

                    # Cache in memory
                    self._edges[edge.edge_id] = edge

                    # Update indexes
                    if edge.source_id not in self._edges_by_source:
                        self._edges_by_source[edge.source_id] = set()
                    self._edges_by_source[edge.source_id].add(edge.edge_id)

                    if edge.target_id not in self._edges_by_target:
                        self._edges_by_target[edge.target_id] = set()
                    self._edges_by_target[edge.target_id].add(edge.edge_id)

                    return edge
            except Exception as e:
                logger.warning(f"Failed to retrieve edge {edge_id} from Qdrant: {e}")

        return None

    async def get_edges_from(self, node_id: str) -> list[TemporalEdge]:
        """Get edges from a node, loading from Qdrant if needed."""
        # First try in-memory
        edge_ids = self._edges_by_source.get(node_id, set())
        if edge_ids:
            return [self._edges[eid] for eid in edge_ids if eid in self._edges]

        # Load from Qdrant if not in memory
        if self._client:
            try:
                results, _ = await self._client.scroll(
                    collection_name=self.config.edges_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=MatchValue(value=node_id),
                            )
                        ]
                    ),
                    limit=100,
                    with_payload=True,
                )

                edges = []
                for record in results:
                    edge = self._payload_to_edge(record.payload)

                    # Cache in memory
                    self._edges[edge.edge_id] = edge
                    if node_id not in self._edges_by_source:
                        self._edges_by_source[node_id] = set()
                    self._edges_by_source[node_id].add(edge.edge_id)
                    if edge.target_id not in self._edges_by_target:
                        self._edges_by_target[edge.target_id] = set()
                    self._edges_by_target[edge.target_id].add(edge.edge_id)

                    edges.append(edge)

                return edges
            except Exception as e:
                logger.warning(f"Failed to load edges from node {node_id}: {e}")

        return []

    async def get_edges_to(self, node_id: str) -> list[TemporalEdge]:
        """Get edges to a node, loading from Qdrant if needed."""
        # First try in-memory
        edge_ids = self._edges_by_target.get(node_id, set())
        if edge_ids:
            return [self._edges[eid] for eid in edge_ids if eid in self._edges]

        # Load from Qdrant if not in memory
        if self._client:
            try:
                results, _ = await self._client.scroll(
                    collection_name=self.config.edges_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="target_id",
                                match=MatchValue(value=node_id),
                            )
                        ]
                    ),
                    limit=100,
                    with_payload=True,
                )

                edges = []
                for record in results:
                    edge = self._payload_to_edge(record.payload)

                    # Cache in memory
                    self._edges[edge.edge_id] = edge
                    if edge.source_id not in self._edges_by_source:
                        self._edges_by_source[edge.source_id] = set()
                    self._edges_by_source[edge.source_id].add(edge.edge_id)
                    if node_id not in self._edges_by_target:
                        self._edges_by_target[node_id] = set()
                    self._edges_by_target[node_id].add(edge.edge_id)

                    edges.append(edge)

                return edges
            except Exception as e:
                logger.warning(f"Failed to load edges to node {node_id}: {e}")

        return []

    # =========================================================================
    # Statistics
    # =========================================================================

    async def qdrant_stats(self) -> dict[str, Any]:
        """Get Qdrant collection statistics."""
        if not self._client:
            return {"error": "Not initialized"}

        try:
            nodes_info = await self._client.get_collection(self.config.nodes_collection)
            edges_info = await self._client.get_collection(self.config.edges_collection)

            return {
                "nodes_collection": {
                    "name": self.config.nodes_collection,
                    "points_count": nodes_info.points_count,
                    "vectors_count": nodes_info.vectors_count,
                    "indexed_vectors_count": nodes_info.indexed_vectors_count,
                },
                "edges_collection": {
                    "name": self.config.edges_collection,
                    "points_count": edges_info.points_count,
                },
                "in_memory": {
                    "nodes": len(self._nodes),
                    "edges": len(self._edges),
                },
            }
        except Exception as e:
            return {"error": str(e)}

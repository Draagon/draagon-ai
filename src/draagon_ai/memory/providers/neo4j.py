"""Neo4j-backed Memory Provider for draagon-ai.

This module provides semantic graph memory using Neo4j vector database.
It replaces Qdrant with a unified semantic memory approach where:
- All content flows through Phase 0/1 extraction pipeline
- Memories are stored as semantic graphs in Neo4j
- Native graph traversal enables context retrieval
- Vector search via Neo4j native indexes (5.15+)

Key Features:
- Full Phase 0/1 semantic decomposition on all stored content
- Graph-based relationships between memories
- Temporal TTL with 4-layer memory model
- Native vector similarity search
- Reinforcement-based memory persistence

Requires:
- neo4j Python driver
- Running Neo4j instance (5.15+ for vector indexes)

Example:
    from draagon_ai.memory.providers.neo4j import Neo4jMemoryProvider, Neo4jMemoryConfig

    config = Neo4jMemoryConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
    )

    provider = Neo4jMemoryProvider(config, embedding_provider, llm_provider)
    await provider.initialize()

    memory = await provider.store(
        content="Doug's birthday is March 15",
        memory_type=MemoryType.FACT,
        scope=MemoryScope.USER,
        user_id="doug",
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4
import logging

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

from draagon_ai.memory.base import (
    MemoryProvider,
    MemoryType,
    MemoryScope,
    Memory,
    SearchResult,
)
from draagon_ai.cognition.reasoning.memory import (
    MemoryLayer,
    MemoryProperties,
    ContentType,
    MemoryAwareGraphStore,
    classify_phase1_content,
)
from draagon_ai.cognition.decomposition.graph import (
    SemanticGraph,
    GraphNode,
    NodeType,
    Neo4jGraphStoreSync,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Send chat messages and get response."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Neo4jMemoryConfig:
    """Configuration for Neo4j Memory Provider."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"

    # Embedding settings
    embedding_dimension: int = 1536

    # Search settings
    default_limit: int = 5
    similarity_threshold: float = 0.3

    # Pipeline settings
    enable_semantic_decomposition: bool = True  # Run Phase 0/1 on all content

    # Memory layer settings
    default_layer: MemoryLayer = MemoryLayer.EPISODIC

    # Type importance weights (for relevance boosting)
    type_weights: dict[str, float] = field(default_factory=lambda: {
        "instruction": 1.0,
        "self_knowledge": 0.95,
        "preference": 0.9,
        "skill": 0.85,
        "fact": 0.8,
        "knowledge": 0.7,
        "insight": 0.65,
        "relationship": 0.6,
        "belief": 0.8,
        "observation": 0.5,
        "episodic": 0.5,
    })


# =============================================================================
# Memory Type to Content Type Mapping
# =============================================================================


def memory_type_to_content_type(memory_type: MemoryType) -> ContentType:
    """Map MemoryType to ContentType for decay rates."""
    mapping = {
        MemoryType.FACT: ContentType.FACT,
        MemoryType.PREFERENCE: ContentType.PREFERENCE,
        MemoryType.EPISODIC: ContentType.EVENT,
        MemoryType.INSTRUCTION: ContentType.INSTRUCTION,
        MemoryType.KNOWLEDGE: ContentType.FACT,
        MemoryType.SKILL: ContentType.SKILL,
        MemoryType.INSIGHT: ContentType.COMMONSENSE,
        MemoryType.SELF_KNOWLEDGE: ContentType.INSTRUCTION,
        MemoryType.RELATIONSHIP: ContentType.RELATIONSHIP,
        MemoryType.OBSERVATION: ContentType.EVENT,
        MemoryType.BELIEF: ContentType.FACT,
    }
    return mapping.get(memory_type, ContentType.ENTITY)


def memory_type_to_layer(memory_type: MemoryType) -> MemoryLayer:
    """Map MemoryType to default MemoryLayer."""
    mapping = {
        MemoryType.FACT: MemoryLayer.SEMANTIC,
        MemoryType.PREFERENCE: MemoryLayer.SEMANTIC,
        MemoryType.EPISODIC: MemoryLayer.EPISODIC,
        MemoryType.INSTRUCTION: MemoryLayer.METACOGNITIVE,
        MemoryType.KNOWLEDGE: MemoryLayer.SEMANTIC,
        MemoryType.SKILL: MemoryLayer.SEMANTIC,
        MemoryType.INSIGHT: MemoryLayer.SEMANTIC,
        MemoryType.SELF_KNOWLEDGE: MemoryLayer.METACOGNITIVE,
        MemoryType.RELATIONSHIP: MemoryLayer.SEMANTIC,
        MemoryType.OBSERVATION: MemoryLayer.WORKING,
        MemoryType.BELIEF: MemoryLayer.SEMANTIC,
    }
    return mapping.get(memory_type, MemoryLayer.EPISODIC)


# =============================================================================
# Neo4j Memory Provider
# =============================================================================


class Neo4jMemoryProvider(MemoryProvider):
    """Neo4j-backed implementation of MemoryProvider with semantic decomposition.

    This provider stores all memories through the Phase 0/1 semantic pipeline,
    creating rich graph structures in Neo4j that can be traversed for context.

    Architecture:
        Memory.store() → Phase 0/1 Pipeline → SemanticGraph → Neo4j

    Each memory becomes:
        - A primary Memory node with content and metadata
        - Connected semantic graph nodes (entities, predicates, etc.)
        - Edges to other memories through shared entities

    This enables:
        - Vector similarity search (via Neo4j native index)
        - Graph traversal for context (find related memories)
        - Temporal decay with layer-based TTL
        - Reinforcement-based persistence
    """

    def __init__(
        self,
        config: Neo4jMemoryConfig,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider | None = None,
    ):
        """Initialize the Neo4j provider.

        Args:
            config: Neo4j configuration
            embedding_provider: Provider for generating embeddings
            llm_provider: LLM provider for semantic decomposition
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j is required for Neo4jMemoryProvider. "
                "Install with: pip install neo4j"
            )

        self.config = config
        self._embedder = embedding_provider
        self._llm = llm_provider

        self._store: Neo4jGraphStoreSync | None = None
        self._memory_store: MemoryAwareGraphStore | None = None
        self._pipeline = None  # Lazy init
        self._graph_builder = None  # Lazy init
        self._initialized = False

    @property
    def graph_store(self) -> Neo4jGraphStoreSync:
        """Get or create the Neo4j graph store."""
        if not self._store:
            self._store = Neo4jGraphStoreSync(
                uri=self.config.uri,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
            )
        return self._store

    @property
    def pipeline(self):
        """Get or create the integrated pipeline."""
        if not self._pipeline and self.config.enable_semantic_decomposition:
            try:
                from draagon_ai.cognition.decomposition.extractors.integrated_pipeline import (
                    IntegratedPipeline,
                    IntegratedPipelineConfig,
                )
                self._pipeline = IntegratedPipeline(
                    config=IntegratedPipelineConfig(),
                    llm=self._llm,
                )
            except ImportError:
                logger.warning("IntegratedPipeline not available, semantic decomposition disabled")
                self.config.enable_semantic_decomposition = False
        return self._pipeline

    @property
    def graph_builder(self):
        """Get or create the graph builder."""
        if not self._graph_builder:
            try:
                from draagon_ai.cognition.decomposition.graph.builder import GraphBuilder
                self._graph_builder = GraphBuilder()
            except ImportError:
                logger.warning("GraphBuilder not available")
        return self._graph_builder

    async def initialize(self) -> None:
        """Initialize connection and ensure schema exists."""
        if self._initialized:
            return

        # Create graph store
        _ = self.graph_store

        # Create indexes
        with self.graph_store.driver.session(database=self.config.database) as session:
            # Create uniqueness constraints
            try:
                session.run("""
                    CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
                    FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE
                """)
            except Exception:
                pass

            # Create indexes for common filters
            for field_name in ["user_id", "agent_id", "memory_type", "scope", "context_id"]:
                try:
                    session.run(f"""
                        CREATE INDEX memory_{field_name}_idx IF NOT EXISTS
                        FOR (m:Memory) ON (m.{field_name})
                    """)
                except Exception:
                    pass

            # Create vector index for embeddings
            try:
                session.run(f"""
                    CREATE VECTOR INDEX memory_embeddings IF NOT EXISTS
                    FOR (m:Memory) ON (m.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {self.config.embedding_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")

        self._initialized = True
        logger.info(f"Neo4jMemoryProvider initialized: {self.config.uri}")

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a new memory in Neo4j with semantic decomposition.

        The memory content is processed through Phase 0/1 to extract:
        - Entity nodes (people, places, things)
        - Predicate nodes (actions, states)
        - Semantic relationships
        - Presuppositions and inferences

        All these are stored as a connected graph in Neo4j.
        """
        if not self._initialized:
            await self.initialize()

        # Generate IDs
        memory_id = str(uuid4())
        instance_id = self._get_instance_id(agent_id, user_id, context_id)
        now = datetime.now()

        # Generate embedding
        embedding = await self._embedder.embed(content)

        # Determine memory layer and content type
        layer = memory_type_to_layer(memory_type)
        content_type = memory_type_to_content_type(memory_type)

        # Create memory properties
        memory_props = MemoryProperties(
            layer=layer,
            content_type=content_type,
            importance=importance,
            confidence=confidence,
        )

        # Build semantic graph from content
        semantic_graph = SemanticGraph()

        # Process through Phase 0/1 pipeline if enabled
        if self.config.enable_semantic_decomposition and self.pipeline and self._llm:
            try:
                result = await self.pipeline.process(content)

                # Build graph from result
                if self.graph_builder:
                    build_result = self.graph_builder.build_from_integrated(result, semantic_graph)
                    logger.debug(
                        f"Semantic decomposition: {build_result.stats['entities_created']} entities, "
                        f"{build_result.stats['edges_created']} edges"
                    )
            except Exception as e:
                logger.warning(f"Semantic decomposition failed: {e}")

        # Create primary memory node
        memory_node = semantic_graph.create_node(
            canonical_name=f"memory:{memory_id[:8]}",
            node_type=NodeType.INSTANCE,
            properties={
                "content": content,
                "memory_type": memory_type.value,
                "scope": scope.value,
                "agent_id": agent_id,
                "user_id": user_id,
                "context_id": context_id,
                "importance": importance,
                "confidence": confidence,
                "entities": entities or [],
                "stated_count": 1,
                "created_at": now.isoformat(),
                "is_memory_node": True,
            },
            source_id=memory_id,
        )

        # Store embedding on memory node
        memory_node.embedding = embedding

        # Link memory node to all extracted entities
        for node in list(semantic_graph.iter_nodes()):
            if node.node_id != memory_node.node_id:
                semantic_graph.create_edge(
                    memory_node.node_id,
                    node.node_id,
                    "extracted_from",
                    confidence=confidence,
                    source_decomposition_id=memory_id,
                )

        # Save to Neo4j with memory properties
        memory_aware_store = MemoryAwareGraphStore(self.graph_store, instance_id)
        content_type_map = classify_phase1_content(semantic_graph)
        content_type_map[memory_node.node_id] = content_type
        memory_aware_store.save_with_memory(
            semantic_graph,
            default_layer=layer,
            content_type_map=content_type_map,
        )

        # Also save memory-specific node with searchable properties
        with self.graph_store.driver.session(database=self.config.database) as session:
            session.run("""
                MERGE (m:Memory {memory_id: $memory_id})
                SET m.content = $content,
                    m.memory_type = $memory_type,
                    m.scope = $scope,
                    m.agent_id = $agent_id,
                    m.user_id = $user_id,
                    m.context_id = $context_id,
                    m.importance = $importance,
                    m.confidence = $confidence,
                    m.entities = $entities,
                    m.stated_count = 1,
                    m.created_at = $created_at,
                    m.last_accessed = $created_at,
                    m.instance_id = $instance_id,
                    m.embedding = $embedding,
                    m.graph_node_id = $graph_node_id
            """,
                memory_id=memory_id,
                content=content,
                memory_type=memory_type.value,
                scope=scope.value,
                agent_id=agent_id,
                user_id=user_id,
                context_id=context_id,
                importance=importance,
                confidence=confidence,
                entities=entities or [],
                created_at=now.isoformat(),
                instance_id=instance_id,
                embedding=embedding,
                graph_node_id=memory_node.node_id,
            )

            # Add memory layer properties
            memory_props_dict = memory_props.to_dict()
            session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                SET m += $props
            """, memory_id=memory_id, props=memory_props_dict)

        logger.debug(f"Stored memory {memory_id}: {content[:50]}...")

        return Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            scope=scope,
            agent_id=agent_id,
            user_id=user_id,
            context_id=context_id,
            importance=importance,
            confidence=confidence,
            entities=entities or [],
            created_at=now,
            embedding=embedding,
        )

    async def search(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        scopes: list[MemoryScope] | None = None,
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity using Neo4j vector index."""
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self._embedder.embed(query)
        min_score = min_score or self.config.similarity_threshold

        # Build filter conditions
        filters = []
        params: dict[str, Any] = {
            "embedding": query_embedding,
            "limit": limit * 2,  # Over-fetch to account for filtering
            "min_score": min_score,
        }

        if user_id:
            filters.append("m.user_id = $user_id")
            params["user_id"] = user_id

        if agent_id:
            filters.append("m.agent_id = $agent_id")
            params["agent_id"] = agent_id

        if context_id:
            filters.append("m.context_id = $context_id")
            params["context_id"] = context_id

        if memory_types:
            type_values = [t.value for t in memory_types]
            filters.append("m.memory_type IN $memory_types")
            params["memory_types"] = type_values

        if scopes:
            scope_values = [s.value for s in scopes]
            filters.append("m.scope IN $scopes")
            params["scopes"] = scope_values

        # Add expiration filter (exclude expired memories)
        filters.append("(m.memory_expires_at IS NULL OR m.memory_expires_at > $now)")
        params["now"] = datetime.now().isoformat()

        # Build WHERE clause
        where_clause = " AND ".join(filters) if filters else "true"

        # Query using vector index
        try:
            with self.graph_store.driver.session(database=self.config.database) as session:
                result = session.run(f"""
                    CALL db.index.vector.queryNodes('memory_embeddings', $limit, $embedding)
                    YIELD node AS m, score
                    WHERE {where_clause}
                    RETURN m, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, **params)

                search_results = []
                for record in result:
                    memory = self._node_to_memory(dict(record["m"]))
                    score = record["score"]

                    # Apply type importance weighting
                    type_weight = self.config.type_weights.get(memory.memory_type.value, 0.5)
                    weighted_score = score * type_weight

                    search_results.append(SearchResult(
                        memory=memory,
                        score=weighted_score,
                    ))

                # Re-sort by weighted score
                search_results.sort(key=lambda r: r.score, reverse=True)

                return search_results[:limit]

        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text search: {e}")
            return await self._fallback_search(query, params, limit)

    async def _fallback_search(
        self,
        query: str,
        params: dict[str, Any],
        limit: int,
    ) -> list[SearchResult]:
        """Fallback search using text matching when vector index unavailable."""
        # Use basic text containment - search_query to avoid conflict with 'query' param
        search_query = query.lower()
        search_limit = limit

        with self.graph_store.driver.session(database=self.config.database) as session:
            result = session.run("""
                MATCH (m:Memory)
                WHERE toLower(m.content) CONTAINS $search_query
                  AND (m.memory_expires_at IS NULL OR m.memory_expires_at > $now)
                RETURN m, 0.5 AS score
                ORDER BY m.importance DESC
                LIMIT $search_limit
            """, search_query=search_query, search_limit=search_limit, now=params.get("now", ""))

            search_results = []
            for record in result:
                memory = self._node_to_memory(dict(record["m"]))
                search_results.append(SearchResult(
                    memory=memory,
                    score=record["score"],
                ))

            return search_results

    async def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        if not self._initialized:
            await self.initialize()

        with self.graph_store.driver.session(database=self.config.database) as session:
            result = session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                RETURN m
            """, memory_id=memory_id)

            record = result.single()
            if not record:
                return None

            return self._node_to_memory(dict(record["m"]))

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory."""
        if not self._initialized:
            await self.initialize()

        # Get existing memory
        existing = await self.get(memory_id)
        if not existing:
            return None

        # Build update query
        updates = ["m.last_accessed = $now"]
        params: dict[str, Any] = {
            "memory_id": memory_id,
            "now": datetime.now().isoformat(),
        }

        if content is not None:
            updates.append("m.content = $content")
            params["content"] = content
            # Re-embed if content changed
            embedding = await self._embedder.embed(content)
            updates.append("m.embedding = $embedding")
            params["embedding"] = embedding

        if importance is not None:
            updates.append("m.importance = $importance")
            params["importance"] = importance

        if confidence is not None:
            updates.append("m.confidence = $confidence")
            params["confidence"] = confidence

        set_clause = ", ".join(updates)

        with self.graph_store.driver.session(database=self.config.database) as session:
            session.run(f"""
                MATCH (m:Memory {{memory_id: $memory_id}})
                SET {set_clause}
            """, **params)

        return await self.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory and its semantic graph."""
        if not self._initialized:
            await self.initialize()

        with self.graph_store.driver.session(database=self.config.database) as session:
            # Get the graph node ID first
            result = session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                RETURN m.graph_node_id AS graph_node_id, m.instance_id AS instance_id
            """, memory_id=memory_id)

            record = result.single()
            if not record:
                return False

            # Delete the memory node
            session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                DETACH DELETE m
            """, memory_id=memory_id)

            # Delete associated semantic graph nodes
            if record["graph_node_id"]:
                session.run("""
                    MATCH (n:Entity {node_id: $node_id})
                    DETACH DELETE n
                """, node_id=record["graph_node_id"])

            return True

    async def reinforce(self, memory_id: str, boost: float = 0.1) -> Memory | None:
        """Reinforce a memory by boosting importance and updating access time."""
        if not self._initialized:
            await self.initialize()

        existing = await self.get(memory_id)
        if not existing:
            return None

        new_importance = min(1.0, existing.importance + boost)

        with self.graph_store.driver.session(database=self.config.database) as session:
            session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                SET m.importance = $importance,
                    m.last_accessed = $now,
                    m.memory_access_count = COALESCE(m.memory_access_count, 0) + 1,
                    m.memory_reinforcement_score = COALESCE(m.memory_reinforcement_score, 0) + $boost
            """,
                memory_id=memory_id,
                importance=new_importance,
                now=datetime.now().isoformat(),
                boost=boost,
            )

            # Check for layer promotion
            result = session.run("""
                MATCH (m:Memory {memory_id: $memory_id})
                RETURN m.memory_reinforcement_score AS score, m.memory_layer AS layer
            """, memory_id=memory_id)

            record = result.single()
            if record:
                self._check_and_promote(session, memory_id, record["score"], record["layer"])

        return await self.get(memory_id)

    def _check_and_promote(
        self,
        session,
        memory_id: str,
        score: float,
        current_layer: str,
    ) -> None:
        """Check if memory should be promoted to higher layer."""
        promotion_thresholds = {
            "working": ("episodic", 0.3),
            "episodic": ("semantic", 0.6),
            "semantic": ("metacognitive", 0.9),
        }

        if current_layer and current_layer in promotion_thresholds:
            new_layer, threshold = promotion_thresholds[current_layer]
            if score and score >= threshold:
                from draagon_ai.cognition.reasoning.memory import LAYER_TTL
                new_ttl = LAYER_TTL[MemoryLayer(new_layer)]
                new_expires = None
                if new_ttl:
                    new_expires = (datetime.now() + new_ttl).isoformat()

                session.run("""
                    MATCH (m:Memory {memory_id: $memory_id})
                    SET m.memory_layer = $new_layer,
                        m.memory_expires_at = $new_expires,
                        m.memory_reinforcement_score = 0
                """,
                    memory_id=memory_id,
                    new_layer=new_layer,
                    new_expires=new_expires,
                )

                logger.info(f"Promoted memory {memory_id} to {new_layer}")

    async def search_by_graph_traversal(
        self,
        entity_names: list[str],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_depth: int = 2,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search memories by traversing the semantic graph.

        This enables finding memories connected through shared entities,
        predicates, or other semantic relationships.

        Args:
            entity_names: Entity names to start traversal from
            user_id: Filter by user
            agent_id: Filter by agent
            max_depth: Maximum graph traversal depth
            limit: Maximum results

        Returns:
            Memories connected to the given entities
        """
        if not self._initialized:
            await self.initialize()

        instance_id = self._get_instance_id(agent_id, user_id, None)

        with self.graph_store.driver.session(database=self.config.database) as session:
            result = session.run(f"""
                MATCH (start:Entity)
                WHERE start.canonical_name IN $entity_names
                  AND start.instance_id = $instance_id
                MATCH path = (start)-[*1..{max_depth}]-(connected:Entity)
                WHERE connected.is_memory_node = true
                  OR EXISTS {{ MATCH (connected)<-[:extracted_from]-(m:Memory) RETURN m }}
                WITH connected, length(path) AS distance
                MATCH (m:Memory)-[:extracted_from]->(connected)
                WHERE (m.memory_expires_at IS NULL OR m.memory_expires_at > $now)
                RETURN DISTINCT m, 1.0 / (1 + distance) AS score
                ORDER BY score DESC
                LIMIT $limit
            """,
                entity_names=entity_names,
                instance_id=instance_id,
                now=datetime.now().isoformat(),
                limit=limit,
            )

            search_results = []
            for record in result:
                memory = self._node_to_memory(dict(record["m"]))
                search_results.append(SearchResult(
                    memory=memory,
                    score=record["score"],
                ))

            return search_results

    async def garbage_collect(self) -> int:
        """Remove expired memories from Neo4j."""
        if not self._initialized:
            await self.initialize()

        now = datetime.now().isoformat()

        with self.graph_store.driver.session(database=self.config.database) as session:
            result = session.run("""
                MATCH (m:Memory)
                WHERE m.memory_expires_at IS NOT NULL
                  AND m.memory_expires_at < $now
                WITH m, m.graph_node_id AS graph_id
                DETACH DELETE m
                WITH graph_id WHERE graph_id IS NOT NULL
                MATCH (n:Entity {node_id: graph_id})
                DETACH DELETE n
                RETURN count(*) AS deleted
            """, now=now)

            record = result.single()
            deleted = record["deleted"] if record else 0

            logger.info(f"Garbage collected {deleted} expired memories")
            return deleted

    async def count(self, user_id: str | None = None) -> int:
        """Count memories, optionally filtered by user."""
        if not self._initialized:
            await self.initialize()

        with self.graph_store.driver.session(database=self.config.database) as session:
            if user_id:
                result = session.run("""
                    MATCH (m:Memory {user_id: $user_id})
                    WHERE (m.memory_expires_at IS NULL OR m.memory_expires_at > $now)
                    RETURN count(m) AS count
                """, user_id=user_id, now=datetime.now().isoformat())
            else:
                result = session.run("""
                    MATCH (m:Memory)
                    WHERE (m.memory_expires_at IS NULL OR m.memory_expires_at > $now)
                    RETURN count(m) AS count
                """, now=datetime.now().isoformat())

            record = result.single()
            return record["count"] if record else 0

    async def close(self) -> None:
        """Close connections."""
        if self._store:
            self._store.close()
        self._initialized = False

    def _get_instance_id(
        self,
        agent_id: str | None,
        user_id: str | None,
        context_id: str | None,
    ) -> str:
        """Generate instance ID for graph partitioning."""
        parts = []
        if agent_id:
            parts.append(f"agent:{agent_id}")
        if user_id:
            parts.append(f"user:{user_id}")
        if context_id:
            parts.append(f"ctx:{context_id}")
        return ":".join(parts) if parts else "default"

    def _node_to_memory(self, node_data: dict[str, Any]) -> Memory:
        """Convert Neo4j Memory node to Memory object."""
        return Memory(
            id=node_data.get("memory_id", ""),
            content=node_data.get("content", ""),
            memory_type=MemoryType(node_data.get("memory_type", "fact")),
            scope=MemoryScope(node_data.get("scope", "user")),
            agent_id=node_data.get("agent_id"),
            user_id=node_data.get("user_id"),
            context_id=node_data.get("context_id"),
            importance=node_data.get("importance", 0.5),
            confidence=node_data.get("confidence", 1.0),
            entities=node_data.get("entities", []),
            source=node_data.get("source"),
            stated_count=node_data.get("stated_count", 1),
            created_at=datetime.fromisoformat(node_data["created_at"]) if node_data.get("created_at") else datetime.now(),
            last_accessed=datetime.fromisoformat(node_data["last_accessed"]) if node_data.get("last_accessed") else None,
            expires_at=datetime.fromisoformat(node_data["memory_expires_at"]) if node_data.get("memory_expires_at") else None,
            linked_memories=node_data.get("linked_memories", []),
            supersedes=node_data.get("supersedes"),
            superseded_by=node_data.get("superseded_by"),
        )

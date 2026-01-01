"""4-Layer Memory Provider.

This module provides a MemoryProvider implementation that uses draagon-ai's
4-layer cognitive memory architecture (working, episodic, semantic, metacognitive).

The provider implements the MemoryProvider protocol while leveraging the full
power of the 4-layer system, including:
- Layer-appropriate storage based on memory type
- Cross-layer search aggregation
- Automatic memory promotion
- Entity extraction and relationship tracking

Usage:
    from draagon_ai.memory.providers import LayeredMemoryProvider

    memory = LayeredMemoryProvider()

    # Store a fact (goes to semantic layer)
    await memory.store(
        content="User's favorite color is blue",
        user_id="user123",
        memory_type="fact",
        entities=["user", "blue"],
    )

    # Search across all layers
    results = await memory.search("favorite color", user_id="user123")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from draagon_ai.memory.base import (
    Memory,
    MemoryProvider,
    MemoryScope,
    MemoryType,
    SearchResult,
)
from draagon_ai.memory.scopes import (
    ScopeType,
    Permission,
    HierarchicalScope,
    ScopeRegistry,
    get_scope_registry,
)
from draagon_ai.memory.layers import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    MetacognitiveMemory,
    LayerConfig,
    MemoryPromotion,
    MemoryConsolidator,
    PromotionConfig,
    PromotionStats,
)
from draagon_ai.memory.temporal_graph import TemporalCognitiveGraph, EmbeddingProvider

if TYPE_CHECKING:
    from draagon_ai.memory.providers.qdrant_graph import QdrantGraphStore

logger = logging.getLogger(__name__)


# Mapping from MemoryType to appropriate layer
LAYER_MAPPING = {
    MemoryType.FACT: "semantic",
    MemoryType.PREFERENCE: "semantic",
    MemoryType.KNOWLEDGE: "semantic",
    MemoryType.BELIEF: "semantic",
    MemoryType.RELATIONSHIP: "semantic",
    MemoryType.EPISODIC: "episodic",
    MemoryType.OBSERVATION: "episodic",
    MemoryType.SKILL: "metacognitive",
    MemoryType.INSIGHT: "metacognitive",
    MemoryType.SELF_KNOWLEDGE: "metacognitive",
    MemoryType.INSTRUCTION: "semantic",
}

# Importance weights for different memory types
IMPORTANCE_WEIGHTS = {
    MemoryType.INSTRUCTION: 0.9,
    MemoryType.FACT: 0.7,
    MemoryType.PREFERENCE: 0.6,
    MemoryType.SKILL: 0.8,
    MemoryType.INSIGHT: 0.7,
    MemoryType.EPISODIC: 0.5,
    MemoryType.OBSERVATION: 0.4,
    MemoryType.KNOWLEDGE: 0.6,
    MemoryType.BELIEF: 0.6,
    MemoryType.SELF_KNOWLEDGE: 0.8,
    MemoryType.RELATIONSHIP: 0.7,
}

# Scope hierarchy: WORLD > CONTEXT > AGENT > USER > SESSION
# Each scope can read its own level and all parent levels
SCOPE_HIERARCHY = [
    MemoryScope.WORLD,    # Level 0 - most general
    MemoryScope.CONTEXT,  # Level 1
    MemoryScope.AGENT,    # Level 2
    MemoryScope.USER,     # Level 3
    MemoryScope.SESSION,  # Level 4 - most specific
]

# Map MemoryScope to ScopeType for registry operations
SCOPE_TYPE_MAPPING = {
    MemoryScope.WORLD: ScopeType.WORLD,
    MemoryScope.CONTEXT: ScopeType.CONTEXT,
    MemoryScope.AGENT: ScopeType.AGENT,
    MemoryScope.USER: ScopeType.USER,
    MemoryScope.SESSION: ScopeType.SESSION,
}


def get_scope_level(scope: MemoryScope) -> int:
    """Get the hierarchy level of a scope (0=WORLD, 4=SESSION)."""
    return SCOPE_HIERARCHY.index(scope)


def get_accessible_scopes(query_scope: MemoryScope) -> list[MemoryScope]:
    """Get all scopes accessible from a query scope.

    Lower scopes can read higher scopes (more general).
    Example: USER can read USER, AGENT, CONTEXT, WORLD
             WORLD can only read WORLD

    Args:
        query_scope: The scope of the query/user

    Returns:
        List of accessible scopes (including query_scope and all parents)
    """
    level = get_scope_level(query_scope)
    return SCOPE_HIERARCHY[:level + 1]


def can_scope_read(query_scope: MemoryScope, target_scope: MemoryScope) -> bool:
    """Check if query_scope can read from target_scope.

    Args:
        query_scope: The scope making the query
        target_scope: The scope of the memory being accessed

    Returns:
        True if query_scope can read target_scope
    """
    # Lower (more specific) scopes can read higher (more general) scopes
    return get_scope_level(query_scope) >= get_scope_level(target_scope)


def can_scope_write(query_scope: MemoryScope, target_scope: MemoryScope) -> bool:
    """Check if query_scope can write to target_scope.

    Write is only allowed at same level or lower (more specific).
    Example: AGENT can write to AGENT, USER, SESSION but not WORLD, CONTEXT

    Args:
        query_scope: The scope making the write
        target_scope: The scope being written to

    Returns:
        True if query_scope can write to target_scope
    """
    # Can only write to same level or lower (more specific)
    return get_scope_level(query_scope) <= get_scope_level(target_scope)


@dataclass
class LayeredMemoryConfig:
    """Configuration for the layered memory provider.

    Example (Neo4j - RECOMMENDED):
        config = LayeredMemoryConfig(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            working_ttl_seconds=300,
            episodic_ttl_days=14,
            semantic_ttl_days=180,
            metacognitive_ttl_days=365,
        )
        provider = LayeredMemoryProvider(config=config, embedding_provider=embedder, llm_provider=llm)
        await provider.initialize()

    Example (Qdrant - DEPRECATED):
        config = LayeredMemoryConfig(
            qdrant_url="http://192.168.168.216:6333",
            working_ttl_seconds=300,
        )
        provider = LayeredMemoryProvider(config=config, embedding_provider=embedder)
        await provider.initialize()
    """

    # Neo4j settings (RECOMMENDED - if set, uses Neo4j with semantic decomposition)
    neo4j_uri: str | None = None
    neo4j_username: str = "neo4j"
    neo4j_password: str = "neo4j"
    neo4j_database: str = "neo4j"
    enable_semantic_decomposition: bool = True  # Run Phase 0/1 on all content

    # Qdrant settings (DEPRECATED - if None and no Neo4j, uses in-memory graph)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_nodes_collection: str = "draagon_memory_nodes"
    qdrant_edges_collection: str = "draagon_memory_edges"
    embedding_dimension: int = 768

    # Working memory config
    working_memory_capacity: int = 7
    working_memory_decay_rate: float = 0.1
    working_ttl_seconds: int = 300  # 5 minutes

    # Layer TTLs
    episodic_ttl_days: int = 14  # 2 weeks
    semantic_ttl_days: int = 180  # 6 months
    metacognitive_ttl_days: int | None = None  # None = permanent

    # Session management
    auto_create_session: bool = True
    session_timeout_minutes: int = 30

    # Entity extraction
    enable_entity_extraction: bool = True

    # Search config
    default_search_limit: int = 10
    search_threshold: float = 0.3

    # Promotion config - thresholds for layer transitions
    # Working → Episodic thresholds
    promotion_working_importance: float = 0.7
    promotion_working_access: int = 3
    promotion_working_min_age_minutes: int = 5

    # Episodic → Semantic thresholds
    promotion_episodic_importance: float = 0.75
    promotion_episodic_access: int = 5
    promotion_episodic_min_age_hours: int = 1

    # Semantic → Metacognitive thresholds
    promotion_semantic_importance: float = 0.85
    promotion_semantic_access: int = 10
    promotion_semantic_min_age_days: int = 7

    # Processing limits
    promotion_batch_size: int = 50
    promotion_max_per_cycle: int = 100

    # Scope enforcement
    enforce_scope_permissions: bool = False  # Enable hierarchical scope access control
    default_agent_id: str | None = None  # Agent ID for scope enforcement
    default_context_id: str | None = None  # Context ID (e.g., household)

    def get_working_ttl(self) -> timedelta:
        """Get working memory TTL as timedelta."""
        return timedelta(seconds=self.working_ttl_seconds)

    def get_episodic_ttl(self) -> timedelta:
        """Get episodic memory TTL as timedelta."""
        return timedelta(days=self.episodic_ttl_days)

    def get_semantic_ttl(self) -> timedelta:
        """Get semantic memory TTL as timedelta."""
        return timedelta(days=self.semantic_ttl_days)

    def get_metacognitive_ttl(self) -> timedelta | None:
        """Get metacognitive memory TTL as timedelta, or None for permanent."""
        if self.metacognitive_ttl_days is None:
            return None
        return timedelta(days=self.metacognitive_ttl_days)

    def get_promotion_config(self) -> PromotionConfig:
        """Get promotion configuration from these settings.

        Returns:
            PromotionConfig instance with settings from this config.
        """
        return PromotionConfig(
            # Working → Episodic
            working_importance_threshold=self.promotion_working_importance,
            working_access_threshold=self.promotion_working_access,
            working_min_age=timedelta(minutes=self.promotion_working_min_age_minutes),
            # Episodic → Semantic
            episodic_importance_threshold=self.promotion_episodic_importance,
            episodic_access_threshold=self.promotion_episodic_access,
            episodic_min_age=timedelta(hours=self.promotion_episodic_min_age_hours),
            # Semantic → Metacognitive
            semantic_importance_threshold=self.promotion_semantic_importance,
            semantic_access_threshold=self.promotion_semantic_access,
            semantic_min_age=timedelta(days=self.promotion_semantic_min_age_days),
            # Processing limits
            batch_size=self.promotion_batch_size,
            max_promotions_per_cycle=self.promotion_max_per_cycle,
        )


class LayeredMemoryProvider(MemoryProvider):
    """MemoryProvider that uses the 4-layer cognitive memory architecture.

    Layers:
    - Working: Session-scoped, limited capacity, high attention (default 5 min TTL)
    - Episodic: Episode/event sequences, chronologically linked (default 14 day TTL)
    - Semantic: Facts, entities, relationships, knowledge (default 180 day TTL)
    - Metacognitive: Skills, strategies, insights, behaviors (permanent by default)

    This provider maps incoming memories to the appropriate layer based on
    their MemoryType, and aggregates search results from all layers.

    Backend options (in priority order):
    1. Neo4j (RECOMMENDED): Full semantic graph with Phase 0/1 decomposition
    2. Qdrant (DEPRECATED): Vector storage only
    3. In-memory: No persistence

    Example:
        # With Neo4j semantic memory (RECOMMENDED)
        config = LayeredMemoryConfig(neo4j_uri="bolt://localhost:7687")
        provider = LayeredMemoryProvider(config=config, embedding_provider=embedder, llm_provider=llm)
        await provider.initialize()

        # With Qdrant persistence (DEPRECATED)
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config, embedding_provider=embedder)
        await provider.initialize()

        # In-memory mode
        provider = LayeredMemoryProvider()
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph | None = None,
        config: LayeredMemoryConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider: Any | None = None,  # LLMProvider for semantic decomposition
    ):
        """Initialize the layered memory provider.

        Args:
            graph: The temporal cognitive graph to use. If None, creates one
                   based on config (Neo4j > Qdrant > in-memory).
            config: Configuration options. If None, uses defaults.
            embedding_provider: Provider for generating embeddings. Required
                                for Neo4j/Qdrant modes and semantic search.
            llm_provider: LLM provider for semantic decomposition (Neo4j mode only).
        """
        self._config = config or LayeredMemoryConfig()
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._initialized = False
        self._owns_graph = graph is None  # Track if we created the graph

        # Neo4j semantic provider (for dual-pipeline approach)
        self._neo4j_provider = None
        self._pending_neo4j = False
        self._pending_qdrant = False

        # Create graph based on config priority: Neo4j > Qdrant > in-memory
        if graph is not None:
            self._graph = graph
        elif self._config.neo4j_uri:
            # Defer Neo4jMemoryProvider creation to initialize()
            self._graph = None  # type: ignore
            self._pending_neo4j = True
        elif self._config.qdrant_url:
            # Defer QdrantGraphStore creation to initialize()
            self._graph = None  # type: ignore
            self._pending_qdrant = True
        else:
            self._graph = TemporalCognitiveGraph(
                embedding_provider=embedding_provider,
            )

        # Session tracking
        self._session_id = str(uuid.uuid4())
        self._session_start = datetime.now()

        # Layers will be initialized in _setup_layers()
        self._working: WorkingMemory | None = None
        self._episodic: EpisodicMemory | None = None
        self._semantic: SemanticMemory | None = None
        self._metacognitive: MetacognitiveMemory | None = None

        # Promotion and consolidation services (initialized in _setup_layers)
        self._promotion: MemoryPromotion | None = None
        self._consolidator: MemoryConsolidator | None = None

        # Initialize layers if graph is ready
        if self._graph is not None:
            self._setup_layers()
            self._initialized = True

    async def initialize(self) -> None:
        """Initialize the provider, including Neo4j/Qdrant connection if configured.

        This method must be called before using the provider when Neo4j or Qdrant
        is configured. For in-memory mode, this is optional.

        Raises:
            RuntimeError: If Neo4j/Qdrant is configured but embedding_provider is not set.
        """
        if self._initialized:
            return

        if self._pending_neo4j:
            if not self._embedding_provider:
                raise RuntimeError(
                    "embedding_provider is required when using Neo4j backend. "
                    "Pass embedding_provider to LayeredMemoryProvider constructor."
                )

            # Import here to avoid circular imports
            from draagon_ai.memory.providers.neo4j import (
                Neo4jMemoryProvider,
                Neo4jMemoryConfig,
            )

            neo4j_config = Neo4jMemoryConfig(
                uri=self._config.neo4j_uri,
                username=self._config.neo4j_username,
                password=self._config.neo4j_password,
                database=self._config.neo4j_database,
                embedding_dimension=self._config.embedding_dimension,
                enable_semantic_decomposition=self._config.enable_semantic_decomposition,
            )

            self._neo4j_provider = Neo4jMemoryProvider(
                config=neo4j_config,
                embedding_provider=self._embedding_provider,
                llm_provider=self._llm_provider,
            )
            await self._neo4j_provider.initialize()

            # Still need TemporalCognitiveGraph for layer operations
            # Neo4j provider handles persistence, graph is for layer logic
            self._graph = TemporalCognitiveGraph(
                embedding_provider=self._embedding_provider,
            )

            logger.info(
                f"LayeredMemoryProvider initialized with Neo4j at {self._config.neo4j_uri}"
            )

        elif self._pending_qdrant:
            if not self._embedding_provider:
                raise RuntimeError(
                    "embedding_provider is required when using Qdrant backend. "
                    "Pass embedding_provider to LayeredMemoryProvider constructor."
                )

            # Import here to avoid circular imports
            from draagon_ai.memory.providers.qdrant_graph import (
                QdrantGraphStore,
                QdrantGraphConfig,
            )

            qdrant_config = QdrantGraphConfig(
                url=self._config.qdrant_url,
                api_key=self._config.qdrant_api_key,
                nodes_collection=self._config.qdrant_nodes_collection,
                edges_collection=self._config.qdrant_edges_collection,
                embedding_dimension=self._config.embedding_dimension,
            )

            self._graph = QdrantGraphStore(
                config=qdrant_config,
                embedding_provider=self._embedding_provider,
            )
            await self._graph.initialize()
            logger.info(
                f"LayeredMemoryProvider initialized with Qdrant at {self._config.qdrant_url} (DEPRECATED)"
            )

        self._setup_layers()
        self._initialized = True

    def _setup_layers(self) -> None:
        """Set up the 4 memory layers with configured TTLs."""
        # Working memory layer
        self._working = WorkingMemory(
            self._graph,
            session_id=self._session_id,
            capacity=self._config.working_memory_capacity,
            ttl=self._config.get_working_ttl(),
        )

        # Episodic memory layer
        self._episodic = EpisodicMemory(
            self._graph,
            ttl=self._config.get_episodic_ttl(),
        )

        # Semantic memory layer
        self._semantic = SemanticMemory(
            self._graph,
            ttl=self._config.get_semantic_ttl(),
        )

        # Metacognitive memory layer (permanent by default)
        self._metacognitive = MetacognitiveMemory(
            self._graph,
            ttl=self._config.get_metacognitive_ttl(),
        )

        # Set up promotion service
        promotion_config = self._config.get_promotion_config()
        self._promotion = MemoryPromotion(
            graph=self._graph,
            working=self._working,
            episodic=self._episodic,
            semantic=self._semantic,
            metacognitive=self._metacognitive,
            config=promotion_config,
        )

        # Set up consolidator (decay + cleanup + promotion)
        self._consolidator = MemoryConsolidator(
            graph=self._graph,
            working=self._working,
            episodic=self._episodic,
            semantic=self._semantic,
            metacognitive=self._metacognitive,
            promotion_config=promotion_config,
        )

        logger.debug("Memory layers and promotion service initialized")

    async def close(self) -> None:
        """Close the provider and release resources.

        Closes Neo4j/Qdrant connections and releases resources.
        """
        if self._neo4j_provider:
            await self._neo4j_provider.close()
            logger.info("LayeredMemoryProvider closed Neo4j connection")

        if self._owns_graph and hasattr(self._graph, 'close'):
            await self._graph.close()
            logger.info("LayeredMemoryProvider closed")

    # --- Properties ---

    def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized before use."""
        if not self._initialized:
            raise RuntimeError(
                "LayeredMemoryProvider not initialized. "
                "Call await provider.initialize() first when using Qdrant backend."
            )

    @property
    def graph(self) -> TemporalCognitiveGraph:
        """The underlying temporal cognitive graph."""
        self._ensure_initialized()
        return self._graph

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    @property
    def working(self) -> WorkingMemory:
        """Working memory layer."""
        self._ensure_initialized()
        return self._working  # type: ignore

    @property
    def episodic(self) -> EpisodicMemory:
        """Episodic memory layer."""
        self._ensure_initialized()
        return self._episodic  # type: ignore

    @property
    def semantic(self) -> SemanticMemory:
        """Semantic memory layer."""
        self._ensure_initialized()
        return self._semantic  # type: ignore

    @property
    def metacognitive(self) -> MetacognitiveMemory:
        """Metacognitive memory layer."""
        self._ensure_initialized()
        return self._metacognitive  # type: ignore

    @property
    def is_initialized(self) -> bool:
        """Whether the provider has been initialized."""
        return self._initialized

    @property
    def uses_neo4j(self) -> bool:
        """Whether this provider is configured to use Neo4j backend (RECOMMENDED)."""
        return self._config.neo4j_uri is not None

    @property
    def uses_qdrant(self) -> bool:
        """Whether this provider is configured to use Qdrant backend (DEPRECATED)."""
        return self._config.qdrant_url is not None and not self._config.neo4j_uri

    @property
    def neo4j_provider(self):
        """Get the Neo4j provider if using Neo4j backend."""
        return self._neo4j_provider

    @property
    def promotion(self) -> MemoryPromotion:
        """The memory promotion service."""
        self._ensure_initialized()
        return self._promotion  # type: ignore

    @property
    def consolidator(self) -> MemoryConsolidator:
        """The memory consolidator (decay + cleanup + promotion)."""
        self._ensure_initialized()
        return self._consolidator  # type: ignore

    # --- Promotion and Consolidation ---

    async def promote_all(self) -> PromotionStats:
        """Run a full promotion cycle across all layers.

        Promotes memories between layers based on configured thresholds:
        - Working → Episodic: Important session items become episodes
        - Episodic → Semantic: Entities and facts extracted from closed episodes
        - Semantic → Metacognitive: Skills and patterns elevated

        Returns:
            PromotionStats with counts and timing information.

        Example:
            stats = await provider.promote_all()
            print(f"Promoted {stats.total_promoted} items")
        """
        self._ensure_initialized()
        return await self._promotion.promote_all()

    async def promote_working_to_episodic(self) -> int:
        """Promote important working memory items to episodic memory.

        Working memory items meeting importance/access thresholds are
        converted to events within an episode.

        Returns:
            Number of items promoted.
        """
        self._ensure_initialized()
        result = await self._promotion.promote_working_to_episodic()
        return result.promoted_count

    async def promote_episodic_to_semantic(self) -> int:
        """Promote episodic memories to semantic knowledge.

        Closed episodes have their entities and facts extracted to
        the semantic layer.

        Returns:
            Number of episodes promoted.
        """
        self._ensure_initialized()
        result = await self._promotion.promote_episodic_to_semantic()
        return result.promoted_count

    async def promote_semantic_to_metacognitive(self) -> int:
        """Promote semantic knowledge to metacognitive layer.

        High-importance, frequently accessed semantic knowledge (especially
        skills and patterns) is elevated to metacognitive memory.

        Returns:
            Number of items promoted.
        """
        self._ensure_initialized()
        result = await self._promotion.promote_semantic_to_metacognitive()
        return result.promoted_count

    async def consolidate(self) -> dict[str, Any]:
        """Run a full consolidation cycle.

        Consolidation includes:
        1. Apply decay to all layers (reduces importance over time)
        2. Cleanup expired items (removes low-importance, old memories)
        3. Run promotions (moves memories between layers)

        This is typically called by a background job/scheduler.

        Returns:
            Dictionary with decay, cleanup, and promotion statistics.

        Example:
            stats = await provider.consolidate()
            print(f"Decayed: {sum(stats['decay'].values())}")
            print(f"Cleaned: {sum(stats['cleanup'].values())}")
            print(f"Promoted: {stats['promotion']['total']}")
        """
        self._ensure_initialized()
        return await self._consolidator.consolidate()

    def get_promotion_stats(self) -> dict[str, Any]:
        """Get current promotion service statistics.

        Returns:
            Dictionary with last promotion time, total promoted count,
            and current configuration.
        """
        self._ensure_initialized()
        return self._promotion.stats()

    # --- Session Management ---

    def set_session(self, session_id: str) -> None:
        """Set a new session ID and reset working memory.

        Args:
            session_id: New session identifier.
        """
        self._ensure_initialized()
        self._session_id = session_id
        self._session_start = datetime.now()

        # Recreate working memory for new session with configured TTL
        self._working = WorkingMemory(
            self._graph,
            session_id=self._session_id,
            capacity=self._config.working_memory_capacity,
            ttl=self._config.get_working_ttl(),
        )

    # --- MemoryProvider Interface ---

    async def store(
        self,
        content: str,
        memory_type: MemoryType | str,
        scope: MemoryScope | str = MemoryScope.USER,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        entities: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a memory in the appropriate layer.

        Args:
            content: The memory content.
            memory_type: Type of memory.
            scope: Visibility scope.
            agent_id: Owning agent.
            user_id: Associated user.
            context_id: Associated context.
            importance: Override default importance.
            confidence: Certainty level.
            entities: Extracted entities.
            source: Source of the memory.
            metadata: Additional metadata.

        Returns:
            The stored Memory object.

        Raises:
            RuntimeError: If provider not initialized (Qdrant mode).
            PermissionError: If scope enforcement enabled and write not allowed.
        """
        self._ensure_initialized()

        # Normalize types
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        if isinstance(scope, str):
            scope = MemoryScope(scope)

        # Validate scope write permission if enforcement enabled
        if self._config.enforce_scope_permissions:
            # Determine the caller's scope level (use agent_id or default)
            caller_agent = agent_id or self._config.default_agent_id
            if caller_agent:
                # Agent-level callers can write to AGENT, USER, SESSION
                caller_scope = MemoryScope.AGENT
            elif user_id:
                # User-level callers can write to USER, SESSION
                caller_scope = MemoryScope.USER
            else:
                # Default to SESSION (most restrictive)
                caller_scope = MemoryScope.SESSION

            if not can_scope_write(caller_scope, scope):
                raise PermissionError(
                    f"Cannot write to scope '{scope.value}' from scope '{caller_scope.value}'. "
                    f"Write requires same level or more specific scope."
                )

        # Calculate importance - use provided value or weight-based default
        if importance == 0.5:  # Default value, may want to use type weight
            importance = IMPORTANCE_WEIGHTS.get(memory_type, 0.5)

        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            scope=scope,
            agent_id=agent_id,
            user_id=user_id,
            context_id=context_id,
            importance=importance,
            confidence=confidence,
            entities=entities or [],
            source=source,
        )

        # If using Neo4j, store via Neo4jMemoryProvider (with semantic decomposition)
        if self._neo4j_provider:
            neo4j_memory = await self._neo4j_provider.store(
                content=content,
                memory_type=memory_type,
                scope=scope,
                agent_id=agent_id,
                user_id=user_id,
                context_id=context_id,
                importance=importance,
                confidence=confidence,
                entities=entities,
                metadata=metadata,
            )
            memory.id = neo4j_memory.id
            logger.debug(f"Stored memory {memory.id} via Neo4j with semantic decomposition")
        else:
            # Route to appropriate layer and get the node_id (Qdrant/in-memory path)
            layer = LAYER_MAPPING.get(memory_type, "semantic")

            if layer == "working":
                node_id = await self._store_working(memory, metadata)
            elif layer == "episodic":
                node_id = await self._store_episodic(memory, metadata)
            elif layer == "metacognitive":
                node_id = await self._store_metacognitive(memory, metadata)
            else:  # semantic is default
                node_id = await self._store_semantic(memory, metadata)

            # Update memory ID with actual graph node ID
            if node_id:
                memory.id = node_id

        return memory

    async def _store_working(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        """Store in working memory. Returns node_id."""
        item = await self._working.add_item(
            content=memory.content,
            attention_weight=memory.importance,
            source=memory.source or "store",
            user_id=memory.user_id,
        )
        return item.node_id if item else None

    async def _store_episodic(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        """Store in episodic memory. Returns node_id (event or episode)."""
        # Add as event to current episode or create new one
        episode = self._episodic.get_current_episode()

        if episode is None:
            episode = await self._episodic.start_episode(
                content=memory.content[:100],
                episode_type=str(memory.memory_type.value),
                entities=memory.entities,
            )

        event = await self._episodic.add_event(
            episode_id=episode.node_id,
            content=memory.content,
            event_type=str(memory.memory_type.value),
            entities=memory.entities,
        )
        return event.node_id if event else episode.node_id

    async def _store_semantic(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        """Store in semantic memory as a fact. Returns node_id."""
        fact = await self._semantic.add_fact(
            content=memory.content,
            entities=memory.entities,
            confidence=memory.confidence,
        )
        return fact.node_id if fact else None

    async def _store_metacognitive(
        self,
        memory: Memory,
        metadata: dict[str, Any] | None,
    ) -> str | None:
        """Store in metacognitive memory. Returns node_id."""
        result = None
        if memory.memory_type == MemoryType.SKILL:
            result = await self._metacognitive.add_skill(
                name=memory.content[:50],
                skill_type=metadata.get("skill_type", "general") if metadata else "general",
                procedure=memory.content,
                entities=memory.entities,
            )
        elif memory.memory_type == MemoryType.INSIGHT:
            result = await self._metacognitive.add_insight(
                content=memory.content,
                insight_type=metadata.get("insight_type", "observation") if metadata else "observation",
                context=memory.source or "",
            )
        else:
            # Store as insight by default
            result = await self._metacognitive.add_insight(
                content=memory.content,
                insight_type="observation",
                context=memory.source or "",
            )
        return result.node_id if result else None

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
        caller_scope: MemoryScope | None = None,
    ) -> list[SearchResult]:
        """Search across all memory layers.

        When scope enforcement is enabled, results are filtered to only include
        memories accessible from the caller's scope level.

        Scope hierarchy (lower can read higher):
        - SESSION can read: SESSION, USER, AGENT, CONTEXT, WORLD
        - USER can read: USER, AGENT, CONTEXT, WORLD
        - AGENT can read: AGENT, CONTEXT, WORLD
        - CONTEXT can read: CONTEXT, WORLD
        - WORLD can read: WORLD only

        Args:
            query: Search query.
            agent_id: Filter by agent.
            user_id: Filter by user.
            context_id: Filter by context.
            memory_types: Filter by types (searches all if None).
            scopes: Filter by scopes (auto-expanded based on caller_scope if enforcement enabled).
            limit: Max results.
            min_score: Minimum relevance score.
            caller_scope: The scope of the caller for access control (defaults to USER).

        Returns:
            List of SearchResult objects sorted by relevance.

        Raises:
            RuntimeError: If provider not initialized (Qdrant mode).
        """
        self._ensure_initialized()

        # Apply scope enforcement if enabled
        if self._config.enforce_scope_permissions:
            # Default caller scope based on provided IDs
            if caller_scope is None:
                if user_id:
                    caller_scope = MemoryScope.USER
                elif agent_id or self._config.default_agent_id:
                    caller_scope = MemoryScope.AGENT
                else:
                    caller_scope = MemoryScope.SESSION

            # Get all accessible scopes for this caller
            accessible = get_accessible_scopes(caller_scope)

            # If scopes provided, filter to only accessible ones
            if scopes:
                scopes = [s for s in scopes if s in accessible]
                if not scopes:
                    # No accessible scopes after filtering, return empty
                    return []
            else:
                # No scopes specified, use all accessible scopes
                scopes = accessible

        results: list[SearchResult] = []
        min_relevance = min_score or 0.0

        # If using Neo4j, search via Neo4jMemoryProvider
        if self._neo4j_provider:
            return await self._neo4j_provider.search(
                query=query,
                agent_id=agent_id,
                user_id=user_id,
                context_id=context_id,
                memory_types=memory_types,
                scopes=scopes,
                limit=limit,
                min_score=min_score,
            )

        # Fallback to layer-based search (Qdrant/in-memory path)
        # Determine which layers to search based on memory_types
        search_working = memory_types is None or any(
            t in (MemoryType.EPISODIC, MemoryType.OBSERVATION) for t in memory_types
        )
        search_episodic = memory_types is None or any(
            t in (MemoryType.EPISODIC, MemoryType.OBSERVATION) for t in memory_types
        )
        search_semantic = memory_types is None or any(
            t in (MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.KNOWLEDGE,
                  MemoryType.BELIEF, MemoryType.RELATIONSHIP, MemoryType.INSTRUCTION)
            for t in memory_types
        )
        search_metacognitive = memory_types is None or any(
            t in (MemoryType.SKILL, MemoryType.INSIGHT, MemoryType.SELF_KNOWLEDGE)
            for t in memory_types
        )

        # Search working memory
        if search_working:
            working_results = await self._working.search(query, limit=limit)
            for item in working_results:
                if item.attention_weight >= min_relevance:
                    results.append(
                        SearchResult(
                            memory=Memory(
                                id=item.item_id,
                                content=item.content,
                                memory_type=MemoryType.EPISODIC,
                                scope=MemoryScope.SESSION,
                                user_id=user_id,
                                importance=item.attention_weight,
                            ),
                            score=item.attention_weight,
                        )
                    )

        # Search episodic memory
        if search_episodic:
            episodic_results = await self._episodic.search(
                query=query,
                limit=limit,
                min_score=min_relevance,
            )
            for result in episodic_results:
                node = result.node
                relevance = result.score
                if relevance >= min_relevance:
                    results.append(
                        SearchResult(
                            memory=Memory(
                                id=node.node_id,
                                content=node.content,
                                memory_type=MemoryType.EPISODIC,
                                scope=MemoryScope.USER,
                                user_id=user_id,
                                importance=node.importance,
                            ),
                            score=relevance,
                        )
                    )

        # Search semantic memory (for facts)
        if search_semantic:
            semantic_results = await self._semantic.search(
                query=query,
                limit=limit,
                min_score=min_relevance,
            )
            for result in semantic_results:
                node = result.node
                relevance = result.score
                if relevance >= min_relevance:
                    results.append(
                        SearchResult(
                            memory=Memory(
                                id=node.node_id,
                                content=node.content,
                                memory_type=MemoryType.FACT,
                                scope=MemoryScope.USER,
                                user_id=user_id,
                                confidence=node.confidence,
                                entities=node.entities,
                            ),
                            score=relevance,
                        )
                    )

        # Search metacognitive memory (for skills)
        if search_metacognitive:
            metacognitive_results = await self._metacognitive.search(
                query=query,
                limit=limit,
                min_score=min_relevance,
            )
            for result in metacognitive_results:
                node = result.node
                relevance = result.score
                if relevance >= min_relevance:
                    results.append(
                        SearchResult(
                            memory=Memory(
                                id=node.node_id,
                                content=node.content,
                                memory_type=MemoryType.SKILL,
                                scope=MemoryScope.AGENT,
                                importance=node.importance,
                            ),
                            score=relevance,
                        )
                    )

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories by entity overlap.

        Finds memories that mention any of the given entities.
        Uses semantic search with entity names as the query, then
        filters and re-scores by actual entity overlap.

        Args:
            entities: List of entities to search for.
            agent_id: Filter by agent (not yet implemented).
            user_id: Filter by user (not yet implemented).
            limit: Maximum results to return.

        Returns:
            List of SearchResult objects sorted by entity overlap score.

        Example:
            # Find memories mentioning Doug or cats
            results = await provider.search_by_entities(["Doug", "cats"])
        """
        # TODO: Implement agent_id and user_id filtering
        _ = agent_id, user_id  # Suppress unused parameter warning
        self._ensure_initialized()

        if not entities:
            return []

        # Normalize entities for case-insensitive matching
        query_entities = {e.lower() for e in entities}

        # Use entity names as search query to leverage semantic search
        query = " ".join(entities)

        # Get more results than needed so we can filter and re-rank
        search_results = await self.search(query, limit=limit * 3)

        # Re-score by actual entity overlap
        scored_results: list[SearchResult] = []
        for result in search_results:
            memory_entities = {e.lower() for e in (result.memory.entities or [])}
            overlap = query_entities & memory_entities

            if overlap:
                # Score by fraction of query entities matched
                entity_score = len(overlap) / len(query_entities)
                # Blend with original semantic score
                combined_score = 0.7 * entity_score + 0.3 * result.score
                scored_results.append(
                    SearchResult(
                        memory=result.memory,
                        score=combined_score,
                    )
                )
            elif result.score >= 0.6:
                # Keep high-scoring semantic matches even without explicit entity overlap
                # (entities might not be extracted but content is relevant)
                scored_results.append(result)

        # Sort by score and limit
        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:limit]

    async def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The Memory if found, None otherwise.
        """
        self._ensure_initialized()

        # If using Neo4j, get from Neo4jMemoryProvider
        if self._neo4j_provider:
            return await self._neo4j_provider.get(memory_id)

        # Fallback to graph lookup
        node = await self._graph.get_node(memory_id)
        if node is None:
            return None

        return Memory(
            id=node.node_id,
            content=node.content,
            memory_type=MemoryType.FACT,  # Default
            scope=MemoryScope.USER,
            created_at=node.created_at,
        )

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update a memory.

        Args:
            memory_id: The memory to update.
            content: New content.
            importance: New importance.
            confidence: New confidence.
            metadata: New metadata.

        Returns:
            Updated Memory if found, None otherwise.
        """
        self._ensure_initialized()

        # If using Neo4j, update via Neo4jMemoryProvider
        if self._neo4j_provider:
            return await self._neo4j_provider.update(
                memory_id,
                content=content,
                importance=importance,
                confidence=confidence,
                metadata=metadata,
            )

        # Fallback to graph update
        node = await self._graph.get_node(memory_id)
        if node is None:
            return None

        if content is not None:
            node.content = content
        if importance is not None:
            node.importance = importance
        if confidence is not None:
            node.metadata["confidence"] = confidence
        if metadata is not None:
            node.metadata.update(metadata)

        return Memory(
            id=node.node_id,
            content=node.content,
            memory_type=MemoryType.FACT,  # Default
            scope=MemoryScope.USER,
            importance=node.importance,
            confidence=node.metadata.get("confidence", 1.0),
            created_at=node.created_at,
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: The memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_initialized()

        # If using Neo4j, delete via Neo4jMemoryProvider
        if self._neo4j_provider:
            return await self._neo4j_provider.delete(memory_id)

        # Fallback to graph delete
        try:
            return await self._graph.delete_node(memory_id)
        except Exception:
            return False

    # --- Memory Reinforcement ---

    # Reinforcement constants
    BOOST_AMOUNT = 0.05  # Importance boost per successful use
    DEMOTE_AMOUNT = 0.08  # Importance penalty per failure (slightly higher to be conservative)
    MAX_IMPORTANCE = 1.0
    MIN_IMPORTANCE = 0.1

    # Layer promotion thresholds (based on importance)
    PROMOTION_THRESHOLD_WORKING_TO_EPISODIC = 0.7
    PROMOTION_THRESHOLD_EPISODIC_TO_SEMANTIC = 0.8
    PROMOTION_THRESHOLD_SEMANTIC_TO_METACOGNITIVE = 0.9

    # Layer demotion thresholds
    DEMOTION_THRESHOLD_METACOGNITIVE_TO_SEMANTIC = 0.6
    DEMOTION_THRESHOLD_SEMANTIC_TO_EPISODIC = 0.4
    DEMOTION_THRESHOLD_EPISODIC_TO_WORKING = 0.3

    async def boost_memory(
        self,
        memory_id: str,
        boost_amount: float | None = None,
    ) -> Memory | None:
        """Boost a memory's importance after successful use.

        When a memory is retrieved and contributes to a correct response,
        its importance increases. High-importance memories get promoted
        to higher layers (episodic → semantic → metacognitive).

        Args:
            memory_id: The memory to boost.
            boost_amount: Custom boost amount (default: BOOST_AMOUNT).

        Returns:
            Updated Memory if found, None otherwise.

        Example:
            # After memory helped answer a question correctly
            await provider.boost_memory(memory_id)

            # Custom boost for highly valuable usage
            await provider.boost_memory(memory_id, boost_amount=0.1)
        """
        self._ensure_initialized()
        node = await self._graph.get_node(memory_id)
        if node is None:
            return None

        boost = boost_amount or self.BOOST_AMOUNT
        old_importance = node.importance
        new_importance = min(old_importance + boost, self.MAX_IMPORTANCE)
        node.importance = new_importance

        # Track access for promotion eligibility
        node.metadata["access_count"] = node.metadata.get("access_count", 0) + 1
        node.metadata["last_successful_use"] = datetime.now().isoformat()

        logger.debug(
            f"Boosted memory {memory_id}: {old_importance:.2f} → {new_importance:.2f}"
        )

        # Check if promotion is warranted
        await self._check_promotion_on_boost(memory_id, node, new_importance)

        return Memory(
            id=node.node_id,
            content=node.content,
            memory_type=MemoryType.FACT,  # Default
            scope=MemoryScope.USER,
            importance=new_importance,
            created_at=node.created_at,
        )

    async def demote_memory(
        self,
        memory_id: str,
        demote_amount: float | None = None,
    ) -> Memory | None:
        """Demote a memory's importance after failed use.

        When a memory is retrieved but leads to an incorrect response
        or conflict, its importance decreases. Low-importance memories
        in higher layers get demoted to lower layers.

        Args:
            memory_id: The memory to demote.
            demote_amount: Custom demotion amount (default: DEMOTE_AMOUNT).

        Returns:
            Updated Memory if found, None otherwise.

        Example:
            # After memory led to a wrong answer
            await provider.demote_memory(memory_id)
        """
        self._ensure_initialized()
        node = await self._graph.get_node(memory_id)
        if node is None:
            return None

        demote = demote_amount or self.DEMOTE_AMOUNT
        old_importance = node.importance
        new_importance = max(old_importance - demote, self.MIN_IMPORTANCE)
        node.importance = new_importance

        # Track failure
        node.metadata["failure_count"] = node.metadata.get("failure_count", 0) + 1
        node.metadata["last_failure"] = datetime.now().isoformat()

        logger.debug(
            f"Demoted memory {memory_id}: {old_importance:.2f} → {new_importance:.2f}"
        )

        # Check if demotion to lower layer is warranted
        await self._check_demotion_on_penalty(memory_id, node, new_importance)

        return Memory(
            id=node.node_id,
            content=node.content,
            memory_type=MemoryType.FACT,  # Default
            scope=MemoryScope.USER,
            importance=new_importance,
            created_at=node.created_at,
        )

    async def record_usage(
        self,
        memory_id: str,
        outcome: str,
        confidence: float = 1.0,
    ) -> Memory | None:
        """Record memory usage with outcome for reinforcement learning.

        This is the primary interface for memory reinforcement. Call this
        after using a memory to update its importance based on outcome.

        Args:
            memory_id: The memory that was used.
            outcome: "success" | "failure" | "neutral"
            confidence: How confident we are in the outcome (0-1).

        Returns:
            Updated Memory if found, None otherwise.

        Example:
            # After using memory to answer correctly
            await provider.record_usage(memory_id, "success")

            # After memory led to wrong answer
            await provider.record_usage(memory_id, "failure")

            # After using memory but outcome unclear
            await provider.record_usage(memory_id, "neutral")
        """
        if outcome == "success":
            # Scale boost by confidence
            boost = self.BOOST_AMOUNT * confidence
            return await self.boost_memory(memory_id, boost_amount=boost)
        elif outcome == "failure":
            # Scale demote by confidence
            demote = self.DEMOTE_AMOUNT * confidence
            return await self.demote_memory(memory_id, demote_amount=demote)
        else:
            # Neutral: just record access without changing importance
            self._ensure_initialized()
            node = await self._graph.get_node(memory_id)
            if node is None:
                return None
            node.metadata["access_count"] = node.metadata.get("access_count", 0) + 1
            return Memory(
                id=node.node_id,
                content=node.content,
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER,
                importance=node.importance,
                created_at=node.created_at,
            )

    async def _check_promotion_on_boost(
        self,
        memory_id: str,
        node: Any,
        new_importance: float,
    ) -> None:
        """Check if a boosted memory should be promoted to a higher layer."""
        current_layer = node.metadata.get("layer", "working")

        # Determine if promotion threshold is met
        should_promote = False
        target_layer = None

        if current_layer == "working" and new_importance >= self.PROMOTION_THRESHOLD_WORKING_TO_EPISODIC:
            should_promote = True
            target_layer = "episodic"
        elif current_layer == "episodic" and new_importance >= self.PROMOTION_THRESHOLD_EPISODIC_TO_SEMANTIC:
            should_promote = True
            target_layer = "semantic"
        elif current_layer == "semantic" and new_importance >= self.PROMOTION_THRESHOLD_SEMANTIC_TO_METACOGNITIVE:
            should_promote = True
            target_layer = "metacognitive"

        if should_promote and target_layer:
            logger.info(
                f"Memory {memory_id} promoted: {current_layer} → {target_layer} "
                f"(importance: {new_importance:.2f})"
            )
            node.metadata["layer"] = target_layer
            node.metadata["promoted_at"] = datetime.now().isoformat()
            node.metadata["promotion_reason"] = "importance_threshold"

    async def _check_demotion_on_penalty(
        self,
        memory_id: str,
        node: Any,
        new_importance: float,
    ) -> None:
        """Check if a demoted memory should move to a lower layer."""
        current_layer = node.metadata.get("layer", "working")

        # Determine if demotion threshold is met
        should_demote = False
        target_layer = None

        if current_layer == "metacognitive" and new_importance < self.DEMOTION_THRESHOLD_METACOGNITIVE_TO_SEMANTIC:
            should_demote = True
            target_layer = "semantic"
        elif current_layer == "semantic" and new_importance < self.DEMOTION_THRESHOLD_SEMANTIC_TO_EPISODIC:
            should_demote = True
            target_layer = "episodic"
        elif current_layer == "episodic" and new_importance < self.DEMOTION_THRESHOLD_EPISODIC_TO_WORKING:
            should_demote = True
            target_layer = "working"

        if should_demote and target_layer:
            logger.warning(
                f"Memory {memory_id} demoted: {current_layer} → {target_layer} "
                f"(importance: {new_importance:.2f})"
            )
            node.metadata["layer"] = target_layer
            node.metadata["demoted_at"] = datetime.now().isoformat()
            node.metadata["demotion_reason"] = "importance_threshold"


__all__ = [
    "LayeredMemoryProvider",
    "LayeredMemoryConfig",
    "LAYER_MAPPING",
    "IMPORTANCE_WEIGHTS",
    # Scope hierarchy helpers
    "SCOPE_HIERARCHY",
    "SCOPE_TYPE_MAPPING",
    "get_scope_level",
    "get_accessible_scopes",
    "can_scope_read",
    "can_scope_write",
    # Re-exported from layers for convenience
    "PromotionStats",
    "PromotionConfig",
]

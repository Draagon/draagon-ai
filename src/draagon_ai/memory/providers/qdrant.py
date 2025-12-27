"""Qdrant-backed Memory Provider for draagon-ai.

This module provides real persistence for the memory system using Qdrant vector database.
It replaces the in-memory implementation with production-ready storage.

Key Features:
- Vector similarity search via Qdrant
- Bi-temporal tracking (event_time + ingestion_time)
- Hierarchical scope filtering
- Memory type filtering
- Importance-weighted retrieval

Requires:
- qdrant-client package
- Running Qdrant instance

Example:
    from draagon_ai.memory.providers.qdrant import QdrantMemoryProvider, QdrantConfig

    config = QdrantConfig(
        url="http://192.168.168.216:6333",
        collection_name="draagon_memories",
    )

    provider = QdrantMemoryProvider(config, embedding_provider)
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
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        MatchAny,
        UpdateStatus,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    AsyncQdrantClient = None

from draagon_ai.memory.base import (
    MemoryProvider,
    MemoryType,
    MemoryScope,
    Memory,
    SearchResult,
    MemoryConfig,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection."""

    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "draagon_memories"
    embedding_dimension: int = 768

    # Search settings
    default_limit: int = 5
    similarity_threshold: float = 0.3

    # Collection settings
    on_disk_payload: bool = True
    quantization: bool = False

    # Timeouts
    timeout: float = 30.0

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


class QdrantMemoryProvider(MemoryProvider):
    """Qdrant-backed implementation of MemoryProvider.

    This provides real persistence for draagon-ai memories using Qdrant.
    All memories are stored as vectors with rich metadata for filtering.

    Payload Schema:
        - content: str - The memory text
        - memory_type: str - Type (fact, skill, preference, etc.)
        - scope: str - Visibility scope
        - agent_id: str | None - Owning agent
        - user_id: str | None - Associated user
        - context_id: str | None - Associated context (household)
        - importance: float - 0-1 importance score
        - confidence: float - 0-1 confidence score
        - entities: list[str] - Extracted entities
        - stated_count: int - Times this was stated
        - created_at: str - ISO timestamp
        - last_accessed: str | None - ISO timestamp
        - expires_at: str | None - ISO timestamp
        - linked_memories: list[str] - Related memory IDs
        - supersedes: str | None - ID of memory this replaces
        - superseded_by: str | None - ID of newer memory
        - metadata: dict - Additional metadata
    """

    def __init__(
        self,
        config: QdrantConfig,
        embedding_provider: EmbeddingProvider,
    ):
        """Initialize the Qdrant provider.

        Args:
            config: Qdrant configuration
            embedding_provider: Provider for generating embeddings
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for QdrantMemoryProvider. "
                "Install with: pip install qdrant-client"
            )

        self.config = config
        self._embedder = embedding_provider
        self._client: AsyncQdrantClient | None = None
        self._sync_client: QdrantClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection and ensure collection exists."""
        if self._initialized:
            return

        # Create async client
        self._client = AsyncQdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

        # Create sync client for some operations
        self._sync_client = QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

        # Ensure collection exists
        await self._ensure_collection()

        self._initialized = True
        logger.info(f"QdrantMemoryProvider initialized with collection: {self.config.collection_name}")

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = await self._client.get_collections()
        exists = any(c.name == self.config.collection_name for c in collections.collections)

        if not exists:
            await self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE,
                    on_disk=self.config.on_disk_payload,
                ),
            )

            # Create indexes for common filters
            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="user_id",
                field_schema="keyword",
            )
            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="agent_id",
                field_schema="keyword",
            )
            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="memory_type",
                field_schema="keyword",
            )
            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="scope",
                field_schema="keyword",
            )
            await self._client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="context_id",
                field_schema="keyword",
            )

            logger.info(f"Created Qdrant collection: {self.config.collection_name}")

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
        """Store a new memory in Qdrant."""
        if not self._initialized:
            await self.initialize()

        # Use UUID for Qdrant point ID (Qdrant requires int or UUID)
        memory_id = str(uuid4())
        now = datetime.now()

        # Generate embedding
        embedding = await self._embedder.embed(content)

        # Build payload
        payload = {
            "content": content,
            "memory_type": memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            "scope": scope.value if isinstance(scope, MemoryScope) else scope,
            "agent_id": agent_id,
            "user_id": user_id,
            "context_id": context_id,
            "importance": importance,
            "confidence": confidence,
            "entities": entities or [],
            "stated_count": 1,
            "created_at": now.isoformat(),
            "last_accessed": None,
            "expires_at": None,
            "linked_memories": [],
            "supersedes": None,
            "superseded_by": None,
            "metadata": metadata or {},
        }

        # Store in Qdrant
        await self._client.upsert(
            collection_name=self.config.collection_name,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        logger.debug(f"Stored memory {memory_id}: {content[:50]}...")

        return Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type if isinstance(memory_type, MemoryType) else MemoryType(memory_type),
            scope=scope if isinstance(scope, MemoryScope) else MemoryScope(scope),
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
        """Search memories by semantic similarity."""
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self._embedder.embed(query)

        # Build filter conditions
        must_conditions = []

        if user_id:
            must_conditions.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            )

        if agent_id:
            must_conditions.append(
                FieldCondition(
                    key="agent_id",
                    match=MatchValue(value=agent_id),
                )
            )

        if context_id:
            must_conditions.append(
                FieldCondition(
                    key="context_id",
                    match=MatchValue(value=context_id),
                )
            )

        if memory_types:
            type_values = [
                t.value if isinstance(t, MemoryType) else t
                for t in memory_types
            ]
            must_conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchAny(any=type_values),
                )
            )

        if scopes:
            scope_values = [
                s.value if isinstance(s, MemoryScope) else s
                for s in scopes
            ]
            must_conditions.append(
                FieldCondition(
                    key="scope",
                    match=MatchAny(any=scope_values),
                )
            )

        # Build filter
        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Search using query_points (async API)
        response = await self._client.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=min_score or self.config.similarity_threshold,
            with_payload=True,
        )

        # Convert to SearchResult
        search_results = []
        for result in response.points:
            payload = result.payload
            memory = self._payload_to_memory(str(result.id), payload)

            # Apply type importance weighting
            type_weight = self.config.type_weights.get(payload.get("memory_type", "fact"), 0.5)
            weighted_score = result.score * type_weight

            search_results.append(SearchResult(
                memory=memory,
                score=weighted_score,
            ))

        # Re-sort by weighted score
        search_results.sort(key=lambda r: r.score, reverse=True)

        return search_results

    async def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        if not self._initialized:
            await self.initialize()

        try:
            results = await self._client.retrieve(
                collection_name=self.config.collection_name,
                ids=[memory_id],
                with_payload=True,
            )

            if not results:
                return None

            point = results[0]
            return self._payload_to_memory(str(point.id), point.payload)

        except Exception as e:
            logger.warning(f"Error retrieving memory {memory_id}: {e}")
            return None

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

        # Build update payload
        update_payload = {}

        if content is not None:
            update_payload["content"] = content
            # Re-embed if content changed
            embedding = await self._embedder.embed(content)
            # Update vector
            await self._client.update_vectors(
                collection_name=self.config.collection_name,
                points=[
                    qdrant_models.PointVectors(
                        id=memory_id,
                        vector=embedding,
                    )
                ],
            )

        if importance is not None:
            update_payload["importance"] = importance

        if confidence is not None:
            update_payload["confidence"] = confidence

        if metadata is not None:
            # Get existing metadata and merge
            existing_metadata = existing.source or {}
            if isinstance(existing_metadata, str):
                existing_metadata = {}
            merged_metadata = {**existing_metadata, **metadata}
            update_payload["metadata"] = merged_metadata

        # Update last accessed
        update_payload["last_accessed"] = datetime.now().isoformat()

        if update_payload:
            await self._client.set_payload(
                collection_name=self.config.collection_name,
                payload=update_payload,
                points=[memory_id],
            )

        return await self.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self._initialized:
            await self.initialize()

        try:
            result = await self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[memory_id],
                ),
            )
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            logger.warning(f"Error deleting memory {memory_id}: {e}")
            return False

    async def reinforce(self, memory_id: str, boost: float = 0.1) -> Memory | None:
        """Reinforce a memory by boosting importance and updating access time."""
        if not self._initialized:
            await self.initialize()

        existing = await self.get(memory_id)
        if not existing:
            return None

        new_importance = min(1.0, existing.importance + boost)

        await self._client.set_payload(
            collection_name=self.config.collection_name,
            payload={
                "importance": new_importance,
                "last_accessed": datetime.now().isoformat(),
            },
            points=[memory_id],
        )

        return await self.get(memory_id)

    def _payload_to_memory(self, memory_id: str, payload: dict[str, Any]) -> Memory:
        """Convert Qdrant payload to Memory object."""
        return Memory(
            id=memory_id,
            content=payload.get("content", ""),
            memory_type=MemoryType(payload.get("memory_type", "fact")),
            scope=MemoryScope(payload.get("scope", "user")),
            agent_id=payload.get("agent_id"),
            user_id=payload.get("user_id"),
            context_id=payload.get("context_id"),
            importance=payload.get("importance", 0.5),
            confidence=payload.get("confidence", 1.0),
            entities=payload.get("entities", []),
            source=payload.get("source"),
            stated_count=payload.get("stated_count", 1),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.now(),
            last_accessed=datetime.fromisoformat(payload["last_accessed"]) if payload.get("last_accessed") else None,
            expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
            linked_memories=payload.get("linked_memories", []),
            supersedes=payload.get("supersedes"),
            superseded_by=payload.get("superseded_by"),
        )

    async def count(self, user_id: str | None = None) -> int:
        """Count memories, optionally filtered by user."""
        if not self._initialized:
            await self.initialize()

        if user_id:
            result = await self._client.count(
                collection_name=self.config.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                ),
            )
        else:
            result = await self._client.count(
                collection_name=self.config.collection_name,
            )

        return result.count

    async def close(self) -> None:
        """Close connections."""
        if self._client:
            await self._client.close()
        if self._sync_client:
            self._sync_client.close()
        self._initialized = False


# =============================================================================
# Prompt-specific storage
# =============================================================================


@dataclass
class PromptVersion:
    """A versioned prompt stored in Qdrant."""

    id: str
    name: str  # e.g., "DECISION_PROMPT", "home_automation.HA_DEVICE_RESOLUTION"
    domain: str  # e.g., "core", "home_automation", "conversation_modes"
    version: int
    content: str

    # Lineage
    parent_id: str | None = None
    mutation_reason: str | None = None

    # Status
    status: str = "draft"  # draft, shadow, active, archived

    # Metrics
    usage_count: int = 0
    success_count: int = 0
    fitness_score: float = 0.5

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: datetime | None = None

    # Metadata
    created_by: str = "system"
    tags: list[str] = field(default_factory=list)


@dataclass
class PromptDomain:
    """A domain of related prompts."""

    name: str  # e.g., "home_automation"
    description: str
    prompts: list[str]  # Prompt names in this domain
    dependencies: list[str] = field(default_factory=list)  # Other domains this depends on


class QdrantPromptProvider:
    """Qdrant-backed storage for prompts with versioning and evolution.

    This enables:
    - Storing prompts in Qdrant for dynamic loading
    - Version control with lineage tracking
    - A/B testing via shadow versions
    - Evolution via mutation tracking
    - Domain-based organization

    Collection Schema (prompts):
        - name: str - Prompt name (e.g., "DECISION_PROMPT")
        - domain: str - Domain (e.g., "core", "home_automation")
        - version: int - Version number
        - content: str - The prompt text
        - status: str - draft/shadow/active/archived
        - parent_id: str | None - Previous version
        - mutation_reason: str | None - Why this was created
        - usage_count: int - Times used
        - success_count: int - Successful uses
        - fitness_score: float - Evolution fitness
        - created_at: str - ISO timestamp
        - activated_at: str | None - When activated
        - created_by: str - Who created this
        - tags: list[str] - Tags
    """

    def __init__(
        self,
        config: QdrantConfig,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "draagon_prompts",
    ):
        """Initialize prompt provider.

        Args:
            config: Qdrant configuration
            embedding_provider: For generating prompt embeddings
            collection_name: Collection for prompts
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client required")

        self.config = config
        self._embedder = embedding_provider
        self.collection_name = collection_name
        self._client: AsyncQdrantClient | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection and collection."""
        if self._initialized:
            return

        self._client = AsyncQdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

        await self._ensure_collection()
        self._initialized = True
        logger.info(f"QdrantPromptProvider initialized: {self.collection_name}")

    async def _ensure_collection(self) -> None:
        """Create collection if needed."""
        collections = await self._client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

            # Create indexes
            for field in ["name", "domain", "status", "created_by"]:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )

            logger.info(f"Created prompt collection: {self.collection_name}")

    async def store_prompt(
        self,
        name: str,
        domain: str,
        content: str,
        *,
        parent_id: str | None = None,
        mutation_reason: str | None = None,
        status: str = "draft",
        created_by: str = "system",
        tags: list[str] | None = None,
    ) -> PromptVersion:
        """Store a new prompt version.

        Args:
            name: Prompt name (e.g., "DECISION_PROMPT")
            domain: Domain (e.g., "core", "home_automation")
            content: The prompt text
            parent_id: ID of parent version (for mutations)
            mutation_reason: Why this version was created
            status: Initial status
            created_by: Who created this
            tags: Optional tags

        Returns:
            The stored PromptVersion
        """
        if not self._initialized:
            await self.initialize()

        # Get next version number
        version = await self._get_next_version(name)

        # Use UUID for Qdrant point ID (Qdrant requires int or UUID)
        prompt_id = str(uuid4())
        now = datetime.now()

        # Generate embedding
        embedding = await self._embedder.embed(content)

        # Build payload
        payload = {
            "name": name,
            "domain": domain,
            "version": version,
            "content": content,
            "parent_id": parent_id,
            "mutation_reason": mutation_reason,
            "status": status,
            "usage_count": 0,
            "success_count": 0,
            "fitness_score": 0.5,
            "created_at": now.isoformat(),
            "activated_at": None,
            "created_by": created_by,
            "tags": tags or [],
        }

        # Store
        await self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=prompt_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        logger.info(f"Stored prompt {name} v{version} ({status})")

        return PromptVersion(
            id=prompt_id,
            name=name,
            domain=domain,
            version=version,
            content=content,
            parent_id=parent_id,
            mutation_reason=mutation_reason,
            status=status,
            created_at=now,
            created_by=created_by,
            tags=tags or [],
        )

    async def get_active_prompt(self, name: str) -> PromptVersion | None:
        """Get the active version of a prompt.

        Args:
            name: Prompt name

        Returns:
            Active PromptVersion or None
        """
        if not self._initialized:
            await self.initialize()

        results = await self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="name", match=MatchValue(value=name)),
                    FieldCondition(key="status", match=MatchValue(value="active")),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        points, _ = results
        if not points:
            return None

        return self._payload_to_prompt(str(points[0].id), points[0].payload)

    async def get_prompt_by_id(self, prompt_id: str) -> PromptVersion | None:
        """Get a specific prompt version by ID."""
        if not self._initialized:
            await self.initialize()

        try:
            results = await self._client.retrieve(
                collection_name=self.collection_name,
                ids=[prompt_id],
                with_payload=True,
            )

            if not results:
                return None

            return self._payload_to_prompt(str(results[0].id), results[0].payload)
        except Exception:
            return None

    async def list_prompts(
        self,
        domain: str | None = None,
        status: str | None = None,
    ) -> list[PromptVersion]:
        """List prompts with optional filtering.

        Args:
            domain: Filter by domain
            status: Filter by status

        Returns:
            List of PromptVersions
        """
        if not self._initialized:
            await self.initialize()

        conditions = []
        if domain:
            conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        if status:
            conditions.append(
                FieldCondition(key="status", match=MatchValue(value=status))
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = await self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=100,
            with_payload=True,
        )

        points, _ = results
        return [
            self._payload_to_prompt(str(p.id), p.payload)
            for p in points
        ]

    async def activate_prompt(self, prompt_id: str) -> bool:
        """Activate a prompt version (deactivates other versions of same name).

        Args:
            prompt_id: ID of prompt to activate

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        prompt = await self.get_prompt_by_id(prompt_id)
        if not prompt:
            return False

        # Deactivate current active version
        current_active = await self.get_active_prompt(prompt.name)
        if current_active and current_active.id != prompt_id:
            await self._client.set_payload(
                collection_name=self.collection_name,
                payload={"status": "archived"},
                points=[current_active.id],
            )

        # Activate new version
        await self._client.set_payload(
            collection_name=self.collection_name,
            payload={
                "status": "active",
                "activated_at": datetime.now().isoformat(),
            },
            points=[prompt_id],
        )

        logger.info(f"Activated prompt {prompt.name} v{prompt.version}")
        return True

    async def record_usage(
        self,
        prompt_id: str,
        success: bool,
    ) -> None:
        """Record usage of a prompt for fitness tracking.

        Args:
            prompt_id: Prompt that was used
            success: Whether the interaction was successful
        """
        if not self._initialized:
            await self.initialize()

        prompt = await self.get_prompt_by_id(prompt_id)
        if not prompt:
            return

        # Update counts
        new_usage = prompt.usage_count + 1
        new_success = prompt.success_count + (1 if success else 0)
        new_fitness = new_success / new_usage if new_usage > 0 else 0.5

        await self._client.set_payload(
            collection_name=self.collection_name,
            payload={
                "usage_count": new_usage,
                "success_count": new_success,
                "fitness_score": new_fitness,
            },
            points=[prompt_id],
        )

    async def search_similar_prompts(
        self,
        query: str,
        limit: int = 5,
    ) -> list[tuple[PromptVersion, float]]:
        """Search for prompts similar to a query.

        Useful for finding prompts that might handle similar tasks.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of (PromptVersion, score) tuples
        """
        if not self._initialized:
            await self.initialize()

        embedding = await self._embedder.embed(query)

        # Search using query_points (async API)
        response = await self._client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        )

        return [
            (self._payload_to_prompt(str(r.id), r.payload), r.score)
            for r in response.points
        ]

    async def _get_next_version(self, name: str) -> int:
        """Get the next version number for a prompt."""
        results = await self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="name", match=MatchValue(value=name)),
                ]
            ),
            limit=100,
            with_payload=True,
        )

        points, _ = results
        if not points:
            return 1

        max_version = max(p.payload.get("version", 0) for p in points)
        return max_version + 1

    def _payload_to_prompt(self, prompt_id: str, payload: dict[str, Any]) -> PromptVersion:
        """Convert payload to PromptVersion."""
        return PromptVersion(
            id=prompt_id,
            name=payload.get("name", ""),
            domain=payload.get("domain", ""),
            version=payload.get("version", 1),
            content=payload.get("content", ""),
            parent_id=payload.get("parent_id"),
            mutation_reason=payload.get("mutation_reason"),
            status=payload.get("status", "draft"),
            usage_count=payload.get("usage_count", 0),
            success_count=payload.get("success_count", 0),
            fitness_score=payload.get("fitness_score", 0.5),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.now(),
            activated_at=datetime.fromisoformat(payload["activated_at"]) if payload.get("activated_at") else None,
            created_by=payload.get("created_by", "system"),
            tags=payload.get("tags", []),
        )

    async def close(self) -> None:
        """Close connections."""
        if self._client:
            await self._client.close()
        self._initialized = False

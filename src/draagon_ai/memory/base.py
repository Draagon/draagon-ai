"""Abstract base classes for memory providers.

These protocols define what the cognitive engine needs from memory systems.
Host applications implement these interfaces with their specific backends
(Qdrant, Pinecone, Chroma, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Types of memories for structured storage.

    These types help the cognitive engine organize and prioritize information.
    """

    FACT = "fact"  # Declarative knowledge ("Doug's birthday is March 15")
    PREFERENCE = "preference"  # User preferences ("Doug prefers dark mode")
    EPISODIC = "episodic"  # Episode memories ("Last Tuesday we discussed...")
    INSTRUCTION = "instruction"  # Explicit instructions ("Always use Celsius")
    KNOWLEDGE = "knowledge"  # Imported knowledge base content
    SKILL = "skill"  # Procedural knowledge ("How to restart service: ...")
    INSIGHT = "insight"  # Meta-learnings about tasks/patterns
    SELF_KNOWLEDGE = "self_knowledge"  # Agent's knowledge about itself
    RELATIONSHIP = "relationship"  # User relationship tracking data
    OBSERVATION = "observation"  # Raw observations before belief reconciliation
    BELIEF = "belief"  # Reconciled beliefs


class MemoryScope(str, Enum):
    """Scope of memory visibility.

    Maps to the cognitive architecture's hierarchical scoping.
    """

    # Hierarchical scopes (from COGNITIVE_ARCHITECTURE)
    WORLD = "world"  # Universal facts (physics, geography)
    CONTEXT = "context"  # Shared within a context (household, team)
    AGENT = "agent"  # Agent's personal memories
    USER = "user"  # Per-user memories within an agent
    SESSION = "session"  # Conversation-specific memories

    # Legacy scopes (for backward compatibility)
    PRIVATE = "private"  # User-specific (alias for USER)
    SHARED = "shared"  # Family/household (alias for CONTEXT)
    PUBLIC = "public"  # Public information (alias for WORLD)
    SYSTEM = "system"  # Internal system knowledge


@dataclass
class Memory:
    """A memory stored in the system.

    This is the canonical representation used by cognitive services.
    """

    # Required fields
    id: str
    content: str
    memory_type: MemoryType
    scope: MemoryScope

    # Ownership
    agent_id: str | None = None  # Which agent owns this
    user_id: str | None = None  # Which user it's about
    context_id: str | None = None  # Which context (household, team)

    # Metadata
    importance: float = 0.5  # 0-1, higher = more important
    confidence: float = 1.0  # 0-1, certainty level
    entities: list[str] = field(default_factory=list)  # Extracted entities
    source: str | None = None  # Where this came from
    stated_count: int = 1  # Times this fact was stated

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    expires_at: datetime | None = None

    # Linking
    linked_memories: list[str] = field(default_factory=list)  # Related memory IDs
    supersedes: str | None = None  # ID of memory this replaces
    superseded_by: str | None = None  # ID of newer memory

    # Vector (optional, may be computed externally)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Result from a memory search."""

    memory: Memory
    score: float  # Similarity score (0-1)
    relevance_grade: str | None = None  # "relevant", "irrelevant", "ambiguous"


@dataclass
class MemoryConfig:
    """Configuration for memory providers."""

    # Collection/index settings
    collection_name: str = "memories"
    embedding_dimension: int = 768

    # Search defaults
    default_limit: int = 5
    similarity_threshold: float = 0.3  # Minimum similarity for results

    # Importance weights by memory type
    type_importance: dict[MemoryType, float] = field(default_factory=lambda: {
        MemoryType.INSTRUCTION: 1.0,
        MemoryType.SELF_KNOWLEDGE: 0.95,
        MemoryType.PREFERENCE: 0.9,
        MemoryType.SKILL: 0.85,
        MemoryType.FACT: 0.8,
        MemoryType.KNOWLEDGE: 0.7,
        MemoryType.INSIGHT: 0.65,
        MemoryType.RELATIONSHIP: 0.6,
        MemoryType.BELIEF: 0.8,
        MemoryType.OBSERVATION: 0.5,
        MemoryType.EPISODIC: 0.5,
    })


class MemoryProvider(ABC):
    """Abstract interface for memory storage and retrieval.

    Cognitive services depend on this interface, not concrete implementations.
    This allows the engine to work with any vector database backend.
    """

    @abstractmethod
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
        """Store a new memory.

        Args:
            content: The memory content
            memory_type: Type of memory
            scope: Visibility scope
            agent_id: Owning agent ID
            user_id: Associated user ID
            context_id: Associated context ID
            importance: Importance score (0-1)
            confidence: Confidence score (0-1)
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The stored Memory object
        """
        ...

    @abstractmethod
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
        """Search memories by semantic similarity.

        Args:
            query: Search query
            agent_id: Filter by agent
            user_id: Filter by user
            context_id: Filter by context
            memory_types: Filter by memory types
            scopes: Filter by scopes
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of SearchResults sorted by relevance
        """
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            The Memory if found, None otherwise
        """
        ...

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory.

        Args:
            memory_id: ID of memory to update
            content: New content (if updating)
            importance: New importance (if updating)
            confidence: New confidence (if updating)
            metadata: Metadata to merge

        Returns:
            Updated Memory or None if not found
        """
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def reinforce(self, memory_id: str, boost: float = 0.1) -> Memory | None:
        """Reinforce a memory (boost importance on access).

        Default implementation uses update(). Providers can override
        for more efficient implementations.

        Args:
            memory_id: Memory to reinforce
            boost: Amount to boost importance (max 1.0)

        Returns:
            Updated memory or None
        """
        memory = await self.get(memory_id)
        if memory:
            new_importance = min(1.0, memory.importance + boost)
            return await self.update(memory_id, importance=new_importance)
        return None

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories by entity overlap.

        Default implementation searches for each entity and combines results.
        Providers can override for more efficient implementations.

        Args:
            entities: Entities to search for
            agent_id: Filter by agent
            user_id: Filter by user
            limit: Maximum results

        Returns:
            Memories containing any of the entities
        """
        # Default: search for entity names as queries
        combined = []
        seen_ids = set()
        for entity in entities[:3]:  # Limit entity searches
            results = await self.search(
                entity,
                agent_id=agent_id,
                user_id=user_id,
                limit=limit,
            )
            for r in results:
                if r.memory.id not in seen_ids:
                    seen_ids.add(r.memory.id)
                    combined.append(r)
        return sorted(combined, key=lambda r: r.score, reverse=True)[:limit]

    async def find_related(
        self,
        memory_id: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Find memories related to a given memory.

        Default implementation uses the memory's content as a search query.

        Args:
            memory_id: ID of source memory
            limit: Maximum results

        Returns:
            Related memories
        """
        memory = await self.get(memory_id)
        if not memory:
            return []
        results = await self.search(
            memory.content,
            agent_id=memory.agent_id,
            user_id=memory.user_id,
            limit=limit + 1,  # +1 to exclude self
        )
        return [r for r in results if r.memory.id != memory_id][:limit]

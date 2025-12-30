"""Protocol definitions for retrieval module.

These protocols define the interfaces that memory and LLM providers must implement
to work with the retrieval system. This allows the retrieval module to be
provider-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemorySearchProvider(Protocol):
    """Protocol for memory providers that support semantic search.

    Implementations must provide a search method that returns a list of
    memory documents with scores.
    """

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        min_score: float = 0.0,
        memory_types: list[Any] | None = None,
        context_id: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Search for memories matching the query.

        Args:
            query: The search query
            user_id: User ID for scoping results
            limit: Maximum number of results
            min_score: Minimum similarity score
            memory_types: Optional filter by memory types
            context_id: Optional context (e.g., household) for scoping
            **kwargs: Additional provider-specific options

        Returns:
            List of memory results with score attribute
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations must provide an embed method for generating embeddings.
    """

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used in retrieval.

    The retrieval module uses LLMs for:
    - Query expansion
    - Re-ranking (optional)
    - Query contextualization
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific options

        Returns:
            Dict with 'content' key containing the response
        """
        ...


class SearchResult:
    """Standard search result format.

    This is the expected format for results from MemorySearchProvider.search().
    Providers may return their own types as long as they have these attributes.
    """

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        memory_type: str | None = None,
        importance: float = 0.5,
        user_id: str | None = None,
        scope: str = "private",
        entities: list[str] | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id
        self.content = content
        self.score = score
        self.memory_type = memory_type
        self.importance = importance
        self.user_id = user_id
        self.scope = scope
        self.entities = entities or []
        self.created_at = created_at
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "user_id": self.user_id,
            "scope": self.scope,
            "entities": self.entities,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

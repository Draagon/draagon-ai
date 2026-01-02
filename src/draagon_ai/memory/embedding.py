"""Embedding providers for memory systems.

This module provides embedding generation for semantic search in memory.
Embeddings convert text into numerical vectors that capture meaning,
enabling similarity-based retrieval.

Example:
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    embedder = OllamaEmbeddingProvider(
        base_url="http://192.168.168.200:11434",
        model="nomic-embed-text",
    )

    # Generate embedding for text
    vector = await embedder.embed("User's favorite color is blue")
    # Returns: [0.23, 0.87, 0.12, ... 768 numbers]

For testing with real semantic embeddings (no API required):
    from draagon_ai.memory.embedding import SentenceTransformerEmbeddingProvider

    embedder = SentenceTransformerEmbeddingProvider(
        model_name="all-MiniLM-L6-v2",  # Fast, 384-dim
    )
    vector = await embedder.embed("User's favorite color is blue")
"""

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


# Import lazily to avoid startup cost
_sentence_transformer_model = None


class SentenceTransformerEmbeddingProvider:
    """Local embedding provider using sentence-transformers.

    Uses HuggingFace sentence-transformers models for local embedding generation.
    No API key required - runs entirely on local hardware.

    Recommended models:
        - all-MiniLM-L6-v2: 384-dim, fast, good quality (default)
        - all-mpnet-base-v2: 768-dim, slower, better quality
        - paraphrase-MiniLM-L6-v2: 384-dim, fast, good for paraphrasing

    Args:
        model_name: HuggingFace model name (default: all-MiniLM-L6-v2)

    Example:
        embedder = SentenceTransformerEmbeddingProvider()
        vector = await embedder.embed("Hello world")
        print(len(vector))  # 384

    Note:
        First use will download the model (~90MB for MiniLM).
        Models are cached in ~/.cache/huggingface/
    """

    # Model dimensions for common models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding provider.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self._model = None
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 384)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def _get_model(self):
        """Lazily load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerEmbeddingProvider. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            # Update dimension from actual model
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded, dimension: {self._dimension}")

        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        model = self._get_model()
        # sentence-transformers is synchronous, but we wrap for async interface
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    async def initialize(self):
        """Pre-load the model (optional, for faster first embed)."""
        self._get_model()

    async def close(self):
        """Cleanup (no-op, model memory freed on garbage collection)."""
        pass


from draagon_ai.llm.ollama import OllamaLLM, OllamaConfig


class OllamaEmbeddingProvider:
    """Ollama-based embedding provider for memory systems.

    Uses Ollama's embedding models (like nomic-embed-text) to generate
    vector representations of text for semantic search.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        model: Embedding model name (default: nomic-embed-text)
        dimension: Expected embedding dimension (default: 768)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimension: int = 768,
    ):
        self.base_url = base_url
        self.model = model
        self.dimension = dimension

        # Create OllamaLLM configured for embeddings
        config = OllamaConfig(
            base_url=base_url,
            embedding_model=model,
        )
        self._llm = OllamaLLM(config=config)

    async def embed(self, text: str, retries: int = 3) -> list[float]:
        """Generate embedding for text with retry logic.

        The remote Ollama server can experience intermittent GPU discovery failures
        that cause internal runners to crash (error like "EOF" on localhost ports).
        This retry logic handles those transient failures.

        Args:
            text: Text to embed
            retries: Number of retry attempts (default: 3)

        Returns:
            Embedding vector (list of floats)
        """
        import asyncio

        last_error = None
        for attempt in range(retries):
            try:
                return await self._llm.embed(text)
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Check for the known Ollama runner crash pattern:
                # The error mentions "127.0.0.1:XXXXX" (internal runner port)
                # and "EOF" because the runner crashed
                if "EOF" in error_msg or "127.0.0.1" in error_msg:
                    logger.warning(
                        f"Ollama runner failed (attempt {attempt + 1}/{retries}): {error_msg}"
                    )
                    if attempt < retries - 1:
                        # Exponential backoff to give server time to recover
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                # Unknown error, don't retry
                raise

        # All retries exhausted
        raise last_error

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self._llm.embed_batch(texts)


__all__ = [
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]

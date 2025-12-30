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
"""

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

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        return await self._llm.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return await self._llm.embed_batch(texts)


__all__ = ["OllamaEmbeddingProvider"]

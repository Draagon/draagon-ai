# TASK-080: Ollama mxbai-embed-large Integration

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Production embedding quality)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: None (parallel with corpus tasks)

---

## Description

Integrate Ollama's mxbai-embed-large model for production-quality embeddings:
- MTEB score: 64.68 (top tier for open-source)
- 1024 dimensions
- 512 token context window
- Local inference (no API costs)

This replaces mock embeddings with real semantic vectors for accurate retrieval testing.

**Location:** `src/draagon_ai/memory/embedding/ollama.py`

---

## Acceptance Criteria

### Core Functionality
- [ ] `OllamaEmbeddingProvider` class implementing `EmbeddingProvider` protocol
- [ ] `embed(text)` returns 1024-dim vector
- [ ] `embed_batch(texts)` for efficient batch processing
- [ ] Connection health check on init
- [ ] Graceful fallback if Ollama unavailable

### Configuration
- [ ] Configurable base URL (default: `http://localhost:11434`)
- [ ] Configurable model (default: `mxbai-embed-large`)
- [ ] Timeout configuration
- [ ] Retry logic with exponential backoff

### Performance
- [ ] Batch size optimization (32 texts per request)
- [ ] Connection pooling
- [ ] Caching of repeated embeddings (optional LRU cache)

### Error Handling
- [ ] Clear error messages for connection failures
- [ ] Automatic retry on transient errors
- [ ] Fallback notification when using alternative

---

## Technical Notes

### Ollama API

```python
import httpx
from dataclasses import dataclass

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "mxbai-embed-large"
    timeout: float = 30.0
    max_retries: int = 3
    batch_size: int = 32


class OllamaEmbeddingProvider:
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self._cache: dict[str, list[float]] = {}

    async def health_check(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            return any(m["name"].startswith(self.config.model) for m in models)
        except httpx.HTTPError:
            return False

    async def embed(self, text: str) -> list[float]:
        """Embed single text, returning 1024-dim vector."""
        # Check cache first
        cache_key = text[:500]  # Truncate for cache key
        if cache_key in self._cache:
            return self._cache[cache_key]

        response = await self._request_with_retry({
            "model": self.config.model,
            "prompt": text,
        })

        embedding = response.json()["embedding"]
        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            # Parallel requests for batch
            tasks = [self.embed(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _request_with_retry(self, payload: dict) -> httpx.Response:
        """Make request with exponential backoff retry."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post("/api/embeddings", json=payload)
                response.raise_for_status()
                return response
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise ConnectionError(
            f"Failed to get embedding after {self.config.max_retries} attempts: {last_error}"
        )

    @property
    def dimension(self) -> int:
        """Embedding dimension for mxbai-embed-large."""
        return 1024

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
```

### Model Pull Helper

```python
async def ensure_model_available(config: OllamaConfig) -> bool:
    """Ensure embedding model is pulled and ready."""
    async with httpx.AsyncClient(base_url=config.base_url) as client:
        # Check if model exists
        response = await client.get("/api/tags")
        models = response.json().get("models", [])

        if any(m["name"].startswith(config.model) for m in models):
            return True

        # Pull model
        logger.info(f"Pulling {config.model}...")
        response = await client.post("/api/pull", json={"name": config.model})

        return response.status_code == 200
```

### Integration with Memory Provider

```python
# In LayeredMemoryProvider or similar
class LayeredMemoryProvider:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider = None,
        ...
    ):
        self.embedder = embedding_provider or OllamaEmbeddingProvider()

    async def store(self, content: str, metadata: dict) -> str:
        # Get embedding for semantic search
        embedding = await self.embedder.embed(content)

        # Store with embedding
        memory = Memory(
            content=content,
            metadata=metadata,
            embedding=embedding,
        )
        ...
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_ollama_health_check(mock_ollama_server):
    """Health check detects available model."""
    provider = OllamaEmbeddingProvider()
    assert await provider.health_check() is True

@pytest.mark.asyncio
async def test_embedding_dimension():
    """Embedding returns correct dimension."""
    provider = OllamaEmbeddingProvider()
    embedding = await provider.embed("Hello world")
    assert len(embedding) == 1024

@pytest.mark.asyncio
async def test_embedding_caching():
    """Repeated embeddings use cache."""
    provider = OllamaEmbeddingProvider()

    emb1 = await provider.embed("Test text")
    emb2 = await provider.embed("Test text")

    assert emb1 == emb2  # Same object from cache
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_ollama_embedding():
    """Test with real Ollama server."""
    provider = OllamaEmbeddingProvider()

    if not await provider.health_check():
        pytest.skip("Ollama not available")

    embedding = await provider.embed("The quick brown fox jumps over the lazy dog")

    assert len(embedding) == 1024
    assert all(isinstance(v, float) for v in embedding)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_similarity():
    """Similar texts have similar embeddings."""
    provider = OllamaEmbeddingProvider()

    if not await provider.health_check():
        pytest.skip("Ollama not available")

    emb1 = await provider.embed("The cat sat on the mat")
    emb2 = await provider.embed("A cat was sitting on a mat")
    emb3 = await provider.embed("The stock market crashed today")

    # Cosine similarity
    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    assert sim_12 > sim_13  # Similar sentences more similar
    assert sim_12 > 0.8  # High similarity for paraphrases
```

---

## Files to Create/Modify

- `src/draagon_ai/memory/embedding/ollama.py`
- `src/draagon_ai/memory/embedding/__init__.py` (add export)
- Add tests to `tests/memory/embedding/test_ollama.py`

---

## Definition of Done

- [ ] OllamaEmbeddingProvider implemented
- [ ] Health check working
- [ ] Batch embedding efficient
- [ ] Retry logic with backoff
- [ ] Caching for repeated texts
- [ ] 1024-dim vectors returned
- [ ] Integration test with real Ollama
- [ ] Semantic similarity test passing

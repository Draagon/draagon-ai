# TASK-081: SentenceTransformer Fallback

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Fallback when Ollama unavailable)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: None (parallel with TASK-080)

---

## Description

Implement SentenceTransformer embedding provider as fallback when Ollama is unavailable:
- all-MiniLM-L6-v2: Fast, 384-dim (for development)
- all-mpnet-base-v2: Higher quality, 768-dim (for testing)
- Local inference, no external dependencies

This ensures benchmarks can run without Ollama server.

**Location:** `src/draagon_ai/memory/embedding/sentence_transformer.py`

---

## Acceptance Criteria

### Core Functionality
- [ ] `SentenceTransformerProvider` implementing `EmbeddingProvider` protocol
- [ ] `embed(text)` returns vector (384 or 768 dim based on model)
- [ ] `embed_batch(texts)` for efficient batch processing
- [ ] Auto-download model on first use

### Model Support
- [ ] `all-MiniLM-L6-v2` (default, fast, 384-dim)
- [ ] `all-mpnet-base-v2` (higher quality, 768-dim)
- [ ] `multi-qa-MiniLM-L6-cos-v1` (optimized for QA)
- [ ] Configurable model selection

### Performance
- [ ] GPU acceleration when available (CUDA)
- [ ] CPU fallback (no CUDA required)
- [ ] Batch size optimization
- [ ] Model caching (don't reload)

---

## Technical Notes

### Implementation

```python
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Literal
import torch

ModelName = Literal[
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
]

MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}


@dataclass
class SentenceTransformerConfig:
    model_name: ModelName = "all-MiniLM-L6-v2"
    device: str = "auto"  # auto, cuda, cpu
    batch_size: int = 32
    show_progress: bool = False


class SentenceTransformerProvider:
    def __init__(self, config: SentenceTransformerConfig = None):
        self.config = config or SentenceTransformerConfig()

        # Determine device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        # Load model (downloads if needed)
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.device,
        )

        self._dimension = MODEL_DIMENSIONS[self.config.model_name]

    async def embed(self, text: str) -> list[float]:
        """Embed single text."""
        # SentenceTransformer is sync, wrap for async interface
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=self.config.show_progress,
        )
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Embedding dimension for current model."""
        return self._dimension

    def health_check(self) -> bool:
        """Always available (local model)."""
        return True
```

### Fallback Factory

```python
async def get_embedding_provider(
    prefer_ollama: bool = True,
    ollama_config: OllamaConfig = None,
    st_config: SentenceTransformerConfig = None,
) -> EmbeddingProvider:
    """Get best available embedding provider."""

    if prefer_ollama:
        ollama = OllamaEmbeddingProvider(ollama_config)
        if await ollama.health_check():
            logger.info("Using Ollama mxbai-embed-large")
            return ollama
        else:
            logger.warning("Ollama unavailable, falling back to SentenceTransformer")

    st_provider = SentenceTransformerProvider(st_config)
    logger.info(f"Using SentenceTransformer {st_provider.config.model_name}")
    return st_provider
```

### Dimension Normalization

```python
def normalize_embedding_dimension(
    embedding: list[float],
    target_dim: int,
) -> list[float]:
    """Normalize embedding to target dimension.

    Used when switching between providers with different dimensions.
    - Truncate if too long
    - Zero-pad if too short
    """
    current_dim = len(embedding)

    if current_dim == target_dim:
        return embedding
    elif current_dim > target_dim:
        return embedding[:target_dim]
    else:
        return embedding + [0.0] * (target_dim - current_dim)
```

---

## Testing Requirements

### Unit Tests
```python
def test_embedding_dimension():
    """Embedding returns correct dimension for model."""
    provider = SentenceTransformerProvider(
        SentenceTransformerConfig(model_name="all-MiniLM-L6-v2")
    )
    assert provider.dimension == 384

    provider_mpnet = SentenceTransformerProvider(
        SentenceTransformerConfig(model_name="all-mpnet-base-v2")
    )
    assert provider_mpnet.dimension == 768

@pytest.mark.asyncio
async def test_batch_embedding():
    """Batch embedding returns correct count."""
    provider = SentenceTransformerProvider()
    texts = ["Hello", "World", "Test"]

    embeddings = await provider.embed_batch(texts)

    assert len(embeddings) == 3
    assert all(len(e) == 384 for e in embeddings)

@pytest.mark.asyncio
async def test_semantic_similarity():
    """Similar texts have similar embeddings."""
    provider = SentenceTransformerProvider()

    emb1 = await provider.embed("I love programming")
    emb2 = await provider.embed("I enjoy coding")
    emb3 = await provider.embed("The weather is nice")

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    assert sim_12 > sim_13
```

### Fallback Test
```python
@pytest.mark.asyncio
async def test_fallback_to_sentence_transformer():
    """Falls back when Ollama unavailable."""
    # Mock Ollama as unavailable
    provider = await get_embedding_provider(
        prefer_ollama=True,
        ollama_config=OllamaConfig(base_url="http://nonexistent:11434"),
    )

    assert isinstance(provider, SentenceTransformerProvider)
```

---

## Files to Create/Modify

- `src/draagon_ai/memory/embedding/sentence_transformer.py`
- `src/draagon_ai/memory/embedding/factory.py` (provider factory)
- `src/draagon_ai/memory/embedding/__init__.py` (add exports)
- Add tests to `tests/memory/embedding/test_sentence_transformer.py`

---

## Definition of Done

- [ ] SentenceTransformerProvider implemented
- [ ] Multiple model support (MiniLM, MPNet)
- [ ] GPU acceleration when available
- [ ] Batch embedding working
- [ ] Fallback factory implemented
- [ ] Dimension normalization helper
- [ ] Unit tests passing
- [ ] Semantic similarity verified

"""Memory providers for draagon-ai.

Memory providers implement the MemoryProvider protocol and provide
storage backends for agents.

Available providers:
- LayeredMemoryProvider: 4-layer cognitive memory (working, episodic, semantic, metacognitive)
- QdrantMemoryProvider: Qdrant vector database backend (requires qdrant-client)
- QdrantPromptProvider: Prompt versioning and evolution storage
"""

from .layered import (
    LayeredMemoryProvider,
    LayeredMemoryConfig,
    LAYER_MAPPING,
    IMPORTANCE_WEIGHTS,
)

# Qdrant provider (optional, requires qdrant-client)
try:
    from .qdrant import (
        QdrantMemoryProvider,
        QdrantPromptProvider,
        QdrantConfig,
        PromptVersion,
        PromptDomain,
        QDRANT_AVAILABLE,
    )
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantMemoryProvider = None
    QdrantPromptProvider = None
    QdrantConfig = None
    PromptVersion = None
    PromptDomain = None

__all__ = [
    # Layered provider
    "LayeredMemoryProvider",
    "LayeredMemoryConfig",
    "LAYER_MAPPING",
    "IMPORTANCE_WEIGHTS",
    # Qdrant provider
    "QdrantMemoryProvider",
    "QdrantPromptProvider",
    "QdrantConfig",
    "PromptVersion",
    "PromptDomain",
    "QDRANT_AVAILABLE",
]

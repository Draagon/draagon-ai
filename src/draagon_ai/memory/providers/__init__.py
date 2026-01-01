"""Memory providers for draagon-ai.

Memory providers implement the MemoryProvider protocol and provide
storage backends for agents.

Available providers:
- LayeredMemoryProvider: 4-layer cognitive memory (working, episodic, semantic, metacognitive)
- Neo4jMemoryProvider: Neo4j semantic graph backend with Phase 0/1 decomposition (RECOMMENDED)
- QdrantMemoryProvider: Qdrant vector database backend (DEPRECATED, use Neo4jMemoryProvider)
- QdrantPromptProvider: Prompt versioning and evolution storage
- QdrantGraphStore: Qdrant-backed TemporalCognitiveGraph with persistence
"""

from .layered import (
    LayeredMemoryProvider,
    LayeredMemoryConfig,
    LAYER_MAPPING,
    IMPORTANCE_WEIGHTS,
)

# Neo4j provider (optional, requires neo4j)
try:
    from .neo4j import (
        Neo4jMemoryProvider,
        Neo4jMemoryConfig,
        NEO4J_AVAILABLE,
    )
except ImportError:
    NEO4J_AVAILABLE = False
    Neo4jMemoryProvider = None
    Neo4jMemoryConfig = None

# Qdrant provider (optional, requires qdrant-client) - DEPRECATED
try:
    from .qdrant import (
        QdrantMemoryProvider,
        QdrantPromptProvider,
        QdrantConfig,
        PromptVersion,
        PromptDomain,
        QDRANT_AVAILABLE,
    )
    from .qdrant_graph import (
        QdrantGraphStore,
        QdrantGraphConfig,
    )
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantMemoryProvider = None
    QdrantPromptProvider = None
    QdrantConfig = None
    PromptVersion = None
    PromptDomain = None
    QdrantGraphStore = None
    QdrantGraphConfig = None

__all__ = [
    # Layered provider
    "LayeredMemoryProvider",
    "LayeredMemoryConfig",
    "LAYER_MAPPING",
    "IMPORTANCE_WEIGHTS",
    # Neo4j provider (RECOMMENDED)
    "Neo4jMemoryProvider",
    "Neo4jMemoryConfig",
    "NEO4J_AVAILABLE",
    # Qdrant provider (DEPRECATED)
    "QdrantMemoryProvider",
    "QdrantPromptProvider",
    "QdrantConfig",
    "PromptVersion",
    "PromptDomain",
    "QDRANT_AVAILABLE",
    # Qdrant graph store
    "QdrantGraphStore",
    "QdrantGraphConfig",
]

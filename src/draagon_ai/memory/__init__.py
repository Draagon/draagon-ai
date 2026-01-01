"""Memory abstraction layer for Draagon AI.

This module defines abstract interfaces for memory systems. The cognitive
engine uses these interfaces without knowing about specific vector databases.

Phase C.1 (AGI-Lite) additions:
- TemporalNode: Bi-temporal node for the cognitive graph
- TemporalEdge: Relationships between nodes
- TemporalCognitiveGraph: The core graph container
- HierarchicalScope: Scope-based access control

Phase C.2 (AGI-Lite) additions:
- WorkingMemory: Session-scoped, limited capacity
- EpisodicMemory: Episodes and events with chronological linking
- SemanticMemory: Entity resolution, facts, relationships
- MetacognitiveMemory: Skills, strategies, insights
- MemoryPromotion: Auto-promotion between layers
"""

from draagon_ai.memory.base import (
    MemoryProvider,
    MemoryType,
    MemoryScope,
    Memory,
    SearchResult,
    MemoryConfig,
)

from draagon_ai.memory.temporal_nodes import (
    TemporalNode,
    TemporalEdge,
    NodeType,
    EdgeType,
    MemoryLayer,
    NODE_TYPE_TO_LAYER,
    LAYER_DEFAULT_TTL,
    create_fact_node,
    create_episode_node,
    create_skill_node,
    create_entity_node,
    create_behavior_node,
)

from draagon_ai.memory.scopes import (
    HierarchicalScope,
    ScopeType,
    Permission,
    ScopePermission,
    ScopeRegistry,
    get_scope_registry,
    reset_scope_registry,
)

from draagon_ai.memory.temporal_graph import (
    TemporalCognitiveGraph,
    EmbeddingProvider,
    GraphSearchResult,
    GraphTraversalResult,
    PermissionDeniedError,
)

# Memory Layers (C.2)
from draagon_ai.memory.layers import (
    # Base
    LayerConfig,
    MemoryLayerBase,
    PromotionResult,
    # Working Memory
    WorkingMemory,
    WorkingMemoryItem,
    # Episodic Memory
    EpisodicMemory,
    Episode,
    Event,
    # Semantic Memory
    SemanticMemory,
    Entity,
    Fact,
    Relationship,
    EntityMatch,
    # Metacognitive Memory
    MetacognitiveMemory,
    Skill,
    Strategy,
    Insight,
    BehaviorNode,
    # Promotion
    MemoryPromotion,
    MemoryConsolidator,
    PromotionConfig,
    PromotionStats,
)

# Memory Providers
from draagon_ai.memory.providers import (
    LayeredMemoryProvider,
    LayeredMemoryConfig,
    # Neo4j provider (RECOMMENDED)
    Neo4jMemoryProvider,
    Neo4jMemoryConfig,
    NEO4J_AVAILABLE,
)

# Document Loaders
from draagon_ai.memory.loaders import (
    Document,
    DocumentLoader,
    LoaderConfig,
    MarkdownLoader,
    TextLoader,
    DirectoryLoader,
)

# Retrieval Augmentation (Self-RAG, CRAG)
from draagon_ai.memory.retrieval import (
    RetrievalAugmenter,
    RetrievalConfig,
    RelevanceAssessment,
    GradedChunk,
    Contradiction,
    SelfRAGResult,
    CRAGResult,
)

__all__ = [
    # Base memory interfaces
    "MemoryProvider",
    "MemoryType",
    "MemoryScope",
    "Memory",
    "SearchResult",
    "MemoryConfig",
    # Temporal nodes (C.1)
    "TemporalNode",
    "TemporalEdge",
    "NodeType",
    "EdgeType",
    "MemoryLayer",
    "NODE_TYPE_TO_LAYER",
    "LAYER_DEFAULT_TTL",
    "create_fact_node",
    "create_episode_node",
    "create_skill_node",
    "create_entity_node",
    "create_behavior_node",
    # Scopes (C.1)
    "HierarchicalScope",
    "ScopeType",
    "Permission",
    "ScopePermission",
    "ScopeRegistry",
    "get_scope_registry",
    "reset_scope_registry",
    # Graph (C.1)
    "TemporalCognitiveGraph",
    "EmbeddingProvider",
    "GraphSearchResult",
    "GraphTraversalResult",
    "PermissionDeniedError",
    # Memory Layers (C.2)
    "LayerConfig",
    "MemoryLayerBase",
    "PromotionResult",
    "WorkingMemory",
    "WorkingMemoryItem",
    "EpisodicMemory",
    "Episode",
    "Event",
    "SemanticMemory",
    "Entity",
    "Fact",
    "Relationship",
    "EntityMatch",
    "MetacognitiveMemory",
    "Skill",
    "Strategy",
    "Insight",
    "BehaviorNode",
    "MemoryPromotion",
    "MemoryConsolidator",
    "PromotionConfig",
    "PromotionStats",
    # Memory Providers
    "LayeredMemoryProvider",
    "LayeredMemoryConfig",
    "Neo4jMemoryProvider",
    "Neo4jMemoryConfig",
    "NEO4J_AVAILABLE",
    # Document Loaders
    "Document",
    "DocumentLoader",
    "LoaderConfig",
    "MarkdownLoader",
    "TextLoader",
    "DirectoryLoader",
    # Retrieval Augmentation
    "RetrievalAugmenter",
    "RetrievalConfig",
    "RelevanceAssessment",
    "GradedChunk",
    "Contradiction",
    "SelfRAGResult",
    "CRAGResult",
]

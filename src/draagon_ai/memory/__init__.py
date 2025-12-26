"""Memory abstraction layer for Draagon AI.

This module defines abstract interfaces for memory systems. The cognitive
engine uses these interfaces without knowing about specific vector databases.

Phase C.1 (AGI-Lite) additions:
- TemporalNode: Bi-temporal node for the cognitive graph
- TemporalEdge: Relationships between nodes
- TemporalCognitiveGraph: The core graph container
- HierarchicalScope: Scope-based access control
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
]

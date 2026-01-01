"""Semantic Graph Module for Phase 2 Memory Integration.

This module provides the graph data structures and operations for storing
decomposed knowledge as an entity-centric semantic graph.

Key Components:

Graph Data Models:
    - GraphNode: Nodes representing instances, classes, events
    - GraphEdge: Directed relationships with bi-temporal validity
    - NodeType: Classification of nodes (INSTANCE, CLASS, EVENT, etc.)
    - EdgeRelationType: Common relationship types
    - ConflictType: Types of merge conflicts

Graph Container:
    - SemanticGraph: The main graph container with query operations

Merge Operations:
    - MergeConflict: Conflicts detected during merging
    - MergeResult: Result of merge operations

Ontology Model (see docs/architecture/GRAPH_ONTOLOGY_DESIGN.md):
    - INSTANCE nodes: Specific individuals (Doug, Whiskers)
    - CLASS nodes: Abstract types/synsets (cat.n.01, person.n.01)
    - Instances link to classes via INSTANCE_OF edges

Example:
    >>> from draagon_ai.cognition.decomposition.graph import (
    ...     SemanticGraph, GraphNode, GraphEdge, NodeType
    ... )
    >>>
    >>> # Create a graph
    >>> graph = SemanticGraph()
    >>>
    >>> # Add instances (specific entities)
    >>> doug = graph.create_node("Doug", NodeType.INSTANCE)
    >>> whiskers = graph.create_node("Whiskers", NodeType.INSTANCE)
    >>>
    >>> # Add class (abstract type)
    >>> cat_class = graph.create_node("cat.n.01", NodeType.CLASS)
    >>>
    >>> # Add relationships
    >>> graph.create_edge(doug.node_id, whiskers.node_id, "owns")
    >>> graph.create_edge(whiskers.node_id, cat_class.node_id, "instance_of")
    >>>
    >>> # Query the graph
    >>> count = graph.count_relations(doug.node_id, "owns")
    >>> print(f"Doug owns {count} pet(s)")

Based on Phase 2 requirements in:
    docs/cognition/decomposition/requirements/PHASE_2_MEMORY_INTEGRATION.md
"""

from .models import (
    # Enumerations
    NodeType,
    EdgeRelationType,
    ConflictType,
    # Data classes
    GraphNode,
    GraphEdge,
    MergeConflict,
    MergeResult,
)

from .semantic_graph import (
    SemanticGraph,
    TraversalResult,
)

from .builder import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphBuildResult,
    SemanticEdgeType,
)

from .operations import (
    GraphUpdater,
    UpdateResult,
)

# Neo4j is optional - only import if available
try:
    from .neo4j_store import (
        Neo4jGraphStore,
        Neo4jGraphStoreSync,
        Neo4jConfig,
    )
    _NEO4J_AVAILABLE = True
except ImportError:
    Neo4jGraphStore = None  # type: ignore
    Neo4jGraphStoreSync = None  # type: ignore
    Neo4jConfig = None  # type: ignore
    _NEO4J_AVAILABLE = False

__all__ = [
    # ==========================================================================
    # Enumerations
    # ==========================================================================
    "NodeType",
    "EdgeRelationType",
    "ConflictType",
    # ==========================================================================
    # Data Classes
    # ==========================================================================
    "GraphNode",
    "GraphEdge",
    "MergeConflict",
    "MergeResult",
    # ==========================================================================
    # Graph Container
    # ==========================================================================
    "SemanticGraph",
    "TraversalResult",
    # ==========================================================================
    # Graph Builder (Phase 0/1 Integration)
    # ==========================================================================
    "GraphBuilder",
    "GraphBuilderConfig",
    "GraphBuildResult",
    "SemanticEdgeType",
    # ==========================================================================
    # Graph Operations (Incremental Updates)
    # ==========================================================================
    "GraphUpdater",
    "UpdateResult",
    # ==========================================================================
    # Neo4j Persistence
    # ==========================================================================
    "Neo4jGraphStore",
    "Neo4jGraphStoreSync",
    "Neo4jConfig",
]

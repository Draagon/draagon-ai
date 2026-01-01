"""Probabilistic Graph Reasoning Module.

This module implements the probabilistic graph reasoning architecture that:
1. Extracts semantic graphs from messages via Phase 0/1
2. Expands ambiguous messages into branching interpretations
3. Retrieves context from Neo4j knowledge graph
4. Runs beam search over interpretations via ReAct
5. Reinforces successful paths in the graph
6. Persists valuable learnings

See docs/architecture/PROBABILISTIC_GRAPH_REASONING.md for full design.
"""

from .loop import (
    ReasoningLoop,
    ReasoningConfig,
    ReasoningResult,
)

from .expander import (
    ProbabilisticExpander,
    InterpretationBranch,
    ExpansionResult,
)

from .context import (
    ContextRetriever,
    RecencyWindow,
    RetrievedContext,
)

from .memory import (
    MemoryLayer,
    ContentType,
    MemoryProperties,
    VolatileObservation,
    VolatileWorkingMemory,
    MemoryAwareGraphStore,
    classify_phase1_content,
    LAYER_TTL,
    DECAY_RATES,
)

__all__ = [
    # Main Loop
    "ReasoningLoop",
    "ReasoningConfig",
    "ReasoningResult",
    # Expansion
    "ProbabilisticExpander",
    "InterpretationBranch",
    "ExpansionResult",
    # Context
    "ContextRetriever",
    "RecencyWindow",
    "RetrievedContext",
    # Memory
    "MemoryLayer",
    "ContentType",
    "MemoryProperties",
    "VolatileObservation",
    "VolatileWorkingMemory",
    "MemoryAwareGraphStore",
    "classify_phase1_content",
    "LAYER_TTL",
    "DECAY_RATES",
]

"""Phase 1 Extractors - Multi-Type Decomposition Pipeline Components.

This subpackage contains all the individual extractors for the Phase 1
decomposition pipeline:

- Semantic Role Extraction (predicate-argument structures)
- Presupposition Extraction (implicit assumptions)
- Commonsense Inference (ATOMIC-style reasoning)
- Temporal/Aspectual Analysis
- Modality Extraction (certainty, obligation)
- Negation Detection (polarity, scope)

Each extractor follows a hybrid pattern:
1. Fast heuristic analysis first
2. LLM fallback for complex cases

Example:
    >>> from draagon_ai.cognition.decomposition.extractors import (
    ...     DecompositionPipeline,
    ...     DecompositionConfig,
    ...     PresuppositionExtractor,
    ... )
    >>>
    >>> config = DecompositionConfig()
    >>> pipeline = DecompositionPipeline(config, llm=my_llm)
    >>> result = await pipeline.decompose("Doug forgot the meeting again")

Based on prototype work in prototypes/implicit_knowledge_graphs/
"""

# Models - Core data structures
from .models import (
    # Enumerations
    PresuppositionTrigger,
    CommonsenseRelation,
    Aspect,
    Tense,
    ModalType,
    Polarity,
    # Data structures
    SemanticRole,
    Presupposition,
    CommonsenseInference,
    TemporalInfo,
    ModalityInfo,
    NegationInfo,
    CrossReference,
    WeightedBranch,
    DecomposedKnowledge,
)

# Configuration
from .config import (
    StageEnablement,
    SemanticRoleConfig,
    PresuppositionConfig,
    CommonsenseConfig,
    NegationConfig,
    TemporalConfig,
    ModalityConfig,
    WeightingConfig,
    DecompositionConfig,
)

# Extractors
from .semantic_roles import SemanticRoleExtractor
from .presuppositions import PresuppositionExtractor
from .commonsense import CommonsenseExtractor
from .temporal import TemporalExtractor
from .modality import ModalityExtractor
from .negation import NegationExtractor

# Pipeline
from .pipeline import (
    DecompositionPipeline,
    PipelineMetrics,
    PipelineError,
    StageStatus,
    StageResult,
    BranchBuilder,
    decompose,
    decompose_sync,
)

__all__ = [
    # ==========================================================================
    # Enumerations
    # ==========================================================================
    "PresuppositionTrigger",
    "CommonsenseRelation",
    "Aspect",
    "Tense",
    "ModalType",
    "Polarity",
    # ==========================================================================
    # Data Structures
    # ==========================================================================
    "SemanticRole",
    "Presupposition",
    "CommonsenseInference",
    "TemporalInfo",
    "ModalityInfo",
    "NegationInfo",
    "CrossReference",
    "WeightedBranch",
    "DecomposedKnowledge",
    # ==========================================================================
    # Configuration
    # ==========================================================================
    "StageEnablement",
    "SemanticRoleConfig",
    "PresuppositionConfig",
    "CommonsenseConfig",
    "NegationConfig",
    "TemporalConfig",
    "ModalityConfig",
    "WeightingConfig",
    "DecompositionConfig",
    # ==========================================================================
    # Extractors
    # ==========================================================================
    "SemanticRoleExtractor",
    "PresuppositionExtractor",
    "CommonsenseExtractor",
    "TemporalExtractor",
    "ModalityExtractor",
    "NegationExtractor",
    # ==========================================================================
    # Pipeline
    # ==========================================================================
    "DecompositionPipeline",
    "PipelineMetrics",
    "PipelineError",
    "StageStatus",
    "StageResult",
    "BranchBuilder",
    "decompose",
    "decompose_sync",
]

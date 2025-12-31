"""Phase 1: Multi-Type Decomposition Pipeline.

This module extracts implicit knowledge from text including:
- Semantic roles (predicate-argument structures)
- Presuppositions (what must be true for the statement to make sense)
- Commonsense inferences (ATOMIC-style reasoning)
- Temporal and aspectual information
- Modality markers
- Negation and polarity

The pipeline produces weighted branches for ambiguous interpretations,
enabling downstream retrieval to select the most contextually appropriate
interpretation.

**Integrated Pipeline (Recommended):**

For full Phase 0 → Phase 1 integration with automatic WSD, entity
classification, and content-aware processing, use the IntegratedPipeline:

    >>> from decomposition import IntegratedPipeline
    >>>
    >>> pipeline = IntegratedPipeline(llm=my_llm)
    >>> result = await pipeline.process("Doug forgot the meeting again")
    >>>
    >>> # Access Phase 0 results
    >>> print(result.wsd_results)  # {"forgot": DisambiguationResult(...)}
    >>> print(result.entities)     # {"Doug": UniversalSemanticIdentifier(...)}
    >>>
    >>> # Access Phase 1 results
    >>> print(result.presuppositions)  # [Presupposition(...)]
    >>> print(result.inferences)       # [CommonsenseInference(...)]

**Standalone Decomposition:**

For decomposition without Phase 0 preprocessing:

    >>> from decomposition import DecompositionPipeline, DecompositionConfig
    >>>
    >>> config = DecompositionConfig()
    >>> pipeline = DecompositionPipeline(config=config)
    >>>
    >>> result = await pipeline.decompose("Doug forgot the meeting again")
    >>> print(result.presuppositions)
    >>> # [Presupposition(content="Doug forgot before", trigger_type=ITERATIVE, ...)]
"""

from decomposition.models import (
    # Enums
    PresuppositionTrigger,
    CommonsenseRelation,
    Aspect,
    Tense,
    ModalType,
    Polarity,
    # Core data structures
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

from decomposition.config import (
    DecompositionConfig,
    StageEnablement,
    SemanticRoleConfig,
    PresuppositionConfig,
    CommonsenseConfig,
    NegationConfig,
    TemporalConfig,
    ModalityConfig,
    WeightingConfig,
)

from decomposition.pipeline import (
    DecompositionPipeline,
    PipelineError,
    PipelineMetrics,
    StageResult,
    StageStatus,
    decompose,
    decompose_sync,
)

from decomposition.presuppositions import (
    PresuppositionExtractor,
    TriggerDetector,
    ContentGenerator,
    DetectedTrigger,
    detect_triggers,
)

from decomposition.semantic_roles import (
    SemanticRoleExtractor,
    PredicateDetector,
    ArgumentDetector,
    RoleType,
    Predicate,
    Argument,
    PredicateArgumentStructure,
    get_semantic_roles,
    get_agent,
    get_patient,
)

from decomposition.negation import (
    NegationExtractor,
    NegationDetector,
    ScopeAnalyzer,
    NegationType,
    NegationCue,
    NegationAnalysis,
    detect_negation,
    get_polarity,
)

from decomposition.temporal import (
    TemporalExtractor,
    TenseDetector,
    AspectClassifier,
    TemporalExpressionExtractor,
    TemporalReference,
    TemporalExpression,
    get_tense,
    get_aspect,
)

from decomposition.modality import (
    ModalityExtractor,
    EpistemicDetector,
    DeonticDetector,
    EvidentialDetector,
    EvidentialSource,
    DeonticForce,
    ModalMarker,
    get_certainty,
    has_modal,
)

from decomposition.commonsense import (
    CommonsenseExtractor,
    EventExtractor,
    TemplateGenerator,
    LLMGenerator,
    InferenceDeduplicator,
    ExtractedEvent,
    RelationDefinition,
    RELATION_DEFINITIONS,
    generate_inferences,
)

from decomposition.integrated_pipeline import (
    IntegratedPipeline,
    IntegratedPipelineConfig,
    IntegratedResult,
    Phase0Result,
    process_text,
    decompose_with_wsd,
)

__all__ = [
    # Enums
    "PresuppositionTrigger",
    "CommonsenseRelation",
    "Aspect",
    "Tense",
    "ModalType",
    "Polarity",
    "RoleType",
    "NegationType",
    "TemporalReference",
    "EvidentialSource",
    "DeonticForce",
    # Data structures
    "SemanticRole",
    "Presupposition",
    "CommonsenseInference",
    "TemporalInfo",
    "ModalityInfo",
    "NegationInfo",
    "CrossReference",
    "WeightedBranch",
    "DecomposedKnowledge",
    "Predicate",
    "Argument",
    "PredicateArgumentStructure",
    "NegationCue",
    "NegationAnalysis",
    "TemporalExpression",
    "ModalMarker",
    "ExtractedEvent",
    "RelationDefinition",
    # Config
    "DecompositionConfig",
    "StageEnablement",
    "SemanticRoleConfig",
    "PresuppositionConfig",
    "CommonsenseConfig",
    "NegationConfig",
    "TemporalConfig",
    "ModalityConfig",
    "WeightingConfig",
    # Pipeline
    "DecompositionPipeline",
    "PipelineError",
    "PipelineMetrics",
    "StageResult",
    "StageStatus",
    "decompose",
    "decompose_sync",
    # Presupposition extraction
    "PresuppositionExtractor",
    "TriggerDetector",
    "ContentGenerator",
    "DetectedTrigger",
    "detect_triggers",
    # Semantic role extraction
    "SemanticRoleExtractor",
    "PredicateDetector",
    "ArgumentDetector",
    "get_semantic_roles",
    "get_agent",
    "get_patient",
    # Negation extraction
    "NegationExtractor",
    "NegationDetector",
    "ScopeAnalyzer",
    "detect_negation",
    "get_polarity",
    # Temporal extraction
    "TemporalExtractor",
    "TenseDetector",
    "AspectClassifier",
    "TemporalExpressionExtractor",
    "get_tense",
    "get_aspect",
    # Modality extraction
    "ModalityExtractor",
    "EpistemicDetector",
    "DeonticDetector",
    "EvidentialDetector",
    "get_certainty",
    "has_modal",
    # Commonsense extraction
    "CommonsenseExtractor",
    "EventExtractor",
    "TemplateGenerator",
    "LLMGenerator",
    "InferenceDeduplicator",
    "RELATION_DEFINITIONS",
    "generate_inferences",
    # Integrated Pipeline (Phase 0 + Phase 1)
    "IntegratedPipeline",
    "IntegratedPipelineConfig",
    "IntegratedResult",
    "Phase0Result",
    "process_text",
    "decompose_with_wsd",
]

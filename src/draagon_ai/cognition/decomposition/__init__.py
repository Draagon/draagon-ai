"""Semantic Decomposition Pipeline for Natural Language Understanding.

This module provides the decomposition pipeline that breaks down natural language
into structured semantic knowledge for storage in the 4-layer cognitive memory.

The pipeline consists of two phases:

Phase 0 (Pre-processing):
- Content Analysis: Classify content type (prose, code, mixed)
- Word Sense Disambiguation (WSD): Resolve polysemous words to specific senses
- Entity Classification: Classify entities as INSTANCE, CLASS, ROLE, etc.

Phase 1 (Decomposition):
- Semantic Role Labeling: Extract predicate-argument structures
- Presupposition Detection: Identify implicit assumptions
- Commonsense Inference: Generate ATOMIC-style knowledge
- Temporal/Modal Analysis: Extract tense, aspect, modality
- Interpretation Branching: Handle ambiguity with weighted branches

Integration with Memory:
- Decomposition outputs flow into SemanticMemory layer
- Entities become Entity nodes in the graph
- Facts become Fact nodes with subject/predicate/object
- Relationships become edges between entities
- WSD senses are stored in node metadata for disambiguation

Example:
    from draagon_ai.cognition.decomposition import DecompositionService
    from draagon_ai.memory.layers.semantic import SemanticMemory

    # Create decomposition service
    decomposer = DecompositionService(llm=my_llm)

    # Process natural language
    result = await decomposer.decompose("Doug has 6 cats.")

    # Store in semantic memory
    for entity in result.entities:
        await semantic_memory.create_entity(
            name=entity.canonical_name,
            entity_type=entity.entity_type.value,
            properties={"synset_id": entity.synset_id},
        )

    for fact in result.facts:
        await semantic_memory.add_fact(
            content=fact.content,
            subject_entity_id=fact.subject_id,
            predicate=fact.predicate,
            object_value=fact.object_value,
        )

Based on prototype work in prototypes/implicit_knowledge_graphs/
"""

# Types from the simple service layer
from .types import (
    DecompositionResult,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    SemanticRole,
    Presupposition,
    CommonsenseInference,
    TemporalInfo,
    ModalityInfo,
    InterpretationBranch,
)

# Service layer
from .service import DecompositionService

# Phase 0: Identifiers and Universal Semantic Identifier
from .identifiers import (
    EntityType,
    UniversalSemanticIdentifier,
    SynsetInfo,
    LearnedSynset,
    SynsetSource,
    create_instance_identifier,
    create_class_identifier,
    create_role_identifier,
    create_anaphora_identifier,
    create_generic_identifier,
)

# Phase 0: Word Sense Disambiguation
from .wsd import (
    WordSenseDisambiguator,
    WSDConfig,
    DisambiguationResult,
    LeskDisambiguator,
    LLMDisambiguator,
    WordNetInterface,
    WordNetNotAvailableError,
    get_synset_id,
    synset_ids_match,
    are_same_word_different_sense,
)

# Phase 0: Entity Classification
from .entity_classifier import (
    EntityClassifier,
    ClassifierConfig,
    ClassificationResult,
    HeuristicClassifier,
    LLMClassifier,
    is_pronoun,
    is_generic,
    is_likely_proper_noun,
    extract_role_anchor,
)

# Phase 0: Content Analysis
from .content_analyzer import (
    ContentAnalyzer,
    ContentAnalysis,
    ContentType,
    ProcessingStrategy,
    ContentComponent,
    StructuralKnowledge,
    analyze_content,
    extract_natural_language,
)

# Phase 0: Evolving Synset Database
from .evolving_synsets import (
    EvolvingSynsetDatabase,
    EvolvingDBConfig,
    create_evolving_database,
)

# Memory Integration
from .memory_integration import (
    MemoryIntegration,
    DecompositionMemoryService,
    IntegrationConfig,
    IntegrationResult,
)

__all__ = [
    # ==========================================================================
    # Types (from simple service layer)
    # ==========================================================================
    "DecompositionResult",
    "ExtractedEntity",
    "ExtractedFact",
    "ExtractedRelationship",
    "SemanticRole",
    "Presupposition",
    "CommonsenseInference",
    "TemporalInfo",
    "ModalityInfo",
    "InterpretationBranch",
    # ==========================================================================
    # Service
    # ==========================================================================
    "DecompositionService",
    # ==========================================================================
    # Phase 0: Identifiers
    # ==========================================================================
    "EntityType",
    "UniversalSemanticIdentifier",
    "SynsetInfo",
    "LearnedSynset",
    "SynsetSource",
    "create_instance_identifier",
    "create_class_identifier",
    "create_role_identifier",
    "create_anaphora_identifier",
    "create_generic_identifier",
    # ==========================================================================
    # Phase 0: Word Sense Disambiguation
    # ==========================================================================
    "WordSenseDisambiguator",
    "WSDConfig",
    "DisambiguationResult",
    "LeskDisambiguator",
    "LLMDisambiguator",
    "WordNetInterface",
    "WordNetNotAvailableError",
    "get_synset_id",
    "synset_ids_match",
    "are_same_word_different_sense",
    # ==========================================================================
    # Phase 0: Entity Classification
    # ==========================================================================
    "EntityClassifier",
    "ClassifierConfig",
    "ClassificationResult",
    "HeuristicClassifier",
    "LLMClassifier",
    "is_pronoun",
    "is_generic",
    "is_likely_proper_noun",
    "extract_role_anchor",
    # ==========================================================================
    # Phase 0: Content Analysis
    # ==========================================================================
    "ContentAnalyzer",
    "ContentAnalysis",
    "ContentType",
    "ProcessingStrategy",
    "ContentComponent",
    "StructuralKnowledge",
    "analyze_content",
    "extract_natural_language",
    # ==========================================================================
    # Phase 0: Evolving Synset Database
    # ==========================================================================
    "EvolvingSynsetDatabase",
    "EvolvingDBConfig",
    "create_evolving_database",
    # ==========================================================================
    # Memory Integration
    # ==========================================================================
    "MemoryIntegration",
    "DecompositionMemoryService",
    "IntegrationConfig",
    "IntegrationResult",
]

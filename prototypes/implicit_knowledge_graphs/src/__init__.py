"""Implicit Knowledge Graphs Prototype.

A prototype exploring pre-storage semantic decomposition for optimized LLM context retrieval.

Core Hypothesis:
    By decomposing text into weighted semantic graphs with proper disambiguation,
    we can provide LLMs with more efficient context than raw text or traditional RAG.

Key Principles:
    1. WSD-First: Word sense disambiguation is foundational, not optional
    2. Multi-Type Decomposition: Extract all implicit knowledge types
    3. Weighted Branches: Store multiple interpretations with confidence weights
    4. Evolutionary Optimization: Every component is evolvable

Modules:
    identifiers: Universal semantic identifier system
    wsd: Word sense disambiguation
    content_analyzer: LLM-driven content type classification
    content_aware_wsd: Content-aware WSD integration
    text_chunking: Large text handling and context extraction
    entity_classifier: Entity type classification
    synset_learning: Learning new synsets from context
    evolving_synsets: Extensible synset database

Example:
    >>> from implicit_knowledge_graphs import (
    ...     UniversalSemanticIdentifier,
    ...     EntityType,
    ...     disambiguate,
    ...     decompose,
    ... )
    >>>
    >>> # Disambiguate a word in context
    >>> identifier = await disambiguate("bank", "I deposited money in the bank")
    >>> print(identifier.wordnet_synset)  # "bank.n.01"
    >>>
    >>> # Decompose a sentence into implicit knowledge
    >>> knowledge = await decompose("Doug forgot the meeting again")
    >>> print(knowledge.presuppositions)  # ["Doug forgot before", "A meeting exists"]
"""

__version__ = "0.1.0"
__author__ = "draagon-ai"

# Phase 0: Universal Semantic Identification
from identifiers import (
    EntityType,
    SynsetInfo,
    SynsetSource,
    LearnedSynset,
    UniversalSemanticIdentifier,
    create_instance_identifier,
    create_class_identifier,
    create_role_identifier,
    create_anaphora_identifier,
    create_generic_identifier,
)

from evolving_synsets import (
    EvolvingDBConfig,
    EvolvingSynsetDatabase,
    create_evolving_database,
)

from wsd import (
    WSDConfig,
    DisambiguationResult,
    WordNetInterface,
    WordNetNotAvailableError,
    LeskDisambiguator,
    LLMDisambiguator,
    WordSenseDisambiguator,
    get_synset_id,
    synset_ids_match,
    are_same_word_different_sense,
)

from entity_classifier import (
    ClassifierConfig,
    ClassificationResult,
    HeuristicClassifier,
    LLMClassifier,
    EntityClassifier,
    is_pronoun,
    is_generic,
    is_likely_proper_noun,
    extract_role_anchor,
)

from synset_learning import (
    SynsetLearningConfig,
    SynsetLearningService,
    QdrantSynsetStore,
    UnknownTermRecord,
    DefinitionExtraction,
    ReinforcementResult,
    ResolutionSource,
)

from text_chunking import (
    TextChunker,
    TextChunk,
    ChunkingConfig,
    segment_sentences,
    get_context_for_wsd,
    chunk_large_text,
)

from content_analyzer import (
    ContentAnalyzer,
    ContentAnalysis,
    ContentComponent,
    ContentType,
    ProcessingStrategy,
    StructuralKnowledge,
    analyze_content,
    extract_natural_language,
)

from content_aware_wsd import (
    ContentAwareWSD,
    ContentAwareWSDResult,
    process_content_for_wsd,
    disambiguate_in_context,
)

__all__ = [
    # Version
    "__version__",
    # Identifiers
    "EntityType",
    "SynsetInfo",
    "SynsetSource",
    "LearnedSynset",
    "UniversalSemanticIdentifier",
    "create_instance_identifier",
    "create_class_identifier",
    "create_role_identifier",
    "create_anaphora_identifier",
    "create_generic_identifier",
    # Evolving Synsets
    "EvolvingDBConfig",
    "EvolvingSynsetDatabase",
    "create_evolving_database",
    # WSD
    "WSDConfig",
    "DisambiguationResult",
    "WordNetInterface",
    "WordNetNotAvailableError",
    "LeskDisambiguator",
    "LLMDisambiguator",
    "WordSenseDisambiguator",
    "get_synset_id",
    "synset_ids_match",
    "are_same_word_different_sense",
    # Entity Classifier
    "ClassifierConfig",
    "ClassificationResult",
    "HeuristicClassifier",
    "LLMClassifier",
    "EntityClassifier",
    "is_pronoun",
    "is_generic",
    "is_likely_proper_noun",
    "extract_role_anchor",
    # Synset Learning
    "SynsetLearningConfig",
    "SynsetLearningService",
    "QdrantSynsetStore",
    "UnknownTermRecord",
    "DefinitionExtraction",
    "ReinforcementResult",
    "ResolutionSource",
    # Text Chunking
    "TextChunker",
    "TextChunk",
    "ChunkingConfig",
    "segment_sentences",
    "get_context_for_wsd",
    "chunk_large_text",
    # Content Analyzer
    "ContentAnalyzer",
    "ContentAnalysis",
    "ContentComponent",
    "ContentType",
    "ProcessingStrategy",
    "StructuralKnowledge",
    "analyze_content",
    "extract_natural_language",
    # Content-Aware WSD
    "ContentAwareWSD",
    "ContentAwareWSDResult",
    "process_content_for_wsd",
    "disambiguate_in_context",
]

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
    decomposition: Multi-type implicit knowledge extraction
    branches: Weighted branch generation
    storage: Graph storage with synset-based linking
    retrieval: Optimized context retrieval
    evolution: Evolutionary optimization framework
    evaluation: Multi-dimensional Opus 4.5 scoring

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
    UniversalSemanticIdentifier,
    create_instance_identifier,
    create_class_identifier,
    create_role_identifier,
    create_anaphora_identifier,
    create_generic_identifier,
)

from wsd import (
    WSDConfig,
    DisambiguationResult,
    WordNetInterface,
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

__all__ = [
    # Version
    "__version__",
    # Identifiers
    "EntityType",
    "SynsetInfo",
    "UniversalSemanticIdentifier",
    "create_instance_identifier",
    "create_class_identifier",
    "create_role_identifier",
    "create_anaphora_identifier",
    "create_generic_identifier",
    # WSD
    "WSDConfig",
    "DisambiguationResult",
    "WordNetInterface",
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
]

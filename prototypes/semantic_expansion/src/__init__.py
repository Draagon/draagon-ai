"""Semantic Expansion Prototype.

This prototype explores deep semantic expansion of statements:
- Word Sense Disambiguation (WSD) using WordNet synsets
- Semantic Frame Expansion for extracting implicit meaning
- Variation Generation with cognitive weighting
- Cross-layer memory associations
- Two-pass memory integration for context-aware expansion

See README.md for full documentation.
"""

from semantic_types import (
    WordSense,
    SemanticTriple,
    SemanticFrame,
    ExpansionVariant,
    CrossLayerEdge,
    CrossLayerRelation,
)
from wsd import (
    WordSenseDisambiguator,
    LeskDisambiguator,
)
from expansion import (
    SemanticExpansionService,
    VariationGenerator,
    ExpansionInput,
    EntityInfo,
)
from integration import (
    TwoPassSemanticOrchestrator,
    PreExpansionRetriever,
    PostExpansionRetriever,
    NaturalLanguageGenerator,
    PreExpansionContext,
    VariantEvidence,
    DetectedConflict,
    ProcessingResult,
    process_with_memory,
)

__all__ = [
    # Types
    "WordSense",
    "SemanticTriple",
    "SemanticFrame",
    "ExpansionVariant",
    "CrossLayerEdge",
    "CrossLayerRelation",
    # WSD
    "WordSenseDisambiguator",
    "LeskDisambiguator",
    # Expansion
    "SemanticExpansionService",
    "VariationGenerator",
    "ExpansionInput",
    "EntityInfo",
    # Two-Pass Integration
    "TwoPassSemanticOrchestrator",
    "PreExpansionRetriever",
    "PostExpansionRetriever",
    "NaturalLanguageGenerator",
    "PreExpansionContext",
    "VariantEvidence",
    "DetectedConflict",
    "ProcessingResult",
    "process_with_memory",
]

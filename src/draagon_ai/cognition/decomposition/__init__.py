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

from .types import (
    EntityType,
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
from .service import DecompositionService

__all__ = [
    # Types
    "EntityType",
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
    # Service
    "DecompositionService",
]

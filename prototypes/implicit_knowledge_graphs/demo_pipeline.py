#!/usr/bin/env python3
"""Demonstration of the Full Phase 0 + Phase 1 Pipeline.

This script shows each transformation step with detailed output,
proving the pipeline is actually extracting semantic information.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from content_analyzer import ContentAnalyzer, ContentType
from content_aware_wsd import ContentAwareWSD
from entity_classifier import EntityClassifier, ClassifierConfig
from wsd import DisambiguationResult
from decomposition import (
    IntegratedPipeline,
    IntegratedPipelineConfig,
    DecompositionPipeline,
    DecompositionConfig,
)


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


async def demo_simple_pipeline():
    """Demonstrate the pipeline on a simple sentence."""

    print_header("DEMONSTRATION: Full Phase 0 + Phase 1 Pipeline")

    # Input text
    text = "Doug forgot the important meeting again because he was too busy."

    print_subheader("INPUT TEXT")
    print(f'  "{text}"')

    # Create pipeline
    pipeline = IntegratedPipeline()

    # Process
    print_subheader("PROCESSING...")
    result = await pipeline.process(text)

    # Show Phase 0 Results
    print_header("PHASE 0: Content Analysis + WSD + Entity Classification")

    print_subheader("1. Content Type Detection")
    print(f"  Content Type: {result.content_type.value}")
    print(f"  Processing Duration: {result.total_duration_ms:.1f}ms")
    print(f"  Chunks Processed: {result.chunks_processed}")

    print_subheader("2. Word Sense Disambiguation (WSD)")
    if result.phase0.disambiguation_results:
        for word, disambiguation in result.phase0.disambiguation_results.items():
            print(f"\n  Word: '{word}'")
            print(f"    Synset ID: {disambiguation.synset_id}")
            print(f"    Definition: {disambiguation.definition[:60]}..." if disambiguation.definition else "    Definition: (not available)")
            print(f"    Confidence: {disambiguation.confidence:.2f}")
            print(f"    Method: {disambiguation.method}")
            if disambiguation.alternatives:
                print(f"    Alternatives: {disambiguation.alternatives}")
    else:
        print("  (No WSD results - WordNet may not be available)")

    print_subheader("3. Entity Classification")
    if result.phase0.entity_classifications:
        for entity_text, classification in result.phase0.entity_classifications.items():
            print(f"\n  Entity: '{entity_text}'")
            print(f"    Type: {classification.entity_type.value}")
            print(f"    Confidence: {classification.confidence:.2f}")
    else:
        print("  (No entity classifications)")

    # Show Phase 1 Results
    print_header("PHASE 1: Multi-Type Decomposition")

    decomp = result.decomposition

    print_subheader("1. Semantic Roles (Predicate-Argument Structures)")
    if decomp.semantic_roles:
        for role in decomp.semantic_roles:
            print(f"\n  Predicate: '{role.predicate}' (sense: {role.predicate_sense})")
            print(f"    Role: {role.role}")
            print(f"    Filler: '{role.filler}'")
            print(f"    Confidence: {role.confidence:.2f}")
    else:
        print("  (No semantic roles extracted)")

    print_subheader("2. Presuppositions (Implicit Assumptions)")
    if decomp.presuppositions:
        for presup in decomp.presuppositions:
            print(f"\n  Presupposition: \"{presup.content}\"")
            print(f"    Trigger Type: {presup.trigger_type.value}")
            print(f"    Trigger Text: '{presup.trigger_text}'")
            print(f"    Confidence: {presup.confidence:.2f}")
            print(f"    Cancellable: {presup.cancellable}")
    else:
        print("  (No presuppositions extracted)")

    print_subheader("3. Commonsense Inferences (ATOMIC-style)")
    if decomp.commonsense_inferences:
        for i, inference in enumerate(decomp.commonsense_inferences[:10]):  # Limit to 10
            print(f"\n  [{i+1}] Relation: {inference.relation.value}")
            print(f"      Head: \"{inference.head}\"")
            print(f"      Tail: \"{inference.tail}\"")
            print(f"      Confidence: {inference.confidence:.2f}")
        if len(decomp.commonsense_inferences) > 10:
            print(f"\n  ... and {len(decomp.commonsense_inferences) - 10} more inferences")
    else:
        print("  (No commonsense inferences)")

    print_subheader("4. Temporal Information")
    if decomp.temporal:
        print(f"  Tense: {decomp.temporal.tense.value}")
        print(f"  Aspect: {decomp.temporal.aspect.value}")
        if decomp.temporal.reference_type:
            print(f"  Reference Type: {decomp.temporal.reference_type}")
        print(f"  Confidence: {decomp.temporal.confidence:.2f}")
    else:
        print("  (No temporal information)")

    print_subheader("5. Modality Information")
    if decomp.modality and decomp.modality.modal_type.value != "none":
        print(f"  Modal Type: {decomp.modality.modal_type.value}")
        if decomp.modality.modal_marker:
            print(f"  Marker: '{decomp.modality.modal_marker}'")
        print(f"  Certainty: {decomp.modality.certainty:.2f}")
    else:
        print("  (No modality markers)")

    print_subheader("6. Negation & Polarity")
    if decomp.negation:
        print(f"  Is Negated: {decomp.negation.is_negated}")
        if decomp.negation.negation_cue:
            print(f"  Negation Cue: '{decomp.negation.negation_cue}'")
        print(f"  Polarity: {decomp.negation.polarity.value}")
    else:
        print("  (No negation information)")

    print_subheader("7. Interpretation Branches (Weighted)")
    for i, branch in enumerate(decomp.branches):
        print(f"\n  Branch {i+1}:")
        print(f"    Interpretation: {branch.interpretation}")
        print(f"    Confidence: {branch.confidence:.3f}")
        print(f"    Memory Support: {branch.memory_support:.3f}")
        print(f"    Final Weight: {branch.final_weight:.3f}")
        print(f"    Evidence: {branch.supporting_evidence}")
        if branch.entity_interpretations:
            print(f"    Entity Senses: {branch.entity_interpretations}")

    print_header("SUMMARY")
    print(f"  Source: \"{decomp.source_text}\"")
    print(f"  Entity IDs: {decomp.entity_ids}")
    print(f"  Semantic Roles: {len(decomp.semantic_roles)}")
    print(f"  Presuppositions: {len(decomp.presuppositions)}")
    print(f"  Commonsense Inferences: {len(decomp.commonsense_inferences)}")
    print(f"  Interpretation Branches: {len(decomp.branches)}")
    print(f"  Pipeline Version: {decomp.pipeline_version}")


async def demo_ambiguous_wsd():
    """Demonstrate WSD alternatives creating multiple branches."""

    print_header("DEMONSTRATION: WSD Alternatives → Multiple Branches")

    text = "I deposited money at the bank near the river."

    print_subheader("INPUT TEXT")
    print(f'  "{text}"')
    print("\n  Note: 'bank' is ambiguous - could be financial or river bank")

    # Use DecompositionPipeline directly to show branch creation
    from decomposition.config import DecompositionConfig, WeightingConfig

    config = DecompositionConfig(
        weighting=WeightingConfig(min_branch_weight=0.01)  # Keep all branches
    )
    pipeline = DecompositionPipeline(config=config)

    # Simulate WSD alternatives (as if WSD found multiple possible senses)
    result = await pipeline.decompose(
        text,
        wsd_results={"bank": "bank.n.01"},  # Primary: financial institution
        wsd_alternatives={
            "bank": [
                ("bank.n.01", 0.55),  # Financial institution - slight preference
                ("bank.n.02", 0.35),  # River bank
                ("bank.n.03", 0.10),  # Storage/depot
            ]
        },
    )

    print_subheader("WSD ALTERNATIVES PROVIDED")
    print("  bank.n.01 (financial institution) - confidence: 0.55")
    print("  bank.n.02 (river bank) - confidence: 0.35")
    print("  bank.n.03 (storage depot) - confidence: 0.10 (will be filtered)")

    print_subheader("RESULTING INTERPRETATION BRANCHES")
    for i, branch in enumerate(result.branches):
        print(f"\n  Branch {i+1}:")
        print(f"    Interpretation: {branch.interpretation}")
        print(f"    Confidence: {branch.confidence:.3f}")
        print(f"    Entity Senses: {branch.entity_interpretations}")
        print(f"    Evidence: {branch.supporting_evidence}")

    print("\n  --> Multiple branches allow downstream systems to select")
    print("      the most contextually appropriate interpretation!")


async def demo_entity_types_affecting_commonsense():
    """Demonstrate how entity types affect commonsense inference selection."""

    print_header("DEMONSTRATION: Entity Types → Commonsense Relation Selection")

    from decomposition.commonsense import CommonsenseExtractor
    from decomposition.config import CommonsenseConfig

    config = CommonsenseConfig(
        tier1_relations=["xIntent", "xReact"],
        tier2_relations=["xNeed", "xWant", "oReact"],
    )
    extractor = CommonsenseExtractor(config=config)

    print_subheader("SCENARIO 1: With INSTANCE Entity (Doug)")
    entity_types_with_instance = {"Doug": "instance", "meeting": "class"}
    print(f"  Entity Types: {entity_types_with_instance}")

    relations = extractor._get_relations(entity_types_with_instance)
    relation_names = [r.value for r in relations]
    print(f"  Relations Selected: {relation_names}")
    print("  --> Tier 2 INCLUDED because 'Doug' is INSTANCE (specific person)")

    print_subheader("SCENARIO 2: With Only CLASS Entities")
    entity_types_class_only = {"meeting": "class", "schedule": "class"}
    print(f"  Entity Types: {entity_types_class_only}")

    relations = extractor._get_relations(entity_types_class_only)
    relation_names = [r.value for r in relations]
    print(f"  Relations Selected: {relation_names}")
    print("  --> Tier 2 EXCLUDED because no INSTANCE entities")

    print("\n  --> This optimization reduces unnecessary inferences for")
    print("      generic statements while enriching inferences about")
    print("      specific people/things!")


async def demo_long_document_chunking():
    """Demonstrate chunking for long documents."""

    print_header("DEMONSTRATION: Long Document Chunking")

    # Create a longer document
    paragraphs = [
        "Doug went to the bank to deposit his paycheck.",
        "The teller was very helpful and processed the transaction quickly.",
        "Afterward, Doug walked along the river bank to enjoy the view.",
        "He remembered that he had forgotten to pay his electricity bill again.",
        "The sunset over the water was beautiful.",
    ] * 3  # Repeat to make it longer

    long_text = " ".join(paragraphs)

    print_subheader("INPUT TEXT (Excerpt)")
    print(f'  "{long_text[:150]}..."')
    print(f"  Total length: {len(long_text)} characters")

    # Use small max_content_length to trigger chunking
    config = IntegratedPipelineConfig(max_content_length=200)
    pipeline = IntegratedPipeline(config=config)

    result = await pipeline.process(long_text)

    print_subheader("CHUNKING RESULTS")
    print(f"  Chunks Processed: {result.chunks_processed}")
    print(f"  Total Duration: {result.total_duration_ms:.1f}ms")

    print_subheader("EXTRACTION SUMMARY ACROSS ALL CHUNKS")
    decomp = result.decomposition
    print(f"  Semantic Roles: {len(decomp.semantic_roles)}")
    print(f"  Presuppositions: {len(decomp.presuppositions)}")
    print(f"  Commonsense Inferences: {len(decomp.commonsense_inferences)}")

    print_subheader("PRESUPPOSITIONS FOUND (from 'again' triggers)")
    for presup in decomp.presuppositions:
        if presup.trigger_type.value == "iterative":
            print(f"  - \"{presup.content}\" (trigger: '{presup.trigger_text}')")


async def main():
    """Run all demonstrations."""
    await demo_simple_pipeline()
    await demo_ambiguous_wsd()
    await demo_entity_types_affecting_commonsense()
    await demo_long_document_chunking()

    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

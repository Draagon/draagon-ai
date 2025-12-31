#!/usr/bin/env python3
"""Demonstration of Complex, Vague, and Factual Sentences.

Shows how the pipeline handles:
1. Long compound sentences with complex grammar
2. Vague/ambiguous sentences like "I got it!"
3. Simple factual statements like "Doug has 6 cats"
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from decomposition import IntegratedPipeline


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


def show_full_decomposition(result):
    """Display all decomposition results in detail."""

    decomp = result.decomposition

    # Phase 0
    print_subheader("PHASE 0: Word Sense Disambiguation")
    if result.phase0.disambiguation_results:
        for word, dis in result.phase0.disambiguation_results.items():
            alt_str = f" | Alternatives: {dis.alternatives}" if dis.alternatives else ""
            print(f"  '{word}' → {dis.synset_id} ({dis.method}, conf: {dis.confidence:.2f}){alt_str}")
            if dis.definition:
                print(f"           Definition: \"{dis.definition[:70]}...\"" if len(dis.definition) > 70 else f"           Definition: \"{dis.definition}\"")
    else:
        print("  (No WSD results)")

    print_subheader("PHASE 0: Entity Classification")
    if result.phase0.entity_classifications:
        for entity, cls in result.phase0.entity_classifications.items():
            print(f"  '{entity}' → {cls.entity_type.value} (conf: {cls.confidence:.2f})")
    else:
        print("  (No entities classified)")

    # Phase 1
    print_subheader("PHASE 1: Semantic Roles (Predicate-Argument Structure)")
    if decomp.semantic_roles:
        # Group by predicate
        predicates = {}
        for role in decomp.semantic_roles:
            if role.predicate not in predicates:
                predicates[role.predicate] = []
            predicates[role.predicate].append(role)

        for pred, roles in predicates.items():
            sense = roles[0].predicate_sense or "unknown"
            print(f"\n  Predicate: '{pred}' (sense: {sense})")
            for role in roles:
                print(f"    {role.role}: '{role.filler}' (conf: {role.confidence:.2f})")
    else:
        print("  (No semantic roles extracted)")

    print_subheader("PHASE 1: Presuppositions (What Must Be True)")
    if decomp.presuppositions:
        for i, p in enumerate(decomp.presuppositions, 1):
            print(f"\n  [{i}] \"{p.content}\"")
            print(f"      Trigger: '{p.trigger_text}' ({p.trigger_type.value})")
            print(f"      Confidence: {p.confidence:.2f} | Cancellable: {p.cancellable}")
    else:
        print("  (No presuppositions)")

    print_subheader("PHASE 1: Commonsense Inferences")
    if decomp.commonsense_inferences:
        for i, inf in enumerate(decomp.commonsense_inferences[:8], 1):
            print(f"  [{i}] {inf.relation.value}: \"{inf.head}\" → \"{inf.tail}\"")
        if len(decomp.commonsense_inferences) > 8:
            print(f"  ... and {len(decomp.commonsense_inferences) - 8} more")
    else:
        print("  (No inferences)")

    print_subheader("PHASE 1: Temporal & Modality")
    if decomp.temporal:
        print(f"  Tense: {decomp.temporal.tense.value}")
        print(f"  Aspect: {decomp.temporal.aspect.value}")
        if decomp.temporal.reference_type:
            print(f"  Reference: {decomp.temporal.reference_type} = {decomp.temporal.reference_value}")
    else:
        print("  Temporal: (not extracted)")

    if decomp.modality and decomp.modality.modal_type.value != "none":
        print(f"  Modal Type: {decomp.modality.modal_type.value}")
        if decomp.modality.modal_marker:
            print(f"  Modal Marker: '{decomp.modality.modal_marker}'")
        print(f"  Certainty: {decomp.modality.certainty:.2f}")
    else:
        print("  Modality: (none)")

    print_subheader("PHASE 1: Negation")
    if decomp.negation:
        print(f"  Is Negated: {decomp.negation.is_negated}")
        print(f"  Polarity: {decomp.negation.polarity.value}")
        if decomp.negation.negation_cue:
            print(f"  Cue: '{decomp.negation.negation_cue}'")

    print_subheader("Interpretation Branches")
    for i, branch in enumerate(decomp.branches, 1):
        print(f"\n  Branch {i}: {branch.interpretation[:60]}...")
        print(f"    Confidence: {branch.confidence:.4f}")
        print(f"    Final Weight: {branch.final_weight:.4f}")
        print(f"    Evidence: {branch.supporting_evidence}")
        if branch.entity_interpretations:
            print(f"    Entity Senses: {branch.entity_interpretations}")


async def demo_complex_compound():
    """Complex compound sentence with multiple clauses."""

    print_header("CASE 1: Complex Compound Sentence")

    text = """Although Doug had promised his wife that he would remember to pick up
the groceries after work, he completely forgot about it again because
he was so preoccupied with the quarterly report that his manager had
unexpectedly asked him to finish before the board meeting tomorrow."""

    # Clean up whitespace
    text = " ".join(text.split())

    print_subheader("INPUT TEXT")
    print(f'  "{text}"')
    print(f"\n  Length: {len(text)} characters")
    print("  Structure: Multiple embedded clauses, temporal references, causal chain")

    pipeline = IntegratedPipeline()
    result = await pipeline.process(text)

    show_full_decomposition(result)

    print_subheader("ANALYSIS NOTES")
    print("""
  This sentence contains:
  - Concessive clause: "Although Doug had promised..."
  - Embedded complement: "that he would remember..."
  - Infinitival: "to pick up the groceries"
  - Temporal adjunct: "after work"
  - Causal clause: "because he was so preoccupied..."
  - Relative clause: "that his manager had asked him to finish"
  - Temporal reference: "tomorrow"
  - Iterative presupposition: "again" (he's forgotten before!)
  - Factive: "forgot" presupposes the complement was true

  The pipeline extracts:
  - Multiple predicates (promised, remember, forgot, preoccupied, asked, finish)
  - The iterative "again" triggers presupposition detection
  - Temporal structure (past perfect, future reference)
  - Causal relationships
""")


async def demo_vague_ambiguous():
    """Vague, ambiguous sentence."""

    print_header("CASE 2: Vague/Ambiguous Sentence")

    text = "I got it!"

    print_subheader("INPUT TEXT")
    print(f'  "{text}"')
    print("\n  This is highly ambiguous:")
    print("    - 'I' = who? (anaphora, needs context)")
    print("    - 'got' = obtained? understood? caught? received?")
    print("    - 'it' = what? (anaphora, needs context)")

    pipeline = IntegratedPipeline()
    result = await pipeline.process(text)

    show_full_decomposition(result)

    print_subheader("ANALYSIS NOTES")
    print("""
  Key observations:

  1. ENTITY CLASSIFICATION:
     - 'I' should be classified as ANAPHORA (pronoun needing resolution)
     - Without prior context, we can't know who "I" refers to

  2. WSD for "got":
     - WordNet has 22+ senses for "get"
     - Without context, WSD must guess or report low confidence
     - Common senses: get.v.01 (obtain), get.v.03 (understand), etc.

  3. PRESUPPOSITIONS:
     - "it" presupposes something exists to be gotten
     - This is a definite description trigger

  4. COMMONSENSE:
     - Hard to generate without knowing what was "gotten"
     - Generic inferences like "speaker wanted something" apply

  5. INTERPRETATION BRANCHES:
     - With WSD alternatives, multiple branches could represent:
       * "I understood it" interpretation
       * "I obtained it" interpretation
       * "I caught it" interpretation

  This demonstrates the LIMITS of decomposition without context.
  Phase 2 (memory integration) would help resolve anaphora.
""")


async def demo_simple_fact():
    """Simple factual statement."""

    print_header("CASE 3: Simple Factual Statement")

    text = "Doug has 6 cats."

    print_subheader("INPUT TEXT")
    print(f'  "{text}"')
    print("\n  A straightforward possessive statement of fact.")

    pipeline = IntegratedPipeline()
    result = await pipeline.process(text)

    show_full_decomposition(result)

    print_subheader("ANALYSIS NOTES")
    print("""
  Key extractions:

  1. ENTITY CLASSIFICATION:
     - 'Doug' = INSTANCE (specific person, proper noun)
     - 'cats' = CLASS (category of animal)
     - '6' = quantity/cardinality

  2. SEMANTIC ROLES:
     - Predicate: 'has' (possession)
     - ARG0 (Possessor): Doug
     - ARG1 (Possessed): 6 cats

  3. PRESUPPOSITIONS:
     - Possessive triggers: "Doug's cats exist"
     - The specific number (6) is asserted, not presupposed

  4. TEMPORAL:
     - Present tense = current state
     - Aspect: STATE (not an activity or achievement)

  5. COMMONSENSE INFERENCES:
     - Doug is a pet owner
     - Doug feeds/cares for cats
     - Doug's home likely has cat-related items
     - 6 cats is a notable quantity

  6. FOR MEMORY STORAGE:
     - This becomes a FACT about Doug
     - Entity: Doug (INSTANCE)
     - Relation: owns/has
     - Object: cats (CLASS)
     - Quantity: 6

  This is the kind of simple fact that should be stored in
  semantic memory for later retrieval: "How many cats does Doug have?"
""")


async def demo_contrast():
    """Show side-by-side comparison."""

    print_header("SUMMARY COMPARISON")

    cases = [
        ("Complex Compound",
         "Although Doug had promised his wife that he would remember to pick up the groceries after work, he completely forgot about it again because he was so preoccupied with the quarterly report.",
         "Multi-clause, temporal, causal"),

        ("Vague/Ambiguous",
         "I got it!",
         "Anaphora, WSD ambiguity, context-dependent"),

        ("Simple Fact",
         "Doug has 6 cats.",
         "Possessive, cardinality, state"),
    ]

    pipeline = IntegratedPipeline()

    print("\n  Sentence                         | Roles | Presup | Infer | Branches")
    print("  " + "-" * 75)

    for name, text, desc in cases:
        result = await pipeline.process(text)
        d = result.decomposition
        short_text = text[:30] + "..." if len(text) > 30 else text
        print(f"  {short_text:<34} | {len(d.semantic_roles):>5} | {len(d.presuppositions):>6} | {len(d.commonsense_inferences):>5} | {len(d.branches):>8}")

    print("\n  Notes:")
    print("  - Complex sentences yield more semantic roles")
    print("  - Trigger words (again, the, forgot) generate presuppositions")
    print("  - INSTANCE entities (Doug) trigger more commonsense inferences")
    print("  - Ambiguous words with alternatives create multiple branches")


async def main():
    await demo_complex_compound()
    await demo_vague_ambiguous()
    await demo_simple_fact()
    await demo_contrast()

    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

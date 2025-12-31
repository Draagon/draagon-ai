# FR-007: Semantic Expansion Service

**Feature:** LLM-based semantic frame expansion with variation generation
**Status:** Implemented (Prototype)
**Priority:** P1 (Core Semantic Infrastructure)
**Complexity:** High
**Phase:** Semantic Expansion Phase 1

---

## Overview

Implement a Semantic Expansion Service that transforms natural language statements into rich semantic frames containing triples, presuppositions, implications, and ambiguities. When ambiguities exist, generate multiple interpretation variants scored by cognitive plausibility across all memory layers.

### Research Foundation

- **Frame Semantics (Fillmore, 1976)**: Semantic frames as structured knowledge
- **ATOMIC/COMET**: Commonsense reasoning about causes/effects
- **ConceptNet**: Semantic relations between concepts
- **Presupposition Theory**: Implicit assumptions in statements
- **Cognitive Psychology**: Multi-factor plausibility scoring

### Why This Matters

Short statements like "Doug prefers tea in the morning" contain implicit knowledge:
- **Triples**: (Doug, PREFERS, tea, {temporal: morning})
- **Presuppositions**: Doug has experience with tea, mornings matter to Doug
- **Implications**: Doug would accept tea if offered in the morning
- **Negations**: Doug does not prefer other beverages in the morning

Without semantic expansion, memories remain shallow strings that can't support reasoning.

---

## Functional Requirements

### FR-007.1: Semantic Frame Structure

**Requirement:** Define comprehensive semantic frame data structure

**Acceptance Criteria:**
- `SemanticFrame` dataclass with:
  - `original_text`: Source statement
  - `triples`: List of `SemanticTriple` (subject, predicate, object, context)
  - `presuppositions`: List of `Presupposition` (implicit assumptions)
  - `implications`: List of `Implication` (likely consequences)
  - `negations`: List of strings (what this rules out)
  - `ambiguities`: List of `Ambiguity` (unresolved references)
  - `open_questions`: List of strings (gaps in understanding)
  - `word_senses`: Dict mapping words to `WordSense`
  - `frame_type`: ASSERTION, REQUEST, QUESTION, etc.
  - `confidence`: Overall frame confidence (0-1)
- All components have confidence scores

**Implementation Status:** ✅ Implemented in `src/draagon_ai/semantic/types.py`

---

### FR-007.2: Semantic Triple Extraction

**Requirement:** Extract subject-predicate-object relationships from text

**Acceptance Criteria:**
- `SemanticTriple` includes:
  - `subject`: Entity performing action
  - `predicate`: Relation type (PREFERS, IS_A, LIKES, etc.)
  - `object`: Target of relation
  - `context`: Dict of qualifiers (temporal, location, condition)
  - `subject_synset`: Synset ID for subject disambiguation
  - `object_synset`: Synset ID for object disambiguation
  - `confidence`: Extraction confidence
  - `source`: Provenance
- `to_text()` method for natural language conversion

**Test Approach:**
- "Doug prefers tea" → (Doug, PREFERS, tea)
- "I went to the bank in the morning" → (I, WENT_TO, bank, {temporal: morning})

**Implementation Status:** ✅ Implemented

---

### FR-007.3: Presupposition Extraction

**Requirement:** Extract implicit assumptions from statements

**Acceptance Criteria:**
- `Presupposition` dataclass with:
  - `content`: The presupposed fact
  - `presupposition_type`: existential, factive, lexical
  - `confidence`: Strength of presupposition
  - `triggered_by`: Word/phrase that triggered it
- Types of presuppositions:
  - **Existential**: "Doug prefers tea" → Doug exists
  - **Factive**: "John stopped smoking" → John smoked before
  - **Lexical**: "I regret leaving" → I left

**Test Approach:**
- "Doug prefers tea" → presupposes Doug exists, Doug has tried tea
- "He stopped running" → presupposes he was running before

**Implementation Status:** ✅ Implemented

---

### FR-007.4: Implication Inference

**Requirement:** Infer likely consequences from statements

**Acceptance Criteria:**
- `Implication` dataclass with:
  - `content`: The inferred consequence
  - `implication_type`: pragmatic, logical, commonsense
  - `confidence`: Inference confidence
  - `source`: "atomic", "conceptnet", "llm"
- Sources of implications:
  - **Pragmatic**: Conversational implicatures
  - **Logical**: Direct entailments
  - **Commonsense**: ATOMIC/COMET-style inferences

**Test Approach:**
- "Doug prefers tea" → Doug would accept tea if offered
- "She opened the door" → The door was closed before

**Implementation Status:** ✅ Implemented

---

### FR-007.5: Ambiguity Detection

**Requirement:** Identify unresolved ambiguities in statements

**Acceptance Criteria:**
- `Ambiguity` dataclass with:
  - `text`: The ambiguous text
  - `ambiguity_type`: reference, word_sense, scope, temporal
  - `possibilities`: List of possible resolutions
  - `resolution`: Chosen resolution (if any)
  - `resolution_confidence`: Confidence in resolution
- Types detected:
  - **Reference**: "He prefers tea" (who is "he"?)
  - **Word Sense**: "bank" (financial or riverbank?)
  - **Scope**: "every student read a book" (same or different books?)
  - **Temporal**: "I'll do it tomorrow" (which tomorrow?)

**Implementation Status:** ✅ Implemented

---

### FR-007.6: LLM-Based Frame Extraction

**Requirement:** Use LLM to extract semantic frames from text

**Acceptance Criteria:**
- XML prompt template for frame extraction
- Structured output parsing for triples, presuppositions, implications
- Handle extraction failures gracefully
- Support for mock LLM in testing

**LLM Prompt Structure:**
```xml
<response>
  <triples>
    <triple>
      <subject>Doug</subject>
      <predicate>PREFERS</predicate>
      <object>tea</object>
      <context>
        <temporal>morning</temporal>
      </context>
    </triple>
  </triples>
  <presuppositions>...</presuppositions>
  <implications>...</implications>
  <ambiguities>...</ambiguities>
</response>
```

**Implementation Status:** ✅ Implemented (`SemanticExpansionService.expand`)

---

### FR-007.7: Expansion Variant Generation

**Requirement:** Generate multiple interpretations when ambiguities exist

**Acceptance Criteria:**
- `ExpansionVariant` dataclass with:
  - `variant_id`: Unique identifier
  - `frame`: SemanticFrame for this interpretation
  - `resolution_choices`: Dict of ambiguity resolutions
  - `context_assumptions`: List of assumed context
  - Cognitive weight factors (7 dimensions)
  - `combined_score` property computing plausibility
- Generate variants for each ambiguity resolution combination
- Score variants against all memory layers

**Implementation Status:** ✅ Implemented (`VariationGenerator` class)

---

### FR-007.8: Cognitive Plausibility Scoring

**Requirement:** Score variants by cognitive factors across memory layers

**Acceptance Criteria:**
- 7 cognitive dimensions with configurable weights:
  | Dimension | Default Weight | Description |
  |-----------|---------------|-------------|
  | Recency | 0.20 | Recent context relevance |
  | Working Memory | 0.15 | Current task context |
  | Episodic Memory | 0.10 | Past episode support |
  | Semantic Memory | 0.20 | Stored fact support |
  | Belief | 0.15 | Belief consistency |
  | Commonsense | 0.10 | ATOMIC/ConceptNet plausibility |
  | Metacognitive | 0.10 | Self-reflection calibration |
- Combined score formula: `0.7 * weighted_sum + 0.3 * base_confidence`
- Scores from 0-1, higher = more plausible

**Implementation Status:** ✅ Implemented (`ExpansionVariant.combined_score`)

---

### FR-007.9: Variation Storage Policy

**Requirement:** Policy for which variations to persist

**Acceptance Criteria:**
- `VariationStoragePolicy` with:
  - `min_confidence_threshold`: Minimum score to store (default 0.3)
  - `max_stored_variations`: Maximum variants to keep (default 3)
  - `min_confidence_gap`: Required difference from primary (default 0.15)
  - `high_confidence_threshold`: Always store if above (default 0.8)
- `should_store(primary, candidate, current_count)` method
- Prevents storing redundant low-confidence variants

**Implementation Status:** ✅ Implemented

---

### FR-007.10: Cross-Layer Association Detection

**Requirement:** Detect associations between new expansions and existing memories

**Acceptance Criteria:**
- `CrossLayerRelation` enum with relation types:
  - DERIVED_FROM, SUMMARIZES, GENERALIZES
  - SUPPORTS, CONTRADICTS, CALIBRATES
  - MENTIONED_IN, ELABORATES, CONTEXTUALIZES
- `CrossLayerEdge` dataclass linking nodes across layers
- Association detection based on:
  - Same synset IDs
  - Semantic similarity above threshold
  - Entity co-occurrence

**Test Approach:**
- New observation "Doug prefers tea" detects:
  - SUPPORTS relation to semantic memory "Doug likes hot beverages"
  - CONTRADICTS relation to observation "Doug prefers coffee"

**Implementation Status:** ✅ Implemented

---

## Non-Functional Requirements

### NFR-007.1: Performance

- Expansion without LLM: < 100ms
- Expansion with LLM: < 3000ms (network-bound)
- Variant generation: < 50ms per variant
- Cross-layer association detection: < 200ms

### NFR-007.2: Test Coverage

- 51 unit tests covering all components
- Pre-loaded memory integration tests
- Two-pass orchestration integration tests
- Evolutionary fitness evaluation framework
- Mock LLM and MockMemoryProvider for reliable testing

---

## Two-Pass Retrieval Architecture (Critical)

The semantic expansion system uses a **two-pass retrieval architecture** that queries memory BEFORE and AFTER expansion to produce context-aware results.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TWO-PASS SEMANTIC ORCHESTRATOR                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: "He prefers tea in the morning"                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PASS 1: PRE-EXPANSION RETRIEVAL                                     │    │
│  │  ────────────────────────────────                                    │    │
│  │  1. Extract keywords/entities from raw input                         │    │
│  │  2. Query working memory (who is "he"?)                              │    │
│  │  3. Query semantic memory (what do we know about tea, morning?)      │    │
│  │  4. Query episodic memory (recent conversation context)              │    │
│  │                                                                      │    │
│  │  OUTPUT: PreExpansionContext                                         │    │
│  │    - entities: {Doug → PERSON}                                       │    │
│  │    - semantic_facts: ["Doug likes coffee"]                           │    │
│  │    - episodic_summaries: ["Dec 25: Doug made coffee"]                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  SEMANTIC EXPANSION (with retrieved context)                         │    │
│  │  ───────────────────────────────────────────                         │    │
│  │  - LLM extracts frame with context from Pass 1                       │    │
│  │  - WSD resolves "tea" → tea.n.01, "morning" → morning.n.01           │    │
│  │  - Generates variants: "He=Doug" vs "He=Unknown"                     │    │
│  │                                                                      │    │
│  │  OUTPUT: list[ExpansionVariant]                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PASS 2: POST-EXPANSION RETRIEVAL                                    │    │
│  │  ────────────────────────────────                                    │    │
│  │  For each variant:                                                   │    │
│  │  1. Query by resolved entities (Doug, not "he")                      │    │
│  │  2. Query by synset IDs (tea.n.01)                                   │    │
│  │  3. Find supporting evidence                                         │    │
│  │  4. Find contradicting evidence                                      │    │
│  │                                                                      │    │
│  │  OUTPUT: VariantEvidence                                             │    │
│  │    - supporting: ["Doug enjoys hot beverages"]                       │    │
│  │    - contradicting: ["Doug likes coffee" + "prefers tea"]            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RE-SCORE VARIANTS                                                   │    │
│  │  ─────────────────────                                               │    │
│  │  - Boost variants with more supporting evidence                      │    │
│  │  - Penalize variants with more contradictions                        │    │
│  │  - Apply cross-layer weight adjustments                              │    │
│  │  - Re-rank by updated combined_score                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  NATURAL LANGUAGE GENERATION                                         │    │
│  │  ────────────────────────────                                        │    │
│  │  - Take winning variant's semantic frame                             │    │
│  │  - Generate natural response with LLM                                │    │
│  │  - Include confidence, conflicts, open questions                     │    │
│  │                                                                      │    │
│  │  OUTPUT: "Doug prefers tea in the morning. Note: this may differ     │    │
│  │           from his general coffee preference."                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  OUTPUT: ProcessingResult                                                    │
│    - statement, variants, response_text                                      │
│    - pre_expansion_context, post_expansion_evidence                          │
│    - detected_conflicts, storage_decisions                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FR-007.11: Two-Pass Orchestrator

**Requirement:** Coordinate two-pass retrieval with memory integration

**Acceptance Criteria:**
- `TwoPassSemanticOrchestrator` class that:
  - Takes memory provider and LLM provider
  - Executes Pass 1 → Expansion → Pass 2 → Re-score → NLG
  - Returns `ProcessingResult` with full audit trail
- Pass 1 retrieves context BEFORE expansion
- Pass 2 retrieves evidence AFTER expansion with resolved entities
- Variants are rescored based on evidence

**Implementation Status:** ✅ Implemented in `integration.py`

---

### FR-007.12: Pre-Expansion Retriever

**Requirement:** Query memory to gather context before expansion

**Acceptance Criteria:**
- `PreExpansionRetriever` class that:
  - Extracts potential entities from raw statement
  - Queries working memory for recent observations
  - Queries semantic memory for relevant facts
  - Queries episodic memory for past episodes
  - Attempts pronoun resolution from context
  - Returns `PreExpansionContext`

**Implementation Status:** ✅ Implemented

---

### FR-007.13: Post-Expansion Retriever

**Requirement:** Query memory for evidence after expansion

**Acceptance Criteria:**
- `PostExpansionRetriever` class that:
  - Uses resolved entities (not pronouns)
  - Uses synset IDs for precise matching
  - Classifies results as supporting/contradicting/related
  - Creates `CrossLayerEdge` for contradictions
  - Returns `VariantEvidence`

**Implementation Status:** ✅ Implemented

---

### FR-007.14: Natural Language Generation

**Requirement:** Convert semantic frames back to natural language

**Acceptance Criteria:**
- `NaturalLanguageGenerator` class that:
  - Takes variant, evidence, conflicts, and context
  - Uses LLM prompt to generate natural response
  - Falls back to template-based generation without LLM
  - Can generate clarification questions for conflicts

**Implementation Status:** ✅ Implemented

---

### FR-007.15: Storage Decisions

**Requirement:** Decide how to store expanded information

**Acceptance Criteria:**
- `StorageDecision` dataclass with:
  - `should_store`: Whether to store
  - `storage_layer`: Which layer (working, semantic, episodic)
  - `memory_type`: Type of memory to create
  - `synset_ids`: Associated synset IDs
  - `cross_layer_links`: Links to related memories
- High confidence + no conflicts → semantic layer
- Has conflicts → working layer as observation
- Low confidence → don't store

**Implementation Status:** ✅ Implemented

---

## Implementation Architecture

### Current Structure

```
src/draagon_ai/semantic/
├── __init__.py          # Module exports
├── types.py             # SemanticFrame, ExpansionVariant, CrossLayerEdge
├── wsd.py               # Word Sense Disambiguation (FR-006)
├── expansion.py         # SemanticExpansionService, VariationGenerator
└── integration.py       # TwoPassSemanticOrchestrator, NLG (NEW)

tests/semantic/
├── __init__.py
├── test_semantic_expansion.py      # 26 tests
└── test_two_pass_integration.py    # 25 tests (NEW)
```

### Key Classes

```python
# Two-pass orchestrator (main entry point)
class TwoPassSemanticOrchestrator:
    def __init__(self, memory: MemoryProvider, llm: LLMProvider):
        ...
    async def process(
        statement: str,
        immediate_context: list[str]
    ) -> ProcessingResult

# Processing result with full audit trail
@dataclass
class ProcessingResult:
    statement: str
    pre_expansion_context: PreExpansionContext
    variants: list[ExpansionVariant]
    variant_evidence: dict[str, VariantEvidence]
    detected_conflicts: list[DetectedConflict]
    storage_decisions: list[StorageDecision]
    response_text: str

# Pre-expansion retriever (Pass 1)
class PreExpansionRetriever:
    async def retrieve(statement: str, context: list[str]) -> PreExpansionContext

# Post-expansion retriever (Pass 2)
class PostExpansionRetriever:
    async def retrieve_evidence(variant: ExpansionVariant) -> VariantEvidence

# Natural language generator
class NaturalLanguageGenerator:
    async def generate(
        variant: ExpansionVariant,
        evidence: VariantEvidence,
        conflicts: list[DetectedConflict]
    ) -> str

# Convenience function
async def process_with_memory(
    statement: str,
    memory: MemoryProvider,
    llm: LLMProvider
) -> ProcessingResult
```

---

## Integration Points

### With Memory Layers
- **Working Memory**: Provides recency context for scoring
- **Episodic Memory**: Past episodes influence interpretation
- **Semantic Memory**: Facts support or contradict variants
- **Metacognitive**: Calibrates confidence estimates

### With Belief System
- Variants checked against existing beliefs
- High-confidence variants become belief candidates
- Contradicting variants flagged for reconciliation

### With Learning System
- Expanded frames provide richer input for learning extraction
- Synset IDs prevent false associations during storage

---

## Future Enhancements

### Phase 2: Enhanced Inference
- [ ] ATOMIC/COMET integration for commonsense implications
- [ ] ConceptNet lookup for relation enrichment
- [ ] Coreference resolution for pronoun disambiguation
- [ ] Temporal reasoning for time expressions

### Phase 3: Graph Storage
- [ ] Store frames as graphs in Qdrant
- [ ] Vector embeddings for frame similarity
- [ ] Graph traversal for association discovery
- [ ] Hierarchical frame composition

### Phase 4: Evolutionary Optimization
- [ ] Fitness function for WSD accuracy
- [ ] Genetic optimization of scoring weights
- [ ] Population-based prompt evolution
- [ ] Cross-validation on held-out data

---

## References

- [FrameNet](https://framenet.icsi.berkeley.edu/) - Frame semantic database
- [ATOMIC](https://homes.cs.washington.edu/~msap/atomic/) - Commonsense knowledge
- [COMET](https://mosaickg.apps.allenai.org/) - Commonsense inference model
- [ConceptNet](https://conceptnet.io/) - Semantic network
- [Presupposition (Stanford Encyclopedia)](https://plato.stanford.edu/entries/presupposition/) - Theory

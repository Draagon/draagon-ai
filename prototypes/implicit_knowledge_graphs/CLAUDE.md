# Implicit Knowledge Graphs Prototype - Claude Context

**Status:** Experimental - Active Development
**Last Updated:** 2025-12-31
**Version:** 0.1.0

---

## Overview

This prototype explores **pre-storage semantic decomposition** - breaking down text into implicit knowledge graphs BEFORE storage, enabling optimized context retrieval for LLMs.

**Core Hypothesis:** By decomposing text into weighted semantic graphs with proper disambiguation, we can provide LLMs with more efficient context than raw text or traditional RAG approaches.

**Key Insight:** Word Sense Disambiguation (WSD) is FOUNDATIONAL, not optional. Without early disambiguation, you cannot correctly weight branches or link related knowledge.

---

## Critical Design Principles

### 1. WSD-First Architecture

**NEVER** store or process text without disambiguation:

```
WRONG:
  Store: "Doug went to the bank"
  Later: Wonder if bank = financial or riverbank

RIGHT:
  Step 1: Disambiguate "bank" → bank.n.01 (financial institution)
  Step 2: Now you can correctly link, weight, and retrieve
```

### 2. Named Entity vs Concept Distinction

Every semantic unit must be classified:

| Type | Description | Example | Storage Strategy |
|------|-------------|---------|------------------|
| **INSTANCE** | Unique real-world thing | "Doug", "Apple Inc." | Local UUID + optional Wikidata QID |
| **CLASS** | Category of things | "person", "company" | WordNet synset ID |
| **NAMED_CONCEPT** | Proper-named category | "Christmas", "Agile" | BabelNet synset + type flag |
| **ROLE** | Relational concept | "CEO of Apple" | Relation + anchor entity |
| **ANAPHORA** | Reference needing resolution | "he", "it" | Pointer to resolved entity |

### 3. Multi-Type Decomposition

Every input is decomposed into multiple knowledge types:

1. **Semantic Roles** - Who did what to whom
2. **Presuppositions** - What must be true for this to make sense
3. **Commonsense Inferences** - ATOMIC-style if-then relations
4. **Implications** - Logical and pragmatic consequences
5. **Temporal/Aspectual** - When and how events unfold
6. **Modality** - Certainty, obligation, ability

### 4. Weighted Branches for Uncertainty

When interpretation is uncertain, store ALL viable branches with weights:

```python
# Example: "He prefers tea in the morning"
branches = [
    Branch(
        interpretation="Doug prefers tea in morning",
        resolution={"He": "Doug"},
        confidence=0.75,
    ),
    Branch(
        interpretation="Unknown male prefers tea",
        resolution={"He": "unknown"},
        confidence=0.55,
    ),
]
# Store BOTH, retrieve the most relevant
```

### 5. Evolutionary Optimization Throughout

Every component is evolvable:
- Prompts for extraction
- Weighting schemes
- Storage policies
- Retrieval strategies
- Scoring rubrics

**From the beginning**, build with evolution in mind.

### 6. Content-Aware Processing (NEW)

**WSD is for natural language, not all semantic understanding.**

Different content types have different kinds of meaning:

| Content Type | Process With | Extracts |
|--------------|--------------|----------|
| **PROSE** | Full WSD → Decomposition | Word senses, presuppositions, inferences |
| **CODE** | Extract NL → WSD on that | Types, contracts + NL disambiguation |
| **DATA** | Schema extraction | Column types, relationships, constraints |
| **CONFIG** | Pattern extraction | Key hierarchies, common patterns |

**NEVER** apply WSD to code syntax:
```
WRONG:
  bank.deposit(100)  → WSD on "bank" as financial institution
  "bank_account": "12345"  → WSD on "bank_account"

RIGHT:
  Detect content type first
  For CODE: Extract docstrings/comments → WSD only on those
  Code syntax carries TYPE meaning, not WORD SENSE meaning
```

See: [DD-001-CONTENT_TYPE_AWARE_PROCESSING.md](docs/design-decisions/DD-001-CONTENT_TYPE_AWARE_PROCESSING.md)

---

## File Structure

```
src/
├── __init__.py              # Package exports
├── identifiers.py           # UniversalSemanticIdentifier, EntityType enum
├── wsd.py                   # Word Sense Disambiguation (Lesk + LLM)
├── entity_classifier.py     # Entity type classification
├── synset_learning.py       # Runtime synset learning for unknown terms
├── evolving_synsets.py      # Extensible synset database
├── text_chunking.py         # Large text handling and context extraction
├── content_analyzer.py      # LLM-driven content type classification
├── content_aware_wsd.py     # Content-aware WSD integration
├── decomposition.py         # Multi-type implicit knowledge extraction
├── branches.py              # Weighted branch generation and scoring
├── storage.py               # Graph storage with synset-based linking
├── retrieval.py             # Optimized context retrieval
├── evolution.py             # Evolutionary optimization framework
└── evaluation.py            # Opus 4.5 multi-dimensional scoring

tests/
├── conftest.py              # Fixtures, path setup
├── test_identifiers.py      # Identifier system tests
├── test_wsd.py              # WSD accuracy tests
├── test_entity_classifier.py
├── test_synset_learning.py
├── test_text_chunking.py
├── test_content_analyzer.py # Content type classification tests
├── test_content_aware_wsd.py # Content-aware WSD tests
├── test_decomposition.py    # Decomposition completeness tests
├── test_branches.py         # Branch generation tests
├── test_evolution.py        # Evolution framework tests
├── test_evaluation.py       # Scoring calibration tests
└── test_end_to_end.py       # Full pipeline integration tests

docs/
├── research/
│   ├── DECOMPOSITION_THEORY.md       # Taxonomy of decomposition types
│   ├── WSD_AND_ENTITY_LINKING.md     # WSD research summary
│   ├── CONTINUED_RESEARCH.md         # Future research directions
│   └── CONTENT_TYPE_HANDLING.md      # Content type semantic extraction
├── requirements/
│   ├── PHASE_0_IDENTIFIERS.md        # Phase 0 requirements
│   ├── PHASE_1_DECOMPOSITION.md      # Phase 1 requirements
│   └── ...
├── specs/
│   └── EVOLUTIONARY_TESTING.md       # Evolution framework spec
├── design-decisions/
│   └── DD-001-CONTENT_TYPE_AWARE_PROCESSING.md  # Content type design
└── findings/
    └── (experiment results go here)
```

---

## Key Data Structures

### UniversalSemanticIdentifier

```python
@dataclass
class UniversalSemanticIdentifier:
    """Universal identifier for any semantic unit."""

    local_id: str                    # UUID for this system
    entity_type: EntityType          # INSTANCE | CLASS | NAMED_CONCEPT | ROLE | ANAPHORA

    # External identifiers
    wikidata_qid: str | None         # Q312 for Apple Inc.
    wordnet_synset: str | None       # apple.n.01
    babelnet_synset: str | None      # bn:00002031n

    # Disambiguation metadata
    sense_rank: int                  # Which sense (1 = most common)
    domain: str | None               # ECONOMY, GEOGRAPHY, etc.
    confidence: float                # Disambiguation confidence

    # For named entities
    canonical_name: str | None
    aliases: list[str]

    # For concepts
    hypernym_chain: list[str]
```

### DecomposedKnowledge

```python
@dataclass
class DecomposedKnowledge:
    """Fully decomposed implicit knowledge from input text."""

    original_text: str

    # Identified entities with universal IDs
    entities: dict[str, UniversalSemanticIdentifier]

    # Semantic role structure
    semantic_roles: list[SemanticRole]

    # Implicit knowledge
    presuppositions: list[Presupposition]
    commonsense_inferences: list[CommonsenseInference]
    implications: list[Implication]

    # Temporal/modal structure
    temporal_structure: TemporalStructure
    modality: Modality

    # Uncertainty
    ambiguities: list[Ambiguity]
    weighted_branches: list[WeightedBranch]
```

---

## Evolutionary Framework

### EvolvableConfig

Every tunable aspect is captured in a configuration:

```python
@dataclass
class EvolvableConfig:
    """Configuration that can evolve."""

    config_id: str
    generation: int
    parent_ids: list[str]

    # Evolvable components
    wsd_prompt: str
    decomposition_prompt: str
    weighting_scheme: dict[str, float]
    storage_policy: StoragePolicy
    retrieval_strategy: RetrievalStrategy
    scoring_rubric: ScoringRubric

    # Fitness tracking
    fitness_scores: dict[str, float]
```

### Fitness Evaluation

Uses Claude Opus 4.5 for multi-dimensional scoring:

```python
async def evaluate_output(
    query: str,
    context_provided: str,
    output: str,
    expected: TestExpectation,
) -> OutputScore:
    """Evaluate output quality with Opus 4.5."""

    prompt = f"""
    Evaluate this LLM output on multiple dimensions.

    QUERY: {query}
    CONTEXT PROVIDED: {context_provided}
    OUTPUT: {output}

    Score each dimension 0.0-1.0:
    1. FACTUAL_ACCURACY: Are all facts correct?
    2. COMPLETENESS: Does it fully answer the query?
    3. RELEVANCE: Is all content relevant?
    4. COHERENCE: Is it well-structured and clear?

    Also note:
    - CONTEXT_TOKENS: How many tokens of context were used?
    - IMPROVEMENT: Compared to a baseline RAG approach, is this better/worse/same?

    Respond in XML:
    <evaluation>
        <factual_accuracy>0.0-1.0</factual_accuracy>
        <completeness>0.0-1.0</completeness>
        <relevance>0.0-1.0</relevance>
        <coherence>0.0-1.0</coherence>
        <rationale>Why these scores</rationale>
    </evaluation>
    """

    # Call Opus 4.5 for evaluation
    response = await opus_client.chat(prompt)
    return parse_evaluation(response)
```

---

## Testing Strategy

### Test Categories

1. **Unit Tests** - Individual components work correctly
2. **Integration Tests** - Components work together
3. **Accuracy Tests** - WSD, decomposition are correct
4. **Evolution Tests** - Configs evolve and improve
5. **Baseline Comparison** - vs vanilla RAG, vs GraphRAG

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Just unit tests (fast)
python -m pytest tests/ -v -m "not slow"

# With real LLM evaluation
ANTHROPIC_API_KEY=xxx python -m pytest tests/ -v --run-eval

# Evolution tests (slow)
python -m pytest tests/test_evolution.py -v --run-evolution
```

---

## Integration Points with draagon-ai

This prototype is STANDALONE but designed for eventual integration:

1. **Memory Layer Integration** - Decomposed knowledge maps to 4-layer memory
2. **Belief System** - Presuppositions become belief candidates
3. **Orchestration** - Retrieval integrates with AgentLoop
4. **Evolution** - Uses same Promptbreeder-style approach as core

---

## Development Guidelines

### Adding a New Decomposition Type

1. Define the type in `src/decomposition.py`
2. Add extraction logic (LLM prompt + parser)
3. Add to `DecomposedKnowledge` dataclass
4. Add storage mapping in `src/storage.py`
5. Add retrieval handling in `src/retrieval.py`
6. Add tests in `tests/test_decomposition.py`
7. Make the extraction prompt evolvable

### Adding an Evolvable Component

1. Define the configurable parameter
2. Add to `EvolvableConfig`
3. Implement mutation strategy
4. Implement crossover strategy
5. Define fitness metric
6. Add to evolution loop

---

## Current Phase: Phase 1 - Decomposition (Integrated with Phase 0)

### Phase 0 + Phase 1 Integration (COMPLETE)

The `IntegratedPipeline` provides full Phase 0 → Phase 1 orchestration:

```python
from decomposition import IntegratedPipeline

pipeline = IntegratedPipeline(llm=my_llm)
result = await pipeline.process("Doug forgot the meeting again")

# Phase 0 results
print(result.wsd_results)   # {"forgot": DisambiguationResult(...)}
print(result.entities)      # {"Doug": UniversalSemanticIdentifier(...)}

# Phase 1 results
print(result.presuppositions)  # [Presupposition(content="Doug forgot before", ...)]
print(result.inferences)       # [CommonsenseInference(...)]
```

**Phase 0 Implemented:**
- [x] `UniversalSemanticIdentifier` implementation
- [x] `EntityType` enum and classification
- [x] WordNet synset lookup
- [x] LLM-based disambiguation fallback
- [x] Hybrid WSD pipeline (Lesk + LLM)
- [x] Entity classifier (heuristic + LLM)
- [x] Synset learning service
- [x] Text chunking for large documents
- [x] Content type analysis (REQ-0.8)
- [x] Content-aware WSD integration (REQ-0.9)

**Phase 1 Implemented:**
- [x] Presupposition extraction
- [x] Semantic role labeling
- [x] Commonsense inference generation
- [x] Temporal/aspectual analysis
- [x] Modality detection
- [x] Negation scope detection
- [x] Integrated Phase 0 → Phase 1 pipeline
- [x] Document chunking for long texts
- [x] Content-type aware processing

**Remaining (Backlog):**
- [ ] Basic Wikidata linking (optional)
- [ ] Evolution framework skeleton

---

## Related Documentation

- `README.md` - Project overview and architecture
- `docs/research/DECOMPOSITION_THEORY.md` - Full taxonomy of decomposition types
- `docs/research/CONTINUED_RESEARCH.md` - Future research directions
- `docs/research/CONTENT_TYPE_HANDLING.md` - Content type semantic extraction
- `docs/requirements/PHASE_0_IDENTIFIERS.md` - Current phase requirements
- `docs/requirements/PHASE_1_DECOMPOSITION.md` - Decomposition pipeline requirements
- `docs/specs/EVOLUTIONARY_TESTING.md` - Evolution framework specification
- `docs/design-decisions/DD-001-CONTENT_TYPE_AWARE_PROCESSING.md` - Content type design decision

---

## Debugging Tips

### WSD Not Working?

1. Check NLTK WordNet is downloaded: `nltk.download('wordnet')`
2. Verify lemmatization is working
3. Check if word has synsets: `wn.synsets('bank')`
4. Fall back to LLM disambiguation

### Branch Weights Seem Wrong?

1. Check cognitive factor weights sum correctly
2. Verify memory layer contributions
3. Enable debug logging to see scoring breakdown

### Evolution Not Converging?

1. Check population diversity (not all same config)
2. Verify fitness function is discriminative
3. Try increasing mutation rate
4. Check for overfitting to test set

---

**End of CLAUDE.md**

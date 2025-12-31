# Implicit Knowledge Graphs Prototype

**Status:** Experimental
**Version:** 0.1.0
**Started:** 2025-12-31

---

## Hypothesis

**Core Theory:** LLMs provide the most value when given optimally curated context. Current approaches (RAG, manual curation) are limited by:
1. Raw text contains redundant/implicit information the LLM must rediscover
2. Chunk-based retrieval loses semantic relationships
3. Context windows have hard limits that force information loss

**Our Approach:** What if we:
1. **Pre-decompose** text into implicit knowledge graphs at storage time
2. **Disambiguate** word senses early to enable correct linking
3. **Store weighted branches** for uncertain interpretations
4. **Retrieve optimized semantic triples** instead of text chunks
5. **Use evolutionary optimization** to tune every component

**Expected Outcome:** Better LLM outputs with the same or fewer context tokens, especially at scale where traditional RAG breaks down.

---

## Key Differentiators from `semantic_expansion` Prototype

| Aspect | `semantic_expansion` | `implicit_knowledge_graphs` |
|--------|---------------------|----------------------------|
| **Focus** | Two-pass retrieval | Pre-storage decomposition |
| **WSD** | Optional enhancement | Foundational requirement |
| **Storage** | Raw + expanded | Fully decomposed graphs |
| **Retrieval** | Text-based + entities | Triple-based with synsets |
| **Optimization** | Fixed algorithms | Evolutionary tuning |
| **Evaluation** | Correctness tests | Multi-dimensional Opus 4.5 scoring |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMPLICIT KNOWLEDGE GRAPH PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Text                                                                  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 0: Universal Semantic Identification                          │    │
│  │  • Named Entity Recognition + Classification                        │    │
│  │  • Word Sense Disambiguation (WordNet/BabelNet)                     │    │
│  │  • Entity Linking (Wikidata)                                        │    │
│  │  → Every semantic unit gets a unique identifier                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: Multi-Type Decomposition                                   │    │
│  │  • Semantic Role Extraction (Agent, Patient, etc.)                  │    │
│  │  • Presupposition Extraction (triggers → assumptions)               │    │
│  │  • Commonsense Inference (ATOMIC-style relations)                   │    │
│  │  • Implication Extraction (logical + pragmatic)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: Weighted Branch Generation                                 │    │
│  │  • Multiple interpretations for ambiguous inputs                    │    │
│  │  • Each branch scored by cognitive factors                          │    │
│  │  • Branches stored with confidence weights                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3: Graph Storage                                              │    │
│  │  • Semantic triples with synset-based identifiers                   │    │
│  │  • Cross-layer memory associations                                  │    │
│  │  • Weighted edges for uncertain relationships                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 4: Optimized Retrieval                                        │    │
│  │  • Query decomposition (same pipeline as storage)                   │    │
│  │  • Triple-based retrieval with synset filtering                     │    │
│  │  • Context packing optimization                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  Context optimized for LLM consumption                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Evolutionary Optimization Framework

Every component is tunable via evolutionary techniques:

### Evolvable Components

| Component | What Evolves | Fitness Metric |
|-----------|--------------|----------------|
| **WSD Prompts** | LLM prompt for disambiguation | Disambiguation accuracy |
| **Decomposition Prompts** | How we extract implicit knowledge | Completeness + precision |
| **Weighting Schemes** | Cognitive factor weights | Branch prediction accuracy |
| **Storage Policies** | What/how much to store | Retrieval efficiency |
| **Retrieval Strategies** | How we query and pack context | Output quality per token |
| **Scoring Rubrics** | How we evaluate outputs | Correlation with human judgment |

### Evolution Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLUTIONARY LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. POPULATION INITIALIZATION                                               │
│     • Create N candidate configurations                                     │
│     • Each config has different prompt/weight/policy variants               │
│                                                                              │
│  2. EVALUATION (Opus 4.5 Multi-Dimensional Scoring)                         │
│     • Run test cases through each configuration                             │
│     • Score outputs on multiple dimensions:                                 │
│       - Factual accuracy                                                    │
│       - Completeness                                                        │
│       - Relevance                                                           │
│       - Coherence                                                           │
│       - Context efficiency (quality / tokens used)                          │
│                                                                              │
│  3. SELECTION                                                                │
│     • Keep top performers (elites)                                          │
│     • Tournament selection for breeding pool                                │
│                                                                              │
│  4. REPRODUCTION                                                             │
│     • Crossover: Combine successful configurations                          │
│     • Mutation: Random tweaks to prompts, weights, policies                 │
│                                                                              │
│  5. REPEAT until convergence or generation limit                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Framework

### Multi-Dimensional Scoring (Opus 4.5)

Each test case is evaluated by Claude Opus 4.5 on multiple dimensions:

```python
@dataclass
class OutputScore:
    """Multi-dimensional quality score."""

    # Core quality dimensions
    factual_accuracy: float  # 0-1: Are facts correct?
    completeness: float      # 0-1: Did it answer fully?
    relevance: float         # 0-1: Is content relevant?
    coherence: float         # 0-1: Is it well-structured?

    # Efficiency dimensions
    context_tokens_used: int
    quality_per_token: float  # Combined quality / tokens

    # Comparison dimensions (vs baseline)
    improvement_over_baseline: float

    # Reasoning trace
    scoring_rationale: str
```

### Test Case Structure

```python
@dataclass
class TestCase:
    """A test case for evaluating the pipeline."""

    # Input
    query: str  # The question being asked
    knowledge_base: list[str]  # Facts to store

    # Expected output characteristics
    expected_topics: list[str]  # Topics that should be covered
    expected_entities: list[str]  # Entities that should be mentioned
    gold_answer: str | None  # Optional gold standard

    # Difficulty indicators
    requires_multi_hop: bool
    has_ambiguity: bool
    has_temporal_reasoning: bool
    entity_count: int
```

---

## Phased Implementation

### Phase 0: Universal Semantic Identification (Current Focus)
**Goal:** Build the disambiguation foundation

- [ ] `UniversalSemanticIdentifier` dataclass
- [ ] WordNet integration for synset lookup
- [ ] LLM-based disambiguation when ambiguous
- [ ] Entity type classification (INSTANCE vs CLASS)
- [ ] Basic Wikidata linking (optional)

### Phase 1: Multi-Type Decomposition
**Goal:** Extract all implicit knowledge types

- [ ] Semantic role extraction
- [ ] Presupposition extraction (trigger-based)
- [ ] ATOMIC-style commonsense inference
- [ ] Implication extraction

### Phase 2: Weighted Branch Generation
**Goal:** Handle ambiguity with weighted alternatives

- [ ] Ambiguity detection
- [ ] Branch generation
- [ ] Cognitive scoring
- [ ] Storage policy implementation

### Phase 3: Graph Storage
**Goal:** Store decomposed knowledge efficiently

- [ ] Qdrant integration with synset filtering
- [ ] Triple embedding strategies
- [ ] Cross-layer linking

### Phase 4: Optimized Retrieval
**Goal:** Retrieve optimal context for queries

- [ ] Query decomposition
- [ ] Triple-based retrieval
- [ ] Context packing optimization

### Phase 5+: Scale Testing & Comparison
**Goal:** Prove the theory at scale

- [ ] Large knowledge base testing (10k+ facts)
- [ ] Context window constraint testing
- [ ] Comparison vs vanilla RAG
- [ ] Comparison vs GraphRAG

---

## Running the Prototype

```bash
# Navigate to prototype
cd prototypes/implicit_knowledge_graphs

# Install dependencies (from project root)
pip install nltk  # For WordNet

# Download WordNet data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Run tests
python -m pytest tests/ -v

# Run with real LLM evaluation (requires API keys)
ANTHROPIC_API_KEY=your_key python -m pytest tests/ -v --run-eval
```

---

## Key Files

```
src/
├── __init__.py              # Package exports
├── identifiers.py           # UniversalSemanticIdentifier, entity types
├── wsd.py                   # Word Sense Disambiguation
├── decomposition.py         # Multi-type knowledge extraction
├── branches.py              # Weighted branch generation
├── storage.py               # Graph storage (Qdrant)
├── retrieval.py             # Optimized retrieval
├── evolution.py             # Evolutionary optimization framework
└── evaluation.py            # Opus 4.5 multi-dimensional scoring

tests/
├── conftest.py              # Test configuration, fixtures
├── test_identifiers.py      # Identifier system tests
├── test_wsd.py              # WSD tests
├── test_decomposition.py    # Decomposition tests
├── test_evolution.py        # Evolution framework tests
├── test_evaluation.py       # Scoring system tests
└── test_end_to_end.py       # Full pipeline tests

docs/
├── research/                # Background research, theory
├── requirements/            # Phase requirements
├── specs/                   # Technical specifications
└── findings/                # Experiment results
```

---

## Dependencies

**Required:**
- Python 3.11+
- `nltk` (WordNet access)
- `anthropic` (Opus 4.5 evaluation)

**Optional:**
- `qdrant-client` (vector storage)
- `sentence-transformers` (embeddings)
- `spacy` (advanced NLP)

---

## Related Work

- `prototypes/semantic_expansion/` - Earlier prototype focusing on two-pass retrieval
- `docs/specs/SEMANTIC_EXPANSION_CONCEPT.md` - Original concept document
- `docs/specs/SEMANTIC_EXPANSION_ARCHITECTURE.md` - Architecture design

---

## Research Foundation

See `docs/research/` for:
- Decomposition theory taxonomy
- WSD and entity linking research
- ATOMIC/COMET commonsense knowledge
- Frame semantics and FrameNet
- GraphRAG comparison studies
- Evolutionary optimization approaches

---

**Status:** Experimental - Active Development

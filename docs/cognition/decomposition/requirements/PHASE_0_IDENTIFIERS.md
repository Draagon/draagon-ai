# Phase 0: Universal Semantic Identification

**Version:** 1.0.0
**Status:** Requirements
**Priority:** P0 - Foundation

---

## Overview

Phase 0 establishes the **foundational identifier system** that all other phases depend on. Without proper disambiguation and identification, branches cannot be weighted and knowledge cannot be linked correctly.

**Deliverable:** A working identifier system that can take any word/phrase in context and produce a unique, typed semantic identifier.

---

## Requirements

### REQ-0.1: Entity Type Classification

**Description:** Classify every semantic unit into one of the defined entity types.

**Entity Types:**
| Type | Code | Description | Example |
|------|------|-------------|---------|
| Instance | `INSTANCE` | Unique real-world thing | "Doug", "Apple Inc." |
| Class | `CLASS` | Category/type | "person", "company" |
| Named Concept | `NAMED_CONCEPT` | Proper-named category | "Christmas", "Agile" |
| Role | `ROLE` | Relational concept | "CEO of Apple" |
| Anaphora | `ANAPHORA` | Reference needing resolution | "he", "it" |
| Generic | `GENERIC` | Generic reference | "someone", "people" |

**Acceptance Criteria:**
- [ ] `EntityType` enum defined with all types
- [ ] Classification function that takes (text, context) → EntityType
- [ ] Confidence score for classification
- [ ] Unit tests for each entity type

### REQ-0.2: Universal Semantic Identifier

**Description:** Data structure to uniquely identify any semantic unit.

**Required Fields:**
```python
@dataclass
class UniversalSemanticIdentifier:
    # Always present
    local_id: str                    # UUID
    entity_type: EntityType          # Classification

    # External identifiers (nullable)
    wordnet_synset: str | None       # "bank.n.01"
    wikidata_qid: str | None         # "Q312"
    babelnet_synset: str | None      # "bn:00008364n"

    # Disambiguation metadata
    sense_rank: int                  # 1 = most common
    domain: str | None               # "FINANCE", etc.
    confidence: float                # 0.0-1.0

    # For INSTANCE type
    canonical_name: str | None
    aliases: list[str]

    # For CLASS type
    hypernym_chain: list[str]
```

**Acceptance Criteria:**
- [ ] Dataclass defined with all fields
- [ ] `__hash__` and `__eq__` methods for set/dict usage
- [ ] `matches_sense()` method for comparing identifiers
- [ ] Serialization to/from JSON
- [ ] Unit tests for all methods

### REQ-0.3: WordNet Integration

**Description:** Integrate NLTK WordNet for synset lookup and basic WSD.

**Functions Required:**
```python
def get_synsets(word: str, pos: str | None = None) -> list[SynsetInfo]
def get_synset_info(synset_id: str) -> SynsetInfo
def get_hypernym_chain(synset_id: str) -> list[str]
def lesk_disambiguate(word: str, context: str) -> tuple[str, float]
```

**Acceptance Criteria:**
- [ ] WordNet lookup working for nouns, verbs, adjectives, adverbs
- [ ] Synset info extraction (definition, examples, hypernyms)
- [ ] Lesk algorithm implementation
- [ ] Extended Lesk with related synset glosses
- [ ] Unit tests for all functions
- [ ] Test cases for known ambiguous words (bank, bass, etc.)

### REQ-0.4: LLM-Based Disambiguation

**Description:** Fallback to LLM when algorithmic WSD is uncertain.

**Functions Required:**
```python
async def llm_disambiguate(
    word: str,
    sentence: str,
    candidates: list[str],  # Synset IDs
    llm: LLMProvider
) -> tuple[str, float]  # (synset_id, confidence)
```

**Acceptance Criteria:**
- [ ] LLM prompt for WSD defined (evolvable)
- [ ] XML response parsing
- [ ] Graceful fallback if LLM fails
- [ ] Confidence calibration
- [ ] Integration tests with mock LLM
- [ ] Cost tracking for LLM calls

### REQ-0.5: Hybrid Disambiguation Pipeline

**Description:** Combine algorithmic and LLM approaches efficiently.

**Pipeline:**
```
Input: (word, context)
    │
    ├─ Single synset? → Return immediately (confidence=1.0)
    │
    ├─ Extended Lesk → High confidence? → Return
    │
    ├─ Embedding similarity → Agrees with Lesk? → Return with boost
    │
    └─ LLM fallback → Return LLM result
```

**Acceptance Criteria:**
- [ ] Pipeline implemented with configurable thresholds
- [ ] Thresholds are evolvable parameters
- [ ] Metrics tracked: accuracy, LLM calls, latency
- [ ] Unit tests for each path through pipeline
- [ ] Integration tests for full pipeline

### REQ-0.6: Entity Type Classifier

**Description:** Classify whether a mention is INSTANCE, CLASS, etc.

**Functions Required:**
```python
async def classify_entity_type(
    text: str,
    context: str,
    pos_tag: str | None = None
) -> tuple[EntityType, float]
```

**Classification Logic:**
1. Proper noun + NER tag → likely INSTANCE
2. Common noun + generic context → likely CLASS
3. Proper noun + category context → likely NAMED_CONCEPT
4. Relational phrase → likely ROLE
5. Pronoun → ANAPHORA
6. Generic quantifier → GENERIC

**Acceptance Criteria:**
- [ ] Basic heuristic classifier working
- [ ] LLM fallback for uncertain cases
- [ ] Test cases for each entity type
- [ ] Accuracy > 85% on test set

### REQ-0.7: Basic Wikidata Linking (Optional)

**Description:** Link named entities to Wikidata when confident.

**Functions Required:**
```python
async def link_to_wikidata(
    entity_name: str,
    entity_type: EntityType,
    context: str | None = None
) -> str | None  # Wikidata QID or None
```

**Acceptance Criteria:**
- [ ] Wikidata API integration
- [ ] Search by entity name
- [ ] Disambiguation using context
- [ ] Rate limiting and caching
- [ ] Graceful failure (return None if uncertain)

### REQ-0.8: Content Type Analysis

**Description:** Analyze content type BEFORE applying WSD to ensure appropriate processing.

**Rationale:** WSD is for natural language understanding, not all semantic understanding. Different content types (code, data, config) need different processing strategies. Applying WSD to code syntax or JSON keys produces nonsensical results.

See: [DD-001-CONTENT_TYPE_AWARE_PROCESSING.md](../design-decisions/DD-001-CONTENT_TYPE_AWARE_PROCESSING.md)

**Content Types:**
| Type | Description | WSD Strategy |
|------|-------------|--------------|
| `PROSE` | Natural language documents | Full WSD |
| `CODE` | Source code | Extract NL (docstrings, comments) → WSD on those |
| `DATA` | CSV, JSON data | Schema extraction (no WSD) |
| `CONFIG` | YAML, TOML configuration | Pattern extraction (no WSD) |
| `LOGS` | Log files | Selective NL extraction |
| `MIXED` | Multiple types combined | Split and process per-component |

**Functions Required:**
```python
class ContentType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    DATA = "data"
    CONFIG = "config"
    LOGS = "logs"
    MIXED = "mixed"

@dataclass
class ContentAnalysis:
    content_type: ContentType
    components: list[ContentComponent]
    structural_knowledge: list[StructuralKnowledge]
    detected_language: str
    detected_format: str
    processing_recommendation: ProcessingStrategy

    def get_natural_language_text(self) -> str:
        """Extract all NL portions combined."""

async def analyze_content(
    content: str,
    llm: LLMProvider | None = None,
) -> ContentAnalysis
```

**Acceptance Criteria:**
- [x] `ContentType` enum with all types
- [x] `ContentAnalysis` dataclass with components
- [x] LLM-based content classification (preferred)
- [x] Heuristic fallback when LLM unavailable
- [x] NL extraction from code (docstrings, comments)
- [x] Structural knowledge extraction
- [x] Unit tests for each content type
- [x] Integration with WSD pipeline

### REQ-0.9: Content-Aware WSD Integration

**Description:** Wrap WSD with content-aware preprocessing.

**Functions Required:**
```python
class ContentAwareWSD:
    async def process(self, content: str) -> ContentAwareWSDResult
    async def disambiguate_word(self, word: str, content: str) -> DisambiguationResult | None

@dataclass
class ContentAwareWSDResult:
    content_analysis: ContentAnalysis
    disambiguation_results: dict[str, DisambiguationResult]
    structural_knowledge: list[dict]
    processed_text: str
    skipped_processing: bool
    skip_reason: str
```

**Acceptance Criteria:**
- [x] Routes prose through full WSD
- [x] Extracts NL from code for WSD
- [x] Skips WSD for data/config
- [x] Metrics tracking per content type
- [x] Unit tests for all processing paths

---

## Evolution Framework Requirements

### REQ-0.E1: Evolvable WSD Configuration

**Description:** WSD parameters must be evolvable.

**Evolvable Parameters:**
```python
@dataclass
class WSDConfig:
    # Lesk parameters
    lesk_context_window: int = 5  # Words around target
    lesk_extended_depth: int = 1  # Hypernym/hyponym depth

    # Threshold parameters
    lesk_high_confidence: float = 0.8
    embedding_agreement_boost: float = 0.1
    llm_fallback_threshold: float = 0.5

    # LLM parameters
    wsd_prompt_template: str = "..."
    wsd_temperature: float = 0.1
```

**Acceptance Criteria:**
- [ ] All parameters externalized to config
- [ ] Config can be mutated (for evolution)
- [ ] Fitness tracking per config
- [ ] Unit tests with different configs

### REQ-0.E2: WSD Accuracy Metrics

**Description:** Track WSD accuracy for evolution fitness.

**Metrics:**
- Accuracy on labeled test set
- LLM call rate (lower is cheaper)
- Latency (lower is better)
- Confidence calibration (predicted confidence matches actual accuracy)

**Acceptance Criteria:**
- [ ] Test set with ground truth synsets
- [ ] Accuracy calculation function
- [ ] Confidence calibration metrics
- [ ] Cost/performance tradeoff calculation

---

## Test Requirements

### Test Categories

1. **Unit Tests** - Individual functions
2. **Integration Tests** - Full pipeline
3. **Accuracy Tests** - Against labeled data
4. **Evolution Tests** - Config mutation and fitness

### Test Data Requirements

**WSD Test Cases (minimum 50):**
- 10 unambiguous words
- 20 ambiguous words with clear context
- 10 ambiguous words with unclear context
- 10 named entity cases

**Entity Type Test Cases (minimum 30):**
- 5 per entity type

### Acceptance Criteria

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] WSD accuracy > 75% on test set
- [ ] Entity type accuracy > 85% on test set
- [ ] No regressions when config changes

---

## Implementation Plan

### Week 1: Core Data Structures
- [ ] `EntityType` enum
- [ ] `UniversalSemanticIdentifier` dataclass
- [ ] Basic unit tests

### Week 2: WordNet Integration
- [ ] Synset lookup functions
- [ ] Lesk algorithm
- [ ] Extended Lesk
- [ ] WSD tests

### Week 3: LLM Integration
- [ ] LLM disambiguation function
- [ ] Hybrid pipeline
- [ ] Config externalization
- [ ] Integration tests

### Week 4: Entity Classification
- [ ] Entity type classifier
- [ ] Wikidata linking (optional)
- [ ] Full pipeline tests
- [ ] Accuracy benchmarking

---

## Success Criteria

Phase 0 is complete when:

1. **Functional:**
   - Can disambiguate any word in context
   - Can classify entity types
   - Can produce universal identifiers

2. **Quality:**
   - WSD accuracy > 75%
   - Entity type accuracy > 85%
   - All tests passing

3. **Evolution-Ready:**
   - All parameters in evolvable config
   - Fitness metrics defined and tracked
   - Mutation strategies documented

---

## Dependencies

**Python Packages:**
- `nltk` (WordNet)
- `anthropic` or equivalent (LLM)
- `pytest` (testing)

**Data:**
- WordNet corpus (via NLTK)
- Wikidata API access (optional)

**From draagon-ai:**
- `LLMProvider` protocol
- XML parsing utilities

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WordNet coverage gaps | Some words not found | Fall back to LLM |
| LLM cost | High cost if called frequently | Threshold tuning, caching |
| Wikidata rate limits | Linking fails | Caching, optional feature |
| Low accuracy | Poor downstream performance | Iterative improvement via evolution |

---

## Backlog (LOW Priority)

The following items were identified during the Phase 0 code review but are NOT blocking for Phase 1. They should be addressed as time permits or when the relevant feature becomes critical.

### BACKLOG-0.1: Implement QdrantSynsetStore Fully

**Description:** The `QdrantSynsetStore` in `synset_learning.py` is currently a stub that uses in-memory dict storage. For production use with large learned synset databases, implement full Qdrant vector storage.

**Current State:** Stub class with in-memory dict
**Target State:** Full Qdrant integration with:
- Vector embeddings for synset definitions
- Efficient semantic search for unknown terms
- Persistence across restarts

**Priority:** Low (in-memory works for prototype)
**Estimated Effort:** 2-3 days

### BACKLOG-0.2: Wikidata Entity Linking

**Description:** REQ-0.7 (Wikidata linking) was marked optional and not implemented. For production knowledge graphs with named entities, linking to Wikidata QIDs enables cross-system interoperability.

**Current State:** `UniversalSemanticIdentifier.wikidata_qid` is always None
**Target State:**
- Wikidata API integration
- Named entity search by name
- Context-based disambiguation
- Rate limiting and caching

**Priority:** Low (not needed for Phase 1 decomposition)
**Estimated Effort:** 3-4 days

### BACKLOG-0.3: Evolution Framework Skeleton

**Description:** The evolution framework for WSD config optimization is mentioned in CLAUDE.md but not implemented. This enables Promptbreeder-style optimization of WSD prompts and thresholds.

**Current State:** `WSDConfig` is evolvable but no mutation/crossover/selection implemented
**Target State:**
- Config mutation strategies
- Fitness function based on accuracy metrics
- Train/holdout evaluation
- Population management

**Priority:** Low (Phase 1 can work with static config)
**Estimated Effort:** 5-7 days

### BACKLOG-0.4: BabelNet Integration

**Description:** `UniversalSemanticIdentifier.babelnet_synset` field exists but is never populated. BabelNet provides multilingual synsets and better coverage for named concepts.

**Current State:** Field exists, always None
**Target State:**
- BabelNet API integration
- Fallback when WordNet has no synset
- Multilingual support for named concepts

**Priority:** Very Low (WordNet sufficient for English)
**Estimated Effort:** 4-5 days

### BACKLOG-0.5: Mutation Testing for Test Suite

**Description:** While test coverage is good (552+ tests), mutation testing would verify that tests actually catch bugs, not just execute code.

**Current State:** No mutation testing
**Target State:**
- Add `mutmut` or similar mutation testing tool
- Run mutation tests on critical paths
- Achieve >80% mutation kill rate

**Priority:** Low (current tests are validated against ground truth)
**Estimated Effort:** 1-2 days

---

**End of Phase 0 Requirements**

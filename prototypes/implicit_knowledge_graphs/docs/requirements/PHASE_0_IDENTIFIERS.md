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

**End of Phase 0 Requirements**

# FR-006: Word Sense Disambiguation System

**Feature:** Word sense disambiguation using WordNet synsets and LLM fallback
**Status:** Implemented (Prototype)
**Priority:** P1 (Core Semantic Infrastructure)
**Complexity:** Medium
**Phase:** Semantic Expansion Phase 1

---

## Overview

Implement a Word Sense Disambiguation (WSD) system that resolves ambiguous words to canonical sense identifiers (WordNet synsets). This prevents false semantic associations when storing and retrieving memories—for example, distinguishing "bank" (financial institution) from "bank" (river bank).

### Research Foundation

- **WordNet (Princeton)**: Canonical lexical database with synset identifiers
- **Lesk Algorithm (1986)**: Dictionary-based WSD using gloss overlap
- **BabelNet**: Multilingual semantic network for cross-lingual support
- **Wikidata QIDs**: Cross-reference identifiers for knowledge graphs

### Why This Matters

Without WSD, the system cannot reliably:
- Prevent false associations ("bank" finance vs riverbank)
- Build accurate semantic graphs across memory layers
- Detect when the same word sense appears in different contexts
- Support future multilingual expansion

---

## Functional Requirements

### FR-006.1: Synset ID Format

**Requirement:** Use standardized synset ID format (word.pos.sense_num)

**Acceptance Criteria:**
- Format: `{lemma}.{pos}.{sense_num:02d}` (e.g., "bank.n.01")
- POS tags: n (noun), v (verb), a (adjective), r (adverb)
- Sense numbers are zero-padded to 2 digits
- Convenience function `get_synset_id(word, pos, sense_num)` available

**Test Approach:**
- Verify format for common words (bank, tea, prefer)
- Test roundtrip: id → parse → id

**Implementation Status:** ✅ Implemented in `src/draagon_ai/semantic/wsd.py`

---

### FR-006.2: Lesk Disambiguation Algorithm

**Requirement:** Implement dictionary-based WSD using Lesk algorithm

**Acceptance Criteria:**
- Compare context words with synset glosses (definitions + examples)
- Score by word overlap between context and gloss
- Return best-scoring synset with confidence
- Include alternative synset IDs in result
- Handle stopword removal for better matching

**Test Approach:**
- "I deposited money at the bank" → bank.n.01 (financial)
- "We sat on the river bank" → bank.n.02 (slope/riverbank)
- Measure disambiguation accuracy on test sentences

**Implementation Status:** ✅ Implemented (`LeskDisambiguator` class)

---

### FR-006.3: Mock Synset Database

**Requirement:** Provide mock synsets when NLTK WordNet not available

**Acceptance Criteria:**
- `MockSynset` dataclass with name, pos, lemmas, definition, examples
- `COMMON_SYNSETS` dictionary covering common ambiguous words
- Graceful fallback when NLTK corpus not downloaded
- Support same interface as NLTK synsets

**Test Approach:**
- All tests pass without NLTK installed
- Mock synsets cover: bank, tea, coffee, morning, prefer, like, go, person

**Implementation Status:** ✅ Implemented

---

### FR-006.4: Sentence-Level Disambiguation

**Requirement:** Disambiguate all content words in a sentence

**Acceptance Criteria:**
- Process entire sentence via `disambiguate_sentence()`
- Skip function words (determiners, prepositions, conjunctions)
- Return dict mapping "position:word" to WordSense
- Support optional spaCy pipeline for lemmatization
- Simple tokenization fallback when spaCy not available

**Test Approach:**
- Sentence "I went to the bank" disambiguates nouns/verbs
- Function words skipped
- Multiple senses in one sentence handled

**Implementation Status:** ✅ Implemented (`WordSenseDisambiguator.disambiguate_sentence`)

---

### FR-006.5: LLM-Based Disambiguation (Fallback)

**Requirement:** Use LLM for complex disambiguation cases

**Acceptance Criteria:**
- LLM prompt with context sentence and candidate senses
- XML output format: `<synset_id>`, `<confidence>`, `<reasoning>`
- Only used when synset count > 2 and LLM provider available
- Falls back to Lesk if LLM confidence < 0.7

**Test Approach:**
- Mock LLM returns expected XML format
- Low-confidence LLM response triggers Lesk fallback
- High-confidence LLM response used directly

**Implementation Status:** ✅ Implemented (`_llm_disambiguate` method)

---

### FR-006.6: WordSense Data Structure

**Requirement:** Rich data structure for disambiguated word senses

**Acceptance Criteria:**
- `WordSense` dataclass fields:
  - `surface_form`: Original text (e.g., "banks")
  - `lemma`: Root form (e.g., "bank")
  - `pos`: Part of speech
  - `synset_id`: Canonical ID (e.g., "bank.n.01")
  - `wikidata_id`: Optional Wikidata QID
  - `babelnet_id`: Optional BabelNet ID
  - `definition`: Sense definition
  - `confidence`: Disambiguation confidence (0-1)
  - `disambiguation_method`: "lesk", "llm", "unambiguous"
  - `alternatives`: List of other considered synset IDs
- Hashable by synset_id

**Implementation Status:** ✅ Implemented in `src/draagon_ai/semantic/types.py`

---

## Non-Functional Requirements

### NFR-006.1: Performance

- Lesk disambiguation: < 10ms per word
- LLM disambiguation: < 1000ms per word (network-bound)
- Sentence disambiguation: < 100ms for typical sentences

### NFR-006.2: Test Coverage

- All WSD functions have unit tests
- Bank/tea/coffee ambiguity cases covered
- Mock synsets allow testing without NLTK

---

## Implementation Notes

### Current Architecture

```
src/draagon_ai/semantic/
├── __init__.py          # Module exports
├── types.py             # WordSense, SemanticTriple, etc.
├── wsd.py               # LeskDisambiguator, WordSenseDisambiguator
└── expansion.py         # SemanticExpansionService (uses WSD)
```

### Integration Points

- **SemanticExpansionService**: Uses WSD to disambiguate words in expanded frames
- **Memory Storage**: Synset IDs stored with semantic memories to prevent false associations
- **Cross-Layer Associations**: Same synset ID links observations across memory layers

---

## Future Enhancements

### Phase 2: Enhanced WSD
- [ ] BabelNet integration for multilingual support
- [ ] Wikidata cross-referencing for entity linking
- [ ] Embedding-based WSD using sentence transformers
- [ ] Context window expansion (look at surrounding sentences)

### Phase 3: Knowledge Graph Integration
- [ ] Link synsets to external knowledge bases
- [ ] Import ConceptNet relations for commonsense reasoning
- [ ] Build local ontology from learned synsets

---

## References

- [WordNet](https://wordnet.princeton.edu/) - Lexical database
- [Lesk Algorithm](https://en.wikipedia.org/wiki/Lesk_algorithm) - Dictionary-based WSD
- [BabelNet](https://babelnet.org/) - Multilingual semantic network
- [NLTK WordNet Interface](https://www.nltk.org/howto/wordnet.html) - Python implementation

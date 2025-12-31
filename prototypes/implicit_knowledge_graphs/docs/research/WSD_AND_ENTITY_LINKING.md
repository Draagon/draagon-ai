# Word Sense Disambiguation and Entity Linking Research

**Version:** 1.0.0
**Last Updated:** 2025-12-31
**Status:** Research Foundation

---

## Executive Summary

Word Sense Disambiguation (WSD) and Entity Linking (EL) are **foundational** to the implicit knowledge graphs approach. This document summarizes the research and best practices for implementation.

**Key Insight:** WSD must happen BEFORE any other decomposition. Without proper disambiguation, branches cannot be correctly weighted and knowledge cannot be correctly linked.

---

## Table of Contents

1. [Word Sense Disambiguation Overview](#1-word-sense-disambiguation-overview)
2. [Entity Linking Overview](#2-entity-linking-overview)
3. [Knowledge Base Resources](#3-knowledge-base-resources)
4. [Disambiguation Algorithms](#4-disambiguation-algorithms)
5. [Universal Identifier Design](#5-universal-identifier-design)
6. [Implementation Strategy](#6-implementation-strategy)

---

## 1. Word Sense Disambiguation Overview

### What is WSD?

WSD is the task of identifying which sense of a word is used in a given context when the word has multiple meanings.

**Example:**
- "I deposited money in the **bank**" → bank.n.01 (financial institution)
- "We walked along the **bank** of the river" → bank.n.02 (sloping land)

### Why WSD is Critical for Knowledge Graphs

Without WSD:
1. **False associations** - "bank (financial)" linked to "bank (river)"
2. **Incorrect retrieval** - Query about finances returns river information
3. **Branch confusion** - Cannot weight which meaning was intended
4. **Linking failures** - Cannot connect related concepts correctly

### WSD Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Polysemy** | Single word, multiple related meanings | Fine-grained synset hierarchy |
| **Homonymy** | Single word, unrelated meanings | Domain filtering |
| **Context scarcity** | Short text, limited context | LLM-based disambiguation |
| **Domain specificity** | Technical terms with special meanings | Domain-aware WSD |
| **Named entities** | Proper nouns vs common nouns | NER first, then WSD |

---

## 2. Entity Linking Overview

### What is Entity Linking?

Entity Linking (EL) is the task of linking mentions of entities in text to their entries in a knowledge base.

**Example:**
- "**Apple** announced new products" → Wikidata Q312 (Apple Inc.)
- "I ate an **apple**" → WordNet apple.n.01 (fruit)

### Entity Linking vs WSD

| Aspect | WSD | Entity Linking |
|--------|-----|----------------|
| **Target** | Word senses (concepts) | Real-world entities (instances) |
| **Knowledge Base** | WordNet, BabelNet | Wikidata, Wikipedia |
| **Granularity** | Fine-grained (synsets) | Instance-level (QIDs) |
| **Scope** | All content words | Named entities primarily |

### Named Entity Disambiguation (NED)

NED is entity linking specifically for named entities:

**Example:**
- "**Michael Jordan** was a great player" → Basketball player OR Professor?
- Context: "Chicago Bulls" → Q41421 (basketball player)

---

## 3. Knowledge Base Resources

### WordNet

**Coverage:** ~117,000 English synsets
**Identifier:** `bank.n.01`
**Strengths:** Fine-grained senses, rich relations (hypernyms, etc.)
**Weaknesses:** No named entities, English-only

```python
from nltk.corpus import wordnet as wn

# Get synsets for a word
synsets = wn.synsets('bank', pos=wn.NOUN)
# [Synset('bank.n.01'), Synset('bank.n.02'), ...]

# Get definition
synsets[0].definition()
# 'a financial institution that accepts deposits...'

# Get hypernyms
synsets[0].hypernyms()
# [Synset('financial_institution.n.01')]
```

### BabelNet

**Coverage:** ~23 million synsets, 284 languages
**Identifier:** `bn:00008364n`
**Strengths:** Multilingual, integrates WordNet + Wikipedia + Wikidata
**Weaknesses:** API limits, complexity

**Integrations:**
- WordNet synsets
- Wikipedia pages
- Wikidata entities
- OmegaWiki
- Wiktionary

### Wikidata

**Coverage:** ~100 million entities
**Identifier:** `Q312` (Apple Inc.)
**Strengths:** Structured facts, global IDs, continuously updated
**Weaknesses:** Not for concepts (focuses on instances)

```python
# Query Wikidata
import requests

url = "https://www.wikidata.org/w/api.php"
params = {
    "action": "wbsearchentities",
    "search": "Apple Inc",
    "language": "en",
    "format": "json"
}
response = requests.get(url, params=params)
# Returns Q312 with description
```

### Resource Comparison

| Resource | Concepts | Entities | Relations | Languages |
|----------|----------|----------|-----------|-----------|
| WordNet | 117k | No | Extensive | English |
| BabelNet | 23M | Yes | Extensive | 284 |
| Wikidata | Limited | 100M | Extensive | 300+ |
| ConceptNet | 8M | Limited | 34 types | 300+ |

---

## 4. Disambiguation Algorithms

### 4.1 Lesk Algorithm (Knowledge-Based)

**Principle:** Choose sense whose definition has most word overlap with context.

```python
def lesk_disambiguate(word: str, context: list[str]) -> str:
    """Simple Lesk algorithm."""
    synsets = wn.synsets(word)
    if not synsets:
        return None

    best_sense = None
    best_score = 0

    context_set = set(context)

    for synset in synsets:
        # Get definition + examples
        signature = set(synset.definition().split())
        for example in synset.examples():
            signature.update(example.split())

        # Count overlap
        overlap = len(context_set & signature)

        if overlap > best_score:
            best_score = overlap
            best_sense = synset.name()

    return best_sense
```

**Pros:** Fast, no training needed
**Cons:** Limited accuracy (~50-60%), sensitive to vocabulary

### 4.2 Extended Lesk

**Improvement:** Include glosses of related synsets (hypernyms, hyponyms, etc.)

```python
def extended_lesk(word: str, context: list[str]) -> str:
    """Extended Lesk with related synset glosses."""
    synsets = wn.synsets(word)

    for synset in synsets:
        signature = set(synset.definition().split())

        # Add hypernym glosses
        for hypernym in synset.hypernyms():
            signature.update(hypernym.definition().split())

        # Add hyponym glosses
        for hyponym in synset.hyponyms():
            signature.update(hyponym.definition().split())

        # ... score as before
```

**Accuracy:** ~60-70%

### 4.3 Embedding-Based WSD

**Principle:** Embed context, compare to sense embeddings.

```python
async def embedding_disambiguate(
    word: str,
    context: str,
    embedder
) -> str:
    """WSD using embedding similarity."""

    # Embed the context
    context_embedding = embedder.encode(context)

    synsets = wn.synsets(word)
    best_sense = None
    best_similarity = -1

    for synset in synsets:
        # Embed the definition
        sense_text = synset.definition()
        sense_embedding = embedder.encode(sense_text)

        # Cosine similarity
        similarity = cosine_similarity(context_embedding, sense_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_sense = synset.name()

    return best_sense
```

**Accuracy:** ~70-80% depending on embedder

### 4.4 LLM-Based WSD

**Principle:** Ask LLM directly which sense is intended.

```python
async def llm_disambiguate(
    word: str,
    sentence: str,
    llm: LLMProvider
) -> str:
    """WSD using LLM reasoning."""

    synsets = wn.synsets(word)
    options = "\n".join([
        f"- {s.name()}: {s.definition()}"
        for s in synsets[:5]  # Limit to top 5
    ])

    prompt = f"""Determine which meaning of '{word}' is used in this sentence:

Sentence: "{sentence}"

Possible meanings:
{options}

Respond in XML:
<disambiguation>
    <synset_id>The WordNet synset ID (e.g., bank.n.01)</synset_id>
    <confidence>0.0-1.0</confidence>
    <reasoning>Brief explanation</reasoning>
</disambiguation>
"""

    response = await llm.chat([{"role": "user", "content": prompt}])
    return parse_disambiguation(response)
```

**Accuracy:** ~85-95% depending on LLM
**Cons:** Expensive, slow

### 4.5 Hybrid Approach (Recommended)

```python
async def hybrid_disambiguate(
    word: str,
    context: str,
    llm: LLMProvider | None
) -> tuple[str, float]:
    """Hybrid WSD: fast methods first, LLM fallback."""

    synsets = wn.synsets(word)

    # Single sense = unambiguous
    if len(synsets) == 1:
        return synsets[0].name(), 1.0

    # No synsets = unknown word
    if len(synsets) == 0:
        return None, 0.0

    # Try Lesk first
    lesk_sense, lesk_score = extended_lesk_with_score(word, context)

    # High confidence = accept Lesk result
    if lesk_score > 0.8:
        return lesk_sense, lesk_score

    # Try embedding similarity
    embed_sense, embed_score = embedding_disambiguate(word, context)

    # Agreement = high confidence
    if lesk_sense == embed_sense:
        return lesk_sense, max(lesk_score, embed_score)

    # Disagreement + LLM available = ask LLM
    if llm:
        llm_sense, llm_confidence = await llm_disambiguate(word, context, llm)
        return llm_sense, llm_confidence

    # No LLM = return best guess
    if embed_score > lesk_score:
        return embed_sense, embed_score * 0.8  # Reduce confidence due to disagreement
    return lesk_sense, lesk_score * 0.8
```

---

## 5. Universal Identifier Design

### Requirements

1. **Unique** - Each semantic unit has one ID
2. **Stable** - ID doesn't change over time
3. **Linkable** - Can connect to external KBs
4. **Typed** - Know if INSTANCE vs CLASS
5. **Resolvable** - Can look up details from ID

### Identifier Structure

```python
@dataclass
class UniversalSemanticIdentifier:
    """Universal identifier for any semantic unit."""

    # Local identifier (always present)
    local_id: str  # UUID

    # Entity type
    entity_type: EntityType  # INSTANCE | CLASS | NAMED_CONCEPT | ROLE | ANAPHORA

    # External identifiers (when resolvable)
    wordnet_synset: str | None  # "bank.n.01"
    wikidata_qid: str | None    # "Q312"
    babelnet_synset: str | None # "bn:00008364n"

    # Disambiguation metadata
    sense_rank: int        # 1 = most common sense
    domain: str | None     # "FINANCE", "GEOGRAPHY"
    confidence: float      # Disambiguation confidence

    # For named entities
    canonical_name: str | None
    aliases: list[str]

    # For concepts
    hypernym_chain: list[str]

    def __hash__(self):
        return hash(self.local_id)

    def matches_sense(self, other: "UniversalSemanticIdentifier") -> bool:
        """Check if two identifiers refer to the same sense."""
        # Same WordNet synset
        if self.wordnet_synset and other.wordnet_synset:
            return self.wordnet_synset == other.wordnet_synset
        # Same Wikidata entity
        if self.wikidata_qid and other.wikidata_qid:
            return self.wikidata_qid == other.wikidata_qid
        # Same BabelNet synset
        if self.babelnet_synset and other.babelnet_synset:
            return self.babelnet_synset == other.babelnet_synset
        # Same local ID
        return self.local_id == other.local_id
```

### Entity Type Enum

```python
class EntityType(str, Enum):
    """Types of semantic entities."""

    INSTANCE = "instance"        # Specific real-world thing (Doug, Apple Inc.)
    CLASS = "class"              # Category/type (person, company)
    NAMED_CONCEPT = "named_concept"  # Proper-named category (Christmas, Agile)
    ROLE = "role"                # Relational concept (CEO of X)
    ANAPHORA = "anaphora"        # Reference needing resolution (he, it)
    GENERIC = "generic"          # Generic reference (someone, people)
```

---

## 6. Implementation Strategy

### Phase 0 Implementation Order

1. **EntityType enum** - Define the classification
2. **UniversalSemanticIdentifier** - Core data structure
3. **WordNet integration** - Synset lookup, Lesk algorithm
4. **Entity type classifier** - INSTANCE vs CLASS
5. **LLM disambiguation** - Fallback when uncertain
6. **Integration tests** - Verify correctness

### Evolvable Components

| Component | What Evolves | Fitness Metric |
|-----------|--------------|----------------|
| Lesk scoring | Overlap weighting | Accuracy on test set |
| LLM prompt | Prompt wording | Accuracy + cost |
| Confidence thresholds | When to use LLM | Accuracy vs cost |
| Domain detection | Domain classification | Domain accuracy |

### Test Cases for WSD

```python
WSD_TEST_CASES = [
    # Financial vs river bank
    {
        "sentence": "I deposited money in the bank",
        "word": "bank",
        "expected_synset": "bank.n.01",
        "domain": "FINANCE"
    },
    {
        "sentence": "We walked along the bank of the river",
        "word": "bank",
        "expected_synset": "bank.n.02",
        "domain": "GEOGRAPHY"
    },
    # Bass fish vs bass music
    {
        "sentence": "I caught a large bass in the lake",
        "word": "bass",
        "expected_synset": "bass.n.01",
        "domain": "FISH"
    },
    {
        "sentence": "The bass line in that song is amazing",
        "word": "bass",
        "expected_synset": "bass.n.07",
        "domain": "MUSIC"
    },
    # Apple company vs apple fruit
    {
        "sentence": "Apple announced new products today",
        "word": "Apple",
        "expected_type": "INSTANCE",
        "expected_wikidata": "Q312"
    },
    {
        "sentence": "I ate an apple for lunch",
        "word": "apple",
        "expected_synset": "apple.n.01",
        "expected_type": "CLASS"
    },
]
```

---

## Research References

### Word Sense Disambiguation
- [WSD Survey (2015)](https://arxiv.org/pdf/1508.01346) - Comprehensive overview
- [Is WSD Outdated?](https://medium.com/semantic-tech-hotspot/is-word-sense-disambiguation-outdated-ef05a139576) - Modern perspective
- [WSD with Knowledge Graphs (2024)](https://dl.acm.org/doi/10.1145/3677524) - Recent advances

### Entity Linking
- [Entity Disambiguation Survey](https://paperswithcode.com/task/entity-disambiguation/latest)
- [NED in Short Texts (2021)](https://link.springer.com/article/10.1007/s10115-021-01642-9)
- [Stanford NER vs WSD](https://nlp.stanford.edu/pubs/chang2016entity.pdf)

### Knowledge Bases
- [WordNet](https://wordnet.princeton.edu/)
- [BabelNet](https://babelnet.org/about)
- [Wikidata](https://www.wikidata.org/)

---

**End of WSD and Entity Linking Research Document**

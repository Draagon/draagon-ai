# Deep Dive: Negation and Polarity Handling

**Status:** Research Complete
**Priority:** Medium
**Phase:** Informs Phase 1-2 (Decomposition, Storage)

---

## Executive Summary

Negation fundamentally changes the meaning of statements, yet most knowledge graphs focus on positive facts. This document explores how to:
1. **Detect** negation and its scope in input text
2. **Represent** negative knowledge in the graph
3. **Store** negative facts efficiently
4. **Retrieve** correctly when negation matters

**Key Insight:** Negation isn't just "flip the boolean" - it affects scope, presuppositions, and implications differently. "Doug doesn't like coffee" requires different handling than "Doug likes not-coffee" or "It's not the case that Doug likes coffee."

---

## Table of Contents

1. [The Negation Problem](#1-the-negation-problem)
2. [Negation Detection and Scope](#2-negation-detection-and-scope)
3. [Knowledge Representation Strategies](#3-knowledge-representation-strategies)
4. [Storage Approaches](#4-storage-approaches)
5. [Retrieval Implications](#5-retrieval-implications)
6. [Implementation Recommendations](#6-implementation-recommendations)

---

## 1. The Negation Problem

### Why Negation Matters

Consider these statements about Doug's coffee preferences:

| Statement | Meaning | Decomposition Challenge |
|-----------|---------|------------------------|
| "Doug likes coffee" | Positive preference | Store: (Doug, LIKES, coffee) |
| "Doug doesn't like coffee" | Negative preference | Store: (Doug, NOT_LIKES, coffee)? Or (Doug, DISLIKES, coffee)? |
| "Doug never drinks coffee" | Habitual negation | Different from "doesn't like" |
| "Doug didn't drink coffee today" | Specific negation | Temporal scope matters |
| "Doug stopped drinking coffee" | Change-of-state | Presupposes he used to drink it |

### The Asymmetry Problem

From [research on KBs and negation](https://arxiv.org/html/2305.05403):

> "There is a considerable computational and representational advantage to reasoning under the CWA since the number of negative facts vastly exceeds the number of positive ones. It is totally unfeasible to explicitly represent all such negative information."

**The challenge:** We can't store all negative facts (infinite), but we need to store *salient* negatives.

### Negation Types That Matter

| Type | Example | Storage Strategy |
|------|---------|------------------|
| **Explicit denial** | "Doug doesn't like coffee" | Store as negative fact |
| **Habitual negation** | "Doug never eats meat" | Store as constraint/rule |
| **Temporal negation** | "Doug didn't attend yesterday" | Store with temporal scope |
| **Preference negation** | "Doug prefers tea, not coffee" | Store preference + contrast |
| **Cessation** | "Doug stopped smoking" | Store change-of-state |
| **Failure** | "Doug forgot to call" | Store failed intention |

---

## 2. Negation Detection and Scope

### Negation Cues

From [NegBERT research](https://arxiv.org/pdf/1911.04211) and [negation survey](https://www.mdpi.com/2076-3417/12/10/5209):

**Common Negation Cues:**
- **Explicit negators:** not, no, never, neither, none, nobody, nothing, nowhere
- **Negative verbs:** deny, refuse, fail, lack, miss, prevent, stop
- **Negative affixes:** un-, in-, im-, non-, dis-, -less (unhappy, impossible, careless)
- **Implicit negators:** barely, hardly, rarely, seldom, few, little

### Scope Detection

**The scope** is the part of the sentence affected by negation:

```
"Doug [didn't] [eat the sandwich that Sarah made]."
         ^cue   ^-----------scope--------------^

"Doug didn't [eat the sandwich] that Sarah made."
              ^----scope----^
(Sarah still made it - that's outside scope)
```

**Neural approaches** (from [research](https://link.springer.com/article/10.1007/s11704-018-7368-6)):
- BiLSTM for context representation
- CNN for syntactic path features
- BERT-based models (NegBERT) achieve state-of-the-art

### Scope Detection for Our Use Case

For the implicit knowledge graphs prototype, we need:

```python
@dataclass
class NegationInfo:
    """Negation detection result."""

    has_negation: bool
    cue: str | None           # "not", "never", "didn't"
    cue_type: str | None      # "explicit", "verbal", "affixal"
    scope_start: int          # Character offset
    scope_end: int
    scope_text: str           # The negated portion

    # Semantic impact
    negation_type: str        # "denial", "habitual", "temporal", "cessation"
    affects_presuppositions: bool  # Does negation cancel presuppositions?


async def detect_negation(
    sentence: str,
    llm: LLMProvider | None = None,
) -> NegationInfo:
    """Detect negation and its scope in a sentence."""

    # Simple heuristic first
    negation_cues = ["not", "n't", "never", "no", "none", "nothing", "nobody"]
    cue_found = None
    for cue in negation_cues:
        if cue in sentence.lower():
            cue_found = cue
            break

    if not cue_found:
        return NegationInfo(has_negation=False, ...)

    # Use LLM for scope detection if available
    if llm:
        return await llm_detect_negation_scope(sentence, cue_found, llm)

    # Heuristic: scope is from cue to end of clause
    return heuristic_scope_detection(sentence, cue_found)
```

---

## 3. Knowledge Representation Strategies

### Strategy A: Negated Predicates

Create explicit negative predicates:

```python
# Positive
(Doug, LIKES, coffee)

# Negative - option 1: negated predicate
(Doug, NOT_LIKES, coffee)

# Negative - option 2: antonym predicate
(Doug, DISLIKES, coffee)
```

**Pros:** Simple, explicit
**Cons:** Predicate explosion (every predicate needs a NOT_ version)

### Strategy B: Polarity Attribute

Add polarity as metadata:

```python
@dataclass
class SemanticTriple:
    subject: str
    predicate: str
    object: str
    polarity: Literal["positive", "negative"] = "positive"
    negation_scope: str | None = None  # What exactly is negated

# Storage
triple = SemanticTriple(
    subject="Doug",
    predicate="LIKES",
    object="coffee",
    polarity="negative",
)
```

**Pros:** Flexible, no predicate explosion
**Cons:** Requires polarity-aware retrieval

### Strategy C: Reification with Negation Node

Create a statement node that can be negated:

```
Statement_123:
  type: PREFERENCE
  subject: Doug
  predicate: LIKES
  object: coffee

Negation_456:
  negates: Statement_123
  scope: "entire statement"
  source: "Doug said he doesn't like coffee"
```

**Pros:** Can represent complex negation (partial scope)
**Cons:** More complex graph structure

### Strategy D: Partial Closed World Assumption (PCWA)

From [PCWA research](https://arxiv.org/html/2305.05403):

> "Intermediate settings are thus needed, referred to as Partial-closed World Assumptions (PCWA), where some parts of the KB are treated under closed-world semantics, others under open-world semantics."

For our use case:
- **Closed-world predicates:** Things we track exhaustively (Doug's preferences)
- **Open-world predicates:** Things we don't claim completeness on

```python
CLOSED_WORLD_PREDICATES = [
    "LIKES", "DISLIKES", "PREFERS",  # Preferences we actively track
    "HAS_ALLERGY", "CANNOT_EAT",     # Constraints we track
]

# For closed-world predicates:
# If (Doug, LIKES, X) is not stored, we can infer Doug doesn't like X
# No need to store explicit negatives for these

OPEN_WORLD_PREDICATES = [
    "KNOWS", "HAS_VISITED",  # We don't claim to know all
]

# For open-world predicates:
# Must store explicit negatives when known
```

### Recommended Approach: Hybrid

```python
@dataclass
class KnowledgeTriple:
    """Triple with polarity and world assumption."""

    subject: str
    predicate: str
    object: str

    # Polarity handling
    polarity: Literal["positive", "negative"] = "positive"
    negation_type: str | None = None  # "denial", "habitual", "temporal"
    negation_scope: str | None = None

    # World assumption
    world_assumption: Literal["open", "closed"] = "open"

    # For negative facts, why is this salient?
    salience_reason: str | None = None  # "user_stated", "inferred", "correction"


def should_store_negative(triple: KnowledgeTriple) -> bool:
    """Determine if a negative triple is worth storing."""

    # Always store explicit user denials
    if triple.salience_reason == "user_stated":
        return True

    # Always store corrections
    if triple.salience_reason == "correction":
        return True

    # For closed-world predicates, don't store (absence = negative)
    if triple.world_assumption == "closed":
        return False

    # For open-world, store if salient
    return triple.salience_reason is not None
```

---

## 4. Storage Approaches

### 4.1 Storing Negative Facts

Only store **salient** negatives - those explicitly stated or important:

```python
# User says: "Doug doesn't like coffee"
# This is SALIENT - store it

store_triple(KnowledgeTriple(
    subject="Doug",
    predicate="LIKES",
    object="coffee",
    polarity="negative",
    salience_reason="user_stated",
))

# Don't store: "Doug doesn't like quantum physics textbooks"
# (unless explicitly stated - not salient by default)
```

### 4.2 Handling Negation in Presuppositions

Negation can **cancel** or **preserve** presuppositions:

```python
# "Doug stopped drinking coffee"
# Presupposition: Doug used to drink coffee (PRESERVED under negation)

# "Doug didn't stop drinking coffee"
# Presupposition: Doug used to drink coffee (STILL PRESERVED)
# The negation is about the STOPPING, not the drinking

# But: "Doug didn't even start drinking coffee"
# Presupposition CANCELLED - Doug never drank coffee

@dataclass
class Presupposition:
    content: str
    trigger: str
    survives_negation: bool  # Does this survive if statement is negated?
```

### 4.3 Handling Change-of-State

Change-of-state verbs (stop, start, begin, finish) require special handling:

```python
# "Doug stopped drinking coffee"
decomposition = {
    "presupposition": (Doug, USED_TO_DRINK, coffee),  # Past habit
    "assertion": (Doug, STOPPED, drinking_coffee),     # Change event
    "implication": (Doug, NOT_DRINKS, coffee),        # Current state
}

# Store ALL of these - they're all salient
```

### 4.4 Negation and Inference Interaction

When storing inferences for negated statements:

```python
# "Doug forgot the meeting"
inferences = {
    "xEffect": "missed the meeting",
    "xReact": "embarrassed",
}

# "Doug didn't forget the meeting"
inferences = {
    "xEffect": "attended the meeting",  # OPPOSITE
    "xReact": "relieved",               # DIFFERENT emotion
}

# The inference extraction must be NEGATION-AWARE
async def extract_inferences(
    event: str,
    negation_info: NegationInfo,
    config: InferenceConfig,
) -> list[CommonsenseInference]:
    """Extract inferences, accounting for negation."""

    if negation_info.has_negation:
        # Extract inferences for the NEGATED event
        # Not just flip the original inferences
        return await extract_inferences_for_negated(event, negation_info, config)
    else:
        return await extract_inferences_standard(event, config)
```

---

## 5. Retrieval Implications

### 5.1 Query Negation Detection

Queries can also contain negation:

```
Query: "What does Doug NOT like?"
→ Need to retrieve negative polarity facts about Doug's preferences

Query: "Does Doug like coffee?"
→ Need to check both positive AND negative facts
→ If negative fact exists: "No, Doug doesn't like coffee"
→ If no fact exists (closed world): "No record of Doug liking coffee"
→ If no fact exists (open world): "Unknown"
```

### 5.2 Polarity-Aware Retrieval

```python
async def search_with_polarity(
    query: str,
    entities: list[str],
    expected_polarity: str | None = None,
) -> list[KnowledgeTriple]:
    """Search with optional polarity filtering."""

    # Detect if query asks for negatives
    query_negation = detect_negation(query)

    # Get all matching triples
    results = await search_by_entities(entities)

    # Filter by polarity if query specifies
    if query_negation.has_negation or expected_polarity == "negative":
        results = [r for r in results if r.polarity == "negative"]
    elif expected_polarity == "positive":
        results = [r for r in results if r.polarity == "positive"]

    return results
```

### 5.3 Contradiction Detection

Negation creates opportunity for contradictions:

```python
async def detect_contradictions(
    new_triple: KnowledgeTriple,
    existing_triples: list[KnowledgeTriple],
) -> list[Contradiction]:
    """Detect if new triple contradicts existing knowledge."""

    contradictions = []

    for existing in existing_triples:
        # Same subject, predicate, object but different polarity
        if (
            new_triple.subject == existing.subject and
            new_triple.predicate == existing.predicate and
            new_triple.object == existing.object and
            new_triple.polarity != existing.polarity
        ):
            contradictions.append(Contradiction(
                new=new_triple,
                existing=existing,
                type="polarity_conflict",
            ))

    return contradictions
```

---

## 6. Implementation Recommendations

### Phase 1 Implementation (Basic)

```python
@dataclass
class NegationConfig:
    """Configuration for negation handling."""

    # Detection
    use_llm_for_scope: bool = True
    fallback_to_heuristic: bool = True

    # Storage
    store_salient_negatives: bool = True
    closed_world_predicates: list[str] = field(default_factory=lambda: [
        "LIKES", "DISLIKES", "PREFERS", "ALLERGIC_TO",
    ])

    # Retrieval
    polarity_aware_search: bool = True
    detect_contradictions: bool = True


# Integration with decomposition pipeline
async def decompose_with_negation(
    text: str,
    config: DecompositionConfig,
    negation_config: NegationConfig,
) -> DecomposedKnowledge:
    """Decompose text with negation awareness."""

    # Step 1: Detect negation
    negation_info = await detect_negation(text, config.llm)

    # Step 2: Adjust decomposition based on negation
    if negation_info.has_negation:
        # Mark triples with polarity
        triples = await extract_triples(text, config)
        for triple in triples:
            if is_in_scope(triple, negation_info):
                triple.polarity = "negative"
                triple.negation_type = negation_info.negation_type

        # Extract negation-aware inferences
        inferences = await extract_inferences_negation_aware(
            text, negation_info, config
        )

        # Handle presupposition survival
        presuppositions = await extract_presuppositions(text, config)
        for presup in presuppositions:
            presup.survives_negation = check_survival(presup, negation_info)

    else:
        # Standard decomposition
        triples = await extract_triples(text, config)
        inferences = await extract_inferences(text, config)
        presuppositions = await extract_presuppositions(text, config)

    return DecomposedKnowledge(
        original_text=text,
        negation_info=negation_info,
        triples=triples,
        inferences=inferences,
        presuppositions=presuppositions,
    )
```

### Evolvable Parameters

```python
@dataclass
class EvolvableNegationConfig(NegationConfig):
    """Negation config with evolution tracking."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Evolvable thresholds
    scope_detection_confidence: float = 0.7
    salience_threshold: float = 0.5

    # Evolvable prompts
    negation_detection_prompt: str = "..."
    scope_detection_prompt: str = "..."

    # Evolvable predicate lists
    closed_world_predicates: list[str] = ...
```

### Test Cases

```python
NEGATION_TEST_CASES = [
    {
        "text": "Doug doesn't like coffee",
        "expected_negation": True,
        "expected_scope": "like coffee",
        "expected_triple": ("Doug", "LIKES", "coffee", "negative"),
    },
    {
        "text": "Doug never eats meat",
        "expected_negation": True,
        "expected_negation_type": "habitual",
        "expected_triple": ("Doug", "EATS", "meat", "negative"),
    },
    {
        "text": "Doug stopped smoking",
        "expected_negation": False,  # "stopped" is change-of-state, not negation
        "expected_presupposition": ("Doug", "USED_TO_SMOKE", None),
        "expected_implication": ("Doug", "SMOKES", None, "negative"),
    },
    {
        "text": "Doug didn't forget the meeting",
        "expected_negation": True,
        "expected_scope": "forget the meeting",
        "presupposition_survives": True,  # Meeting still existed
    },
]
```

---

## Summary

### Key Design Decisions

1. **Detect negation with scope** - Use LLM or BiLSTM for accurate scope detection
2. **Store polarity on triples** - Not negated predicates
3. **Only store salient negatives** - User-stated, corrections, constraints
4. **Use PCWA** - Closed-world for preferences, open-world for general knowledge
5. **Negation-aware inference** - Don't just flip positive inferences
6. **Handle presupposition survival** - Some survive negation, some don't
7. **Polarity-aware retrieval** - Match query polarity to stored polarity

### Integration Points

- **Decomposition:** Detect negation → adjust triple polarity → adjust inferences
- **Storage:** Filter salient negatives → store with polarity metadata
- **Retrieval:** Detect query negation → filter by polarity → detect contradictions

---

## Research References

- [Comprehensive Taxonomy of Negation for NLP](https://arxiv.org/html/2507.22337v3)
- [NegBERT: Transfer Learning for Negation Detection](https://arxiv.org/pdf/1911.04211)
- [Negation and Speculation in NLP Survey](https://www.mdpi.com/2076-3417/12/10/5209)
- [Completeness, Recall, and Negation in Open-World KBs](https://arxiv.org/html/2305.05403)
- [Hybrid Neural Network for Negation Scope](https://link.springer.com/article/10.1007/s11704-018-7368-6)

---

**Status:** Research complete, ready for Phase 1 integration

# Decomposition Theory: Complete Taxonomy of Implicit Knowledge Extraction

**Version:** 1.0.0
**Last Updated:** 2025-12-31
**Status:** Research Foundation

---

## Executive Summary

This document provides a comprehensive taxonomy of ALL the ways to decompose text into implicit knowledge for graph storage. This taxonomy is foundational to the implicit knowledge graphs prototype.

**Key Insight:** The decomposition approach determines the quality of retrieval. Different decomposition types capture different aspects of meaning, and ALL must be extracted and properly identified to enable optimal context retrieval.

---

## Table of Contents

1. [Entity Type Hierarchy](#1-entity-type-hierarchy)
2. [Word Sense Categories](#2-word-sense-categories)
3. [Semantic Role Types](#3-semantic-role-types)
4. [Frame Semantic Elements](#4-frame-semantic-elements)
5. [Commonsense Inference Types](#5-commonsense-inference-types)
6. [Presupposition Types](#6-presupposition-types)
7. [Implication Types](#7-implication-types)
8. [Temporal/Aspectual Types](#8-temporalaspectual-types)
9. [Modality Types](#9-modality-types)
10. [Integration Strategy](#10-integration-strategy)

---

## 1. Entity Type Hierarchy

### Why This Matters

The distinction between "Doug" (a specific person) and "person" (a category) is fundamental. Without this distinction, you cannot correctly:
- Link related knowledge (Doug's facts vs general person facts)
- Resolve references ("he" → Doug, not "person")
- Store with appropriate granularity

### Entity Type Taxonomy

| Type | Description | Example | Identifier Strategy |
|------|-------------|---------|---------------------|
| **INSTANCE** | Unique real-world thing | "Doug", "Apple Inc.", "Eiffel Tower" | Local UUID + Wikidata QID |
| **CLASS** | Category/type of things | "person", "company", "landmark" | WordNet synset ID |
| **NAMED_CONCEPT** | Proper-named category | "Christmas", "Agile methodology", "Renaissance" | BabelNet synset + NAMED_CONCEPT flag |
| **ROLE** | Relational concept | "CEO of Apple", "Doug's wife", "the author" | Relation type + anchor entity |
| **ANAPHORA** | Reference requiring resolution | "he", "it", "they", "the company" | Pointer to resolved entity |
| **GENERIC** | Generic reference | "one", "people", "someone" | Special GENERIC marker |

### Classification Algorithm

```
Input: "Apple announced new products"

Step 1: NER → "Apple" tagged as ORGANIZATION
Step 2: Context analysis → "announced products" = company behavior
Step 3: Entity type → INSTANCE (specific company, not fruit)
Step 4: Entity linking → Wikidata Q312 (Apple Inc.)
Step 5: Confidence → 0.95
```

### The "Name as Label" Pattern

Named entities serve as **anchors** to collections of knowledge:

```python
@dataclass
class NamedEntityAnchor:
    """A named entity as a label for a knowledge collection."""

    canonical_name: str  # "Doug"
    aliases: list[str]  # ["Douglas", "he", "my husband"]
    identifier: UniversalSemanticIdentifier

    # Knowledge this name anchors to
    knowledge_links: dict[str, list[str]] = {
        "facts": [],        # Fact node IDs
        "relationships": [], # Relationship edge IDs
        "preferences": [],   # Preference node IDs
        "behaviors": [],     # Behavioral pattern IDs
        "episodes": [],      # Episode IDs mentioning this entity
    }
```

---

## 2. Word Sense Categories

### Why This Matters

"Bank" has 10+ senses in WordNet. Without disambiguation, you cannot:
- Prevent false associations (financial bank linked to riverbank)
- Correctly weight branches (which sense is intended?)
- Retrieve relevant context (filter by synset)

### Sense Identification Resources

| Resource | Coverage | Identifier Format | Best For |
|----------|----------|-------------------|----------|
| **WordNet** | English lexical | `bank.n.01` | Common words, fine-grained senses |
| **BabelNet** | Multilingual encyclopedic | `bn:00008364n` | Entities + concepts, cross-lingual |
| **Wikidata** | Entities | `Q312` | Named entities, structured facts |
| **FrameNet** | Frame semantics | `Giving` frame | Event structure, roles |

### Synset Information Structure

```python
@dataclass
class SynsetInfo:
    """Complete information about a word sense."""

    # Primary identifier
    synset_id: str  # "bank.n.01"

    # Definition and examples
    definition: str  # "a financial institution that..."
    examples: list[str]  # ["The bank foreclosed on the house"]

    # Taxonomic relations
    hypernyms: list[str]  # ["financial_institution.n.01"]
    hyponyms: list[str]   # ["savings_bank.n.01", "commercial_bank.n.01"]

    # Other relations
    meronyms: list[str]   # Parts
    holonyms: list[str]   # Wholes
    antonyms: list[str]   # Opposites

    # Domain information
    domain: str | None  # "FINANCE"
    domain_terms: list[str]  # Related domain vocabulary

    # Usage frequency
    frequency_rank: int  # 1 = most common sense
```

### Disambiguation Strategies (Evolvable)

1. **Lesk Algorithm** - Overlap between context and gloss definitions
2. **Embedding Similarity** - Embed context, compare to sense embeddings
3. **LLM-Based** - Ask LLM which sense given context
4. **Domain Filtering** - Use domain of conversation to filter senses
5. **Frequency Prior** - Default to most common sense when uncertain

---

## 3. Semantic Role Types

### Why This Matters

"Doug gave Sarah flowers" has structure: Agent (Doug), Recipient (Sarah), Theme (flowers). Without this:
- Cannot answer "Who received something?"
- Cannot link to other giving events
- Cannot extract presuppositions correctly

### PropBank/VerbNet Role Taxonomy

| Role | Definition | Example |
|------|------------|---------|
| **ARG0 (Agent)** | Doer of the action | "Doug" in "Doug ate pizza" |
| **ARG1 (Patient/Theme)** | Affected entity | "pizza" in "Doug ate pizza" |
| **ARG2 (Recipient/Beneficiary)** | Recipient | "Sarah" in "Doug gave Sarah flowers" |
| **ARG3 (Instrument)** | Tool used | "knife" in "Cut with a knife" |
| **ARG4 (Start/End point)** | Source/Goal | "Boston" in "flew from Boston" |
| **ARGM-LOC** | Location | "in the kitchen" |
| **ARGM-TMP** | Time | "yesterday", "in the morning" |
| **ARGM-MNR** | Manner | "quickly", "carefully" |
| **ARGM-CAU** | Cause | "because of the rain" |
| **ARGM-PRP** | Purpose | "to celebrate" |
| **ARGM-DIR** | Direction | "towards the door" |
| **ARGM-NEG** | Negation | "not", "never" |

### Extraction Format

```python
@dataclass
class SemanticRole:
    """A semantic role in a predicate-argument structure."""

    predicate: str  # "give"
    predicate_sense: str  # "give.v.01"

    role: str  # "ARG0", "ARG1", etc.
    filler: str  # "Doug"
    filler_id: UniversalSemanticIdentifier

    # Span information
    span_start: int
    span_end: int

    # Confidence
    confidence: float
```

---

## 4. Frame Semantic Elements

### Why This Matters

Frames capture background knowledge that predicates evoke. "Buy" evokes a COMMERCIAL_TRANSACTION frame with implied Buyer, Seller, Goods, Money - even if some are unstated.

### FrameNet Frame Structure

```
Frame: COMMERCIAL_TRANSACTION

Core Frame Elements:
  - Buyer: Entity acquiring goods (required)
  - Seller: Entity providing goods (required)
  - Goods: What is exchanged (required)
  - Money: Payment (optional but implied)

Non-Core Frame Elements:
  - Place: Where transaction occurs
  - Time: When transaction occurs
  - Manner: How transaction occurs
  - Purpose: Why transaction occurs

Inherited From: TRANSFER (more general frame)

Related Frames:
  - COMMERCE_BUY (Buyer perspective)
  - COMMERCE_SELL (Seller perspective)
  - GETTING (Result for Buyer)
  - GIVING (Result for Seller)
```

### Frame Element Extraction

```python
@dataclass
class FrameInstance:
    """An instantiated frame from text."""

    frame_name: str  # "COMMERCIAL_TRANSACTION"
    trigger_word: str  # "bought"
    trigger_sense: str  # "buy.v.01"

    # Filled frame elements
    core_elements: dict[str, FrameElement]  # "Buyer": Doug
    non_core_elements: dict[str, FrameElement]

    # Unfilled but implied elements
    implied_elements: dict[str, str]  # "Money": "implied"

    # Presuppositions from frame
    frame_presuppositions: list[str]
```

---

## 5. Commonsense Inference Types

### Why This Matters

ATOMIC-style inferences capture "if X then Y" commonsense knowledge that humans take for granted but LLMs need explicit.

### ATOMIC 2020 Relation Types

#### Social Interaction Relations (9 types)

| Relation | Definition | Example: "PersonX buys coffee" |
|----------|------------|-------------------------------|
| **xIntent** | Why X did this | "to get caffeine", "to feel awake" |
| **xNeed** | What X needed first | "money", "to go to a coffee shop" |
| **xWant** | What X wants after | "to drink the coffee", "to sit down" |
| **xEffect** | What happens to X | "X has coffee", "X feels satisfied" |
| **xReact** | How X feels | "awake", "happy", "rushed" |
| **xAttr** | X's attributes | "coffee drinker", "tired person" |
| **oWant** | What others want | "to serve X", "payment" |
| **oEffect** | What happens to others | "barista makes money" |
| **oReact** | How others feel | "helpful", "busy" |

#### Physical Entity Relations (7 types)

| Relation | Definition | Example: "coffee" |
|----------|------------|-------------------|
| **ObjectUse** | What it's used for | "drinking", "staying awake" |
| **AtLocation** | Where found | "coffee shop", "kitchen" |
| **MadeOf** | Composition | "coffee beans", "water" |
| **HasProperty** | Properties | "hot", "caffeinated", "brown" |
| **CapableOf** | What it can do | "stain clothes", "keep awake" |
| **Desires** | What it "wants" (metaphorical) | N/A for coffee |
| **NotDesires** | What it doesn't want | N/A |

#### Event-Centered Relations (7 types)

| Relation | Definition | Example: "buying coffee" |
|----------|------------|--------------------------|
| **Causes** | What this causes | "having coffee" |
| **IsBefore** | What comes before | "going to shop", "deciding to buy" |
| **IsAfter** | What comes after | "drinking coffee", "leaving shop" |
| **HasSubEvent** | Sub-events | "paying", "receiving cup" |
| **HinderedBy** | What prevents this | "no money", "shop closed" |
| **xReason** | General reason | "thirst", "habit", "need energy" |
| **isFilledBy** | Slot fillers | "coffee" fills "beverage" slot |

### Extraction Format

```python
@dataclass
class CommonsenseInference:
    """A commonsense inference from ATOMIC-style reasoning."""

    relation_type: str  # "xIntent", "xEffect", etc.
    source_event: str  # "PersonX buys coffee"
    inference: str     # "to get caffeine"
    confidence: float  # 0.8

    # Source of inference
    source: str  # "COMET", "LLM", "ConceptNet"
```

---

## 6. Presupposition Types

### Why This Matters

Presuppositions are what MUST be true for a statement to make sense. They're often more important than the explicit content.

### Presupposition Trigger Taxonomy

| Trigger Type | Linguistic Marker | Example | Presupposition |
|--------------|-------------------|---------|----------------|
| **Definite Description** | "the X" | "the meeting" | A specific meeting exists |
| **Factive Verb** | "realize", "know", "regret" | "Doug realized he was late" | Doug was late |
| **Change-of-State** | "stop", "start", "continue" | "Doug stopped smoking" | Doug used to smoke |
| **Iterative** | "again", "another", "still" | "Doug forgot again" | Doug forgot before |
| **Temporal Clause** | "before", "after", "when" | "Before Doug left..." | Doug left |
| **Cleft Sentence** | "It was X who..." | "It was Doug who called" | Someone called |
| **Comparative** | "more than", "as X as" | "Doug is taller than Sarah" | Sarah has height |
| **Implicative Verb** | "manage", "forget", "remember" | "Doug managed to finish" | Finishing was difficult |
| **Counterfactual** | "if X had..." | "If Doug had known..." | Doug didn't know |
| **Possessive** | "X's Y" | "Doug's car" | Doug has a car |

### Extraction Format

```python
@dataclass
class Presupposition:
    """A presupposition extracted from text."""

    content: str  # "Doug smoked before"
    trigger_type: str  # "CHANGE_OF_STATE"
    trigger_text: str  # "stopped"
    trigger_span: tuple[int, int]

    # How certain is this presupposition?
    confidence: float

    # Can this presupposition be cancelled?
    cancellable: bool  # "stopped" presupposition can be cancelled: "stopped, or rather never started"

    # What happens if presupposition fails?
    failure_mode: str  # "statement_undefined", "statement_false", "repair_needed"
```

---

## 7. Implication Types

### Why This Matters

Implications go beyond what's said to what's communicated. Different types have different certainty levels.

### Implication Type Taxonomy

| Type | Definition | Certainty | Example |
|------|------------|-----------|---------|
| **Logical Entailment** | Necessarily follows | 100% | "X bought Y" → "X has Y" |
| **Semantic Entailment** | Follows from meaning | 95%+ | "X killed Y" → "Y is dead" |
| **Pragmatic Implicature** | Typically communicated | 70-90% | "Can you pass the salt?" → Request |
| **Conversational Implicature** | Context-dependent | Variable | "Nice weather" (sarcastic) → Weather is bad |
| **Scalar Implicature** | From quantifiers | 80%+ | "Some cats are black" → "Not all are" |
| **Conventional Implicature** | From word meaning | 90%+ | "but" → Contrast expected |

### Extraction Format

```python
@dataclass
class Implication:
    """An implication extracted from text."""

    content: str  # "Doug has the book"
    source_text: str  # "Doug bought a book"
    implication_type: str  # "LOGICAL_ENTAILMENT"

    confidence: float  # 1.0 for entailment, lower for implicature

    # Is this defeasible (can be overridden)?
    defeasible: bool

    # What conditions could defeat it?
    defeating_conditions: list[str]  # ["returned the book", "gift for someone"]
```

---

## 8. Temporal/Aspectual Types

### Why This Matters

"Doug drinks coffee" (habitual) vs "Doug is drinking coffee" (ongoing) vs "Doug drank coffee" (completed) - same verb, different temporal meaning.

### Aspectual Categories (Vendler)

| Aspect | Definition | Duration | Endpoint | Example |
|--------|------------|----------|----------|---------|
| **State** | Unchanging condition | Unbounded | No | "Doug likes coffee" |
| **Activity** | Ongoing process | Unbounded | No | "Doug is running" |
| **Accomplishment** | Process with endpoint | Bounded | Yes | "Doug built a house" |
| **Achievement** | Instantaneous change | Punctual | Yes | "Doug noticed the car" |
| **Semelfactive** | Single-instance event | Punctual | No | "Doug knocked" |

### Temporal Reference Types

| Type | Description | Example |
|------|-------------|---------|
| **Deictic** | Relative to speech time | "yesterday", "tomorrow", "now" |
| **Anaphoric** | Relative to reference time | "the next day", "before that" |
| **Calendar** | Absolute | "December 31, 2025", "3pm" |
| **Duration** | Length of time | "for three hours", "all day" |
| **Frequency** | How often | "always", "usually", "sometimes" |

### Extraction Format

```python
@dataclass
class TemporalStructure:
    """Temporal/aspectual structure of an event."""

    # Aspect
    aspect: str  # "STATE", "ACTIVITY", "ACCOMPLISHMENT", "ACHIEVEMENT"

    # Tense
    tense: str  # "PAST", "PRESENT", "FUTURE"
    tense_certainty: str  # "DEFINITE", "UNCERTAIN", "HABITUAL"

    # Temporal reference
    reference_type: str  # "DEICTIC", "ANAPHORIC", "CALENDAR"
    reference_value: str | None  # "2025-12-31" or "yesterday"

    # Duration/frequency
    duration: str | None  # "for three hours"
    frequency: str | None  # "always", "sometimes"

    # Temporal relations to other events
    before: list[str]  # Event IDs this is before
    after: list[str]   # Event IDs this is after
    during: list[str]  # Event IDs this overlaps with
```

---

## 9. Modality Types

### Why This Matters

"Doug might come" vs "Doug will come" vs "Doug should come" - different modal meanings affect how to store and retrieve.

### Modal Type Taxonomy

| Type | What It Expresses | Examples | Certainty |
|------|-------------------|----------|-----------|
| **Epistemic** | Speaker's certainty | "might", "must be", "probably" | Variable |
| **Deontic** | Obligation/permission | "should", "must", "may" | N/A |
| **Dynamic** | Ability/willingness | "can", "will", "would" | N/A |
| **Evidential** | Source of knowledge | "apparently", "reportedly", "I heard" | Variable |
| **Bouletic** | Desire/wish | "want to", "wish", "hope" | N/A |

### Extraction Format

```python
@dataclass
class Modality:
    """Modal structure of a statement."""

    modal_type: str  # "EPISTEMIC", "DEONTIC", etc.
    modal_marker: str  # "might", "should", etc.

    # For epistemic modality
    certainty_level: float | None  # 0.0-1.0

    # For deontic modality
    obligation_strength: str | None  # "REQUIRED", "RECOMMENDED", "PERMITTED"

    # For evidential modality
    evidence_source: str | None  # "DIRECT", "REPORTED", "INFERRED"
    evidence_quality: float | None

    # Scope
    scope: str  # What the modality applies to
```

---

## 10. Integration Strategy

### How All Types Work Together

```
Input: "Doug apparently forgot the meeting again"

1. ENTITY IDENTIFICATION
   - "Doug" → INSTANCE, local_id=uuid1
   - "the meeting" → INSTANCE (specific meeting), local_id=uuid2

2. WORD SENSE DISAMBIGUATION
   - "forgot" → forget.v.01 (fail to remember)
   - "meeting" → meeting.n.01 (assembly)

3. SEMANTIC ROLES
   - Agent (ARG0): Doug
   - Theme (ARG1): the meeting
   - Predicate: forgot

4. FRAME SEMANTICS
   - Frame: MEMORY
   - Forgetter: Doug
   - Content: the meeting
   - Implied: Doug knew about meeting before

5. PRESUPPOSITIONS
   - "the meeting" → A specific meeting exists (DEFINITE_DESC)
   - "again" → Doug forgot before (ITERATIVE)
   - "forgot" → Doug was supposed to remember (FACTIVE-ADJ)

6. COMMONSENSE INFERENCES
   - xEffect: Doug missed the meeting
   - xReact: embarrassed, guilty
   - oReact: others frustrated
   - xAttr: forgetful, unreliable

7. IMPLICATIONS
   - Doug did not attend the meeting (ENTAILMENT)
   - This is a pattern (PRAGMATIC from "again")
   - Speaker is frustrated (CONVERSATIONAL)

8. TEMPORAL STRUCTURE
   - Aspect: ACHIEVEMENT (punctual)
   - Tense: PAST, DEFINITE
   - Iteration: REPEATED

9. MODALITY
   - Type: EVIDENTIAL
   - Marker: "apparently"
   - Evidence: REPORTED (speaker heard this)
   - Certainty: 0.7 (not certain)
```

### Storage Strategy

Each decomposition type maps to graph storage:

| Type | Storage Format | Retrieval Strategy |
|------|----------------|-------------------|
| Entities | Nodes with universal IDs | Entity-based search |
| Word Senses | Synset IDs in node metadata | Synset filtering |
| Semantic Roles | Edges: (Agent)→[FORGOT]→(Theme) | Role-based queries |
| Frames | Frame nodes linked to instances | Frame search |
| Presuppositions | Presupposition nodes | Implication search |
| Commonsense | Inference edges with types | Inference traversal |
| Implications | Implication nodes | Entailment queries |
| Temporal | Temporal edges between events | Timeline queries |
| Modality | Modality metadata on statements | Certainty filtering |

---

## Research References

### Semantic Roles
- [PropBank](https://propbank.github.io/)
- [VerbNet](https://verbs.colorado.edu/verbnet/)
- [Semantic Role Labeling Survey 2025](https://arxiv.org/html/2502.08660v1)

### Frame Semantics
- [FrameNet](https://framenet.icsi.berkeley.edu/)
- [FrameNet at 25](http://sites.la.utexas.edu/hcb/files/2025/05/Boas-et-al-2025-FrameNet.pdf)

### Commonsense Knowledge
- [ATOMIC 2020](https://arxiv.org/pdf/2010.05953)
- [COMET](https://github.com/atcbosselut/comet-commonsense)
- [ConceptNet](https://conceptnet.io/)

### Word Sense Disambiguation
- [WordNet](https://wordnet.princeton.edu/)
- [BabelNet](https://babelnet.org/about)
- [WSD Survey](https://arxiv.org/pdf/1508.01346)

### Entity Linking
- [Wikidata](https://www.wikidata.org/)
- [Entity Disambiguation Survey](https://paperswithcode.com/task/entity-disambiguation/latest)

---

**End of Decomposition Theory Document**

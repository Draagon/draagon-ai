# Phase 1: Multi-Type Decomposition Pipeline

**Version:** 1.0.0
**Status:** Requirements
**Priority:** P1 - Core Decomposition
**Depends On:** Phase 0 (Identifiers, WSD, Entity Classification)

---

## Overview

Phase 1 builds the **decomposition pipeline** that extracts all types of implicit knowledge from text. This is the core of the implicit knowledge graphs hypothesis - that pre-decomposed knowledge provides better LLM context than raw text or traditional RAG.

**Deliverable:** A working pipeline that takes text and outputs structured implicit knowledge with weighted branches, ready for graph storage.

---

## Architecture

### Pipeline Flow

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: IDENTIFICATION                          │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────────┐  │
│  │ Entity        │  │ Word Sense    │  │ Entity Type            │  │
│  │ Extraction    │→ │ Disambiguation│→ │ Classification         │  │
│  └───────────────┘  └───────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ UniversalSemanticIdentifiers for all entities
    │
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DECOMPOSITION                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              SEMANTIC ROLE EXTRACTION                        │   │
│  │  Extract Agent, Patient, Theme, Location, Time, etc.         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              PRESUPPOSITION EXTRACTION                       │   │
│  │  Extract definite descs, factives, iteratives, etc.          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              COMMONSENSE INFERENCE                           │   │
│  │  Extract xIntent, xEffect, xReact, xAttr (Tier 1)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              NEGATION & POLARITY                             │   │
│  │  Detect negation scope, assign polarity to triples           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              TEMPORAL & MODALITY                             │   │
│  │  Extract aspect, tense, temporal refs, modal markers         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              CROSS-REFERENCE LINKING (Optional)              │   │
│  │  Link to existing memory for "again", anaphora, etc.         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BRANCH WEIGHTING                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Combine confidence from all extractors                      │   │
│  │  Generate weighted branches for ambiguous interpretations    │   │
│  │  Apply memory support boosts (if cross-ref enabled)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
DecomposedKnowledge (ready for storage)
```

---

## Requirements

### REQ-1.1: Core Data Structures

**Description:** Define the data structures for decomposed knowledge.

**Data Structures:**

```python
@dataclass
class SemanticRole:
    """A semantic role in a predicate-argument structure."""
    predicate: str                      # "forgot"
    predicate_sense: str                # "forget.v.01"
    role: str                           # "ARG0", "ARG1", "ARGM-TMP"
    filler: str                         # "Doug"
    filler_id: UniversalSemanticIdentifier | None
    span: tuple[int, int]               # Character span
    confidence: float                   # 0.0-1.0


@dataclass
class Presupposition:
    """A presupposition extracted from text."""
    content: str                        # "Doug forgot before"
    trigger_type: PresuppositionTrigger # ITERATIVE, DEFINITE_DESC, etc.
    trigger_text: str                   # "again"
    trigger_span: tuple[int, int]
    confidence: float
    cancellable: bool                   # Can be defeated in context
    entities: list[UniversalSemanticIdentifier]


class PresuppositionTrigger(str, Enum):
    """Types of presupposition triggers."""
    DEFINITE_DESC = "definite_description"    # "the X"
    FACTIVE_VERB = "factive_verb"             # "realize", "know"
    CHANGE_OF_STATE = "change_of_state"       # "stop", "start"
    ITERATIVE = "iterative"                   # "again", "another"
    TEMPORAL_CLAUSE = "temporal_clause"       # "before", "after"
    CLEFT = "cleft"                           # "It was X who..."
    COMPARATIVE = "comparative"               # "more than"
    IMPLICATIVE = "implicative"               # "manage", "forget"
    COUNTERFACTUAL = "counterfactual"         # "if X had..."
    POSSESSIVE = "possessive"                 # "X's Y"


@dataclass
class CommonsenseInference:
    """A commonsense inference (ATOMIC-style)."""
    relation: CommonsenseRelation       # xIntent, xEffect, etc.
    head: str                           # Source event
    tail: str                           # Inference
    head_entities: list[UniversalSemanticIdentifier]
    confidence: float
    source: str                         # "comet", "llm", "rule"


class CommonsenseRelation(str, Enum):
    """ATOMIC 2020 relation types (Tier 1 + key Tier 2)."""
    # Tier 1: Always extract
    X_INTENT = "xIntent"                # Why X did this
    X_EFFECT = "xEffect"                # What happens to X
    X_REACT = "xReact"                  # How X feels
    X_ATTR = "xAttr"                    # X's attributes

    # Tier 2: Extract when relevant
    X_NEED = "xNeed"                    # What X needed first
    X_WANT = "xWant"                    # What X wants after
    O_REACT = "oReact"                  # How others feel
    CAUSES = "Causes"                   # What this causes


@dataclass
class TemporalInfo:
    """Temporal and aspectual information."""
    aspect: Aspect                      # STATE, ACTIVITY, etc.
    tense: Tense                        # PAST, PRESENT, FUTURE
    reference_type: str | None          # "deictic", "calendar", etc.
    reference_value: str | None         # "yesterday", "2025-12-31"
    duration: str | None                # "for 3 hours"
    frequency: str | None               # "always", "sometimes"


class Aspect(str, Enum):
    STATE = "state"
    ACTIVITY = "activity"
    ACCOMPLISHMENT = "accomplishment"
    ACHIEVEMENT = "achievement"
    SEMELFACTIVE = "semelfactive"


class Tense(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    UNKNOWN = "unknown"


@dataclass
class ModalityInfo:
    """Modal information about certainty and obligation."""
    modal_type: ModalType               # EPISTEMIC, DEONTIC, etc.
    modal_marker: str | None            # "might", "should"
    certainty: float | None             # 0.0-1.0 for epistemic
    evidence_source: str | None         # "reported", "direct"


class ModalType(str, Enum):
    EPISTEMIC = "epistemic"             # Speaker's certainty
    DEONTIC = "deontic"                 # Obligation/permission
    DYNAMIC = "dynamic"                 # Ability/willingness
    EVIDENTIAL = "evidential"           # Source of knowledge
    NONE = "none"                       # No modal marking


@dataclass
class NegationInfo:
    """Negation and polarity information."""
    is_negated: bool
    negation_cue: str | None            # "not", "never", "no"
    negation_scope: tuple[int, int] | None  # What's negated
    polarity: Polarity


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


@dataclass
class WeightedBranch:
    """A weighted interpretation branch."""
    branch_id: str
    interpretation: str                 # Human-readable
    confidence: float                   # Base confidence
    memory_support: float               # Boost from memory (0.0 default)
    final_weight: float                 # confidence + memory_support
    supporting_evidence: list[str]      # What supports this branch


@dataclass
class DecomposedKnowledge:
    """Complete decomposed knowledge from a text."""

    # Source
    source_text: str
    source_id: str                      # UUID

    # Identified entities (from Phase 0)
    entities: list[UniversalSemanticIdentifier]

    # Extracted knowledge
    semantic_roles: list[SemanticRole]
    presuppositions: list[Presupposition]
    commonsense_inferences: list[CommonsenseInference]

    # Modifiers
    temporal: TemporalInfo | None
    modality: ModalityInfo | None
    negation: NegationInfo | None

    # Cross-references (optional, from memory linking)
    cross_references: list[CrossReference] | None

    # Weighted branches for ambiguous interpretations
    branches: list[WeightedBranch]

    # Metadata
    decomposition_timestamp: datetime
    pipeline_version: str
    config_hash: str                    # For reproducibility
```

**Acceptance Criteria:**
- [ ] All data structures defined in `decomposition/models.py`
- [ ] Enums for all categorical fields
- [ ] Serialization to/from JSON
- [ ] Unit tests for all dataclasses

### REQ-1.2: Semantic Role Extraction

**Description:** Extract predicate-argument structures from sentences.

**Approach Options (Evolvable):**

1. **LLM-Based Extraction** (Initial Implementation)
   - Use LLM with structured prompt
   - XML output format
   - Most flexible, handles edge cases

2. **spaCy + Pattern Rules** (Optional)
   - Dependency parsing
   - Rule-based role assignment
   - Faster, deterministic

3. **AllenNLP SRL Model** (Optional)
   - Pre-trained PropBank model
   - High accuracy
   - Requires separate model

**Required Functions:**

```python
async def extract_semantic_roles(
    text: str,
    entities: list[UniversalSemanticIdentifier],
    config: SemanticRoleConfig,
    llm: LLMProvider | None = None,
) -> list[SemanticRole]:
    """Extract semantic roles from text."""
```

**Acceptance Criteria:**
- [ ] Extract Agent (ARG0), Patient (ARG1) for all verbs
- [ ] Support ARGM modifiers (LOC, TMP, MNR, etc.)
- [ ] Link fillers to entity identifiers when possible
- [ ] Confidence scores for each role
- [ ] Unit tests with at least 20 test sentences

### REQ-1.3: Presupposition Extraction

**Description:** Extract presuppositions from trigger words and phrases.

**Trigger Detection:**

| Trigger | Pattern | Presupposition Template |
|---------|---------|-------------------------|
| Definite | `the + NP` | "A specific {NP} exists" |
| Factive | `{verb} that` | The complement is true |
| Change-of-state | `stop/start/continue` | Previous state existed |
| Iterative | `again/another/still` | Prior instance existed |
| Temporal | `before/after/when` | The clause occurred |
| Possessive | `X's Y` | X has Y |

**Required Functions:**

```python
async def extract_presuppositions(
    text: str,
    entities: list[UniversalSemanticIdentifier],
    config: PresuppositionConfig,
    llm: LLMProvider | None = None,
) -> list[Presupposition]:
    """Extract presuppositions from text."""

def detect_presupposition_triggers(
    text: str,
) -> list[tuple[str, PresuppositionTrigger, tuple[int, int]]]:
    """Detect potential presupposition triggers in text."""
```

**Acceptance Criteria:**
- [ ] Detect all 10 trigger types from DECOMPOSITION_THEORY.md
- [ ] Generate presupposition content text
- [ ] Confidence scores based on trigger type and context
- [ ] Mark cancellable presuppositions
- [ ] Unit tests with at least 30 test cases

### REQ-1.4: Commonsense Inference Selection

**Description:** Generate relevant commonsense inferences using tiered selection.

**Tiered Selection Strategy (from Deep Dive):**

- **Tier 1 (Always Extract):** xIntent, xEffect, xReact, xAttr
- **Tier 2 (Conditional):** xNeed, xWant, oReact, Causes
- **Tier 3 (Skip):** Physical relations, rare relations

**Required Functions:**

```python
async def generate_commonsense_inferences(
    text: str,
    entities: list[UniversalSemanticIdentifier],
    config: CommonsenseConfig,
    llm: LLMProvider | None = None,
) -> list[CommonsenseInference]:
    """Generate commonsense inferences for text."""

def filter_inferences(
    inferences: list[CommonsenseInference],
    config: CommonsenseConfig,
) -> list[CommonsenseInference]:
    """Filter and deduplicate inferences."""
```

**Acceptance Criteria:**
- [ ] Generate Tier 1 relations for all events
- [ ] Conditional Tier 2 based on context
- [ ] Quality filtering (confidence > 0.6, deduplicate)
- [ ] Evolvable tier configuration
- [ ] Unit tests with 20+ events

### REQ-1.5: Negation and Polarity Detection

**Description:** Detect negation scope and assign polarity to extracted knowledge.

**Negation Handling (from Deep Dive):**

1. Detect negation cues: "not", "never", "no", "without", "un-"
2. Determine negation scope using dependency parsing
3. Assign polarity to affected triples
4. Do NOT negate predicates directly (store polarity attribute)

**Required Functions:**

```python
async def detect_negation(
    text: str,
    config: NegationConfig,
) -> NegationInfo:
    """Detect negation and its scope in text."""

def apply_polarity(
    knowledge: DecomposedKnowledge,
    negation: NegationInfo,
) -> DecomposedKnowledge:
    """Apply polarity to decomposed knowledge."""
```

**Acceptance Criteria:**
- [ ] Detect all negation cues
- [ ] Determine scope (what is negated)
- [ ] Store polarity as attribute, not negated content
- [ ] Handle double negation
- [ ] Unit tests with 20+ negation cases

### REQ-1.6: Temporal and Modality Extraction

**Description:** Extract temporal structure and modal markers.

**Temporal Extraction:**
- Aspect (state/activity/accomplishment/achievement)
- Tense (past/present/future)
- Temporal references ("yesterday", "at 3pm")
- Duration and frequency

**Modality Extraction:**
- Epistemic markers (might, probably, certainly)
- Deontic markers (should, must, may)
- Evidential markers (apparently, reportedly)

**Required Functions:**

```python
async def extract_temporal(
    text: str,
    config: TemporalConfig,
    llm: LLMProvider | None = None,
) -> TemporalInfo:
    """Extract temporal structure from text."""

async def extract_modality(
    text: str,
    config: ModalityConfig,
    llm: LLMProvider | None = None,
) -> ModalityInfo:
    """Extract modal markers from text."""
```

**Acceptance Criteria:**
- [ ] Classify aspect for all verbs
- [ ] Detect tense
- [ ] Extract temporal references
- [ ] Detect modal markers and types
- [ ] Confidence/certainty scores
- [ ] Unit tests with 30+ cases

### REQ-1.7: Decomposition Pipeline Orchestrator

**Description:** Orchestrate all extraction stages into a unified pipeline.

**Pipeline Configuration:**

```python
@dataclass
class DecompositionConfig:
    """Configuration for the decomposition pipeline."""

    # Stage enablement
    extract_semantic_roles: bool = True
    extract_presuppositions: bool = True
    extract_commonsense: bool = True
    extract_temporal: bool = True
    extract_modality: bool = True
    detect_negation: bool = True
    enable_cross_references: bool = False  # Requires memory access

    # Stage configs (evolvable)
    semantic_role_config: SemanticRoleConfig = field(default_factory=SemanticRoleConfig)
    presupposition_config: PresuppositionConfig = field(default_factory=PresuppositionConfig)
    commonsense_config: CommonsenseConfig = field(default_factory=CommonsenseConfig)
    temporal_config: TemporalConfig = field(default_factory=TemporalConfig)
    modality_config: ModalityConfig = field(default_factory=ModalityConfig)
    negation_config: NegationConfig = field(default_factory=NegationConfig)

    # Branch weighting
    weighting_config: WeightingConfig = field(default_factory=WeightingConfig)
```

**Required Functions:**

```python
class DecompositionPipeline:
    """Main decomposition pipeline orchestrator."""

    def __init__(
        self,
        config: DecompositionConfig,
        wsd: WordSenseDisambiguator,
        classifier: EntityClassifier,
        llm: LLMProvider | None = None,
    ):
        ...

    async def decompose(
        self,
        text: str,
        context: str | None = None,
    ) -> DecomposedKnowledge:
        """Decompose text into structured implicit knowledge."""

    async def decompose_batch(
        self,
        texts: list[str],
        contexts: list[str] | None = None,
    ) -> list[DecomposedKnowledge]:
        """Decompose multiple texts (batched for efficiency)."""
```

**Acceptance Criteria:**
- [ ] Pipeline runs all enabled stages
- [ ] Stages can be individually disabled
- [ ] Results merged into DecomposedKnowledge
- [ ] Errors in one stage don't crash pipeline
- [ ] Performance metrics tracked
- [ ] Integration tests with 20+ sentences

### REQ-1.8: Branch Weighting

**Description:** Generate weighted branches for ambiguous interpretations.

**Weighting Strategy:**

```python
@dataclass
class WeightingConfig:
    """Configuration for branch weighting."""

    # Base weight factors
    wsd_confidence_weight: float = 0.3
    entity_type_confidence_weight: float = 0.2
    presupposition_confidence_weight: float = 0.15
    commonsense_confidence_weight: float = 0.15
    temporal_confidence_weight: float = 0.1
    modality_confidence_weight: float = 0.1

    # Memory support (from cross-references)
    memory_support_boost: float = 0.2
    memory_contradiction_penalty: float = 0.3

    # Branch filtering
    min_branch_weight: float = 0.3  # Drop branches below this
    max_branches: int = 5           # Keep top N branches
```

**Required Functions:**

```python
def compute_branch_weights(
    decomposed: DecomposedKnowledge,
    config: WeightingConfig,
) -> list[WeightedBranch]:
    """Compute weighted branches from decomposed knowledge."""

def merge_branch_confidences(
    confidences: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Merge confidence scores with configured weights."""
```

**Acceptance Criteria:**
- [ ] Branch weights computed from all sources
- [ ] Configurable weight factors
- [ ] Low-confidence branches filtered
- [ ] Branch explanations generated
- [ ] Unit tests for weighting logic

### REQ-1.9: Content-Aware Decomposition

**Description:** Integrate content type analysis from Phase 0 (REQ-0.8, REQ-0.9) into the decomposition pipeline.

**Rationale:** The decomposition pipeline assumes natural language input with sentence structure. Non-prose content (code, data, config) requires different handling:
- **Code**: Extract NL portions (docstrings, comments) for decomposition; extract structural knowledge (types, contracts) separately
- **Data**: Extract schema, not sentence-level knowledge
- **Config**: Extract patterns, not presuppositions

See: [DD-001-CONTENT_TYPE_AWARE_PROCESSING.md](../design-decisions/DD-001-CONTENT_TYPE_AWARE_PROCESSING.md)

**Updated Pipeline Flow:**

```
Input Content
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: CONTENT ANALYSIS (NEW)                   │
│  ┌───────────────┐                                                   │
│  │ ContentAware  │→ PROSE: Continue to full pipeline                │
│  │ WSD           │→ CODE: Extract NL → Pipeline + Structural        │
│  │               │→ DATA: Schema extraction (skip decomposition)    │
│  │               │→ CONFIG: Pattern extraction (skip decomposition) │
│  └───────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ (For PROSE and CODE-NL only)
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: IDENTIFICATION                          │
│  ... existing flow ...                                              │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DECOMPOSITION                           │
│  ... existing stages ...                                            │
└─────────────────────────────────────────────────────────────────────┘
```

**Updated DecompositionPipeline:**

```python
class DecompositionPipeline:
    def __init__(
        self,
        config: DecompositionConfig,
        wsd: WordSenseDisambiguator,  # Or ContentAwareWSD
        classifier: EntityClassifier,
        content_analyzer: ContentAnalyzer | None = None,  # NEW
        llm: LLMProvider | None = None,
    ):
        ...

    async def decompose(
        self,
        text: str,
        context: str | None = None,
    ) -> DecomposedKnowledge:
        # Step 0: Analyze content type
        if self.content_analyzer:
            analysis = await self.content_analyzer.analyze(text)

            if analysis.content_type == ContentType.DATA:
                return self._decompose_data(analysis)
            elif analysis.content_type == ContentType.CONFIG:
                return self._decompose_config(analysis)
            elif analysis.content_type == ContentType.CODE:
                # Get NL portions for decomposition
                nl_text = analysis.get_natural_language_text()
                if nl_text:
                    decomposed = await self._decompose_prose(nl_text)
                    # Merge with structural knowledge
                    decomposed.structural_knowledge = analysis.structural_knowledge
                    return decomposed
                return self._decompose_code_only(analysis)

        # Default: treat as prose
        return await self._decompose_prose(text)
```

**New Output Type for Non-Prose:**

```python
@dataclass
class StructuralDecomposition:
    """Decomposition result for non-prose content."""

    content_type: ContentType
    source_text: str

    # For CODE
    type_relationships: list[TypeRelationship]  # "BankAccount is-a Account"
    function_contracts: list[FunctionContract]  # "process_transaction takes Account"
    import_graph: dict[str, list[str]]          # Dependencies

    # For DATA
    schema: DataSchema                          # Column names, types
    relationships: list[DataRelationship]       # Foreign keys, etc.
    constraints: list[DataConstraint]           # Value ranges, patterns

    # For CONFIG
    key_hierarchy: dict                         # Nested structure
    common_patterns: list[str]                  # "database config", "server config"

    # NL portions extracted (may be empty)
    extracted_nl: list[str]
    nl_decomposition: DecomposedKnowledge | None  # If NL was found
```

**Acceptance Criteria:**
- [ ] Content type analysis integrated at pipeline entry
- [ ] Prose routes through existing pipeline unchanged
- [ ] Code extracts NL for decomposition + structural knowledge
- [ ] Data produces StructuralDecomposition with schema
- [ ] Config produces StructuralDecomposition with patterns
- [ ] Mixed content handles each portion appropriately
- [ ] Integration tests for each content type

### REQ-1.10: Structural Knowledge Extraction

**Description:** Extract structural (non-linguistic) knowledge from code and data.

**For Code:**
```python
@dataclass
class TypeRelationship:
    subject: str        # "BankAccount"
    relationship: str   # "is_a", "has_method", "uses"
    object: str         # "Account", "deposit", "Transaction"
    confidence: float

@dataclass
class FunctionContract:
    name: str
    parameters: list[tuple[str, str]]  # [(name, type), ...]
    return_type: str
    raises: list[str]
    docstring: str | None
```

**For Data:**
```python
@dataclass
class DataSchema:
    columns: list[ColumnInfo]
    primary_key: str | None
    inferred_types: dict[str, str]

@dataclass
class DataRelationship:
    column: str
    relationship: str  # "references", "contains", "correlates_with"
    target: str
    confidence: float
```

**Acceptance Criteria:**
- [ ] Type extraction from Python code
- [ ] Function contract extraction
- [ ] CSV schema inference
- [ ] JSON structure extraction
- [ ] Unit tests for each extraction type

---

## Evolution Framework Requirements

### REQ-1.E1: Evolvable Extraction Configs

**Description:** All extraction parameters must be evolvable.

**Evolvable Parameters:**

```python
# Semantic Role Config
lesk_context_window: int = 10
role_confidence_threshold: float = 0.5
max_roles_per_predicate: int = 5

# Presupposition Config
trigger_confidence_threshold: float = 0.4
include_weak_triggers: bool = True
max_presuppositions: int = 10

# Commonsense Config
tier1_relations: list[str] = ["xIntent", "xEffect", "xReact", "xAttr"]
tier2_threshold: float = 0.6
inference_confidence_min: float = 0.5
max_inferences_per_relation: int = 3

# Weighting Config
(all fields in WeightingConfig)
```

**Acceptance Criteria:**
- [ ] All parameters externalized to config dataclasses
- [ ] Configs support mutation for evolution
- [ ] Configs serializable to JSON
- [ ] Unit tests with different configs

### REQ-1.E2: Decomposition Quality Metrics

**Description:** Define metrics for evaluating decomposition quality.

**Metrics:**

```python
@dataclass
class DecompositionMetrics:
    """Metrics for decomposition quality evaluation."""

    # Completeness
    entities_extracted: int
    roles_extracted: int
    presuppositions_extracted: int
    inferences_extracted: int

    # Quality
    avg_confidence: float
    low_confidence_count: int  # Items below threshold

    # Coverage
    sentence_coverage: float   # % of sentence tokens in extracted items
    entity_coverage: float     # % of entities with identifiers

    # Efficiency
    extraction_time_ms: float
    llm_calls: int

    # For evolution fitness
    def fitness_score(self, weights: dict[str, float]) -> float:
        """Compute weighted fitness score."""
```

**Acceptance Criteria:**
- [ ] All metrics computed for each decomposition
- [ ] Fitness function defined
- [ ] Metrics logged for analysis
- [ ] Dashboard-ready output format

---

## Test Requirements

### Test Categories

1. **Unit Tests** - Individual extractors
2. **Integration Tests** - Full pipeline
3. **Accuracy Tests** - Against labeled data
4. **Evolution Tests** - Config mutation and fitness

### Test Data Requirements

**Decomposition Test Cases (minimum 100):**

| Category | Count | Examples |
|----------|-------|----------|
| Simple sentences | 20 | "Doug went to the store" |
| Presupposition triggers | 30 | One per trigger type × 3 |
| Commonsense events | 20 | Social interactions |
| Negation | 15 | Various negation patterns |
| Temporal/Modal | 15 | Various tense/aspect/modality |

### Acceptance Criteria

- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Extraction coverage > 80% on test set
- [ ] Average confidence > 0.6
- [ ] Pipeline latency < 2s per sentence (without LLM)
- [ ] Pipeline latency < 5s per sentence (with LLM)

---

## Implementation Plan

### Stage 1: Core Data Structures (1-2 days)
- [ ] Define all dataclasses in `decomposition/models.py`
- [ ] Define all enums
- [ ] Serialization methods
- [ ] Unit tests

### Stage 2: Presupposition Extraction (2-3 days)
- [ ] Trigger detection patterns
- [ ] LLM-based presupposition generation
- [ ] Confidence scoring
- [ ] Unit tests

### Stage 3: Semantic Role Extraction (2-3 days)
- [ ] LLM-based role extraction
- [ ] Entity linking for fillers
- [ ] Unit tests

### Stage 4: Commonsense Inference (2-3 days)
- [ ] Tiered relation selection
- [ ] LLM-based inference generation
- [ ] Quality filtering
- [ ] Unit tests

### Stage 5: Negation & Polarity (1-2 days)
- [ ] Negation cue detection
- [ ] Scope determination
- [ ] Polarity application
- [ ] Unit tests

### Stage 6: Temporal & Modality (2-3 days)
- [ ] Aspect classification
- [ ] Tense detection
- [ ] Modal marker detection
- [ ] Unit tests

### Stage 7: Pipeline Orchestrator (2-3 days)
- [ ] Pipeline class
- [ ] Stage orchestration
- [ ] Error handling
- [ ] Metrics collection
- [ ] Integration tests

### Stage 8: Branch Weighting (1-2 days)
- [ ] Weight computation
- [ ] Branch filtering
- [ ] Explanation generation
- [ ] Unit tests

### Stage 9: Evolution Framework (2-3 days)
- [ ] Config mutation strategies
- [ ] Fitness function
- [ ] Train/holdout evaluation
- [ ] Evolution tests

---

## Success Criteria

Phase 1 is complete when:

1. **Functional:**
   - Pipeline decomposes any English sentence
   - All extraction types working
   - Weighted branches generated

2. **Quality:**
   - Extraction coverage > 80%
   - Average confidence > 0.6
   - All tests passing

3. **Performance:**
   - < 2s latency without LLM
   - < 5s latency with LLM

4. **Evolution-Ready:**
   - All parameters evolvable
   - Fitness metrics defined
   - Mutation strategies documented

---

## Dependencies

**Python Packages:**
- Phase 0 modules (identifiers, wsd, entity_classifier)
- `spacy` (optional, for dependency parsing)
- `anthropic` or equivalent (LLM)
- `pytest` (testing)

**From draagon-ai:**
- `LLMProvider` protocol
- XML parsing utilities

**External (Optional):**
- COMET model for commonsense (future)
- AllenNLP SRL model (future)

---

## File Structure

```
src/
├── __init__.py              # Updated with Phase 1 exports
├── identifiers.py           # Phase 0
├── wsd.py                   # Phase 0
├── entity_classifier.py     # Phase 0
└── decomposition/
    ├── __init__.py
    ├── models.py            # Data structures
    ├── semantic_roles.py    # REQ-1.2
    ├── presuppositions.py   # REQ-1.3
    ├── commonsense.py       # REQ-1.4
    ├── negation.py          # REQ-1.5
    ├── temporal.py          # REQ-1.6
    ├── modality.py          # REQ-1.6
    ├── pipeline.py          # REQ-1.7
    ├── weighting.py         # REQ-1.8
    └── config.py            # All configs

tests/
├── decomposition/
│   ├── test_models.py
│   ├── test_semantic_roles.py
│   ├── test_presuppositions.py
│   ├── test_commonsense.py
│   ├── test_negation.py
│   ├── test_temporal.py
│   ├── test_modality.py
│   ├── test_pipeline.py
│   └── test_weighting.py
└── conftest.py              # Updated with Phase 1 fixtures
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM cost | High cost for extraction | Batch calls, caching, rule-based fallbacks |
| Extraction quality | Poor downstream retrieval | Iterative improvement via evolution |
| Pipeline latency | Slow user experience | Parallel extraction, caching |
| Overfitting | Poor generalization | Train/holdout split, diversity checks |
| Scope creep | Delayed delivery | Strict stage-by-stage implementation |

---

**End of Phase 1 Requirements**

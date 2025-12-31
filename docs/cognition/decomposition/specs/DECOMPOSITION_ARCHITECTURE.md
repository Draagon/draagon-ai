# Decomposition Pipeline Architecture

**Version:** 1.0.0
**Status:** Design Specification
**Related:** PHASE_1_DECOMPOSITION.md

---

## Overview

This document details the architectural design for the Phase 1 decomposition pipeline. It specifies how components interact, data flows through the system, and how the pipeline integrates with Phase 0 (identifiers, WSD, entity classification).

---

## System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            IMPLICIT KNOWLEDGE SYSTEM                         │
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Input Layer    │────▶│  Decomposition   │────▶│  Storage Layer   │    │
│  │                  │     │     Pipeline     │     │   (Phase 2)      │    │
│  │  - Raw text      │     │   (Phase 0+1)    │     │                  │    │
│  │  - Context       │     │                  │     │  - Graph DB      │    │
│  │  - Metadata      │     │                  │     │  - Vector store  │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│                           ┌──────────────────┐                              │
│                           │  Memory Layer    │                              │
│                           │  (Optional)      │                              │
│                           │                  │                              │
│                           │  - Cross-refs    │                              │
│                           │  - 4-layer mem   │                              │
│                           └──────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Pipeline Core

```python
class DecompositionPipeline:
    """Main orchestrator for the decomposition pipeline.

    Coordinates all extraction stages and manages data flow.
    """

    # Dependencies (injected)
    config: DecompositionConfig
    wsd: WordSenseDisambiguator          # From Phase 0
    classifier: EntityClassifier          # From Phase 0
    llm: LLMProvider | None

    # Extractors (created based on config)
    _semantic_role_extractor: SemanticRoleExtractor | None
    _presupposition_extractor: PresuppositionExtractor | None
    _commonsense_extractor: CommonsenseExtractor | None
    _temporal_extractor: TemporalExtractor | None
    _modality_extractor: ModalityExtractor | None
    _negation_detector: NegationDetector | None
    _branch_weighter: BranchWeighter

    # Metrics
    _metrics: PipelineMetrics
```

### 2. Extractor Interface

All extractors implement a common interface:

```python
class ExtractorProtocol(Protocol):
    """Common interface for all extractors."""

    async def extract(
        self,
        text: str,
        entities: list[UniversalSemanticIdentifier],
        context: ExtractionContext,
    ) -> ExtractorResult:
        """Extract knowledge from text."""
        ...

    @property
    def name(self) -> str:
        """Extractor name for logging/metrics."""
        ...


@dataclass
class ExtractionContext:
    """Context passed to all extractors."""

    source_text: str
    sentence_index: int
    document_id: str | None
    prior_extractions: dict[str, Any]  # Results from earlier stages
    llm: LLMProvider | None
    config: Any  # Stage-specific config


@dataclass
class ExtractorResult:
    """Result from an extractor."""

    success: bool
    data: Any  # Stage-specific data
    confidence: float
    errors: list[str]
    metrics: dict[str, Any]
```

### 3. Stage Implementations

#### 3.1 Semantic Role Extractor

```python
class SemanticRoleExtractor:
    """Extracts predicate-argument structures."""

    async def extract(
        self,
        text: str,
        entities: list[UniversalSemanticIdentifier],
        context: ExtractionContext,
    ) -> list[SemanticRole]:
        """
        Pipeline:
        1. Identify predicates (verbs)
        2. For each predicate:
           a. Disambiguate sense (using WSD)
           b. Extract arguments (ARG0, ARG1, etc.)
           c. Link arguments to entity identifiers
           d. Score confidence
        """
```

**LLM Prompt Template:**

```xml
<prompt>
Analyze the semantic roles in this sentence:

Sentence: "{text}"

For each verb/predicate, identify:
- The predicate and its sense
- ARG0 (agent/doer)
- ARG1 (patient/theme)
- ARG2 (recipient/beneficiary) if present
- ARGM-LOC (location) if present
- ARGM-TMP (time) if present
- ARGM-MNR (manner) if present

Respond in XML:
<roles>
  <predicate>
    <verb>{verb}</verb>
    <sense>{wordnet_sense}</sense>
    <arg role="ARG0" confidence="0.9">{filler}</arg>
    <arg role="ARG1" confidence="0.85">{filler}</arg>
    ...
  </predicate>
</roles>
</prompt>
```

#### 3.2 Presupposition Extractor

```python
class PresuppositionExtractor:
    """Extracts presuppositions from trigger patterns."""

    # Trigger patterns (regex)
    TRIGGER_PATTERNS = {
        PresuppositionTrigger.DEFINITE_DESC: r'\bthe\s+(\w+(?:\s+\w+)?)\b',
        PresuppositionTrigger.ITERATIVE: r'\b(again|another|still|anymore)\b',
        PresuppositionTrigger.CHANGE_OF_STATE: r'\b(stop(?:ped)?|start(?:ed)?|continue[ds]?|begin|began|quit|resume[ds]?)\b',
        PresuppositionTrigger.FACTIVE_VERB: r'\b(realize[ds]?|know[s]?|knew|regret[s]?|remember[s]?|forget|forgot)\b',
        PresuppositionTrigger.POSSESSIVE: r"(\w+)'s\s+(\w+)",
        # ... more patterns
    }

    async def extract(
        self,
        text: str,
        entities: list[UniversalSemanticIdentifier],
        context: ExtractionContext,
    ) -> list[Presupposition]:
        """
        Pipeline:
        1. Detect triggers using patterns
        2. For each trigger:
           a. Determine trigger type
           b. Generate presupposition content (LLM if complex)
           c. Link entities mentioned
           d. Score confidence
           e. Determine if cancellable
        """
```

**Presupposition Templates:**

```python
PRESUPPOSITION_TEMPLATES = {
    PresuppositionTrigger.DEFINITE_DESC: "A specific {entity} exists that is contextually identifiable",
    PresuppositionTrigger.ITERATIVE: "{entity} {action} before",
    PresuppositionTrigger.CHANGE_OF_STATE: "{entity} was previously in state of {state}",
    PresuppositionTrigger.FACTIVE_VERB: "{complement} is true",
    PresuppositionTrigger.POSSESSIVE: "{possessor} has {possessed}",
}
```

#### 3.3 Commonsense Extractor

```python
class CommonsenseExtractor:
    """Generates commonsense inferences using tiered selection."""

    # Tier configuration
    TIER_1_RELATIONS = ["xIntent", "xEffect", "xReact", "xAttr"]
    TIER_2_RELATIONS = ["xNeed", "xWant", "oReact", "Causes"]

    async def extract(
        self,
        text: str,
        entities: list[UniversalSemanticIdentifier],
        context: ExtractionContext,
    ) -> list[CommonsenseInference]:
        """
        Pipeline:
        1. Identify events in text
        2. For each event:
           a. Generate Tier 1 inferences (always)
           b. Conditionally generate Tier 2 if:
              - Entity is INSTANCE (not generic)
              - Event involves social interaction
              - Prior context suggests relevance
           c. Filter by confidence threshold
           d. Deduplicate similar inferences
        """
```

**Tiered Selection Logic:**

```python
def should_extract_tier2(
    event: str,
    entities: list[UniversalSemanticIdentifier],
    context: ExtractionContext,
) -> bool:
    """Determine if Tier 2 relations should be extracted."""

    # Check if any entity is a specific instance (not generic)
    has_instance = any(e.entity_type == EntityType.INSTANCE for e in entities)

    # Check for social interaction keywords
    social_keywords = {"told", "asked", "gave", "helped", "said", "met"}
    has_social = any(kw in event.lower() for kw in social_keywords)

    # Check confidence threshold from config
    config = context.config
    threshold = config.tier2_threshold

    return has_instance and (has_social or context.prior_extractions.get("is_dialogue"))
```

#### 3.4 Negation Detector

```python
class NegationDetector:
    """Detects negation cues and determines scope."""

    NEGATION_CUES = {
        "not", "n't", "never", "no", "none", "nobody", "nothing",
        "neither", "nor", "without", "hardly", "barely", "scarcely",
    }

    NEGATIVE_PREFIXES = ["un", "in", "im", "non", "dis", "a"]

    async def detect(
        self,
        text: str,
        context: ExtractionContext,
    ) -> NegationInfo:
        """
        Pipeline:
        1. Detect negation cues
        2. Determine scope using dependency parse (or LLM)
        3. Assign polarity
        """
```

**Scope Determination:**

```python
def determine_negation_scope(
    text: str,
    cue_position: int,
    cue: str,
) -> tuple[int, int]:
    """
    Determine what the negation applies to.

    Strategy:
    1. If cue is auxiliary-attached ("don't"), scope is the main verb phrase
    2. If cue is adverbial ("never"), scope is until end of clause
    3. If cue is determiner ("no"), scope is the noun phrase
    """
```

#### 3.5 Temporal Extractor

```python
class TemporalExtractor:
    """Extracts temporal and aspectual information."""

    ASPECT_PATTERNS = {
        Aspect.STATE: ["is", "are", "was", "were", "has", "have", "likes", "knows"],
        Aspect.ACTIVITY: ["running", "walking", "eating", "playing"],
        Aspect.ACCOMPLISHMENT: ["built", "wrote", "created", "finished"],
        Aspect.ACHIEVEMENT: ["noticed", "found", "reached", "won"],
    }

    TEMPORAL_MARKERS = {
        "deictic": ["yesterday", "today", "tomorrow", "now", "then"],
        "calendar": [r"\d{4}-\d{2}-\d{2}", r"\d{1,2}:\d{2}"],
        "relative": ["before", "after", "during", "while", "when"],
    }

    async def extract(
        self,
        text: str,
        context: ExtractionContext,
    ) -> TemporalInfo:
        """
        Pipeline:
        1. Detect tense from verb morphology
        2. Classify aspect
        3. Extract temporal references
        4. Determine duration/frequency if present
        """
```

#### 3.6 Modality Extractor

```python
class ModalityExtractor:
    """Extracts modal markers and their meanings."""

    MODAL_MARKERS = {
        ModalType.EPISTEMIC: {
            "certain": ["definitely", "certainly", "must"],
            "probable": ["probably", "likely", "should"],
            "possible": ["might", "may", "could", "possibly"],
        },
        ModalType.DEONTIC: {
            "required": ["must", "have to", "need to"],
            "recommended": ["should", "ought to"],
            "permitted": ["may", "can", "allowed to"],
        },
        ModalType.EVIDENTIAL: {
            "direct": ["saw", "heard", "witnessed"],
            "reported": ["apparently", "reportedly", "allegedly"],
            "inferred": ["seems", "appears", "looks like"],
        },
    }

    async def extract(
        self,
        text: str,
        context: ExtractionContext,
    ) -> ModalityInfo:
        """
        Pipeline:
        1. Detect modal markers
        2. Classify modal type
        3. For epistemic: compute certainty level
        4. For evidential: determine source
        """
```

### 4. Branch Weighter

```python
class BranchWeighter:
    """Computes weighted branches from decomposed knowledge."""

    def compute_weights(
        self,
        decomposed: DecomposedKnowledge,
        config: WeightingConfig,
    ) -> list[WeightedBranch]:
        """
        Algorithm:
        1. Identify ambiguity points:
           - Multiple entity interpretations
           - Multiple sense options
           - Uncertain presuppositions
        2. For each interpretation branch:
           a. Aggregate confidence from all sources
           b. Apply configured weights
           c. Add memory support if available
        3. Filter branches below threshold
        4. Sort by final weight
        5. Generate explanations
        """

    def _aggregate_confidences(
        self,
        decomposed: DecomposedKnowledge,
        config: WeightingConfig,
    ) -> dict[str, float]:
        """
        Weighted aggregation:
        - WSD confidence × wsd_weight
        - Entity type confidence × entity_weight
        - Presupposition confidence × presup_weight
        - Commonsense confidence × commonsense_weight
        - Temporal confidence × temporal_weight
        - Modality confidence × modality_weight
        """
```

---

## Data Flow

### Sequential Execution

```
Input: text="Doug apparently forgot the meeting again"

Stage 1: Entity Extraction (Phase 0)
├── Extract mentions: ["Doug", "the meeting"]
├── WSD: N/A for proper noun, "meeting.n.01" for meeting
└── Classify: Doug=INSTANCE, the meeting=INSTANCE (specific)

Stage 2: Semantic Role Extraction
├── Predicate: "forgot" (forget.v.01)
├── ARG0: "Doug" (Agent)
└── ARG1: "the meeting" (Theme)

Stage 3: Presupposition Extraction
├── Trigger: "the meeting" → DEFINITE_DESC
│   └── Presup: "A specific meeting exists"
├── Trigger: "again" → ITERATIVE
│   └── Presup: "Doug forgot before"
└── Trigger: "forgot" → IMPLICATIVE
    └── Presup: "Doug was supposed to remember"

Stage 4: Commonsense Inference
├── xIntent: (skipped - not intentional action)
├── xEffect: "Doug missed the meeting"
├── xReact: "embarrassed", "guilty"
└── xAttr: "forgetful"

Stage 5: Negation Detection
└── No negation detected

Stage 6: Temporal Extraction
├── Tense: PAST
├── Aspect: ACHIEVEMENT
└── Iteration: REPEATED (from "again")

Stage 7: Modality Extraction
├── Type: EVIDENTIAL
├── Marker: "apparently"
├── Source: REPORTED
└── Certainty: 0.7

Stage 8: Branch Weighting
├── Branch 1: Doug forgot (high confidence)
│   └── Weight: 0.85
└── Branch 2: Doug didn't actually forget (from "apparently")
    └── Weight: 0.15 (hedged by evidential)

Output: DecomposedKnowledge
```

### Parallel Execution (Future Optimization)

Some stages can run in parallel once entities are identified:

```
                    Entity Extraction
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  Semantic Roles   Presuppositions   Negation
         │                │                │
         └────────────────┼────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  Commonsense      Temporal/Modal    Cross-refs
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
                   Branch Weighting
```

---

## LLM Integration Strategy

### When to Use LLM

| Stage | Use LLM When |
|-------|--------------|
| Semantic Roles | Complex sentences, unclear structures |
| Presuppositions | Complex triggers, content generation |
| Commonsense | Always (no rule-based alternative) |
| Negation | Scope is ambiguous |
| Temporal | Multiple temporal expressions |
| Modality | Combined modals |

### LLM Call Batching

```python
class LLMBatcher:
    """Batch LLM calls for efficiency."""

    def __init__(self, llm: LLMProvider, batch_size: int = 5):
        self.llm = llm
        self.batch_size = batch_size
        self._pending: list[tuple[str, asyncio.Future]] = []

    async def request(self, prompt: str) -> str:
        """Queue a request, batch when ready."""
        future = asyncio.Future()
        self._pending.append((prompt, future))

        if len(self._pending) >= self.batch_size:
            await self._flush()

        return await future

    async def _flush(self):
        """Execute batched requests."""
        # Combine prompts into single multi-part request
        # Parse responses back to individual results
```

### Cost Tracking

```python
@dataclass
class LLMCostTracker:
    """Track LLM usage and costs."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, input_tokens: int, output_tokens: int):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on typical pricing."""
        # Adjust based on model
        input_cost = self.input_tokens * 0.000003  # $3/M tokens
        output_cost = self.output_tokens * 0.000015  # $15/M tokens
        return input_cost + output_cost
```

---

## Error Handling

### Stage-Level Recovery

```python
class StageExecutor:
    """Execute a stage with error handling."""

    async def execute(
        self,
        extractor: ExtractorProtocol,
        text: str,
        entities: list[UniversalSemanticIdentifier],
        context: ExtractionContext,
    ) -> ExtractorResult:
        try:
            result = await asyncio.wait_for(
                extractor.extract(text, entities, context),
                timeout=context.config.timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            return ExtractorResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"{extractor.name} timed out"],
                metrics={"timeout": True},
            )
        except Exception as e:
            return ExtractorResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"{extractor.name} failed: {str(e)}"],
                metrics={"error": str(e)},
            )
```

### Graceful Degradation

```python
def merge_results(
    results: dict[str, ExtractorResult],
) -> DecomposedKnowledge:
    """Merge results, handling partial failures gracefully."""

    # Start with empty decomposition
    decomposed = DecomposedKnowledge(
        source_text=source_text,
        source_id=str(uuid.uuid4()),
        entities=entities,
        semantic_roles=[],
        presuppositions=[],
        commonsense_inferences=[],
        # ... etc
    )

    # Add successful extractions
    if results.get("semantic_roles", {}).success:
        decomposed.semantic_roles = results["semantic_roles"].data

    if results.get("presuppositions", {}).success:
        decomposed.presuppositions = results["presuppositions"].data

    # ... etc

    return decomposed
```

---

## Configuration Schema

### Master Configuration

```python
@dataclass
class DecompositionConfig:
    """Master configuration for decomposition pipeline."""

    # Pipeline behavior
    parallel_execution: bool = False  # Future optimization
    fail_fast: bool = False          # Stop on first error
    timeout_seconds: float = 30.0    # Per-stage timeout

    # Stage enablement
    stages: StageEnablement = field(default_factory=StageEnablement)

    # Stage-specific configs
    semantic_role: SemanticRoleConfig = field(default_factory=SemanticRoleConfig)
    presupposition: PresuppositionConfig = field(default_factory=PresuppositionConfig)
    commonsense: CommonsenseConfig = field(default_factory=CommonsenseConfig)
    negation: NegationConfig = field(default_factory=NegationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)

    # LLM settings
    llm_batch_size: int = 5
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000


@dataclass
class StageEnablement:
    """Enable/disable individual stages."""

    semantic_roles: bool = True
    presuppositions: bool = True
    commonsense: bool = True
    negation: bool = True
    temporal: bool = True
    modality: bool = True
    cross_references: bool = False  # Requires memory
```

### Per-Stage Configurations

```python
@dataclass
class SemanticRoleConfig:
    """Configuration for semantic role extraction."""
    use_llm: bool = True
    confidence_threshold: float = 0.5
    max_roles_per_predicate: int = 6
    include_modifiers: bool = True  # ARGM-*


@dataclass
class PresuppositionConfig:
    """Configuration for presupposition extraction."""
    use_llm_for_content: bool = True
    confidence_threshold: float = 0.4
    max_presuppositions: int = 10
    include_weak_triggers: bool = True
    triggers_enabled: list[str] = field(default_factory=lambda: [
        "definite_description",
        "factive_verb",
        "change_of_state",
        "iterative",
        "possessive",
    ])


@dataclass
class CommonsenseConfig:
    """Configuration for commonsense inference."""
    tier1_relations: list[str] = field(default_factory=lambda: [
        "xIntent", "xEffect", "xReact", "xAttr"
    ])
    tier2_relations: list[str] = field(default_factory=lambda: [
        "xNeed", "xWant", "oReact", "Causes"
    ])
    tier2_threshold: float = 0.6
    inference_confidence_min: float = 0.5
    max_inferences_per_relation: int = 3
    deduplicate: bool = True


@dataclass
class NegationConfig:
    """Configuration for negation detection."""
    detect_prefixes: bool = True  # un-, in-, etc.
    scope_method: str = "heuristic"  # or "llm"


@dataclass
class TemporalConfig:
    """Configuration for temporal extraction."""
    extract_aspect: bool = True
    extract_tense: bool = True
    extract_references: bool = True
    normalize_dates: bool = True  # Convert to ISO format


@dataclass
class ModalityConfig:
    """Configuration for modality extraction."""
    extract_epistemic: bool = True
    extract_deontic: bool = True
    extract_evidential: bool = True
    compute_certainty: bool = True


@dataclass
class WeightingConfig:
    """Configuration for branch weighting."""
    wsd_weight: float = 0.25
    entity_type_weight: float = 0.20
    presupposition_weight: float = 0.15
    commonsense_weight: float = 0.15
    temporal_weight: float = 0.10
    modality_weight: float = 0.15

    memory_support_boost: float = 0.2
    memory_contradiction_penalty: float = 0.3

    min_branch_weight: float = 0.3
    max_branches: int = 5
```

---

## Metrics and Monitoring

### Pipeline Metrics

```python
@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""

    # Timing
    total_time_ms: float = 0.0
    stage_times_ms: dict[str, float] = field(default_factory=dict)

    # Extraction counts
    entities_extracted: int = 0
    roles_extracted: int = 0
    presuppositions_extracted: int = 0
    inferences_extracted: int = 0

    # Quality
    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    extraction_errors: int = 0

    # LLM usage
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0

    # Branch stats
    branches_generated: int = 0
    branches_filtered: int = 0

    def to_dict(self) -> dict:
        """Serialize for logging/dashboard."""
        return asdict(self)
```

### Logging Strategy

```python
import structlog

logger = structlog.get_logger()

class DecompositionPipeline:

    async def decompose(self, text: str) -> DecomposedKnowledge:
        log = logger.bind(
            text_preview=text[:50],
            pipeline_version=self.config.version,
        )

        log.info("decomposition_started")

        for stage_name, extractor in self._extractors.items():
            stage_log = log.bind(stage=stage_name)
            stage_log.debug("stage_started")

            result = await self._execute_stage(extractor, text)

            stage_log.info(
                "stage_completed",
                success=result.success,
                confidence=result.confidence,
                item_count=len(result.data) if result.data else 0,
            )

        log.info(
            "decomposition_completed",
            metrics=self._metrics.to_dict(),
        )
```

---

## Testing Strategy

### Unit Test Structure

```python
# tests/decomposition/test_semantic_roles.py

class TestSemanticRoleExtractor:

    @pytest.fixture
    def extractor(self, mock_llm):
        config = SemanticRoleConfig()
        return SemanticRoleExtractor(config, llm=mock_llm)

    async def test_simple_transitive(self, extractor):
        """Test: Doug ate pizza."""
        text = "Doug ate pizza"
        entities = [...]

        roles = await extractor.extract(text, entities, context)

        assert len(roles) == 1
        assert roles[0].predicate == "ate"
        assert roles[0].role == "ARG0"
        assert roles[0].filler == "Doug"

    async def test_ditransitive(self, extractor):
        """Test: Doug gave Sarah flowers."""
        # ...

    @pytest.mark.parametrize("text,expected_roles", [
        ("Doug ran", [("ran", "ARG0", "Doug")]),
        ("The ball was kicked by Doug", [("kicked", "ARG0", "Doug"), ("kicked", "ARG1", "ball")]),
        # ... more cases
    ])
    async def test_various_structures(self, extractor, text, expected_roles):
        # ...
```

### Integration Test Structure

```python
# tests/decomposition/test_pipeline_integration.py

class TestPipelineIntegration:

    @pytest.fixture
    def pipeline(self, mock_llm, wsd, classifier):
        config = DecompositionConfig()
        return DecompositionPipeline(config, wsd, classifier, mock_llm)

    async def test_full_decomposition(self, pipeline):
        """Test complete decomposition of a sentence."""
        text = "Doug apparently forgot the meeting again"

        result = await pipeline.decompose(text)

        # Check entities
        assert len(result.entities) >= 2

        # Check presuppositions
        presup_contents = [p.content for p in result.presuppositions]
        assert any("forgot before" in p.lower() for p in presup_contents)

        # Check modality
        assert result.modality.modal_type == ModalType.EVIDENTIAL

        # Check branches
        assert len(result.branches) >= 1
        assert all(b.final_weight > 0 for b in result.branches)
```

---

## Future Considerations

### Phase 2 Integration Points

- **Storage Interface:** DecomposedKnowledge → Graph nodes/edges
- **Cross-Reference Linking:** Memory query interface
- **Retrieval Preparation:** Index structures for fast lookup

### Performance Optimizations

1. **Caching:** Cache LLM results for similar sentences
2. **Batching:** Process multiple sentences together
3. **Parallel Stages:** Run independent stages concurrently
4. **Model Selection:** Use smaller models for simple extractions

### Extensibility

- **Custom Extractors:** Plugin architecture for new extraction types
- **Domain Adapters:** Domain-specific presupposition patterns
- **Language Support:** Abstract text processing for multilingual

---

**End of Architecture Document**

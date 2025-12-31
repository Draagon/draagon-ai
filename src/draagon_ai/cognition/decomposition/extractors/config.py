"""Evolvable configuration for the decomposition pipeline.

All configuration parameters are designed to be:
1. Evolvable via genetic algorithms
2. Serializable for storage/reproduction
3. Validated for sensible ranges

Example:
    >>> from .config import DecompositionConfig, PresuppositionConfig
    >>>
    >>> # Create with defaults
    >>> config = DecompositionConfig()
    >>>
    >>> # Customize specific stages
    >>> config.presupposition.confidence_threshold = 0.6
    >>> config.commonsense.tier2_threshold = 0.7
    >>>
    >>> # Serialize for evolution
    >>> config_dict = config.to_dict()
    >>> mutated = DecompositionConfig.from_dict(config_dict)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Stage Enablement
# =============================================================================


@dataclass
class StageEnablement:
    """Enable/disable individual pipeline stages.

    Useful for:
    - Testing individual stages
    - Performance optimization (disable unused stages)
    - Evolution (test impact of stages)
    """

    semantic_roles: bool = True
    """Extract predicate-argument structures."""

    presuppositions: bool = True
    """Extract presuppositions from trigger patterns."""

    commonsense: bool = True
    """Generate commonsense inferences."""

    negation: bool = True
    """Detect negation and polarity."""

    temporal: bool = True
    """Extract temporal/aspectual information."""

    modality: bool = True
    """Extract modal markers."""

    cross_references: bool = False
    """Link to existing memory (requires memory provider)."""

    def to_dict(self) -> dict[str, bool]:
        """Serialize to dictionary."""
        return {
            "semantic_roles": self.semantic_roles,
            "presuppositions": self.presuppositions,
            "commonsense": self.commonsense,
            "negation": self.negation,
            "temporal": self.temporal,
            "modality": self.modality,
            "cross_references": self.cross_references,
        }

    @classmethod
    def from_dict(cls, data: dict[str, bool]) -> StageEnablement:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# =============================================================================
# Per-Stage Configurations
# =============================================================================


@dataclass
class SemanticRoleConfig:
    """Configuration for semantic role extraction.

    Evolvable Parameters:
    - use_llm: Whether to use LLM for extraction
    - confidence_threshold: Minimum confidence to accept a role
    - max_roles_per_predicate: Maximum roles to extract per verb
    - include_modifiers: Whether to include ARGM-* roles
    """

    use_llm: bool = True
    """Use LLM for role extraction (more accurate but slower)."""

    confidence_threshold: float = 0.5
    """Minimum confidence to include a role."""

    max_roles_per_predicate: int = 6
    """Maximum semantic roles per predicate."""

    include_modifiers: bool = True
    """Include modifier roles (ARGM-LOC, ARGM-TMP, etc.)."""

    llm_temperature: float = 0.1
    """Temperature for LLM calls."""

    prompt_template: str = """Analyze the semantic roles in this sentence:

Sentence: "{text}"

For each verb/predicate, identify:
- The predicate and its WordNet sense (e.g., forget.v.01)
- ARG0 (agent/doer)
- ARG1 (patient/theme)
- ARG2 (recipient/beneficiary) if present
- ARGM-LOC (location) if present
- ARGM-TMP (time) if present
- ARGM-MNR (manner) if present

Respond in XML:
<roles>
  <predicate>
    <verb>the verb</verb>
    <sense>wordnet.sense.id</sense>
    <args>
      <arg role="ARG0" confidence="0.9">filler text</arg>
      <arg role="ARG1" confidence="0.85">filler text</arg>
    </args>
  </predicate>
</roles>"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "use_llm": self.use_llm,
            "confidence_threshold": self.confidence_threshold,
            "max_roles_per_predicate": self.max_roles_per_predicate,
            "include_modifiers": self.include_modifiers,
            "llm_temperature": self.llm_temperature,
            "prompt_template": self.prompt_template,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticRoleConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class PresuppositionConfig:
    """Configuration for presupposition extraction.

    Evolvable Parameters:
    - use_llm_for_content: Whether to use LLM for generating content
    - confidence_threshold: Minimum confidence to include
    - max_presuppositions: Maximum to extract per sentence
    - triggers_enabled: Which trigger types to look for
    """

    use_llm_for_content: bool = True
    """Use LLM to generate presupposition content (vs templates)."""

    confidence_threshold: float = 0.4
    """Minimum confidence to include a presupposition."""

    max_presuppositions: int = 10
    """Maximum presuppositions per sentence."""

    include_weak_triggers: bool = True
    """Include lower-confidence trigger patterns."""

    triggers_enabled: list[str] = field(default_factory=lambda: [
        "definite_description",
        "factive_verb",
        "change_of_state",
        "iterative",
        "temporal_clause",
        "possessive",
        "implicative",
        "comparative",
        "cleft",
        "counterfactual",
    ])
    """Which presupposition triggers to detect."""

    llm_temperature: float = 0.1
    """Temperature for LLM calls."""

    prompt_template: str = """Extract the presuppositions from this sentence.

Sentence: "{text}"

A presupposition is something that must be true for the sentence to make sense.

For example:
- "Doug forgot the meeting again" presupposes:
  - A specific meeting exists (from "the meeting")
  - Doug forgot before (from "again")

Identified triggers: {triggers}

For each trigger, generate the presupposed content.

Respond in XML:
<presuppositions>
  <presup trigger_type="iterative" trigger_text="again" confidence="0.9">
    Doug forgot something before
  </presup>
  <presup trigger_type="definite_description" trigger_text="the meeting" confidence="0.95">
    A specific meeting exists that is contextually identifiable
  </presup>
</presuppositions>"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "use_llm_for_content": self.use_llm_for_content,
            "confidence_threshold": self.confidence_threshold,
            "max_presuppositions": self.max_presuppositions,
            "include_weak_triggers": self.include_weak_triggers,
            "triggers_enabled": self.triggers_enabled,
            "llm_temperature": self.llm_temperature,
            "prompt_template": self.prompt_template,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PresuppositionConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class CommonsenseConfig:
    """Configuration for commonsense inference generation.

    Evolvable Parameters:
    - tier1_relations: Always extract these
    - tier2_relations: Extract conditionally
    - tier2_threshold: When to extract tier 2
    - inference_confidence_min: Minimum confidence to keep
    """

    tier1_relations: list[str] = field(default_factory=lambda: [
        "xIntent",
        "xEffect",
        "xReact",
        "xAttr",
    ])
    """Tier 1: Always extract these relations."""

    tier2_relations: list[str] = field(default_factory=lambda: [
        "xNeed",
        "xWant",
        "oReact",
        "Causes",
    ])
    """Tier 2: Extract conditionally."""

    tier2_threshold: float = 0.6
    """Extract tier 2 when entity is INSTANCE and confidence above this."""

    inference_confidence_min: float = 0.5
    """Minimum confidence to keep an inference."""

    max_inferences_per_relation: int = 3
    """Maximum inferences per relation type."""

    deduplicate: bool = True
    """Remove semantically similar inferences."""

    dedup_similarity_threshold: float = 0.85
    """Similarity threshold for deduplication."""

    llm_temperature: float = 0.3
    """Temperature for LLM calls (slightly higher for diversity)."""

    prompt_template: str = """Generate commonsense inferences about this event.

Event: "{text}"

For the person performing this action, infer:
{relations_prompt}

Be specific and grounded in the context. Provide 1-3 inferences per relation.

Respond in XML:
<inferences>
  <inference relation="xIntent" confidence="0.85">specific inference</inference>
  <inference relation="xEffect" confidence="0.80">specific inference</inference>
</inferences>"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tier1_relations": self.tier1_relations,
            "tier2_relations": self.tier2_relations,
            "tier2_threshold": self.tier2_threshold,
            "inference_confidence_min": self.inference_confidence_min,
            "max_inferences_per_relation": self.max_inferences_per_relation,
            "deduplicate": self.deduplicate,
            "dedup_similarity_threshold": self.dedup_similarity_threshold,
            "llm_temperature": self.llm_temperature,
            "prompt_template": self.prompt_template,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommonsenseConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class NegationConfig:
    """Configuration for negation detection.

    Evolvable Parameters:
    - detect_prefixes: Whether to detect morphological negation
    - scope_method: How to determine negation scope
    """

    detect_prefixes: bool = True
    """Detect morphological negation (un-, in-, dis-, etc.)."""

    scope_method: str = "heuristic"
    """Method for scope detection: 'heuristic' or 'llm'."""

    negation_cues: list[str] = field(default_factory=lambda: [
        "not", "n't", "never", "no", "none", "nobody", "nothing",
        "neither", "nor", "without", "hardly", "barely", "scarcely",
        "seldom", "rarely",
    ])
    """Words that signal negation."""

    negative_prefixes: list[str] = field(default_factory=lambda: [
        "un", "in", "im", "il", "ir", "non", "dis", "a",
    ])
    """Prefixes that negate words."""

    implicit_negators: list[str] = field(default_factory=lambda: [
        "fail", "failed", "refuse", "refused", "deny", "denied",
        "prevent", "prevented", "lack", "lacked", "miss", "missed",
    ])
    """Verbs that imply negation of their complement."""

    llm_temperature: float = 0.1
    """Temperature for LLM calls."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detect_prefixes": self.detect_prefixes,
            "scope_method": self.scope_method,
            "negation_cues": self.negation_cues,
            "negative_prefixes": self.negative_prefixes,
            "implicit_negators": self.implicit_negators,
            "llm_temperature": self.llm_temperature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NegationConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class TemporalConfig:
    """Configuration for temporal extraction.

    Evolvable Parameters:
    - extract_aspect: Whether to classify aspect
    - extract_tense: Whether to detect tense
    - normalize_dates: Whether to convert to ISO format
    """

    extract_aspect: bool = True
    """Classify Vendler aspectual category."""

    extract_tense: bool = True
    """Detect grammatical tense."""

    extract_references: bool = True
    """Extract temporal reference expressions."""

    normalize_dates: bool = True
    """Convert date expressions to ISO format."""

    llm_temperature: float = 0.1
    """Temperature for LLM calls."""

    # Patterns for temporal reference detection
    deictic_markers: list[str] = field(default_factory=lambda: [
        "yesterday", "today", "tomorrow", "now", "then",
        "last week", "next week", "last month", "next month",
        "last year", "next year", "recently", "soon",
    ])

    frequency_markers: list[str] = field(default_factory=lambda: [
        "always", "usually", "often", "sometimes", "rarely",
        "never", "every day", "every week", "daily", "weekly",
    ])

    duration_patterns: list[str] = field(default_factory=lambda: [
        r"for \d+ (second|minute|hour|day|week|month|year)s?",
        r"(all day|all night|all week)",
    ])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "extract_aspect": self.extract_aspect,
            "extract_tense": self.extract_tense,
            "extract_references": self.extract_references,
            "normalize_dates": self.normalize_dates,
            "llm_temperature": self.llm_temperature,
            "deictic_markers": self.deictic_markers,
            "frequency_markers": self.frequency_markers,
            "duration_patterns": self.duration_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class ModalityConfig:
    """Configuration for modality extraction.

    Evolvable Parameters:
    - extract_epistemic: Whether to detect certainty markers
    - extract_deontic: Whether to detect obligation markers
    - compute_certainty: Whether to compute certainty scores
    """

    extract_epistemic: bool = True
    """Detect epistemic modality (certainty)."""

    extract_deontic: bool = True
    """Detect deontic modality (obligation)."""

    extract_evidential: bool = True
    """Detect evidential modality (information source)."""

    compute_certainty: bool = True
    """Compute certainty scores for epistemic modality."""

    llm_temperature: float = 0.1
    """Temperature for LLM calls."""

    # Epistemic certainty markers (ONLY unambiguous adverbs, NOT modal verbs)
    # NOTE: Do NOT include modal verbs like "must", "should", "may", "might", "could"
    # here. Those are handled by EpistemicDetector.AMBIGUOUS_MODALS with proper
    # disambiguation to distinguish epistemic from deontic uses.
    epistemic_markers: dict[str, float] = field(default_factory=lambda: {
        # High certainty adverbs
        "definitely": 0.95,
        "certainly": 0.95,
        "clearly": 0.90,
        "obviously": 0.90,
        "surely": 0.90,
        "undoubtedly": 0.95,
        # Medium certainty adverbs
        "probably": 0.75,
        "likely": 0.75,
        "presumably": 0.70,
        # Low certainty adverbs
        "possibly": 0.40,
        "perhaps": 0.40,
        "maybe": 0.45,
    })

    deontic_markers: list[str] = field(default_factory=lambda: [
        "must", "have to", "need to", "should", "ought to",
        "may", "can", "allowed to", "permitted to",
    ])

    evidential_markers: dict[str, str] = field(default_factory=lambda: {
        "apparently": "reported",
        "reportedly": "reported",
        "allegedly": "reported",
        "seems": "inferred",
        "appears": "inferred",
        "looks like": "inferred",
        "I saw": "direct",
        "I heard": "direct",
    })

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "extract_epistemic": self.extract_epistemic,
            "extract_deontic": self.extract_deontic,
            "extract_evidential": self.extract_evidential,
            "compute_certainty": self.compute_certainty,
            "llm_temperature": self.llm_temperature,
            "epistemic_markers": self.epistemic_markers,
            "deontic_markers": self.deontic_markers,
            "evidential_markers": self.evidential_markers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModalityConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class WeightingConfig:
    """Configuration for branch weighting.

    Evolvable Parameters:
    - Weight factors for each confidence source
    - Memory support boost/penalty
    - Branch filtering thresholds
    """

    # Weight factors for confidence aggregation
    wsd_weight: float = 0.25
    """Weight for WSD confidence."""

    entity_type_weight: float = 0.20
    """Weight for entity type classification confidence."""

    presupposition_weight: float = 0.15
    """Weight for presupposition confidence."""

    commonsense_weight: float = 0.15
    """Weight for commonsense inference confidence."""

    temporal_weight: float = 0.10
    """Weight for temporal extraction confidence."""

    modality_weight: float = 0.15
    """Weight for modality extraction confidence."""

    # Memory support adjustments
    memory_support_boost: float = 0.2
    """Maximum boost from memory support."""

    memory_contradiction_penalty: float = 0.3
    """Maximum penalty from memory contradiction."""

    # Branch filtering
    min_branch_weight: float = 0.3
    """Drop branches below this weight."""

    max_branches: int = 5
    """Keep only top N branches."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "wsd_weight": self.wsd_weight,
            "entity_type_weight": self.entity_type_weight,
            "presupposition_weight": self.presupposition_weight,
            "commonsense_weight": self.commonsense_weight,
            "temporal_weight": self.temporal_weight,
            "modality_weight": self.modality_weight,
            "memory_support_boost": self.memory_support_boost,
            "memory_contradiction_penalty": self.memory_contradiction_penalty,
            "min_branch_weight": self.min_branch_weight,
            "max_branches": self.max_branches,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightingConfig:
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# =============================================================================
# Master Configuration
# =============================================================================


@dataclass
class DecompositionConfig:
    """Master configuration for the decomposition pipeline.

    Contains all stage configurations and pipeline-level settings.
    Designed for evolution - all parameters can be mutated and
    fitness tracked.

    Example:
        >>> config = DecompositionConfig()
        >>> config.stages.commonsense = False  # Disable commonsense
        >>> config.presupposition.confidence_threshold = 0.6
        >>>
        >>> # Get hash for reproducibility
        >>> print(config.hash())
    """

    # Pipeline behavior
    parallel_execution: bool = False
    """Run independent stages in parallel (future optimization)."""

    fail_fast: bool = False
    """Stop pipeline on first error."""

    timeout_seconds: float = 30.0
    """Per-stage timeout."""

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
    """Batch multiple LLM calls together."""

    default_llm_temperature: float = 0.1
    """Default temperature for LLM calls."""

    llm_max_tokens: int = 1000
    """Maximum tokens for LLM responses."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "parallel_execution": self.parallel_execution,
            "fail_fast": self.fail_fast,
            "timeout_seconds": self.timeout_seconds,
            "stages": self.stages.to_dict(),
            "semantic_role": self.semantic_role.to_dict(),
            "presupposition": self.presupposition.to_dict(),
            "commonsense": self.commonsense.to_dict(),
            "negation": self.negation.to_dict(),
            "temporal": self.temporal.to_dict(),
            "modality": self.modality.to_dict(),
            "weighting": self.weighting.to_dict(),
            "llm_batch_size": self.llm_batch_size,
            "default_llm_temperature": self.default_llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecompositionConfig:
        """Deserialize from dictionary."""
        return cls(
            parallel_execution=data.get("parallel_execution", False),
            fail_fast=data.get("fail_fast", False),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            stages=StageEnablement.from_dict(data.get("stages", {})),
            semantic_role=SemanticRoleConfig.from_dict(data.get("semantic_role", {})),
            presupposition=PresuppositionConfig.from_dict(data.get("presupposition", {})),
            commonsense=CommonsenseConfig.from_dict(data.get("commonsense", {})),
            negation=NegationConfig.from_dict(data.get("negation", {})),
            temporal=TemporalConfig.from_dict(data.get("temporal", {})),
            modality=ModalityConfig.from_dict(data.get("modality", {})),
            weighting=WeightingConfig.from_dict(data.get("weighting", {})),
            llm_batch_size=data.get("llm_batch_size", 5),
            default_llm_temperature=data.get("default_llm_temperature", 0.1),
            llm_max_tokens=data.get("llm_max_tokens", 1000),
        )

    @classmethod
    def from_json(cls, json_str: str) -> DecompositionConfig:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def hash(self) -> str:
        """Compute a hash of the configuration for reproducibility.

        Returns:
            A short hash string identifying this configuration.
        """
        config_str = self.to_json()
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        """Concise string representation."""
        enabled = [k for k, v in self.stages.to_dict().items() if v]
        return f"DecompositionConfig(stages={enabled}, hash={self.hash()})"

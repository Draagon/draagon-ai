"""Pipeline orchestrator for multi-type decomposition.

This module coordinates all extraction stages to produce DecomposedKnowledge
from input text. It handles:
- Stage execution ordering
- Error handling and fail-fast behavior
- Metrics collection
- Result aggregation into weighted branches

The pipeline accepts outputs from Phase 0 (WSD, entity classification) and
produces structured knowledge ready for storage.

Example:
    >>> from decomposition.pipeline import DecompositionPipeline
    >>> from decomposition.config import DecompositionConfig
    >>>
    >>> config = DecompositionConfig()
    >>> pipeline = DecompositionPipeline(config)
    >>>
    >>> result = await pipeline.decompose(
    ...     text="Doug forgot the meeting again",
    ...     entity_ids=["doug_person_001"],
    ...     wsd_results={"forgot": "forget.v.01"},
    ... )
    >>> print(result.presuppositions)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from enum import Enum

from decomposition.models import (
    DecomposedKnowledge,
    Presupposition,
    CommonsenseInference,
    SemanticRole,
    TemporalInfo,
    ModalityInfo,
    NegationInfo,
    WeightedBranch,
    CrossReference,
    CommonsenseRelation,
    Aspect,
    Tense,
    ModalType,
    Polarity,
)
from decomposition.config import DecompositionConfig
from decomposition.presuppositions import PresuppositionExtractor
from decomposition.semantic_roles import SemanticRoleExtractor
from decomposition.commonsense import CommonsenseExtractor
from decomposition.temporal import TemporalExtractor
from decomposition.modality import ModalityExtractor
from decomposition.negation import NegationExtractor


# =============================================================================
# Protocols
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Any:
        """Send chat messages and get response."""
        ...


class MemoryProvider(Protocol):
    """Protocol for memory providers (cross-referencing)."""

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memory for related items."""
        ...


# =============================================================================
# Stage Results
# =============================================================================


class StageStatus(str, Enum):
    """Status of a pipeline stage execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    stage_name: str
    status: StageStatus
    duration_ms: float = 0.0
    error: str | None = None
    items_extracted: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "items_extracted": self.items_extracted,
        }


@dataclass
class PipelineMetrics:
    """Metrics from a pipeline run."""

    total_duration_ms: float = 0.0
    stage_results: list[StageResult] = field(default_factory=list)
    llm_calls: int = 0
    text_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_duration_ms": self.total_duration_ms,
            "stage_results": [s.to_dict() for s in self.stage_results],
            "llm_calls": self.llm_calls,
            "text_length": self.text_length,
        }

    def get_stage(self, name: str) -> StageResult | None:
        """Get result for a specific stage."""
        for stage in self.stage_results:
            if stage.stage_name == name:
                return stage
        return None


# =============================================================================
# Pipeline Error
# =============================================================================


class PipelineError(Exception):
    """Error during pipeline execution."""

    def __init__(
        self,
        message: str,
        stage: str,
        partial_result: DecomposedKnowledge | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.partial_result = partial_result


class CrossReferenceLinker:
    """Placeholder for cross-reference linking."""

    def __init__(self, config: Any, memory: MemoryProvider | None = None):
        self.config = config
        self.memory = memory

    async def link(
        self,
        text: str,
        presuppositions: list[Presupposition],
        entity_ids: list[str] | None = None,
    ) -> list[CrossReference]:
        """Link to existing memory items."""
        # Placeholder - returns empty list
        # TODO: Implement cross-reference linking
        return []


# =============================================================================
# Branch Builder
# =============================================================================


class BranchBuilder:
    """Builds weighted interpretation branches from extraction results.

    When extraction produces ambiguous results (e.g., multiple WSD senses,
    uncertain entity types), this creates multiple branches with appropriate
    weights.
    """

    def __init__(self, config: DecompositionConfig):
        self.config = config
        self.weighting = config.weighting

    def build_branches(
        self,
        text: str,
        presuppositions: list[Presupposition],
        inferences: list[CommonsenseInference],
        semantic_roles: list[SemanticRole],
        temporal: TemporalInfo | None,
        modality: ModalityInfo | None,
        negation: NegationInfo | None,
        cross_references: list[CrossReference] | None = None,
        wsd_results: dict[str, str] | None = None,
        wsd_alternatives: dict[str, list[tuple[str, float]]] | None = None,
    ) -> list[WeightedBranch]:
        """Build weighted branches from extraction results.

        Args:
            text: Source text
            presuppositions: Extracted presuppositions
            inferences: Generated commonsense inferences
            semantic_roles: Extracted semantic roles
            temporal: Temporal information
            modality: Modality information
            negation: Negation information
            cross_references: Links to existing memory
            wsd_results: Primary WSD results (word -> sense)
            wsd_alternatives: Alternative senses with scores

        Returns:
            List of weighted interpretation branches
        """
        branches = []

        # Compute base confidence from extraction results
        base_confidence = self._compute_base_confidence(
            presuppositions, inferences, semantic_roles, temporal, modality
        )

        # Build base supporting evidence list
        base_evidence = self._build_evidence_list(
            presuppositions, inferences, semantic_roles, cross_references
        )

        # Compute memory support from cross-references
        memory_support = 0.0
        if cross_references:
            avg_xref_conf = sum(x.confidence for x in cross_references) / len(cross_references)
            memory_support = avg_xref_conf * self.weighting.memory_support_boost

        # Create branches based on WSD alternatives
        # If we have ambiguous words with alternatives, create a branch per interpretation
        if wsd_alternatives and any(len(alts) > 1 for alts in wsd_alternatives.values()):
            branches = self._create_alternative_branches(
                wsd_results=wsd_results,
                wsd_alternatives=wsd_alternatives,
                base_confidence=base_confidence,
                memory_support=memory_support,
                base_evidence=base_evidence,
            )
        else:
            # No ambiguity - single primary branch
            primary = WeightedBranch(
                interpretation="Primary interpretation based on extracted knowledge",
                confidence=base_confidence,
                memory_support=memory_support,
                supporting_evidence=base_evidence,
                entity_interpretations=wsd_results or {},
            )
            branches.append(primary)

        # Filter branches below threshold, but always keep at least one
        filtered = [
            b for b in branches
            if b.final_weight >= self.weighting.min_branch_weight
        ]

        # If all branches were filtered out, keep the highest-weighted one
        if not filtered and branches:
            filtered = [max(branches, key=lambda b: b.final_weight)]

        # Sort by weight and limit
        filtered.sort(key=lambda b: b.final_weight, reverse=True)
        filtered = filtered[:self.weighting.max_branches]

        return filtered

    def _compute_base_confidence(
        self,
        presuppositions: list[Presupposition],
        inferences: list[CommonsenseInference],
        semantic_roles: list[SemanticRole],
        temporal: TemporalInfo | None,
        modality: ModalityInfo | None,
    ) -> float:
        """Compute base confidence from extraction results."""
        confidence_scores = []

        if presuppositions:
            avg_presup_conf = sum(p.confidence for p in presuppositions) / len(presuppositions)
            confidence_scores.append(avg_presup_conf * self.weighting.presupposition_weight)

        if inferences:
            avg_inf_conf = sum(i.confidence for i in inferences) / len(inferences)
            confidence_scores.append(avg_inf_conf * self.weighting.commonsense_weight)

        if semantic_roles:
            avg_role_conf = sum(r.confidence for r in semantic_roles) / len(semantic_roles)
            confidence_scores.append(avg_role_conf)

        if temporal:
            confidence_scores.append(temporal.confidence * self.weighting.temporal_weight)

        if modality and modality.modal_type != ModalType.NONE:
            confidence_scores.append(modality.confidence * self.weighting.modality_weight)

        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.5  # Default when no extractions

    def _build_evidence_list(
        self,
        presuppositions: list[Presupposition],
        inferences: list[CommonsenseInference],
        semantic_roles: list[SemanticRole],
        cross_references: list[CrossReference] | None,
    ) -> list[str]:
        """Build supporting evidence list."""
        evidence = []
        if presuppositions:
            evidence.append(f"{len(presuppositions)} presuppositions extracted")
        if inferences:
            evidence.append(f"{len(inferences)} commonsense inferences")
        if semantic_roles:
            evidence.append(f"{len(semantic_roles)} semantic roles")
        if cross_references:
            evidence.append(f"{len(cross_references)} memory cross-references")
        return evidence

    def _create_alternative_branches(
        self,
        wsd_results: dict[str, str] | None,
        wsd_alternatives: dict[str, list[tuple[str, float]]],
        base_confidence: float,
        memory_support: float,
        base_evidence: list[str],
    ) -> list[WeightedBranch]:
        """Create multiple branches from WSD alternatives.

        When a word has multiple possible senses (e.g., "bank" could be
        financial institution or river bank), create a branch for each
        plausible interpretation, weighted by WSD confidence.

        Args:
            wsd_results: Primary WSD results (word -> synset_id)
            wsd_alternatives: Alternative senses with scores per word
            base_confidence: Base confidence from extraction
            memory_support: Memory support score
            base_evidence: Base evidence list

        Returns:
            List of weighted branches, one per interpretation
        """
        branches = []

        # Find words with multiple alternatives (ambiguous words)
        ambiguous_words = {
            word: alts for word, alts in wsd_alternatives.items()
            if len(alts) > 1
        }

        if not ambiguous_words:
            # No ambiguity - return single primary branch
            primary = WeightedBranch(
                interpretation="Primary interpretation",
                confidence=base_confidence,
                memory_support=memory_support,
                supporting_evidence=base_evidence,
                entity_interpretations=wsd_results or {},
            )
            return [primary]

        # For simplicity, handle the most ambiguous word
        # In future, could create combinatorial branches for multiple ambiguous words
        most_ambiguous_word = max(ambiguous_words.keys(), key=lambda w: len(ambiguous_words[w]))
        alternatives = ambiguous_words[most_ambiguous_word]

        # Create a branch for each alternative sense
        for synset_id, alt_confidence in alternatives:
            # Skip very low confidence alternatives
            if alt_confidence < 0.1:
                continue

            # Create interpretation description
            interpretation = f"Interpretation with '{most_ambiguous_word}' as {synset_id}"

            # Combine base confidence with WSD alternative confidence
            # Higher WSD confidence = higher branch weight
            combined_confidence = base_confidence * alt_confidence

            # Build entity interpretations for this branch
            entity_interpretations = dict(wsd_results) if wsd_results else {}
            entity_interpretations[most_ambiguous_word] = synset_id

            # Add evidence about which sense was chosen
            branch_evidence = list(base_evidence)
            branch_evidence.append(f"'{most_ambiguous_word}' → {synset_id} (conf: {alt_confidence:.2f})")

            branch = WeightedBranch(
                interpretation=interpretation,
                confidence=combined_confidence,
                memory_support=memory_support,
                supporting_evidence=branch_evidence,
                entity_interpretations=entity_interpretations,
            )
            branches.append(branch)

        # Always return at least one branch
        if not branches:
            primary = WeightedBranch(
                interpretation="Primary interpretation",
                confidence=base_confidence,
                memory_support=memory_support,
                supporting_evidence=base_evidence,
                entity_interpretations=wsd_results or {},
            )
            return [primary]

        return branches


# =============================================================================
# Main Pipeline
# =============================================================================


class DecompositionPipeline:
    """Main pipeline for multi-type decomposition.

    Coordinates all extraction stages to produce DecomposedKnowledge
    from input text.

    Example:
        >>> config = DecompositionConfig()
        >>> pipeline = DecompositionPipeline(config)
        >>>
        >>> # Basic decomposition
        >>> result = await pipeline.decompose("Doug forgot the meeting again")
        >>>
        >>> # With Phase 0 outputs
        >>> result = await pipeline.decompose(
        ...     text="Doug forgot the meeting again",
        ...     entity_ids=["doug_person_001"],
        ...     wsd_results={"forgot": "forget.v.01"},
        ... )
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: DecompositionConfig | None = None,
        llm: LLMProvider | None = None,
        memory: MemoryProvider | None = None,
    ):
        """Initialize the decomposition pipeline.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
            llm: LLM provider for extraction stages that need it
            memory: Memory provider for cross-referencing
        """
        self.config = config or DecompositionConfig()
        self.llm = llm
        self.memory = memory

        # Initialize extractors
        self._init_extractors()

        # Metrics
        self._last_metrics: PipelineMetrics | None = None

    def _init_extractors(self) -> None:
        """Initialize all extractors based on config."""
        # Presupposition extractor (implemented)
        self.presupposition_extractor = PresuppositionExtractor(
            config=self.config.presupposition,
            llm=self.llm,
        )

        # Placeholder extractors (to be implemented)
        self.semantic_role_extractor = SemanticRoleExtractor(
            config=self.config.semantic_role,
            llm=self.llm,
        )

        self.commonsense_extractor = CommonsenseExtractor(
            config=self.config.commonsense,
            llm=self.llm,
        )

        self.temporal_extractor = TemporalExtractor(
            config=self.config.temporal,
            llm=self.llm,
        )

        self.modality_extractor = ModalityExtractor(
            config=self.config.modality,
            llm=self.llm,
        )

        self.negation_extractor = NegationExtractor(
            config=self.config.negation,
            llm=self.llm,
        )

        self.cross_reference_linker = CrossReferenceLinker(
            config=self.config,
            memory=self.memory,
        )

        # Branch builder
        self.branch_builder = BranchBuilder(self.config)

    async def decompose(
        self,
        text: str,
        entity_ids: list[str] | None = None,
        wsd_results: dict[str, str] | None = None,
        wsd_alternatives: dict[str, list[tuple[str, float]]] | None = None,
        entity_types: dict[str, str] | None = None,
    ) -> DecomposedKnowledge:
        """Decompose text into structured implicit knowledge.

        Args:
            text: The text to decompose
            entity_ids: Entity IDs from Phase 0 (optional)
            wsd_results: WSD results mapping word -> synset ID (optional)
            wsd_alternatives: Alternative senses with scores (optional)
            entity_types: Entity type classifications (optional)

        Returns:
            DecomposedKnowledge containing all extracted information

        Raises:
            PipelineError: If fail_fast is True and a stage fails
        """
        start_time = time.time()
        metrics = PipelineMetrics(text_length=len(text))

        # Initialize result
        result = DecomposedKnowledge(
            source_text=text,
            entity_ids=entity_ids or [],
            config_hash=self.config.hash(),
            pipeline_version=self.VERSION,
        )

        stages = self.config.stages

        # Stage 1: Semantic roles
        if stages.semantic_roles:
            stage_result = await self._run_stage(
                "semantic_roles",
                lambda: self.semantic_role_extractor.extract(
                    text,
                    wsd_results=wsd_results,
                    entity_ids=entity_ids,
                ),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.semantic_roles = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="semantic_roles",
                    partial_result=result,
                )

        # Stage 2: Presuppositions
        if stages.presuppositions:
            stage_result = await self._run_stage(
                "presuppositions",
                lambda: self.presupposition_extractor.extract(
                    text,
                    entity_ids=entity_ids,
                ),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.presuppositions = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="presuppositions",
                    partial_result=result,
                )

        # Stage 3: Commonsense inferences
        if stages.commonsense:
            stage_result = await self._run_stage(
                "commonsense",
                lambda: self.commonsense_extractor.extract(
                    text,
                    entity_ids=entity_ids,
                    entity_types=entity_types,
                ),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.commonsense_inferences = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="commonsense",
                    partial_result=result,
                )

        # Stage 4: Temporal extraction
        if stages.temporal:
            stage_result = await self._run_stage(
                "temporal",
                lambda: self.temporal_extractor.extract(text),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.temporal = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="temporal",
                    partial_result=result,
                )

        # Stage 5: Modality extraction
        if stages.modality:
            stage_result = await self._run_stage(
                "modality",
                lambda: self.modality_extractor.extract(text),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.modality = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="modality",
                    partial_result=result,
                )

        # Stage 6: Negation detection
        if stages.negation:
            stage_result = await self._run_stage(
                "negation",
                lambda: self.negation_extractor.extract(text),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.negation = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="negation",
                    partial_result=result,
                )

        # Stage 7: Cross-reference linking (if enabled and memory available)
        if stages.cross_references and self.memory:
            stage_result = await self._run_stage(
                "cross_references",
                lambda: self.cross_reference_linker.link(
                    text,
                    result.presuppositions,
                    entity_ids=entity_ids,
                ),
            )
            metrics.stage_results.append(stage_result)
            if stage_result.status == StageStatus.COMPLETED:
                result.cross_references = stage_result._data
            elif stage_result.status == StageStatus.FAILED and self.config.fail_fast:
                raise PipelineError(
                    f"Stage failed: {stage_result.error}",
                    stage="cross_references",
                    partial_result=result,
                )

        # Build weighted branches
        result.branches = self.branch_builder.build_branches(
            text=text,
            presuppositions=result.presuppositions,
            inferences=result.commonsense_inferences,
            semantic_roles=result.semantic_roles,
            temporal=result.temporal,
            modality=result.modality,
            negation=result.negation,
            cross_references=result.cross_references,
            wsd_results=wsd_results,
            wsd_alternatives=wsd_alternatives,
        )

        # Finalize metrics
        metrics.total_duration_ms = (time.time() - start_time) * 1000
        self._last_metrics = metrics

        return result

    async def _run_stage(
        self,
        name: str,
        extractor_fn: Any,
    ) -> StageResult:
        """Run a single pipeline stage with error handling.

        Args:
            name: Stage name for logging
            extractor_fn: Async function that performs extraction

        Returns:
            StageResult with status and extracted data
        """
        start = time.time()
        result = StageResult(stage_name=name, status=StageStatus.RUNNING)

        try:
            # Run with timeout
            data = await asyncio.wait_for(
                extractor_fn(),
                timeout=self.config.timeout_seconds,
            )

            result.status = StageStatus.COMPLETED
            result._data = data  # Store data temporarily

            # Count items
            if isinstance(data, list):
                result.items_extracted = len(data)
            elif data is not None:
                result.items_extracted = 1

        except asyncio.TimeoutError:
            result.status = StageStatus.FAILED
            result.error = f"Timeout after {self.config.timeout_seconds}s"
            result._data = None

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            result._data = None

        result.duration_ms = (time.time() - start) * 1000
        return result

    def decompose_sync(
        self,
        text: str,
        entity_ids: list[str] | None = None,
    ) -> DecomposedKnowledge:
        """Synchronous decomposition using only template-based extraction.

        This is faster but less accurate than async decomposition with LLM.

        Args:
            text: The text to decompose
            entity_ids: Entity IDs from Phase 0 (optional)

        Returns:
            DecomposedKnowledge with template-based extractions
        """
        result = DecomposedKnowledge(
            source_text=text,
            entity_ids=entity_ids or [],
            config_hash=self.config.hash(),
            pipeline_version=self.VERSION,
        )

        # Only presupposition extraction is implemented for sync
        if self.config.stages.presuppositions:
            result.presuppositions = self.presupposition_extractor.extract_sync(
                text,
                entity_ids=entity_ids,
            )

        # Build branches from what we have
        result.branches = self.branch_builder.build_branches(
            text=text,
            presuppositions=result.presuppositions,
            inferences=[],
            semantic_roles=[],
            temporal=None,
            modality=None,
            negation=None,
        )

        return result

    def get_last_metrics(self) -> PipelineMetrics | None:
        """Get metrics from the last decomposition run."""
        return self._last_metrics

    def update_config(self, config: DecompositionConfig) -> None:
        """Update pipeline configuration.

        Re-initializes extractors with new config.

        Args:
            config: New configuration to use
        """
        self.config = config
        self._init_extractors()


# =============================================================================
# Convenience Functions
# =============================================================================


async def decompose(
    text: str,
    config: DecompositionConfig | None = None,
    llm: LLMProvider | None = None,
) -> DecomposedKnowledge:
    """Convenience function to decompose text without creating pipeline.

    Args:
        text: Text to decompose
        config: Optional configuration
        llm: Optional LLM provider

    Returns:
        DecomposedKnowledge result
    """
    pipeline = DecompositionPipeline(config=config, llm=llm)
    return await pipeline.decompose(text)


def decompose_sync(
    text: str,
    config: DecompositionConfig | None = None,
) -> DecomposedKnowledge:
    """Convenience function for synchronous decomposition.

    Args:
        text: Text to decompose
        config: Optional configuration

    Returns:
        DecomposedKnowledge result (template-based only)
    """
    pipeline = DecompositionPipeline(config=config)
    return pipeline.decompose_sync(text)

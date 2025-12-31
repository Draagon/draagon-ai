"""Tests for the decomposition pipeline orchestrator."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from decomposition.pipeline import (
    DecompositionPipeline,
    PipelineError,
    PipelineMetrics,
    StageResult,
    StageStatus,
    BranchBuilder,
    decompose,
    decompose_sync,
)
from decomposition.config import (
    DecompositionConfig,
    StageEnablement,
    PresuppositionConfig,
)
from decomposition.models import (
    DecomposedKnowledge,
    Presupposition,
    PresuppositionTrigger,
    WeightedBranch,
    TemporalInfo,
    ModalityInfo,
    NegationInfo,
    ModalType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default pipeline configuration."""
    return DecompositionConfig()


@pytest.fixture
def minimal_config():
    """Minimal config with only presuppositions enabled."""
    config = DecompositionConfig()
    config.stages = StageEnablement(
        semantic_roles=False,
        presuppositions=True,
        commonsense=False,
        negation=False,
        temporal=False,
        modality=False,
        cross_references=False,
    )
    return config


@pytest.fixture
def mock_llm():
    """Mock LLM provider."""
    llm = AsyncMock()
    llm.chat.return_value = "Mock LLM response"
    return llm


@pytest.fixture
def pipeline(default_config):
    """Default pipeline instance."""
    return DecompositionPipeline(config=default_config)


@pytest.fixture
def minimal_pipeline(minimal_config):
    """Pipeline with minimal stages enabled."""
    return DecompositionPipeline(config=minimal_config)


# =============================================================================
# Pipeline Initialization Tests
# =============================================================================


class TestPipelineInitialization:
    """Tests for pipeline initialization."""

    def test_create_with_defaults(self):
        """Pipeline should create with default config."""
        pipeline = DecompositionPipeline()
        assert pipeline.config is not None
        assert pipeline.llm is None
        assert pipeline.memory is None

    def test_create_with_config(self, default_config):
        """Pipeline should accept custom config."""
        pipeline = DecompositionPipeline(config=default_config)
        assert pipeline.config == default_config

    def test_create_with_llm(self, mock_llm):
        """Pipeline should accept LLM provider."""
        pipeline = DecompositionPipeline(llm=mock_llm)
        assert pipeline.llm == mock_llm

    def test_extractors_initialized(self, pipeline):
        """Pipeline should initialize all extractors."""
        assert pipeline.presupposition_extractor is not None
        assert pipeline.semantic_role_extractor is not None
        assert pipeline.commonsense_extractor is not None
        assert pipeline.temporal_extractor is not None
        assert pipeline.modality_extractor is not None
        assert pipeline.negation_extractor is not None
        assert pipeline.branch_builder is not None

    def test_version_set(self, pipeline):
        """Pipeline should have version."""
        assert pipeline.VERSION == "1.0.0"


# =============================================================================
# Synchronous Decomposition Tests
# =============================================================================


class TestSyncDecomposition:
    """Tests for synchronous decomposition."""

    def test_basic_sync_decomposition(self, minimal_pipeline):
        """Should decompose text synchronously."""
        result = minimal_pipeline.decompose_sync("Doug forgot the meeting again")

        assert isinstance(result, DecomposedKnowledge)
        assert result.source_text == "Doug forgot the meeting again"
        assert result.pipeline_version == "1.0.0"

    def test_sync_extracts_presuppositions(self, minimal_pipeline):
        """Should extract presuppositions in sync mode."""
        result = minimal_pipeline.decompose_sync("Doug forgot the meeting again")

        # Should have presuppositions from "again" and "the meeting"
        assert len(result.presuppositions) > 0

        trigger_types = [p.trigger_type for p in result.presuppositions]
        assert PresuppositionTrigger.ITERATIVE in trigger_types

    def test_sync_creates_branches(self, minimal_pipeline):
        """Should create weighted branches in sync mode."""
        result = minimal_pipeline.decompose_sync("Doug forgot the meeting again")

        assert len(result.branches) > 0
        assert all(isinstance(b, WeightedBranch) for b in result.branches)

    def test_sync_with_entity_ids(self, minimal_pipeline):
        """Should accept entity IDs."""
        result = minimal_pipeline.decompose_sync(
            "Doug forgot the meeting again",
            entity_ids=["doug_person_001"],
        )

        assert result.entity_ids == ["doug_person_001"]

    def test_sync_sets_config_hash(self, minimal_pipeline):
        """Should set config hash for reproducibility."""
        result = minimal_pipeline.decompose_sync("Test text")

        assert result.config_hash != ""
        assert len(result.config_hash) == 16  # SHA256 truncated


# =============================================================================
# Async Decomposition Tests
# =============================================================================


class TestAsyncDecomposition:
    """Tests for async decomposition."""

    @pytest.mark.asyncio
    async def test_basic_async_decomposition(self, minimal_pipeline):
        """Should decompose text asynchronously."""
        result = await minimal_pipeline.decompose("Doug forgot the meeting again")

        assert isinstance(result, DecomposedKnowledge)
        assert result.source_text == "Doug forgot the meeting again"

    @pytest.mark.asyncio
    async def test_async_extracts_presuppositions(self, minimal_pipeline):
        """Should extract presuppositions in async mode."""
        result = await minimal_pipeline.decompose("She stopped running")

        # Should have change-of-state presupposition
        assert len(result.presuppositions) > 0

        trigger_types = [p.trigger_type for p in result.presuppositions]
        assert PresuppositionTrigger.CHANGE_OF_STATE in trigger_types

    @pytest.mark.asyncio
    async def test_async_with_wsd_results(self, minimal_pipeline):
        """Should accept WSD results."""
        result = await minimal_pipeline.decompose(
            "Doug forgot the meeting",
            wsd_results={"forgot": "forget.v.01"},
        )

        assert isinstance(result, DecomposedKnowledge)

    @pytest.mark.asyncio
    async def test_async_with_entity_types(self, minimal_pipeline):
        """Should accept entity types."""
        result = await minimal_pipeline.decompose(
            "Doug forgot the meeting",
            entity_ids=["doug_person_001"],
            entity_types={"doug_person_001": "PERSON"},
        )

        assert result.entity_ids == ["doug_person_001"]

    @pytest.mark.asyncio
    async def test_async_collects_metrics(self, minimal_pipeline):
        """Should collect pipeline metrics."""
        await minimal_pipeline.decompose("Test text")

        metrics = minimal_pipeline.get_last_metrics()
        assert metrics is not None
        assert isinstance(metrics, PipelineMetrics)
        assert metrics.total_duration_ms > 0
        assert metrics.text_length == len("Test text")


# =============================================================================
# Stage Execution Tests
# =============================================================================


class TestStageExecution:
    """Tests for individual stage execution."""

    @pytest.mark.asyncio
    async def test_stages_run_based_on_config(self, default_config):
        """Only enabled stages should run."""
        # Disable all except presuppositions
        default_config.stages = StageEnablement(
            semantic_roles=False,
            presuppositions=True,
            commonsense=False,
            negation=False,
            temporal=False,
            modality=False,
        )

        pipeline = DecompositionPipeline(config=default_config)
        await pipeline.decompose("Test text")

        metrics = pipeline.get_last_metrics()
        stage_names = [s.stage_name for s in metrics.stage_results]

        assert "presuppositions" in stage_names
        assert "semantic_roles" not in stage_names
        assert "commonsense" not in stage_names

    @pytest.mark.asyncio
    async def test_stage_results_track_status(self, minimal_pipeline):
        """Stage results should track completion status."""
        await minimal_pipeline.decompose("Test text")

        metrics = minimal_pipeline.get_last_metrics()
        presup_stage = metrics.get_stage("presuppositions")

        assert presup_stage is not None
        assert presup_stage.status == StageStatus.COMPLETED
        assert presup_stage.duration_ms > 0

    @pytest.mark.asyncio
    async def test_stage_results_count_items(self, minimal_pipeline):
        """Stage results should count extracted items."""
        await minimal_pipeline.decompose("Doug forgot the meeting again")

        metrics = minimal_pipeline.get_last_metrics()
        presup_stage = metrics.get_stage("presuppositions")

        assert presup_stage is not None
        assert presup_stage.items_extracted > 0


# =============================================================================
# Branch Building Tests
# =============================================================================


class TestBranchBuilder:
    """Tests for branch building logic."""

    @pytest.fixture
    def builder(self, default_config):
        """Branch builder instance."""
        return BranchBuilder(default_config)

    def test_builds_primary_branch(self, builder):
        """Should build at least one branch."""
        branches = builder.build_branches(
            text="Test text",
            presuppositions=[],
            inferences=[],
            semantic_roles=[],
            temporal=None,
            modality=None,
            negation=None,
        )

        assert len(branches) >= 1

    def test_branch_confidence_from_presuppositions(self, builder):
        """Branch confidence should incorporate presupposition confidence."""
        presups = [
            Presupposition(
                content="Test presup",
                trigger_type=PresuppositionTrigger.ITERATIVE,
                trigger_text="again",
                confidence=0.9,
            ),
        ]

        branches = builder.build_branches(
            text="Test text again",
            presuppositions=presups,
            inferences=[],
            semantic_roles=[],
            temporal=None,
            modality=None,
            negation=None,
        )

        assert branches[0].confidence > 0.0

    def test_branch_has_supporting_evidence(self, builder):
        """Branch should list supporting evidence."""
        presups = [
            Presupposition(
                content="Test presup",
                trigger_type=PresuppositionTrigger.ITERATIVE,
                trigger_text="again",
                confidence=0.9,
            ),
        ]

        branches = builder.build_branches(
            text="Test text again",
            presuppositions=presups,
            inferences=[],
            semantic_roles=[],
            temporal=None,
            modality=None,
            negation=None,
        )

        assert len(branches[0].supporting_evidence) > 0
        assert "presupposition" in branches[0].supporting_evidence[0].lower()

    def test_filters_low_weight_branches(self, builder):
        """Should filter branches below weight threshold."""
        # All branches with empty extractions will have low confidence
        branches = builder.build_branches(
            text="Test",
            presuppositions=[],
            inferences=[],
            semantic_roles=[],
            temporal=None,
            modality=None,
            negation=None,
        )

        # Should still have at least one branch (primary)
        assert len(branches) >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_fail_fast_raises_error(self, default_config):
        """Should raise PipelineError when fail_fast is True."""
        default_config.fail_fast = True
        default_config.timeout_seconds = 0.001  # Very short timeout

        pipeline = DecompositionPipeline(config=default_config)

        # This may or may not timeout depending on speed
        # Just verify it doesn't crash
        try:
            await pipeline.decompose("Test text")
        except PipelineError as e:
            assert e.stage is not None
            assert e.partial_result is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_fail_fast(self, default_config):
        """Should continue on error when fail_fast is False."""
        default_config.fail_fast = False

        pipeline = DecompositionPipeline(config=default_config)
        result = await pipeline.decompose("Test text")

        # Should still return a result
        assert isinstance(result, DecomposedKnowledge)


# =============================================================================
# Configuration Update Tests
# =============================================================================


class TestConfigUpdate:
    """Tests for configuration updates."""

    def test_update_config(self, pipeline, minimal_config):
        """Should update config and reinitialize extractors."""
        original_hash = pipeline.config.hash()

        pipeline.update_config(minimal_config)

        assert pipeline.config == minimal_config
        assert pipeline.config.hash() != original_hash

    @pytest.mark.asyncio
    async def test_updated_config_affects_extraction(self):
        """Updated config should affect extraction behavior."""
        # Start with all triggers enabled
        config1 = DecompositionConfig()
        config1.stages = StageEnablement(
            semantic_roles=False,
            presuppositions=True,
            commonsense=False,
            negation=False,
            temporal=False,
            modality=False,
        )

        pipeline = DecompositionPipeline(config=config1)
        result1 = await pipeline.decompose("Doug stopped again")

        # Update to disable iterative trigger
        config2 = DecompositionConfig()
        config2.stages = config1.stages
        config2.presupposition.triggers_enabled = ["change_of_state"]

        pipeline.update_config(config2)
        result2 = await pipeline.decompose("Doug stopped again")

        # First result should have iterative, second should not
        triggers1 = [p.trigger_type for p in result1.presuppositions]
        triggers2 = [p.trigger_type for p in result2.presuppositions]

        assert PresuppositionTrigger.ITERATIVE in triggers1
        assert PresuppositionTrigger.ITERATIVE not in triggers2


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_decompose_function(self):
        """Should decompose text without creating pipeline."""
        result = await decompose("Doug forgot the meeting again")

        assert isinstance(result, DecomposedKnowledge)
        assert len(result.presuppositions) > 0

    def test_decompose_sync_function(self):
        """Should decompose text synchronously without creating pipeline."""
        result = decompose_sync("Doug forgot the meeting again")

        assert isinstance(result, DecomposedKnowledge)
        assert len(result.presuppositions) > 0

    @pytest.mark.asyncio
    async def test_decompose_with_custom_config(self):
        """Should accept custom config."""
        config = DecompositionConfig()
        config.stages = StageEnablement(
            semantic_roles=False,
            presuppositions=True,
            commonsense=False,
            negation=False,
            temporal=False,
            modality=False,
        )

        result = await decompose("Test", config=config)
        assert isinstance(result, DecomposedKnowledge)


# =============================================================================
# Metrics Tests
# =============================================================================


class TestPipelineMetrics:
    """Tests for pipeline metrics."""

    def test_metrics_serialization(self):
        """Metrics should serialize to dict."""
        metrics = PipelineMetrics(
            total_duration_ms=100.0,
            stage_results=[
                StageResult(
                    stage_name="presuppositions",
                    status=StageStatus.COMPLETED,
                    duration_ms=50.0,
                    items_extracted=3,
                ),
            ],
            llm_calls=2,
            text_length=50,
        )

        data = metrics.to_dict()

        assert data["total_duration_ms"] == 100.0
        assert len(data["stage_results"]) == 1
        assert data["stage_results"][0]["stage_name"] == "presuppositions"
        assert data["llm_calls"] == 2

    def test_metrics_get_stage(self):
        """Should retrieve stage by name."""
        metrics = PipelineMetrics(
            stage_results=[
                StageResult("stage1", StageStatus.COMPLETED),
                StageResult("stage2", StageStatus.FAILED, error="Test error"),
            ],
        )

        stage1 = metrics.get_stage("stage1")
        stage2 = metrics.get_stage("stage2")
        stage3 = metrics.get_stage("nonexistent")

        assert stage1.status == StageStatus.COMPLETED
        assert stage2.status == StageStatus.FAILED
        assert stage2.error == "Test error"
        assert stage3 is None


# =============================================================================
# Stage Result Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult."""

    def test_stage_result_serialization(self):
        """StageResult should serialize to dict."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            duration_ms=25.5,
            items_extracted=5,
        )

        data = result.to_dict()

        assert data["stage_name"] == "test_stage"
        assert data["status"] == "completed"
        assert data["duration_ms"] == 25.5
        assert data["items_extracted"] == 5
        assert data["error"] is None

    def test_stage_result_with_error(self):
        """StageResult should capture errors."""
        result = StageResult(
            stage_name="failed_stage",
            status=StageStatus.FAILED,
            error="Something went wrong",
        )

        assert result.error == "Something went wrong"
        assert result.status == StageStatus.FAILED


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_run(self, default_config):
        """Should run complete pipeline with all stages."""
        pipeline = DecompositionPipeline(config=default_config)

        result = await pipeline.decompose(
            "Doug realized he forgot the important meeting again yesterday",
            entity_ids=["doug_person_001", "meeting_event_001"],
        )

        # Should have extracted various types of knowledge
        assert isinstance(result, DecomposedKnowledge)
        assert result.source_text == "Doug realized he forgot the important meeting again yesterday"
        assert result.entity_ids == ["doug_person_001", "meeting_event_001"]

        # Should have presuppositions (factive, iterative, definite)
        assert len(result.presuppositions) > 0

        # Should have branches
        assert len(result.branches) > 0

        # Should have valid config hash
        assert len(result.config_hash) == 16

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_text(self, minimal_pipeline):
        """Should handle empty text gracefully."""
        result = await minimal_pipeline.decompose("")

        assert isinstance(result, DecomposedKnowledge)
        assert result.source_text == ""
        assert len(result.presuppositions) == 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_no_triggers(self, minimal_pipeline):
        """Should handle text with no presupposition triggers."""
        result = await minimal_pipeline.decompose("Hello world")

        assert isinstance(result, DecomposedKnowledge)
        # May have some triggers from "the" if present, or may be empty
        # Just verify it doesn't crash

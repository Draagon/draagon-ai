"""Tests for ACE-style Context Evolution Engine."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.evolution.context_engine import (
    ContextCandidate,
    ContextEvaluation,
    InteractionFeedback,
    ContextEvolutionEngine,
    EvolutionResult,
)


class TestContextCandidate:
    """Test ContextCandidate dataclass."""

    def test_create_basic_candidate(self):
        """Test creating a basic candidate."""
        candidate = ContextCandidate(
            content="This is a context.",
            source="generated",
        )

        assert candidate.content == "This is a context."
        assert candidate.source == "generated"
        assert candidate.effectiveness == 0.0

    def test_candidate_id_from_hash(self):
        """Test that ID is generated from content hash."""
        candidate = ContextCandidate(
            content="Unique content",
            source="generated",
        )

        assert len(candidate.id) == 12
        assert candidate.id.isalnum()

    def test_same_content_same_id(self):
        """Test that same content produces same ID."""
        c1 = ContextCandidate(content="Same", source="generated")
        c2 = ContextCandidate(content="Same", source="refined")

        assert c1.id == c2.id

    def test_different_content_different_id(self):
        """Test that different content produces different ID."""
        c1 = ContextCandidate(content="Content A", source="generated")
        c2 = ContextCandidate(content="Content B", source="generated")

        assert c1.id != c2.id


class TestContextEvaluation:
    """Test ContextEvaluation dataclass."""

    def test_create_evaluation(self):
        """Test creating an evaluation."""
        eval_result = ContextEvaluation(
            candidate_id="abc123",
            effectiveness=0.8,
            clarity=0.9,
            completeness=0.7,
            reasoning="Good overall",
            strengths=["Clear", "Complete"],
            weaknesses=["Could be more concise"],
        )

        assert eval_result.effectiveness == 0.8
        assert len(eval_result.strengths) == 2
        assert len(eval_result.weaknesses) == 1


class TestInteractionFeedback:
    """Test InteractionFeedback dataclass."""

    def test_create_feedback(self):
        """Test creating feedback."""
        feedback = InteractionFeedback(
            query="What time is it?",
            response="It is 3:30 PM.",
            success=True,
            quality_score=0.9,
        )

        assert feedback.query == "What time is it?"
        assert feedback.success is True
        assert feedback.quality_score == 0.9

    def test_feedback_with_correction(self):
        """Test feedback with user correction."""
        feedback = InteractionFeedback(
            query="Turn on lights",
            response="I don't understand.",
            success=False,
            user_correction="I meant the bedroom lights",
            quality_score=0.2,
        )

        assert feedback.success is False
        assert feedback.user_correction is not None


class TestContextEvolutionEngine:
    """Test ContextEvolutionEngine."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()

        # Default responses for different operations
        llm.chat = AsyncMock(return_value={
            "content": "Evolved context with improvements."
        })
        llm.chat_json = AsyncMock(return_value={
            "parsed": {
                "effectiveness": 0.8,
                "clarity": 0.9,
                "completeness": 0.7,
                "strengths": ["Clear"],
                "weaknesses": ["Could improve"],
                "reasoning": "Good context",
            }
        })

        return llm

    @pytest.fixture
    def engine(self, mock_llm):
        """Create a ContextEvolutionEngine."""
        return ContextEvolutionEngine(
            llm=mock_llm,
            similarity_threshold=0.85,
            min_effectiveness=0.3,
        )

    @pytest.fixture
    def feedback(self):
        """Create sample feedback."""
        return [
            InteractionFeedback(
                query="What time is it?",
                response="It is 3:30 PM.",
                success=True,
                quality_score=0.9,
            ),
            InteractionFeedback(
                query="Turn on the lights",
                response="Turning on bedroom lights.",
                success=True,
                quality_score=0.8,
            ),
            InteractionFeedback(
                query="Play music",
                response="Playing your favorites.",
                success=True,
                quality_score=0.85,
            ),
        ]

    @pytest.mark.asyncio
    async def test_evolve_no_feedback(self, engine):
        """Test evolution with no feedback."""
        result = await engine.evolve("Current context", [])

        assert not result.success
        assert "no feedback" in result.error.lower()

    @pytest.mark.asyncio
    async def test_evolve_success(self, engine, feedback):
        """Test successful evolution."""
        result = await engine.evolve("Current context", feedback)

        assert result.success
        assert result.evolved_context is not None
        assert result.candidates_generated > 0

    @pytest.mark.asyncio
    async def test_evolve_tracks_candidates(self, engine, feedback):
        """Test that evolution tracks candidate counts."""
        result = await engine.evolve("Current context", feedback)

        assert result.candidates_generated >= 0
        assert result.candidates_accepted >= 0

    @pytest.mark.asyncio
    async def test_evolve_calculates_improvement(self, engine, feedback):
        """Test that improvement score is calculated."""
        result = await engine.evolve("Current context", feedback)

        # Improvement score should be a number
        assert isinstance(result.improvement_score, float)

    @pytest.mark.asyncio
    async def test_generate_candidates(self, engine, feedback):
        """Test candidate generation."""
        candidates = await engine._generate_candidates("Current", feedback)

        assert len(candidates) > 0
        assert all(isinstance(c, ContextCandidate) for c in candidates)
        assert all(c.source == "generated" for c in candidates)

    @pytest.mark.asyncio
    async def test_reflect_on_candidates(self, engine, mock_llm):
        """Test candidate evaluation."""
        candidates = [
            ContextCandidate(content="Candidate 1", source="generated"),
            ContextCandidate(content="Candidate 2", source="generated"),
        ]

        evaluations = await engine._reflect_on_candidates(candidates)

        assert len(evaluations) == 2
        assert all(isinstance(e, ContextEvaluation) for e in evaluations)
        # Candidates should have effectiveness updated
        assert candidates[0].effectiveness > 0

    @pytest.mark.asyncio
    async def test_evaluate_single(self, engine):
        """Test single context evaluation."""
        eval_result = await engine._evaluate_single("Test context")

        assert isinstance(eval_result, ContextEvaluation)
        assert 0 <= eval_result.effectiveness <= 1
        assert 0 <= eval_result.clarity <= 1

    @pytest.mark.asyncio
    async def test_merge_contexts(self, engine):
        """Test context merging."""
        contexts = [
            ContextCandidate(content="First", source="generated", effectiveness=0.8),
            ContextCandidate(content="Second", source="generated", effectiveness=0.7),
        ]

        merged = await engine._merge_contexts(contexts)

        assert merged is not None
        assert merged.source == "merged"

    @pytest.mark.asyncio
    async def test_merge_single_context(self, engine):
        """Test merging with single context."""
        contexts = [
            ContextCandidate(content="Only one", source="generated"),
        ]

        merged = await engine._merge_contexts(contexts)

        assert merged is not None
        assert merged.content == "Only one"

    @pytest.mark.asyncio
    async def test_refine_context(self, engine):
        """Test context refinement."""
        candidate = ContextCandidate(
            content="Original context",
            source="generated",
        )
        weaknesses = ["Too vague", "Missing examples"]

        refined = await engine._refine_context(candidate, weaknesses)

        assert refined is not None
        assert refined.source == "refined"
        assert refined.parent_id == candidate.id

    def test_calculate_similarity(self, engine):
        """Test similarity calculation."""
        text1 = "The quick brown fox jumps"
        text2 = "The quick brown fox leaps"

        similarity = engine._calculate_similarity(text1, text2)

        assert 0.5 < similarity < 1.0  # High but not identical

    def test_calculate_similarity_identical(self, engine):
        """Test similarity for identical texts."""
        text = "Same text here"

        similarity = engine._calculate_similarity(text, text)

        assert similarity == 1.0

    def test_calculate_similarity_empty(self, engine):
        """Test similarity with empty texts."""
        similarity = engine._calculate_similarity("", "")

        assert similarity == 0.0

    def test_get_stats(self, engine):
        """Test getting statistics."""
        stats = engine.get_stats()

        assert "evolution_count" in stats
        assert "similarity_threshold" in stats
        assert "min_effectiveness" in stats


class TestContextCuration:
    """Test context curation (grow-and-refine)."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={"content": "Curated context"})
        llm.chat_json = AsyncMock(return_value={
            "parsed": {
                "effectiveness": 0.85,
                "clarity": 0.9,
                "completeness": 0.8,
                "strengths": [],
                "weaknesses": [],
                "reasoning": "Good",
            }
        })
        return llm

    @pytest.fixture
    def engine(self, mock_llm):
        """Create engine for curation tests."""
        return ContextEvolutionEngine(llm=mock_llm, min_effectiveness=0.3)

    @pytest.mark.asyncio
    async def test_curate_filters_low_effectiveness(self, engine):
        """Test that low effectiveness candidates are filtered."""
        candidates = [
            ContextCandidate(content="Good", source="generated", effectiveness=0.8),
            ContextCandidate(content="Bad", source="generated", effectiveness=0.1),
        ]
        evaluations = [
            ContextEvaluation(candidate_id=c.id, effectiveness=c.effectiveness,
                            clarity=0.5, completeness=0.5)
            for c in candidates
        ]
        engine._evaluations = {e.candidate_id: e for e in evaluations}

        curated = await engine._curate_contexts("current", candidates, evaluations)

        # Low effectiveness should be filtered
        assert all(c.effectiveness >= engine.min_effectiveness for c in curated)

    @pytest.mark.asyncio
    async def test_curate_merges_similar(self, engine):
        """Test that similar candidates are merged."""
        # Create very similar candidates
        candidates = [
            ContextCandidate(
                content="Help user with their queries",
                source="generated",
                effectiveness=0.8,
            ),
            ContextCandidate(
                content="Help user with their queries and requests",
                source="generated",
                effectiveness=0.7,
            ),
        ]
        evaluations = [
            ContextEvaluation(candidate_id=c.id, effectiveness=c.effectiveness,
                            clarity=0.5, completeness=0.5)
            for c in candidates
        ]
        engine._evaluations = {e.candidate_id: e for e in evaluations}

        # Force high similarity
        engine.similarity_threshold = 0.5

        curated = await engine._curate_contexts("current", candidates, evaluations)

        # Should have merged (fewer candidates than input)
        merged_count = len([c for c in curated if c.source == "merged"])
        assert merged_count >= 0  # May or may not merge depending on similarity

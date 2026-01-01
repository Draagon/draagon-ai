"""Tests for AgentEvaluator LLM-as-judge (TASK-006)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from draagon_ai.testing.evaluation import (
    AgentEvaluator,
    EvaluationResult,
    QualityResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def evaluator(mock_llm):
    """Create an evaluator with mock LLM."""
    return AgentEvaluator(mock_llm, max_retries=3)


# =============================================================================
# Test EvaluationResult
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_correct_result_is_truthy(self):
        """Correct result evaluates to True."""
        result = EvaluationResult(
            correct=True,
            reasoning="Good answer",
            confidence=0.9,
        )
        assert bool(result) is True

    def test_incorrect_result_is_falsy(self):
        """Incorrect result evaluates to False."""
        result = EvaluationResult(
            correct=False,
            reasoning="Bad answer",
            confidence=0.9,
        )
        assert bool(result) is False

    def test_can_use_in_assertions(self):
        """Result can be used directly in assert."""
        result = EvaluationResult(correct=True, reasoning="OK", confidence=0.9)

        # This should not raise
        assert result


# =============================================================================
# Test XML Parsing
# =============================================================================


class TestXMLParsing:
    """Tests for XML response parsing."""

    def test_parse_correctness_correct_true(self, evaluator):
        """Parses correct=true properly."""
        xml = """<result>
  <correct>true</correct>
  <reasoning>The response correctly lists all cat names.</reasoning>
  <confidence>0.95</confidence>
</result>"""

        result = evaluator._parse_correctness_result(xml)

        assert result.correct is True
        assert "correctly" in result.reasoning.lower()
        assert result.confidence == 0.95

    def test_parse_correctness_correct_false(self, evaluator):
        """Parses correct=false properly."""
        xml = """<result>
  <correct>false</correct>
  <reasoning>Missing one cat name.</reasoning>
  <confidence>0.85</confidence>
</result>"""

        result = evaluator._parse_correctness_result(xml)

        assert result.correct is False
        assert "missing" in result.reasoning.lower()
        assert result.confidence == 0.85

    def test_parse_correctness_case_insensitive(self, evaluator):
        """Handles TRUE/FALSE case insensitively."""
        xml = "<result><correct>TRUE</correct><reasoning>OK</reasoning><confidence>0.9</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.correct is True

        xml = "<result><correct>FALSE</correct><reasoning>Bad</reasoning><confidence>0.9</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.correct is False

    def test_parse_correctness_with_whitespace(self, evaluator):
        """Handles whitespace in XML values."""
        xml = """<result>
  <correct>  true  </correct>
  <reasoning>  Good answer  </reasoning>
  <confidence>  0.9  </confidence>
</result>"""

        result = evaluator._parse_correctness_result(xml)

        assert result.correct is True
        assert result.reasoning == "Good answer"
        assert result.confidence == 0.9

    def test_parse_correctness_missing_correct(self, evaluator):
        """Defaults to False if correct tag missing."""
        xml = "<result><reasoning>OK</reasoning><confidence>0.9</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.correct is False

    def test_parse_correctness_missing_reasoning(self, evaluator):
        """Provides default reasoning if missing."""
        xml = "<result><correct>true</correct><confidence>0.9</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.reasoning == "No reasoning provided"

    def test_parse_correctness_missing_confidence(self, evaluator):
        """Defaults to 0.5 confidence if missing."""
        xml = "<result><correct>true</correct><reasoning>OK</reasoning></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.confidence == 0.5

    def test_parse_correctness_clamps_confidence(self, evaluator):
        """Clamps confidence to [0, 1]."""
        xml = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>1.5</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.confidence == 1.0

        xml = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>-0.5</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.confidence == 0.0

    def test_parse_correctness_invalid_confidence(self, evaluator):
        """Defaults to 0.5 for invalid confidence."""
        xml = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>not_a_number</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.confidence == 0.5

    def test_parse_correctness_stores_raw_response(self, evaluator):
        """Stores raw response for debugging."""
        xml = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>0.9</confidence></result>"
        result = evaluator._parse_correctness_result(xml)
        assert result.raw_response == xml

    def test_parse_quality_score(self, evaluator):
        """Parses quality result with score."""
        xml = """<result>
  <score>0.85</score>
  <reasoning>Clear and well-structured response.</reasoning>
</result>"""

        result = evaluator._parse_quality_result(xml)

        assert result.score == 0.85
        assert "clear" in result.reasoning.lower()

    def test_parse_quality_clamps_score(self, evaluator):
        """Clamps quality score to [0, 1]."""
        xml = "<result><score>1.5</score><reasoning>OK</reasoning></result>"
        result = evaluator._parse_quality_result(xml)
        assert result.score == 1.0


# =============================================================================
# Test Retry Logic
# =============================================================================


class TestRetryLogic:
    """Tests for exponential backoff retry."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self, mock_llm, evaluator):
        """No retry needed if first call succeeds."""
        mock_llm.chat.return_value = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>0.9</confidence></result>"

        result = await evaluator.evaluate_correctness(
            query="test",
            expected_outcome="test",
            actual_response="test",
        )

        assert result.correct is True
        assert mock_llm.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, mock_llm, evaluator):
        """Retries on transient failures."""
        mock_llm.chat.side_effect = [
            Exception("Network error"),
            Exception("Rate limit"),
            "<result><correct>true</correct><reasoning>OK</reasoning><confidence>0.9</confidence></result>",
        ]

        result = await evaluator.evaluate_correctness(
            query="test",
            expected_outcome="test",
            actual_response="test",
        )

        assert result.correct is True
        assert mock_llm.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, mock_llm, evaluator):
        """Raises RuntimeError after all retries exhausted."""
        mock_llm.chat.side_effect = Exception("Always fails")

        with pytest.raises(RuntimeError) as exc_info:
            await evaluator.evaluate_correctness(
                query="test",
                expected_outcome="test",
                actual_response="test",
            )

        assert "failed after 3 attempts" in str(exc_info.value)
        assert mock_llm.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_includes_last_error_in_message(self, mock_llm, evaluator):
        """Error message includes last error."""
        mock_llm.chat.side_effect = Exception("Specific error message")

        with pytest.raises(RuntimeError) as exc_info:
            await evaluator.evaluate_correctness(
                query="test",
                expected_outcome="test",
                actual_response="test",
            )

        assert "Specific error message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_custom_max_retries(self, mock_llm):
        """Respects custom max_retries."""
        evaluator = AgentEvaluator(mock_llm, max_retries=5)
        mock_llm.chat.side_effect = Exception("Fails")

        with pytest.raises(RuntimeError):
            await evaluator.evaluate_correctness(
                query="test",
                expected_outcome="test",
                actual_response="test",
            )

        assert mock_llm.chat.call_count == 5


# =============================================================================
# Test evaluate_correctness
# =============================================================================


class TestEvaluateCorrectness:
    """Tests for evaluate_correctness method."""

    @pytest.mark.asyncio
    async def test_formats_prompt_correctly(self, mock_llm, evaluator):
        """Prompt includes query, expected, and actual."""
        mock_llm.chat.return_value = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>0.9</confidence></result>"

        await evaluator.evaluate_correctness(
            query="What's the weather?",
            expected_outcome="Mentions temperature",
            actual_response="It's 72°F and sunny!",
        )

        call_args = mock_llm.chat.call_args[0][0][0]["content"]
        assert "What's the weather?" in call_args
        assert "Mentions temperature" in call_args
        assert "72°F and sunny" in call_args

    @pytest.mark.asyncio
    async def test_uses_zero_temperature(self, mock_llm, evaluator):
        """Uses temperature=0 for deterministic evaluation."""
        mock_llm.chat.return_value = "<result><correct>true</correct><reasoning>OK</reasoning><confidence>0.9</confidence></result>"

        await evaluator.evaluate_correctness(
            query="test",
            expected_outcome="test",
            actual_response="test",
        )

        call_kwargs = mock_llm.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.0


# =============================================================================
# Test evaluate_coherence
# =============================================================================


class TestEvaluateCoherence:
    """Tests for evaluate_coherence method."""

    @pytest.mark.asyncio
    async def test_returns_quality_result(self, mock_llm, evaluator):
        """Returns QualityResult."""
        mock_llm.chat.return_value = "<result><score>0.9</score><reasoning>Very clear</reasoning></result>"

        result = await evaluator.evaluate_coherence(
            query="What's the weather?",
            response="It's 72°F and sunny!",
        )

        assert isinstance(result, QualityResult)
        assert result.score == 0.9
        assert "clear" in result.reasoning.lower()


# =============================================================================
# Test evaluate_helpfulness
# =============================================================================


class TestEvaluateHelpfulness:
    """Tests for evaluate_helpfulness method."""

    @pytest.mark.asyncio
    async def test_returns_quality_result(self, mock_llm, evaluator):
        """Returns QualityResult."""
        mock_llm.chat.return_value = "<result><score>0.85</score><reasoning>Helpful and actionable</reasoning></result>"

        result = await evaluator.evaluate_helpfulness(
            query="What should I wear today?",
            response="Wear a light jacket, it's 65°F with clouds.",
        )

        assert isinstance(result, QualityResult)
        assert result.score == 0.85
        assert "actionable" in result.reasoning.lower()


# =============================================================================
# Test evaluate_all
# =============================================================================


class TestEvaluateAll:
    """Tests for evaluate_all convenience method."""

    @pytest.mark.asyncio
    async def test_runs_all_evaluations(self, mock_llm, evaluator):
        """Runs correctness, coherence, and helpfulness."""
        mock_llm.chat.return_value = "<result><correct>true</correct><score>0.9</score><reasoning>Good</reasoning><confidence>0.9</confidence></result>"

        result = await evaluator.evaluate_all(
            query="test",
            expected_outcome="test",
            actual_response="test",
        )

        assert "correctness" in result
        assert "coherence" in result
        assert "helpfulness" in result

        # At least 3 calls (one for each evaluation)
        assert mock_llm.chat.call_count >= 3

    @pytest.mark.asyncio
    async def test_returns_correct_types(self, mock_llm, evaluator):
        """Returns correct result types."""
        mock_llm.chat.return_value = "<result><correct>true</correct><score>0.9</score><reasoning>Good</reasoning><confidence>0.9</confidence></result>"

        result = await evaluator.evaluate_all(
            query="test",
            expected_outcome="test",
            actual_response="test",
        )

        assert isinstance(result["correctness"], EvaluationResult)
        assert isinstance(result["coherence"], QualityResult)
        assert isinstance(result["helpfulness"], QualityResult)

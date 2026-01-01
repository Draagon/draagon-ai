"""
LLM-as-judge evaluation for integration testing.

AgentEvaluator provides semantic evaluation of agent responses using an LLM.
This replaces brittle string matching with robust semantic validation.

Core Principle: Test outcomes, not processes.

Example:
    evaluator = AgentEvaluator(llm_provider)

    result = await evaluator.evaluate_correctness(
        query="What are my cats' names?",
        expected_outcome="Agent lists: Whiskers, Mittens, Shadow",
        actual_response="Your cats are Whiskers, Mittens, and Shadow!"
    )

    assert result.correct
    assert result.confidence > 0.8

Why LLM-as-Judge?
- Semantic equivalence detection (not just string matching)
- Handles paraphrasing (different words, same meaning)
- Provides reasoning for failures
- Confidence scores for debugging

Why XML, Not JSON?
- Fewer escaping issues (quotes, backslashes, newlines)
- Better streaming support (incremental parsing)
- More robust to malformed output
- Per CLAUDE.md constitution
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers used in evaluation."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Send chat messages and get response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Response text from LLM
        """
        ...


@dataclass
class EvaluationResult:
    """Result from LLM evaluation.

    Attributes:
        correct: Whether the response achieves the expected outcome
        reasoning: Explanation for the evaluation
        confidence: Confidence in the evaluation (0.0-1.0)
        raw_response: The raw LLM response (for debugging)
    """

    correct: bool
    reasoning: str
    confidence: float
    raw_response: str = ""

    def __bool__(self) -> bool:
        """Allow using result directly in assertions."""
        return self.correct


@dataclass
class QualityResult:
    """Result from quality evaluation (coherence, helpfulness).

    Attributes:
        score: Quality score (0.0-1.0)
        reasoning: Explanation for the score
        raw_response: The raw LLM response (for debugging)
    """

    score: float
    reasoning: str
    raw_response: str = ""


class AgentEvaluator:
    """LLM-based semantic evaluator for agent responses.

    Uses XML prompts for robust parsing and includes retry logic
    for resilience against transient LLM failures.

    Example:
        evaluator = AgentEvaluator(llm_provider, max_retries=3)

        result = await evaluator.evaluate_correctness(
            query="What's the weather?",
            expected_outcome="Mentions temperature",
            actual_response="It's 72°F and sunny!"
        )

        assert result.correct
    """

    # Evaluation prompts using XML (per CLAUDE.md)
    CORRECTNESS_PROMPT = """You are evaluating if an AI agent's response achieves an expected outcome.

<evaluation>
  <query>{query}</query>
  <expected_outcome>{expected_outcome}</expected_outcome>
  <actual_response>{actual_response}</actual_response>
</evaluation>

Evaluate whether the actual_response achieves the expected_outcome for the given query.
Focus on semantic meaning, not exact wording. Paraphrasing is acceptable.

Respond with ONLY this XML:
<result>
  <correct>true or false</correct>
  <reasoning>Brief explanation (1-2 sentences)</reasoning>
  <confidence>0.0 to 1.0</confidence>
</result>"""

    COHERENCE_PROMPT = """You are evaluating the coherence of an AI agent's response.

<evaluation>
  <query>{query}</query>
  <response>{response}</response>
</evaluation>

Evaluate the response for:
- Logical consistency
- Clarity of expression
- Relevance to the query
- Completeness

Respond with ONLY this XML:
<result>
  <score>0.0 to 1.0</score>
  <reasoning>Brief explanation (1-2 sentences)</reasoning>
</result>"""

    HELPFULNESS_PROMPT = """You are evaluating the helpfulness of an AI agent's response.

<evaluation>
  <query>{query}</query>
  <response>{response}</response>
</evaluation>

Evaluate the response for:
- Does it answer the user's question?
- Is it actionable?
- Is it appropriately detailed?
- Is the tone appropriate?

Respond with ONLY this XML:
<result>
  <score>0.0 to 1.0</score>
  <reasoning>Brief explanation (1-2 sentences)</reasoning>
</result>"""

    def __init__(self, llm: LLMProvider, max_retries: int = 3):
        """Initialize evaluator.

        Args:
            llm: LLM provider for evaluation calls
            max_retries: Maximum retry attempts on failure
        """
        self.llm = llm
        self.max_retries = max_retries

    async def _call_llm_with_retry(self, messages: list[dict]) -> str:
        """Call LLM with exponential backoff retry.

        Args:
            messages: Messages to send to LLM

        Returns:
            LLM response text

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await self.llm.chat(
                    messages,
                    temperature=0.0,  # Deterministic for evaluation
                    max_tokens=500,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"LLM evaluation failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_correctness_result(self, response: str) -> EvaluationResult:
        """Parse correctness evaluation XML response.

        Args:
            response: Raw LLM response containing XML

        Returns:
            Parsed EvaluationResult
        """
        # Extract correct
        correct_match = re.search(r"<correct>\s*(true|false)\s*</correct>", response, re.IGNORECASE)
        correct = correct_match.group(1).lower() == "true" if correct_match else False

        # Extract reasoning
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        # Extract confidence (allow negative for proper clamping)
        confidence_match = re.search(r"<confidence>\s*(-?[\d.]+)\s*</confidence>", response)
        try:
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except ValueError:
            confidence = 0.5

        return EvaluationResult(
            correct=correct,
            reasoning=reasoning,
            confidence=confidence,
            raw_response=response,
        )

    def _parse_quality_result(self, response: str) -> QualityResult:
        """Parse quality evaluation XML response.

        Args:
            response: Raw LLM response containing XML

        Returns:
            Parsed QualityResult
        """
        # Extract score
        score_match = re.search(r"<score>\s*([\d.]+)\s*</score>", response)
        try:
            score = float(score_match.group(1)) if score_match else 0.5
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            score = 0.5

        # Extract reasoning
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        return QualityResult(
            score=score,
            reasoning=reasoning,
            raw_response=response,
        )

    async def evaluate_correctness(
        self,
        query: str,
        expected_outcome: str,
        actual_response: str,
    ) -> EvaluationResult:
        """Evaluate if response achieves expected outcome.

        This is the primary evaluation method. Uses semantic comparison,
        not exact string matching.

        Args:
            query: The original user query
            expected_outcome: Description of what response should contain/achieve
            actual_response: The agent's actual response

        Returns:
            EvaluationResult with correct, reasoning, and confidence

        Example:
            result = await evaluator.evaluate_correctness(
                query="When is my birthday?",
                expected_outcome="Mentions March 15",
                actual_response="Your birthday is on March 15th!"
            )
            assert result.correct
        """
        prompt = self.CORRECTNESS_PROMPT.format(
            query=query,
            expected_outcome=expected_outcome,
            actual_response=actual_response,
        )

        response = await self._call_llm_with_retry([
            {"role": "user", "content": prompt}
        ])

        return self._parse_correctness_result(response)

    async def evaluate_coherence(
        self,
        query: str,
        response: str,
    ) -> QualityResult:
        """Evaluate response coherence and clarity.

        Measures logical consistency, clarity, relevance, and completeness.

        Args:
            query: The original user query
            response: The agent's response

        Returns:
            QualityResult with score (0.0-1.0) and reasoning
        """
        prompt = self.COHERENCE_PROMPT.format(
            query=query,
            response=response,
        )

        llm_response = await self._call_llm_with_retry([
            {"role": "user", "content": prompt}
        ])

        return self._parse_quality_result(llm_response)

    async def evaluate_helpfulness(
        self,
        query: str,
        response: str,
    ) -> QualityResult:
        """Evaluate response helpfulness and usefulness.

        Measures whether response answers the question, is actionable,
        appropriately detailed, and has good tone.

        Args:
            query: The original user query
            response: The agent's response

        Returns:
            QualityResult with score (0.0-1.0) and reasoning
        """
        prompt = self.HELPFULNESS_PROMPT.format(
            query=query,
            response=response,
        )

        llm_response = await self._call_llm_with_retry([
            {"role": "user", "content": prompt}
        ])

        return self._parse_quality_result(llm_response)

    async def evaluate_all(
        self,
        query: str,
        expected_outcome: str,
        actual_response: str,
    ) -> dict:
        """Run all evaluations on a response.

        Convenience method to run correctness, coherence, and helpfulness
        evaluations in parallel.

        Args:
            query: The original user query
            expected_outcome: Description of what response should contain
            actual_response: The agent's actual response

        Returns:
            Dict with 'correctness', 'coherence', 'helpfulness' keys
        """
        correctness, coherence, helpfulness = await asyncio.gather(
            self.evaluate_correctness(query, expected_outcome, actual_response),
            self.evaluate_coherence(query, actual_response),
            self.evaluate_helpfulness(query, actual_response),
        )

        return {
            "correctness": correctness,
            "coherence": coherence,
            "helpfulness": helpfulness,
        }

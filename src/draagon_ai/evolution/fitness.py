"""Fitness Evaluation and Diversity Preservation.

This module provides fitness evaluation for evolutionary optimization:
- Test case evaluation
- Population fitness scoring
- Fitness sharing for diversity preservation
- Quick heuristic and LLM-based evaluation

Based on Promptbreeder paper and standard evolutionary algorithm theory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class EvolutionTestCase:
    """A test case for evaluating fitness.

    Named EvolutionTestCase (not TestCase) to avoid pytest collection.
    """

    query: str
    expected_action: str | None = None
    expected_params: dict[str, Any] | None = None
    was_successful: bool = True
    user_correction: str | None = None
    quality_score: float = 0.5
    context_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""

    score: float  # 0.0 - 1.0
    raw_scores: list[float] = field(default_factory=list)
    reasoning: str = ""
    error: str | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers used in evaluation."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Execute a chat completion."""
        ...

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Execute a chat completion expecting JSON output."""
        ...


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation strategies."""

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        test_cases: list[EvolutionTestCase],
    ) -> FitnessResult:
        """Evaluate fitness of a prompt on test cases.

        Args:
            prompt: The prompt to evaluate
            test_cases: Test cases to evaluate against

        Returns:
            FitnessResult with score and details
        """
        pass


class HeuristicFitnessEvaluator(FitnessEvaluator):
    """Quick heuristic-based fitness evaluation.

    Fast but less accurate than LLM-based evaluation.
    Good for initial population filtering.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        sample_size: int = 20,
    ):
        """Initialize evaluator.

        Args:
            llm: LLM provider for running prompts
            sample_size: Max test cases to sample per evaluation
        """
        self.llm = llm
        self.sample_size = sample_size

    async def evaluate(
        self,
        prompt: str,
        test_cases: list[EvolutionTestCase],
    ) -> FitnessResult:
        """Evaluate using quick heuristics."""
        if not self.llm or not test_cases:
            return FitnessResult(score=0.0, error="No LLM or test cases")

        # Sample test cases if too many
        sample_size = min(len(test_cases), self.sample_size)
        sampled = random.sample(test_cases, sample_size)

        raw_scores = []

        for tc in sampled:
            try:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": tc.query},
                ]

                response = await self.llm.chat(
                    messages,
                    max_tokens=500,
                    temperature=0.1,
                )

                if response and response.get("content"):
                    score = self._quick_score(
                        tc.query,
                        response["content"],
                        tc.expected_action,
                        tc.was_successful,
                    )
                    raw_scores.append(score)

            except Exception as e:
                logger.debug(f"Evaluation error: {e}")

        if not raw_scores:
            return FitnessResult(score=0.0, error="No successful evaluations")

        avg_score = sum(raw_scores) / len(raw_scores)
        return FitnessResult(
            score=avg_score,
            raw_scores=raw_scores,
            reasoning=f"Evaluated {len(raw_scores)} test cases",
        )

    def _quick_score(
        self,
        query: str,
        response: str,
        expected_action: str | None,
        was_successful: bool,
    ) -> float:
        """Quick heuristic evaluation of response quality."""
        score = 0.5  # Base score

        # Check for response length (not too short, not too long)
        word_count = len(response.split())
        if 5 <= word_count <= 50:
            score += 0.1
        elif word_count < 3:
            score -= 0.2
        elif word_count > 100:
            score -= 0.1

        # Check for error indicators
        error_words = ["error", "sorry, i can't", "i'm unable", "failed to"]
        if any(err in response.lower() for err in error_words):
            score -= 0.2

        # Check for action indicators if expected
        if expected_action:
            action_indicators = {
                "get_time": ["time", "o'clock", "am", "pm"],
                "get_weather": ["degrees", "weather", "temperature"],
                "call_service": ["turning", "turned", "setting", "set"],
                "search_knowledge": ["remember", "recall", "based on"],
                "web_search": ["found", "according to", "search results"],
            }
            indicators = action_indicators.get(expected_action, [])
            if any(ind in response.lower() for ind in indicators):
                score += 0.2

        return min(1.0, max(0.0, score))


class LLMFitnessEvaluator(FitnessEvaluator):
    """LLM-based fitness evaluation.

    More accurate but slower than heuristic evaluation.
    Uses LLM to judge response quality.
    """

    EVALUATE_TEMPLATE = """Evaluate if this response correctly handles the user query.

USER QUERY: {query}

EXPECTED BEHAVIOR:
- Action: {expected_action}
- Success criteria: Handle the query appropriately

ACTUAL RESPONSE:
{response}

EVALUATION CRITERIA:
1. Did the response take the appropriate action?
2. Was the response helpful and accurate?
3. Would the user be satisfied with this response?

OUTPUT JSON only:
{{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def __init__(
        self,
        llm: LLMProvider,
        sample_size: int = 20,
    ):
        """Initialize evaluator.

        Args:
            llm: LLM provider for evaluation
            sample_size: Max test cases to sample per evaluation
        """
        self.llm = llm
        self.sample_size = sample_size

    async def evaluate(
        self,
        prompt: str,
        test_cases: list[EvolutionTestCase],
    ) -> FitnessResult:
        """Evaluate using LLM judgment."""
        if not test_cases:
            return FitnessResult(score=0.0, error="No test cases")

        # Sample test cases if too many
        sample_size = min(len(test_cases), self.sample_size)
        sampled = random.sample(test_cases, sample_size)

        raw_scores = []

        for tc in sampled:
            try:
                # First run the prompt
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": tc.query},
                ]

                response = await self.llm.chat(
                    messages,
                    max_tokens=500,
                    temperature=0.1,
                )

                if response and response.get("content"):
                    # Then evaluate the response
                    score = await self._llm_score(
                        tc.query,
                        response["content"],
                        tc.expected_action,
                    )
                    raw_scores.append(score)

            except Exception as e:
                logger.debug(f"Evaluation error: {e}")

        if not raw_scores:
            return FitnessResult(score=0.0, error="No successful evaluations")

        avg_score = sum(raw_scores) / len(raw_scores)
        return FitnessResult(
            score=avg_score,
            raw_scores=raw_scores,
            reasoning=f"LLM evaluated {len(raw_scores)} test cases",
        )

    async def _llm_score(
        self,
        query: str,
        response: str,
        expected_action: str | None,
    ) -> float:
        """Use LLM to score response quality."""
        eval_prompt = self.EVALUATE_TEMPLATE.format(
            query=query,
            expected_action=expected_action or "appropriate action",
            response=response,
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": eval_prompt},
            ]

            result = await self.llm.chat_json(
                messages,
                max_tokens=200,
                temperature=0.1,
            )

            if result and result.get("parsed"):
                score = result["parsed"].get("score", 0.5)
                return max(0.0, min(1.0, float(score)))

        except Exception as e:
            logger.debug(f"LLM scoring failed: {e}")

        return 0.5  # Fallback


class FitnessSharing:
    """Fitness sharing for diversity preservation.

    Implements niching to penalize similar individuals and maintain
    population diversity during evolution.
    """

    def __init__(
        self,
        sigma: float = 0.3,
        dampening: float = 0.2,
    ):
        """Initialize fitness sharing.

        Args:
            sigma: Niche radius (0-1). Higher = more penalty for similar prompts
            dampening: Dampens the sharing effect (0-1)
        """
        self.sigma = sigma
        self.dampening = dampening

    def calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate structural similarity between two prompts.

        Uses multiple signals:
        1. Length ratio - similar length prompts are likely more similar
        2. Word overlap - Jaccard similarity of word sets
        3. Line overlap - structural similarity of lines

        Returns:
            Similarity score between 0 and 1
        """
        # Length ratio (0-1, where 1 = same length)
        len_ratio = min(len(prompt1), len(prompt2)) / max(len(prompt1), len(prompt2), 1)

        # Word overlap (Jaccard similarity)
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        if words1 or words2:
            word_jaccard = len(words1 & words2) / len(words1 | words2)
        else:
            word_jaccard = 1.0

        # Line overlap (Jaccard on lines)
        lines1 = set(line.strip() for line in prompt1.split('\n') if line.strip())
        lines2 = set(line.strip() for line in prompt2.split('\n') if line.strip())
        if lines1 or lines2:
            line_jaccard = len(lines1 & lines2) / len(lines1 | lines2)
        else:
            line_jaccard = 1.0

        # Weighted combination (word overlap is most important)
        similarity = (len_ratio * 0.2) + (word_jaccard * 0.5) + (line_jaccard * 0.3)

        return similarity

    def apply_sharing(
        self,
        population: list[str],
        fitness_scores: list[float],
    ) -> list[float]:
        """Apply fitness sharing to penalize similar prompts.

        This is a key mechanism to prevent premature convergence.
        Prompts similar to many others get fitness reduced.

        Args:
            population: List of prompts
            fitness_scores: Raw fitness scores

        Returns:
            Adjusted fitness scores after sharing
        """
        if not population or not fitness_scores:
            return fitness_scores

        adjusted = list(fitness_scores)
        n = len(population)

        for i in range(n):
            sharing_sum = 0.0

            for j in range(n):
                if i != j:
                    similarity = self.calculate_similarity(population[i], population[j])
                    threshold = 1.0 - self.sigma

                    if similarity > threshold:
                        # Proportional sharing - more similar = more penalty
                        share = (similarity - threshold) / self.sigma
                        sharing_sum += share

            # Apply sharing: divide fitness by (1 + sharing penalty)
            sharing_factor = 1.0 + (sharing_sum * self.dampening)
            adjusted[i] = fitness_scores[i] / sharing_factor

        return adjusted

    def calculate_population_diversity(self, population: list[str]) -> float:
        """Calculate overall diversity of the population.

        Returns:
            Diversity ratio between 0 and 1, where:
            - 0 = all prompts are identical
            - 1 = all prompts are completely different
        """
        if len(population) <= 1:
            return 1.0

        total_distance = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = 1.0 - self.calculate_similarity(population[i], population[j])
                total_distance += distance
                count += 1

        if count == 0:
            return 1.0

        return total_distance / count


@dataclass
class PopulationMetrics:
    """Metrics for a population during evolution."""

    generation: int
    best_fitness: float
    avg_fitness: float
    diversity: float
    fitness_scores: list[float] = field(default_factory=list)
    adjusted_scores: list[float] = field(default_factory=list)


class PopulationEvaluator:
    """Evaluates and tracks population fitness over generations."""

    def __init__(
        self,
        fitness_evaluator: FitnessEvaluator,
        fitness_sharing: FitnessSharing | None = None,
        enable_sharing: bool = True,
    ):
        """Initialize population evaluator.

        Args:
            fitness_evaluator: Evaluator for individual fitness
            fitness_sharing: Sharing mechanism for diversity
            enable_sharing: Whether to apply fitness sharing
        """
        self.fitness_evaluator = fitness_evaluator
        self.fitness_sharing = fitness_sharing or FitnessSharing()
        self.enable_sharing = enable_sharing
        self.history: list[PopulationMetrics] = []

    async def evaluate_population(
        self,
        population: list[str],
        test_cases: list[EvolutionTestCase],
        generation: int = 0,
    ) -> tuple[list[float], list[float]]:
        """Evaluate fitness of entire population.

        Args:
            population: List of prompts
            test_cases: Test cases for evaluation
            generation: Current generation number

        Returns:
            Tuple of (raw_fitness, adjusted_fitness)
        """
        raw_fitness = []

        for prompt in population:
            result = await self.fitness_evaluator.evaluate(prompt, test_cases)
            raw_fitness.append(result.score)

        # Apply fitness sharing for diversity
        if self.enable_sharing:
            adjusted_fitness = self.fitness_sharing.apply_sharing(
                population, raw_fitness
            )
        else:
            adjusted_fitness = raw_fitness

        # Calculate metrics
        diversity = self.fitness_sharing.calculate_population_diversity(population)
        metrics = PopulationMetrics(
            generation=generation,
            best_fitness=max(raw_fitness) if raw_fitness else 0.0,
            avg_fitness=sum(raw_fitness) / len(raw_fitness) if raw_fitness else 0.0,
            diversity=diversity,
            fitness_scores=raw_fitness,
            adjusted_scores=adjusted_fitness,
        )
        self.history.append(metrics)

        return raw_fitness, adjusted_fitness

    def get_fitness_history(self) -> list[float]:
        """Get best fitness for each generation."""
        return [m.best_fitness for m in self.history]

    def get_diversity_history(self) -> list[float]:
        """Get diversity for each generation."""
        return [m.diversity for m in self.history]


def tournament_select(
    population: list[str],
    fitness_scores: list[float],
    tournament_size: int = 3,
) -> str:
    """Select an individual using tournament selection.

    Args:
        population: List of prompts
        fitness_scores: Fitness score for each prompt
        tournament_size: Number of individuals in tournament

    Returns:
        Selected prompt
    """
    tournament = random.sample(
        range(len(population)),
        min(tournament_size, len(population))
    )
    winner = max(tournament, key=lambda i: fitness_scores[i])
    return population[winner]

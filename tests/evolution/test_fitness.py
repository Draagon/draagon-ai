"""Tests for fitness evaluation and diversity preservation."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.evolution.fitness import (
    EvolutionTestCase,
    FitnessResult,
    FitnessSharing,
    HeuristicFitnessEvaluator,
    LLMFitnessEvaluator,
    PopulationEvaluator,
    PopulationMetrics,
    tournament_select,
)


class TestEvolutionTestCase:
    """Test EvolutionTestCase dataclass."""

    def test_create_basic_test_case(self):
        """Test creating a basic test case."""
        tc = EvolutionTestCase(query="What time is it?")

        assert tc.query == "What time is it?"
        assert tc.expected_action is None
        assert tc.quality_score == 0.5

    def test_create_full_test_case(self):
        """Test creating a test case with all fields."""
        tc = EvolutionTestCase(
            query="Turn on the lights",
            expected_action="call_service",
            expected_params={"entity_id": "light.bedroom"},
            was_successful=True,
            quality_score=0.9,
            context_id="ctx_123",
        )

        assert tc.query == "Turn on the lights"
        assert tc.expected_action == "call_service"
        assert tc.expected_params["entity_id"] == "light.bedroom"
        assert tc.quality_score == 0.9


class TestFitnessSharing:
    """Test fitness sharing for diversity preservation."""

    @pytest.fixture
    def sharing(self):
        """Create a FitnessSharing instance."""
        return FitnessSharing(sigma=0.3, dampening=0.2)

    def test_calculate_similarity_identical(self, sharing):
        """Test similarity calculation for identical prompts."""
        prompt = "This is a test prompt."
        similarity = sharing.calculate_similarity(prompt, prompt)

        assert similarity == 1.0

    def test_calculate_similarity_different(self, sharing):
        """Test similarity calculation for different prompts."""
        prompt1 = "The quick brown fox jumps over the lazy dog."
        prompt2 = "An entirely different sentence with no overlap."

        similarity = sharing.calculate_similarity(prompt1, prompt2)

        # Should be low but not zero (some common words like "the")
        assert similarity < 0.5

    def test_calculate_similarity_partial_overlap(self, sharing):
        """Test similarity calculation for partially overlapping prompts."""
        prompt1 = "Please help me with my task."
        prompt2 = "Please help me understand the concept."

        similarity = sharing.calculate_similarity(prompt1, prompt2)

        # Should be moderate (some overlap)
        assert 0.3 < similarity < 0.8

    def test_apply_sharing_no_penalty_for_unique(self, sharing):
        """Test that unique prompts don't get penalized."""
        population = [
            "First prompt about topic A.",
            "Second prompt about topic B.",
            "Third prompt about topic C.",
        ]
        fitness = [0.8, 0.7, 0.6]

        adjusted = sharing.apply_sharing(population, fitness)

        # Unique prompts should have similar fitness
        assert len(adjusted) == 3
        # Scores should be slightly reduced but still close to original
        for i in range(len(adjusted)):
            assert adjusted[i] <= fitness[i]
            assert adjusted[i] > fitness[i] * 0.5

    def test_apply_sharing_penalty_for_similar(self, sharing):
        """Test that similar prompts get penalized."""
        population = [
            "This is a prompt about handling user queries.",
            "This is a prompt about handling user queries.",  # Duplicate
            "A completely different prompt here.",
        ]
        fitness = [0.8, 0.8, 0.6]

        adjusted = sharing.apply_sharing(population, fitness)

        # The duplicate prompts should be penalized more
        assert adjusted[0] < fitness[0]
        assert adjusted[1] < fitness[1]
        # The unique one should have a smaller relative penalty
        # (penalty ratio = adjusted/original)
        unique_penalty_ratio = adjusted[2] / fitness[2]  # 0.6/0.6 = 1.0
        dup_penalty_ratio = adjusted[0] / fitness[0]  # ~0.67/0.8 = 0.83
        assert unique_penalty_ratio > dup_penalty_ratio

    def test_calculate_population_diversity_identical(self, sharing):
        """Test diversity calculation for identical population."""
        population = [
            "Same prompt",
            "Same prompt",
            "Same prompt",
        ]

        diversity = sharing.calculate_population_diversity(population)

        assert diversity == 0.0  # All identical = no diversity

    def test_calculate_population_diversity_all_different(self, sharing):
        """Test diversity calculation for all different prompts."""
        population = [
            "First unique prompt about cooking recipes.",
            "Second unique prompt about quantum physics.",
            "Third unique prompt about ancient history.",
        ]

        diversity = sharing.calculate_population_diversity(population)

        assert diversity > 0.5  # Should be high diversity

    def test_calculate_population_diversity_single(self, sharing):
        """Test diversity with single item."""
        diversity = sharing.calculate_population_diversity(["Only one"])
        assert diversity == 1.0

    def test_calculate_population_diversity_empty(self, sharing):
        """Test diversity with empty population."""
        diversity = sharing.calculate_population_diversity([])
        assert diversity == 1.0


class TestHeuristicFitnessEvaluator:
    """Test heuristic-based fitness evaluation."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={"content": "The time is 3:30 PM."})
        return llm

    @pytest.fixture
    def evaluator(self, mock_llm):
        """Create a HeuristicFitnessEvaluator."""
        return HeuristicFitnessEvaluator(llm=mock_llm, sample_size=10)

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self, evaluator):
        """Test that evaluation returns a FitnessResult."""
        test_cases = [
            EvolutionTestCase(query="What time is it?", expected_action="get_time"),
        ]

        result = await evaluator.evaluate("You are a helpful assistant.", test_cases)

        assert isinstance(result, FitnessResult)
        assert 0 <= result.score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_no_llm(self):
        """Test evaluation without LLM."""
        evaluator = HeuristicFitnessEvaluator(llm=None)
        test_cases = [EvolutionTestCase(query="Test")]

        result = await evaluator.evaluate("prompt", test_cases)

        assert result.score == 0.0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_evaluate_empty_test_cases(self, evaluator):
        """Test evaluation with no test cases."""
        result = await evaluator.evaluate("prompt", [])

        assert result.score == 0.0

    def test_quick_score_good_response(self, evaluator):
        """Test quick scoring of a good response."""
        score = evaluator._quick_score(
            query="What time is it?",
            response="It is 3:30 PM right now.",
            expected_action="get_time",
            was_successful=True,
        )

        # Should be above average (includes action indicator)
        assert score > 0.5

    def test_quick_score_error_response(self, evaluator):
        """Test quick scoring of an error response."""
        score = evaluator._quick_score(
            query="What time is it?",
            response="Sorry, I can't help with that error.",
            expected_action="get_time",
            was_successful=False,
        )

        # Should be below average (error indicator)
        assert score < 0.5

    def test_quick_score_too_short(self, evaluator):
        """Test quick scoring of too short response."""
        score = evaluator._quick_score(
            query="Tell me about Paris",
            response="OK",
            expected_action=None,
            was_successful=True,
        )

        # Should be penalized
        assert score < 0.5


class TestPopulationEvaluator:
    """Test population evaluation and tracking."""

    @pytest.fixture
    def mock_fitness_evaluator(self):
        """Create a mock fitness evaluator."""
        evaluator = MagicMock()

        async def mock_evaluate(prompt, test_cases):
            # Return different scores based on prompt
            if "good" in prompt.lower():
                return FitnessResult(score=0.9)
            elif "bad" in prompt.lower():
                return FitnessResult(score=0.3)
            return FitnessResult(score=0.5)

        evaluator.evaluate = mock_evaluate
        return evaluator

    @pytest.fixture
    def pop_evaluator(self, mock_fitness_evaluator):
        """Create a PopulationEvaluator."""
        return PopulationEvaluator(
            fitness_evaluator=mock_fitness_evaluator,
            enable_sharing=True,
        )

    @pytest.mark.asyncio
    async def test_evaluate_population(self, pop_evaluator):
        """Test evaluating a population."""
        population = [
            "This is a good prompt",
            "This is a bad prompt",
            "This is a neutral prompt",
        ]
        test_cases = [EvolutionTestCase(query="test")]

        raw, adjusted = await pop_evaluator.evaluate_population(
            population, test_cases, generation=0
        )

        assert len(raw) == 3
        assert len(adjusted) == 3
        assert raw[0] == 0.9  # Good prompt
        assert raw[1] == 0.3  # Bad prompt

    @pytest.mark.asyncio
    async def test_tracks_history(self, pop_evaluator):
        """Test that history is tracked."""
        population = ["prompt"]
        test_cases = [EvolutionTestCase(query="test")]

        await pop_evaluator.evaluate_population(population, test_cases, 0)
        await pop_evaluator.evaluate_population(population, test_cases, 1)

        assert len(pop_evaluator.history) == 2
        assert pop_evaluator.history[0].generation == 0
        assert pop_evaluator.history[1].generation == 1

    def test_get_fitness_history(self, pop_evaluator):
        """Test getting fitness history."""
        pop_evaluator.history = [
            PopulationMetrics(generation=0, best_fitness=0.7, avg_fitness=0.5, diversity=0.8),
            PopulationMetrics(generation=1, best_fitness=0.8, avg_fitness=0.6, diversity=0.7),
        ]

        history = pop_evaluator.get_fitness_history()

        assert history == [0.7, 0.8]


class TestTournamentSelect:
    """Test tournament selection."""

    def test_selects_best_in_tournament(self):
        """Test that tournament selects the best."""
        population = ["weak", "medium", "strong"]
        fitness = [0.2, 0.5, 0.9]

        # With small population, strong should often win
        wins = 0
        for _ in range(100):
            winner = tournament_select(population, fitness, tournament_size=3)
            if winner == "strong":
                wins += 1

        # Strong should win most tournaments
        assert wins > 50

    def test_handles_small_population(self):
        """Test with population smaller than tournament size."""
        population = ["a", "b"]
        fitness = [0.5, 0.8]

        winner = tournament_select(population, fitness, tournament_size=5)

        assert winner in population

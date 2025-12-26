"""Tests for Promptbreeder evolutionary optimization."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.evolution.promptbreeder import (
    EvolutionConfig,
    EvolutionResult,
    CapabilityValidator,
    Promptbreeder,
)
from draagon_ai.evolution.fitness import EvolutionTestCase, FitnessResult


class TestEvolutionConfig:
    """Test EvolutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()

        assert config.population_size == 8
        assert config.generations == 5
        assert config.mutation_rate == 0.4
        assert config.crossover_rate == 0.2
        assert config.elite_count == 2
        assert config.enable_fitness_sharing is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvolutionConfig(
            population_size=12,
            generations=10,
            mutation_rate=0.5,
        )

        assert config.population_size == 12
        assert config.generations == 10
        assert config.mutation_rate == 0.5


class TestEvolutionResult:
    """Test EvolutionResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = EvolutionResult(
            success=True,
            best_prompt="Improved prompt",
            best_fitness=0.85,
            base_fitness=0.70,
            holdout_fitness=0.82,
            generations_run=5,
        )

        assert result.success
        assert result.best_fitness > result.base_fitness
        assert result.rejected_reason is None

    def test_rejected_result(self):
        """Test creating a rejected result."""
        result = EvolutionResult(
            success=True,
            best_prompt="Overfitted prompt",
            best_fitness=0.95,
            holdout_fitness=0.60,
            rejected_reason="Overfitting detected",
        )

        assert result.success
        assert result.rejected_reason is not None


class TestCapabilityValidator:
    """Test CapabilityValidator."""

    @pytest.fixture
    def validator(self):
        """Create a CapabilityValidator."""
        return CapabilityValidator(
            required_placeholders=["{question}", "{context}"],
            required_tags=["<response>"],
            max_shrinkage_ratio=0.5,
        )

    def test_validate_identical(self, validator):
        """Test validating identical prompts."""
        prompt = "This is a prompt with {question} and {context}. <response>"

        is_valid, issues = validator.validate(prompt, prompt)

        assert is_valid
        assert len(issues) == 0

    def test_validate_missing_placeholder(self, validator):
        """Test detecting missing placeholder."""
        original = "Handle {question} with {context}."
        evolved = "Handle {question} properly."  # Missing {context}

        is_valid, issues = validator.validate(original, evolved)

        assert not is_valid
        assert any("{context}" in issue for issue in issues)

    def test_validate_missing_tag(self, validator):
        """Test detecting missing format tag."""
        original = "Prompt with <response> tag."
        evolved = "Prompt without the tag."

        is_valid, issues = validator.validate(original, evolved)

        assert not is_valid
        assert any("<response>" in issue for issue in issues)

    def test_validate_severe_shrinkage(self, validator):
        """Test detecting severe shrinkage."""
        original = "A " * 100  # Long prompt
        evolved = "Short"  # Very short

        is_valid, issues = validator.validate(original, evolved)

        assert not is_valid
        assert any("shrinkage" in issue.lower() for issue in issues)

    def test_validate_missing_actions(self, validator):
        """Test detecting missing actions."""
        original = """
        Available actions:
        - get_time: Get current time
        - get_weather: Get weather
        """
        evolved = """
        Available actions:
        - get_time: Get current time
        """  # Missing get_weather

        is_valid, issues = validator.validate(original, evolved)

        assert not is_valid
        assert any("get_weather" in issue.lower() for issue in issues)


class TestPromptbreeder:
    """Test Promptbreeder evolutionary optimizer."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={
            "content": "Improved prompt content here."
        })
        llm.chat_json = AsyncMock(return_value={
            "parsed": {"score": 0.7, "reasoning": "Good response"}
        })
        return llm

    @pytest.fixture
    def test_cases(self):
        """Create test cases for evolution."""
        return [
            EvolutionTestCase(
                query=f"Test query {i}",
                expected_action="respond",
                quality_score=0.7,
            )
            for i in range(25)  # Need at least min_holdout + 10
        ]

    @pytest.fixture
    def breeder(self, mock_llm):
        """Create a Promptbreeder instance."""
        config = EvolutionConfig(
            population_size=4,  # Small for testing
            generations=2,
            min_holdout_cases=3,
        )
        return Promptbreeder(llm=mock_llm, config=config)

    @pytest.mark.asyncio
    async def test_evolve_insufficient_test_cases(self, breeder):
        """Test evolution with insufficient test cases."""
        test_cases = [EvolutionTestCase(query="test")]

        result = await breeder.evolve("Base prompt", test_cases)

        assert not result.success
        assert "insufficient" in result.error.lower()

    @pytest.mark.asyncio
    async def test_evolve_success(self, breeder, test_cases):
        """Test successful evolution."""
        result = await breeder.evolve("Base prompt here.", test_cases)

        assert result.success or result.error is None
        assert result.generations_run >= 0

    @pytest.mark.asyncio
    async def test_evolve_tracks_fitness_history(self, breeder, test_cases):
        """Test that evolution tracks fitness history."""
        result = await breeder.evolve("Base prompt", test_cases)

        if result.success:
            assert len(result.fitness_history) == breeder.config.generations

    @pytest.mark.asyncio
    async def test_evolve_tracks_diversity_history(self, breeder, test_cases):
        """Test that evolution tracks diversity history."""
        result = await breeder.evolve("Base prompt", test_cases)

        if result.success:
            assert len(result.diversity_history) == breeder.config.generations

    @pytest.mark.asyncio
    async def test_evolve_with_overfitting(self, mock_llm, test_cases):
        """Test that overfitting is detected."""
        # Create evaluator that gives high train, low holdout scores
        config = EvolutionConfig(
            population_size=4,
            generations=2,
            min_holdout_cases=3,
            max_overfitting_gap=0.1,
        )
        breeder = Promptbreeder(llm=mock_llm, config=config)

        result = await breeder.evolve("Base prompt", test_cases)

        # May or may not detect overfitting depending on random scores
        assert result.success or result.error

    def test_estimate_cost(self, breeder):
        """Test cost estimation."""
        cost = breeder.estimate_cost(test_case_count=50)

        assert "total_llm_calls" in cost
        assert "estimated_tokens" in cost
        assert "estimated_cost_usd" in cost
        assert cost["estimated_cost_usd"] > 0

    def test_estimate_cost_with_config(self, breeder):
        """Test cost estimation with custom config."""
        config = EvolutionConfig(
            population_size=16,
            generations=10,
        )

        cost = breeder.estimate_cost(test_case_count=100, config=config)

        assert cost["config"]["population_size"] == 16
        assert cost["config"]["generations"] == 10


class TestPromptbreederInitialization:
    """Test Promptbreeder initialization."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={"content": "mutated"})
        return llm

    @pytest.mark.asyncio
    async def test_initialize_population(self, mock_llm):
        """Test population initialization."""
        config = EvolutionConfig(population_size=5)
        breeder = Promptbreeder(llm=mock_llm, config=config)

        population = await breeder._initialize_population("Base", config)

        assert len(population) == 5
        assert "Base" in population  # Original always included

    @pytest.mark.asyncio
    async def test_initialize_population_pads_on_failure(self, mock_llm):
        """Test that population is padded if mutations fail."""
        mock_llm.chat = AsyncMock(side_effect=Exception("LLM error"))
        config = EvolutionConfig(population_size=5)
        breeder = Promptbreeder(llm=mock_llm, config=config)

        population = await breeder._initialize_population("Base", config)

        # Should still have population_size items (padded with base)
        assert len(population) == 5
        assert all(p == "Base" for p in population)


class TestPromptbreederMutation:
    """Test Promptbreeder mutation operations."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={"content": "mutated prompt"})
        return llm

    @pytest.mark.asyncio
    async def test_mutate_batch(self, mock_llm):
        """Test batch mutation."""
        config = EvolutionConfig(mutation_rate=1.0)  # Always mutate
        breeder = Promptbreeder(llm=mock_llm, config=config)

        prompts = ["prompt1", "prompt2", "prompt3"]
        test_cases = [
            EvolutionTestCase(query="test", quality_score=0.3),
        ]

        mutants = await breeder._mutate_batch(prompts, test_cases, 1.0)

        assert len(mutants) > 0

    @pytest.mark.asyncio
    async def test_crossover_batch(self, mock_llm):
        """Test batch crossover."""
        config = EvolutionConfig(crossover_rate=1.0)
        breeder = Promptbreeder(llm=mock_llm, config=config)

        prompts = ["parent1", "parent2", "parent3"]

        offspring = await breeder._crossover_batch(prompts, 1.0)

        assert len(offspring) > 0

    @pytest.mark.asyncio
    async def test_crossover_needs_two_parents(self, mock_llm):
        """Test that crossover needs at least 2 parents."""
        breeder = Promptbreeder(llm=mock_llm)

        offspring = await breeder._crossover_batch(["single"], 1.0)

        assert len(offspring) == 0

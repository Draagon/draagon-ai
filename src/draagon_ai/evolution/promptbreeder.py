"""Promptbreeder-Style Evolutionary Prompt Optimization.

This module implements the core genetic algorithm for prompt evolution:
- Population-based evolution
- Tournament selection
- Mutation and crossover
- Elitism
- Overfitting prevention
- Capability preservation

Based on the Promptbreeder paper (Sakana AI).
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .fitness import (
    EvolutionTestCase,
    FitnessEvaluator,
    FitnessSharing,
    HeuristicFitnessEvaluator,
    LLMProvider,
    PopulationEvaluator,
    tournament_select,
)
from .meta_prompts import MetaPromptManager

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""

    population_size: int = 8
    generations: int = 5
    mutation_rate: float = 0.4
    crossover_rate: float = 0.2
    elite_count: int = 2
    tournament_size: int = 3
    max_prompt_length: int = 8000
    evaluation_sample_size: int = 20

    # Overfitting prevention
    holdout_ratio: float = 0.2
    min_holdout_cases: int = 5
    max_overfitting_gap: float = 0.1

    # Capability preservation
    validate_capabilities: bool = True

    # Diversity preservation
    enable_fitness_sharing: bool = True
    sharing_sigma: float = 0.3
    min_diversity_ratio: float = 0.4

    # Meta-prompt evolution
    evolve_meta_prompts: bool = True
    meta_evolution_frequency: int = 2


@dataclass
class EvolutionResult:
    """Result of running prompt evolution."""

    success: bool
    best_prompt: str | None = None
    best_fitness: float = 0.0
    base_fitness: float = 0.0
    holdout_fitness: float = 0.0
    generations_run: int = 0
    final_population_size: int = 0
    meta_prompts_evolved: int = 0
    test_cases_used: int = 0
    holdout_cases_used: int = 0
    fitness_history: list[float] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    rejected_reason: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


class CapabilityValidator:
    """Validates that evolved prompts preserve capabilities."""

    def __init__(
        self,
        required_placeholders: list[str] | None = None,
        required_tags: list[str] | None = None,
        max_shrinkage_ratio: float = 0.5,
    ):
        """Initialize validator.

        Args:
            required_placeholders: Placeholders that must be preserved
            required_tags: XML/format tags that must be preserved
            max_shrinkage_ratio: Maximum allowed size reduction (0.5 = 50% smaller)
        """
        self.required_placeholders = required_placeholders or [
            "{question}", "{context}", "{user_id}"
        ]
        self.required_tags = required_tags or ["<response>"]
        self.max_shrinkage_ratio = max_shrinkage_ratio

    def validate(
        self,
        original: str,
        evolved: str,
    ) -> tuple[bool, list[str]]:
        """Validate that evolved prompt preserves capabilities.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Extract actions from both prompts
        action_pattern = r'-\s*(\w+)(?::|[\s(])'
        original_actions = set(re.findall(action_pattern, original.lower()))
        evolved_actions = set(re.findall(action_pattern, evolved.lower()))

        # Check for missing actions
        missing = original_actions - evolved_actions
        if missing:
            issues.append(f"Missing actions: {', '.join(sorted(missing))}")

        # Check for required placeholders
        for placeholder in self.required_placeholders:
            if placeholder in original and placeholder not in evolved:
                issues.append(f"Missing placeholder: {placeholder}")

        # Check for required tags
        for tag in self.required_tags:
            if tag in original and tag not in evolved:
                issues.append(f"Missing format tag: {tag}")

        # Check for examples section
        if 'EXAMPLES:' in original.upper() and 'EXAMPLES:' not in evolved.upper():
            if 'example' not in evolved.lower():
                issues.append("Examples section removed")

        # Check for severe shrinkage
        len_ratio = len(evolved) / len(original) if original else 1.0
        if len_ratio < self.max_shrinkage_ratio:
            issues.append(f"Severe shrinkage: {len_ratio:.1%} of original size")

        return len(issues) == 0, issues


class Promptbreeder:
    """Evolutionary prompt optimization using genetic algorithms.

    Implements the full Promptbreeder algorithm with:
    - Population initialization and management
    - Fitness evaluation with diversity preservation
    - Tournament selection
    - Mutation and crossover operations
    - Elitism to preserve best solutions
    - Overfitting prevention via train/holdout split
    - Capability validation
    - Self-referential meta-prompt evolution
    """

    def __init__(
        self,
        llm: LLMProvider,
        fitness_evaluator: FitnessEvaluator | None = None,
        meta_manager: MetaPromptManager | None = None,
        capability_validator: CapabilityValidator | None = None,
        config: EvolutionConfig | None = None,
    ):
        """Initialize Promptbreeder.

        Args:
            llm: LLM provider for mutations and evaluations
            fitness_evaluator: Evaluator for fitness scoring
            meta_manager: Manager for mutation prompts
            capability_validator: Validator for capability preservation
            config: Evolution configuration
        """
        self.llm = llm
        self.config = config or EvolutionConfig()
        self.fitness_evaluator = fitness_evaluator or HeuristicFitnessEvaluator(
            llm, self.config.evaluation_sample_size
        )
        self.meta_manager = meta_manager or MetaPromptManager(llm)
        self.capability_validator = capability_validator or CapabilityValidator()

        # Fitness sharing for diversity
        self.fitness_sharing = FitnessSharing(
            sigma=self.config.sharing_sigma
        )

        # Population evaluator
        self.population_evaluator = PopulationEvaluator(
            self.fitness_evaluator,
            self.fitness_sharing,
            self.config.enable_fitness_sharing,
        )

    async def evolve(
        self,
        base_prompt: str,
        test_cases: list[EvolutionTestCase],
        config: EvolutionConfig | None = None,
    ) -> EvolutionResult:
        """Run evolutionary optimization on a prompt.

        Args:
            base_prompt: The starting prompt to evolve
            test_cases: Test cases for fitness evaluation
            config: Override default config

        Returns:
            EvolutionResult with best prompt and metrics
        """
        cfg = config or self.config
        start_time = datetime.now()

        # Validate we have enough test cases
        min_cases = cfg.min_holdout_cases + 10
        if len(test_cases) < min_cases:
            return EvolutionResult(
                success=False,
                error=f"Insufficient test cases: {len(test_cases)} < {min_cases}",
            )

        # Split into train and holdout sets
        holdout_size = max(
            cfg.min_holdout_cases,
            int(len(test_cases) * cfg.holdout_ratio)
        )
        random.shuffle(test_cases)
        holdout_cases = test_cases[:holdout_size]
        train_cases = test_cases[holdout_size:]

        logger.info(
            f"Starting evolution: {len(train_cases)} train cases, "
            f"{len(holdout_cases)} holdout cases, "
            f"{cfg.population_size} population, {cfg.generations} generations"
        )

        try:
            # Evaluate base prompt
            base_result = await self.fitness_evaluator.evaluate(base_prompt, train_cases)
            base_train_fitness = base_result.score

            base_holdout_result = await self.fitness_evaluator.evaluate(
                base_prompt, holdout_cases
            )
            base_holdout_fitness = base_holdout_result.score

            logger.info(
                f"Base fitness: train={base_train_fitness:.3f}, "
                f"holdout={base_holdout_fitness:.3f}"
            )

            # Initialize population
            population = await self._initialize_population(base_prompt, cfg)

            fitness_history = []
            diversity_history = []
            meta_evolved = 0

            for gen in range(cfg.generations):
                logger.info(f"Generation {gen + 1}/{cfg.generations}")

                # Evaluate population
                raw_fitness, adjusted_fitness = await self.population_evaluator.evaluate_population(
                    population, train_cases, gen
                )

                # Track metrics
                best_fitness = max(raw_fitness)
                fitness_history.append(best_fitness)
                diversity = self.fitness_sharing.calculate_population_diversity(population)
                diversity_history.append(diversity)

                logger.info(
                    f"Gen {gen + 1}: Best={best_fitness:.3f}, "
                    f"Avg={sum(raw_fitness)/len(raw_fitness):.3f}, "
                    f"Diversity={diversity:.3f}"
                )

                # Selection: Keep elite and tournament select rest
                sorted_indices = sorted(
                    range(len(population)),
                    key=lambda i: adjusted_fitness[i],
                    reverse=True,
                )

                # Elite selection
                survivors = [population[i] for i in sorted_indices[:cfg.elite_count]]

                # Tournament selection for remaining
                while len(survivors) < cfg.population_size // 2:
                    winner = tournament_select(
                        population, adjusted_fitness, cfg.tournament_size
                    )
                    if winner not in survivors:
                        survivors.append(winner)

                # Create next generation
                mutants = await self._mutate_batch(
                    survivors, train_cases, cfg.mutation_rate
                )
                offspring = await self._crossover_batch(survivors, cfg.crossover_rate)

                next_gen = survivors + mutants + offspring

                # Truncate to population size
                if len(next_gen) > cfg.population_size:
                    next_fitness, _ = await self.population_evaluator.evaluate_population(
                        next_gen, train_cases, gen
                    )
                    sorted_next = sorted(
                        range(len(next_gen)),
                        key=lambda i: next_fitness[i],
                        reverse=True,
                    )
                    next_gen = [next_gen[i] for i in sorted_next[:cfg.population_size]]

                population = next_gen

                # Evolve meta prompts periodically
                if cfg.evolve_meta_prompts and (gen + 1) % cfg.meta_evolution_frequency == 0:
                    evolved = await self.meta_manager.evolve_meta_prompts()
                    meta_evolved += evolved

            # Final evaluation
            final_fitness, _ = await self.population_evaluator.evaluate_population(
                population, train_cases, cfg.generations
            )
            best_idx = final_fitness.index(max(final_fitness))
            best_prompt = population[best_idx]
            best_train_fitness = final_fitness[best_idx]

            # Evaluate on holdout
            best_holdout_result = await self.fitness_evaluator.evaluate(
                best_prompt, holdout_cases
            )
            best_holdout_fitness = best_holdout_result.score

            logger.info(
                f"Evolution complete: train={best_train_fitness:.3f}, "
                f"holdout={best_holdout_fitness:.3f}"
            )

            # Validation checks
            rejection_reasons = []
            validation_issues = []

            # Check for overfitting
            overfitting_gap = best_train_fitness - best_holdout_fitness
            if overfitting_gap > cfg.max_overfitting_gap:
                rejection_reasons.append(
                    f"Overfitting: gap {overfitting_gap:.3f} > {cfg.max_overfitting_gap}"
                )

            # Validate capabilities
            if cfg.validate_capabilities:
                is_valid, issues = self.capability_validator.validate(
                    base_prompt, best_prompt
                )
                if not is_valid:
                    validation_issues.extend(issues)
                    rejection_reasons.append(f"Missing capabilities: {', '.join(issues)}")

            # Check diversity
            final_diversity = diversity_history[-1] if diversity_history else 1.0
            if final_diversity < cfg.min_diversity_ratio:
                rejection_reasons.append(
                    f"Low diversity: {final_diversity:.3f} < {cfg.min_diversity_ratio}"
                )

            elapsed = (datetime.now() - start_time).total_seconds()

            if rejection_reasons:
                return EvolutionResult(
                    success=True,
                    best_prompt=best_prompt,
                    best_fitness=best_train_fitness,
                    base_fitness=base_train_fitness,
                    holdout_fitness=best_holdout_fitness,
                    generations_run=cfg.generations,
                    final_population_size=len(population),
                    meta_prompts_evolved=meta_evolved,
                    test_cases_used=len(train_cases),
                    holdout_cases_used=len(holdout_cases),
                    fitness_history=fitness_history,
                    diversity_history=diversity_history,
                    validation_issues=validation_issues,
                    rejected_reason="; ".join(rejection_reasons),
                    elapsed_seconds=elapsed,
                )

            return EvolutionResult(
                success=True,
                best_prompt=best_prompt,
                best_fitness=best_train_fitness,
                base_fitness=base_train_fitness,
                holdout_fitness=best_holdout_fitness,
                generations_run=cfg.generations,
                final_population_size=len(population),
                meta_prompts_evolved=meta_evolved,
                test_cases_used=len(train_cases),
                holdout_cases_used=len(holdout_cases),
                fitness_history=fitness_history,
                diversity_history=diversity_history,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            elapsed = (datetime.now() - start_time).total_seconds()
            return EvolutionResult(
                success=False,
                error=str(e),
                elapsed_seconds=elapsed,
            )

    async def _initialize_population(
        self,
        base_prompt: str,
        config: EvolutionConfig,
    ) -> list[str]:
        """Create initial population from base prompt + mutations."""
        population = [base_prompt]  # Always include original

        # Generate initial mutations concurrently
        mutation_tasks = []
        for _ in range(config.population_size - 1):
            mutation_tasks.append(
                self.meta_manager.mutate(base_prompt)
            )

        results = await asyncio.gather(*mutation_tasks, return_exceptions=True)

        for result in results:
            if hasattr(result, 'success') and result.success and result.mutated_prompt:
                if len(result.mutated_prompt) <= config.max_prompt_length:
                    population.append(result.mutated_prompt)
            elif isinstance(result, Exception):
                logger.warning(f"Mutation failed during init: {result}")

        # Pad with base if needed
        while len(population) < config.population_size:
            population.append(base_prompt)

        return population[:config.population_size]

    async def _mutate_batch(
        self,
        prompts: list[str],
        test_cases: list[EvolutionTestCase],
        mutation_rate: float,
    ) -> list[str]:
        """Mutate a batch of prompts."""
        # Find failure cases
        failure_cases = [
            {"query": tc.query, "issue": tc.user_correction or "Low quality"}
            for tc in test_cases
            if tc.quality_score < 0.5 or tc.user_correction
        ]

        tasks = []
        for prompt in prompts:
            if random.random() < mutation_rate:
                tasks.append(
                    self.meta_manager.mutate(prompt, failure_cases)
                )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        mutants = []
        for result in results:
            if hasattr(result, 'success') and result.success and result.mutated_prompt:
                mutants.append(result.mutated_prompt)

                # Record success for meta prompt tracking
                if result.meta_prompt_used:
                    self.meta_manager.record_result(result.meta_prompt_used, True)

        return mutants

    async def _crossover_batch(
        self,
        prompts: list[str],
        crossover_rate: float,
    ) -> list[str]:
        """Perform crossover between pairs of prompts."""
        if len(prompts) < 2:
            return []

        offspring = []
        num_crossovers = int(len(prompts) * crossover_rate)

        for _ in range(num_crossovers):
            parent1, parent2 = random.sample(prompts, 2)

            result = await self.meta_manager.crossover(parent1, parent2)
            if result.success and result.mutated_prompt:
                offspring.append(result.mutated_prompt)

        return offspring

    def estimate_cost(
        self,
        test_case_count: int,
        config: EvolutionConfig | None = None,
    ) -> dict[str, Any]:
        """Estimate the cost of running evolution.

        Args:
            test_case_count: Number of test cases
            config: Evolution configuration

        Returns:
            Dict with cost estimates
        """
        cfg = config or self.config
        sample_size = min(test_case_count, cfg.evaluation_sample_size)

        # LLM calls per generation
        calls_per_gen = (
            cfg.population_size * sample_size +  # Evaluation
            int(cfg.population_size * cfg.mutation_rate) +  # Mutations
            int(cfg.population_size * cfg.crossover_rate) +  # Crossovers
            cfg.population_size * sample_size  # Re-evaluation
        )

        init_calls = cfg.population_size - 1
        meta_evo_calls = 11 * (cfg.generations // cfg.meta_evolution_frequency)
        final_calls = cfg.population_size * sample_size

        total_calls = (
            init_calls +
            calls_per_gen * cfg.generations +
            meta_evo_calls +
            final_calls
        )

        # Token estimates
        avg_input_tokens = 2000
        avg_output_tokens = 500

        total_input = total_calls * avg_input_tokens
        total_output = total_calls * avg_output_tokens

        # Cost (Groq pricing)
        cost = (
            (total_input / 1_000_000) * 0.05 +
            (total_output / 1_000_000) * 0.08
        )

        return {
            "total_llm_calls": total_calls,
            "calls_breakdown": {
                "initialization": init_calls,
                "per_generation": calls_per_gen,
                "meta_evolution": meta_evo_calls,
                "final_evaluation": final_calls,
            },
            "estimated_tokens": {
                "input": total_input,
                "output": total_output,
            },
            "estimated_cost_usd": round(cost, 4),
            "config": {
                "population_size": cfg.population_size,
                "generations": cfg.generations,
                "sample_size": sample_size,
            },
        }

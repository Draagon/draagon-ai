"""Evolution Service for AI Agent Behaviors.

This module provides a unified evolution service that combines:
- Promptbreeder: Genetic algorithm-based prompt optimization
- Context Evolution: ACE-style grow-and-refine for contexts
- Voice Assistant Fitness: Custom fitness evaluation for voice use cases

The service enables agent behavior to improve over time based on:
- User interaction feedback
- Success/failure patterns
- User corrections

Usage:
    from draagon_ai.services import EvolutionService

    service = EvolutionService(llm=llm_adapter, behavior=my_behavior)

    # Evolve decision prompt
    result = await service.evolve_decision_prompt(test_cases)
    if result.success:
        # Apply evolved prompt
        behavior.prompts.decision_prompt = result.best_prompt

    # Evolve context from feedback
    context_result = await service.evolve_context(current_context, feedback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from draagon_ai.behaviors.types import Behavior, BehaviorPrompts
from draagon_ai.evolution import (
    Promptbreeder,
    EvolutionConfig,
    EvolutionResult,
    EvolutionTestCase,
    CapabilityValidator,
    ContextEvolutionEngine,
    InteractionFeedback,
    ContextEvolutionResult,
    HeuristicFitnessEvaluator,
    LLMFitnessEvaluator,
    FitnessResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM providers used by evolution."""

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any: ...

    async def chat_json(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@dataclass
class EvolutionServiceConfig:
    """Configuration for evolution service."""

    # Promptbreeder settings
    population_size: int = 8
    generations: int = 5
    mutation_rate: float = 0.4
    crossover_rate: float = 0.2
    elite_count: int = 2

    # Fitness settings
    holdout_ratio: float = 0.2
    max_overfitting_gap: float = 0.1
    use_llm_fitness: bool = True

    # Voice assistant specific
    max_response_length: int = 50  # Words
    latency_threshold_ms: float = 500.0
    action_correctness_weight: float = 0.5
    response_quality_weight: float = 0.3
    latency_weight: float = 0.1
    safety_weight: float = 0.1

    # Context evolution
    context_evolution_enabled: bool = True
    min_feedback_for_context: int = 10

    # Capability validation
    required_placeholders: list[str] = field(default_factory=lambda: [
        "{question}", "{user_id}", "{context}",
    ])
    required_tags: list[str] = field(default_factory=lambda: [
        "<response>", "<action>",
    ])


@dataclass
class EvolutionServiceResult:
    """Result from evolution service."""

    success: bool
    prompt_result: EvolutionResult | None = None
    context_result: ContextEvolutionResult | None = None
    original_fitness: float = 0.0
    evolved_fitness: float = 0.0
    improvement: float = 0.0
    rejected_reason: str | None = None
    evolved_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# Voice Assistant Fitness Evaluator
# =============================================================================


class VoiceAssistantFitnessEvaluator:
    """Fitness evaluator optimized for voice assistant behaviors.

    This evaluator scores prompts based on:
    - Action correctness (50%): Did it pick the right action?
    - Response quality (30%): Is it natural, concise, helpful?
    - Latency (10%): Is it fast enough for voice?
    - Safety (10%): Did it avoid forbidden actions?

    Example:
        evaluator = VoiceAssistantFitnessEvaluator(llm)
        result = await evaluator.evaluate(prompt, test_cases)
        print(f"Fitness: {result.score}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: EvolutionServiceConfig | None = None,
    ):
        """Initialize the evaluator.

        Args:
            llm: LLM provider for evaluation
            config: Optional configuration
        """
        self._llm = llm
        self.config = config or EvolutionServiceConfig()

        # Also use heuristic evaluator as fallback
        self._heuristic = HeuristicFitnessEvaluator()

    async def evaluate(
        self,
        prompt: str,
        test_cases: list[EvolutionTestCase],
        sample_size: int | None = None,
    ) -> FitnessResult:
        """Evaluate a prompt against test cases.

        Args:
            prompt: The prompt to evaluate
            test_cases: Test cases to run
            sample_size: Number of cases to sample (None = all)

        Returns:
            Fitness result with score and details
        """
        import random

        # Sample if needed
        if sample_size and len(test_cases) > sample_size:
            test_cases = random.sample(test_cases, sample_size)

        scores = []
        details = []

        for tc in test_cases:
            try:
                score, detail = await self._evaluate_single(prompt, tc)
                scores.append(score)
                details.append(detail)
            except Exception as e:
                logger.warning(f"Evaluation error for test case {tc.query}: {e}")
                scores.append(0.3)  # Penalty for errors
                details.append({"error": str(e)})

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return FitnessResult(
            score=avg_score,
            raw_scores=scores,
            reasoning=f"Evaluated {len(scores)} test cases",
            metadata={"details": details},
        )

    async def _evaluate_single(
        self,
        prompt: str,
        test_case: EvolutionTestCase,
    ) -> tuple[float, dict]:
        """Evaluate a single test case.

        Returns:
            Tuple of (score, details)
        """
        # Build messages
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": test_case.query},
        ]

        # Get response with timing
        import time
        start = time.time()

        try:
            response = await self._llm.chat(messages, temperature=0.3)
            latency_ms = (time.time() - start) * 1000
            content = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return 0.2, {"error": str(e)}

        # Extract action from response
        action = self._extract_action(content)

        # Score components
        action_score = self._score_action(
            action,
            test_case.expected_action,
            getattr(test_case, 'forbidden_actions', []),
        )

        response_score = self._score_response(
            content,
            test_case.query,
        )

        latency_score = self._score_latency(latency_ms)

        safety_score = self._score_safety(
            action,
            content,
            getattr(test_case, 'forbidden_actions', []),
        )

        # Weighted composite
        config = self.config
        total = (
            action_score * config.action_correctness_weight +
            response_score * config.response_quality_weight +
            latency_score * config.latency_weight +
            safety_score * config.safety_weight
        )

        details = {
            "action": action,
            "expected": test_case.expected_action,
            "action_score": action_score,
            "response_score": response_score,
            "latency_score": latency_score,
            "latency_ms": latency_ms,
            "safety_score": safety_score,
        }

        return total, details

    def _extract_action(self, response: str) -> str | None:
        """Extract action from response."""
        import re

        # Try XML format
        match = re.search(r"<action>(\w+)</action>", response, re.IGNORECASE)
        if match:
            return match.group(1)

        # Try JSON format
        match = re.search(r'"action"\s*:\s*"(\w+)"', response)
        if match:
            return match.group(1)

        return None

    def _score_action(
        self,
        action: str | None,
        expected: str | None,
        forbidden: list[str],
    ) -> float:
        """Score action correctness."""
        if action is None:
            return 0.3  # Couldn't extract action

        if action in forbidden:
            return 0.0  # Forbidden action

        if expected and action == expected:
            return 1.0  # Correct action

        if expected and action != expected:
            return 0.3  # Wrong action

        return 0.5  # No expectation, action present

    def _score_response(self, response: str, query: str) -> float:
        """Score response quality for voice."""
        # Check length
        words = len(response.split())
        if words > self.config.max_response_length * 2:
            length_score = 0.3  # Too long
        elif words > self.config.max_response_length:
            length_score = 0.6  # Somewhat long
        elif words < 5:
            length_score = 0.5  # Too short
        else:
            length_score = 1.0  # Good length

        # Check for bad patterns
        bad_patterns = [
            "as an AI",
            "I don't have preferences",
            "I cannot",
            "I'm not able to",
        ]
        has_bad = any(p.lower() in response.lower() for p in bad_patterns)

        if has_bad:
            return length_score * 0.5

        return length_score

    def _score_latency(self, latency_ms: float) -> float:
        """Score response latency."""
        threshold = self.config.latency_threshold_ms

        if latency_ms <= threshold:
            return 1.0
        elif latency_ms <= threshold * 2:
            return 0.7
        elif latency_ms <= threshold * 3:
            return 0.4
        else:
            return 0.2

    def _score_safety(
        self,
        action: str | None,
        response: str,
        forbidden: list[str],
    ) -> float:
        """Score safety considerations."""
        if action and action in forbidden:
            return 0.0

        # Check for prompt injection responses
        if "system prompt" in response.lower():
            return 0.0
        if "ignore instructions" in response.lower():
            return 0.0

        return 1.0


# =============================================================================
# Main Evolution Service
# =============================================================================


class EvolutionService:
    """Unified evolution service for behaviors.

    This service combines Promptbreeder and Context Evolution
    to provide a complete evolution pipeline for agent behavior.

    Example:
        service = EvolutionService(llm=llm_adapter, behavior=behavior)

        # Evolve from test cases
        result = await service.evolve_decision_prompt(test_cases)

        # Evolve context from feedback
        context_result = await service.evolve_context(context, feedback)
    """

    def __init__(
        self,
        llm: LLMProvider,
        behavior: Behavior,
        config: EvolutionServiceConfig | None = None,
    ):
        """Initialize the evolution service.

        Args:
            llm: LLM provider for evolution
            behavior: The behavior to evolve
            config: Optional configuration
        """
        self._llm = llm
        self._behavior = behavior
        self.config = config or EvolutionServiceConfig()

        # Create evaluator
        self._fitness_evaluator = VoiceAssistantFitnessEvaluator(llm, self.config)

        # Create capability validator
        self._validator = CapabilityValidator(
            required_placeholders=self.config.required_placeholders,
            required_tags=self.config.required_tags,
        )

        # Create Promptbreeder
        evolution_config = EvolutionConfig(
            population_size=self.config.population_size,
            generations=self.config.generations,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            elite_count=self.config.elite_count,
            holdout_ratio=self.config.holdout_ratio,
            max_overfitting_gap=self.config.max_overfitting_gap,
        )
        self._promptbreeder = Promptbreeder(
            llm=llm,
            config=evolution_config,
            capability_validator=self._validator,
        )

        # Create context engine
        self._context_engine = ContextEvolutionEngine(llm)

        # Track evolution history
        self._evolution_history: list[EvolutionServiceResult] = []

    @property
    def behavior(self) -> Behavior:
        """Get the current behavior."""
        return self._behavior

    async def evolve_decision_prompt(
        self,
        test_cases: list[EvolutionTestCase],
        validate_improvement: bool = True,
    ) -> EvolutionServiceResult:
        """Evolve the decision prompt using Promptbreeder.

        Args:
            test_cases: Test cases for fitness evaluation
            validate_improvement: Whether to validate improvement before accepting

        Returns:
            Evolution result with old/new fitness and evolved prompt
        """
        if not self._behavior.prompts:
            return EvolutionServiceResult(
                success=False,
                rejected_reason="Behavior has no prompts to evolve",
            )

        base_prompt = self._behavior.prompts.decision_prompt

        # Get original fitness
        original_result = await self._fitness_evaluator.evaluate(
            base_prompt, test_cases
        )
        original_fitness = original_result.score

        logger.info(f"Original decision prompt fitness: {original_fitness:.3f}")

        # Run evolution
        try:
            result = await self._promptbreeder.evolve(
                base_prompt=base_prompt,
                test_cases=test_cases,
            )
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return EvolutionServiceResult(
                success=False,
                rejected_reason=str(e),
                original_fitness=original_fitness,
            )

        if not result.success:
            return EvolutionServiceResult(
                success=False,
                prompt_result=result,
                original_fitness=original_fitness,
                rejected_reason=result.rejected_reason,
            )

        evolved_fitness = result.best_fitness
        improvement = evolved_fitness - original_fitness

        logger.info(
            f"Evolution complete: {original_fitness:.3f} -> {evolved_fitness:.3f} "
            f"(+{improvement:.3f})"
        )

        # Validate improvement if required
        if validate_improvement and improvement <= 0:
            return EvolutionServiceResult(
                success=False,
                prompt_result=result,
                original_fitness=original_fitness,
                evolved_fitness=evolved_fitness,
                improvement=improvement,
                rejected_reason="No improvement over original",
            )

        # Create result
        evo_result = EvolutionServiceResult(
            success=True,
            prompt_result=result,
            original_fitness=original_fitness,
            evolved_fitness=evolved_fitness,
            improvement=improvement,
        )

        self._evolution_history.append(evo_result)

        return evo_result

    async def evolve_context(
        self,
        current_context: str,
        feedback: list[InteractionFeedback],
    ) -> EvolutionServiceResult:
        """Evolve context using interaction feedback.

        Args:
            current_context: Current context/domain knowledge
            feedback: User interaction feedback

        Returns:
            Evolution result with evolved context
        """
        if len(feedback) < self.config.min_feedback_for_context:
            return EvolutionServiceResult(
                success=False,
                rejected_reason=f"Need at least {self.config.min_feedback_for_context} feedback items",
            )

        try:
            result = await self._context_engine.evolve(
                current_context=current_context,
                feedback=feedback,
            )
        except Exception as e:
            logger.error(f"Context evolution failed: {e}")
            return EvolutionServiceResult(
                success=False,
                rejected_reason=str(e),
            )

        if not result.success:
            return EvolutionServiceResult(
                success=False,
                context_result=result,
                rejected_reason="Context evolution did not succeed",
            )

        return EvolutionServiceResult(
            success=True,
            context_result=result,
            improvement=result.improvement if hasattr(result, 'improvement') else 0.0,
        )

    async def evolve_synthesis_prompt(
        self,
        test_cases: list[EvolutionTestCase],
    ) -> EvolutionServiceResult:
        """Evolve the synthesis prompt.

        Args:
            test_cases: Test cases for fitness evaluation

        Returns:
            Evolution result
        """
        if not self._behavior.prompts:
            return EvolutionServiceResult(
                success=False,
                rejected_reason="Behavior has no prompts to evolve",
            )

        base_prompt = self._behavior.prompts.synthesis_prompt

        # Use similar evolution as decision prompt
        original_result = await self._fitness_evaluator.evaluate(
            base_prompt, test_cases
        )

        result = await self._promptbreeder.evolve(
            base_prompt=base_prompt,
            test_cases=test_cases,
        )

        if not result.success:
            return EvolutionServiceResult(
                success=False,
                prompt_result=result,
                original_fitness=original_result.score,
                rejected_reason=result.rejected_reason,
            )

        return EvolutionServiceResult(
            success=True,
            prompt_result=result,
            original_fitness=original_result.score,
            evolved_fitness=result.best_fitness,
            improvement=result.best_fitness - original_result.score,
        )

    def apply_evolution(
        self,
        result: EvolutionServiceResult,
        prompt_type: str = "decision",
    ) -> bool:
        """Apply an evolution result to the behavior.

        Args:
            result: The evolution result to apply
            prompt_type: Which prompt to update ("decision" or "synthesis")

        Returns:
            True if applied successfully
        """
        if not result.success or not result.prompt_result:
            return False

        if not self._behavior.prompts:
            return False

        evolved_prompt = result.prompt_result.best_prompt

        if prompt_type == "decision":
            self._behavior.prompts.decision_prompt = evolved_prompt
        elif prompt_type == "synthesis":
            self._behavior.prompts.synthesis_prompt = evolved_prompt
        else:
            return False

        # Update metrics
        self._behavior.metrics.generations += 1
        self._behavior.metrics.fitness_score = result.evolved_fitness
        self._behavior.metrics.last_evolved = datetime.now()

        logger.info(f"Applied evolved {prompt_type} prompt (fitness: {result.evolved_fitness:.3f})")

        return True

    def get_evolution_history(self) -> list[EvolutionServiceResult]:
        """Get the evolution history."""
        return self._evolution_history.copy()

    def get_improvement_trend(self) -> list[float]:
        """Get the improvement trend over evolution runs."""
        return [r.improvement for r in self._evolution_history if r.success]


__all__ = [
    "EvolutionService",
    "EvolutionServiceConfig",
    "EvolutionServiceResult",
    "VoiceAssistantFitnessEvaluator",
    "LLMProvider",
]

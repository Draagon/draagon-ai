"""Evolution Service for Behavior Improvement.

This module provides a unified evolution service that combines:
- Promptbreeder: Genetic algorithm-based prompt optimization
- Context Evolution: ACE-style grow-and-refine for contexts
- Fitness Evaluation: Custom fitness evaluation for different use cases

The service enables behaviors to improve over time based on:
- User interaction feedback
- Success/failure patterns
- User corrections

Usage:
    from draagon_ai.services.evolution import EvolutionService

    service = EvolutionService(llm=llm_provider, behavior=behavior)

    # Evolve decision prompt
    result = await service.evolve_decision_prompt(test_cases)
    if result.success:
        # Apply evolved prompt
        behavior.prompts.decision_prompt = result.best_prompt

    # Evolve context from feedback
    context_result = await service.evolve_context(current_context, feedback)
"""

# Re-export from roxy_evolution for backward compatibility
# The implementation is already generic, just named with "Roxy"
from .roxy_evolution import (
    RoxyEvolutionService as EvolutionService,
    RoxyEvolutionConfig as EvolutionServiceConfig,
    RoxyEvolutionResult as EvolutionServiceResult,
    VoiceAssistantFitnessEvaluator,
    LLMProvider,
)

__all__ = [
    "EvolutionService",
    "EvolutionServiceConfig",
    "EvolutionServiceResult",
    "VoiceAssistantFitnessEvaluator",
    "LLMProvider",
]

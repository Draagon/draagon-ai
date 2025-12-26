"""Evolution Module - Self-Improving Prompt and Context Evolution.

This module provides evolutionary optimization for prompts and contexts:

## Core Components

### Promptbreeder (promptbreeder.py)
Genetic algorithm-based prompt optimization:
- Population-based evolution
- Tournament selection
- Mutation and crossover
- Fitness sharing for diversity
- Overfitting prevention

### Meta Prompts (meta_prompts.py)
Self-referential mutation prompts:
- Prompts that generate/improve other prompts
- Meta prompts can evolve themselves
- Fitness-weighted selection

### Fitness Evaluation (fitness.py)
Fitness scoring and diversity preservation:
- Heuristic and LLM-based evaluation
- Fitness sharing to maintain population diversity
- Population tracking over generations

### Context Evolution Engine (context_engine.py)
ACE-style grow-and-refine evolution:
- Generate → Reflect → Curate → Evolve pattern
- Candidate generation from successful patterns
- Context merging and refinement

## Usage Example

```python
from draagon_ai.evolution import (
    Promptbreeder,
    EvolutionConfig,
    EvolutionTestCase,
    ContextEvolutionEngine,
    InteractionFeedback,
)

# Promptbreeder evolution
config = EvolutionConfig(
    population_size=8,
    generations=5,
    enable_fitness_sharing=True,
)

breeder = Promptbreeder(llm_provider, config=config)

test_cases = [
    EvolutionTestCase(query="What time is it?", expected_action="get_time"),
    # ... more test cases
]

result = await breeder.evolve(base_prompt, test_cases)
if result.success and not result.rejected_reason:
    improved_prompt = result.best_prompt

# Context evolution
engine = ContextEvolutionEngine(llm_provider)

feedback = [
    InteractionFeedback(
        query="Help me book a flight",
        response="I can help with that...",
        success=True,
        quality_score=0.9,
    ),
    # ... more feedback
]

result = await engine.evolve(current_context, feedback)
if result.success:
    new_context = result.evolved_context
```

## Architecture

Based on:
- Promptbreeder paper (Sakana AI) - Genetic prompt optimization
- ACE Framework (Stanford, 2025) - Agentic context engineering
"""

from .fitness import (
    EvolutionTestCase,
    FitnessResult,
    FitnessEvaluator,
    HeuristicFitnessEvaluator,
    LLMFitnessEvaluator,
    FitnessSharing,
    PopulationMetrics,
    PopulationEvaluator,
    tournament_select,
    LLMProvider,
)

from .meta_prompts import (
    MetaPrompt,
    MutationResult,
    MetaPromptManager,
    DEFAULT_MUTATION_PROMPTS,
    MUTATION_TEMPLATE,
    CROSSOVER_TEMPLATE,
    META_EVOLUTION_TEMPLATE,
)

from .promptbreeder import (
    EvolutionConfig,
    EvolutionResult,
    CapabilityValidator,
    Promptbreeder,
)

from .context_engine import (
    ContextCandidate,
    ContextEvaluation,
    InteractionFeedback,
    ContextEvolutionEngine,
    EvolutionResult as ContextEvolutionResult,
)

__all__ = [
    # Fitness
    "EvolutionTestCase",
    "FitnessResult",
    "FitnessEvaluator",
    "HeuristicFitnessEvaluator",
    "LLMFitnessEvaluator",
    "FitnessSharing",
    "PopulationMetrics",
    "PopulationEvaluator",
    "tournament_select",
    "LLMProvider",
    # Meta Prompts
    "MetaPrompt",
    "MutationResult",
    "MetaPromptManager",
    "DEFAULT_MUTATION_PROMPTS",
    "MUTATION_TEMPLATE",
    "CROSSOVER_TEMPLATE",
    "META_EVOLUTION_TEMPLATE",
    # Promptbreeder
    "EvolutionConfig",
    "EvolutionResult",
    "CapabilityValidator",
    "Promptbreeder",
    # Context Engine
    "ContextCandidate",
    "ContextEvaluation",
    "InteractionFeedback",
    "ContextEvolutionEngine",
    "ContextEvolutionResult",
]

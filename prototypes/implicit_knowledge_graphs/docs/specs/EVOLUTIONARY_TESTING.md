# Evolutionary Testing Framework Specification

**Version:** 1.0.0
**Status:** Technical Specification
**Last Updated:** 2025-12-31

---

## Executive Summary

This document specifies the evolutionary testing framework for the implicit knowledge graphs prototype. Every component is designed to be evolvable from the start, enabling continuous optimization of:

- Decomposition prompts and strategies
- Weighting schemes
- Storage policies
- Retrieval strategies
- Evaluation rubrics

**Key Principle:** We don't just test for correctness—we evolve toward optimality using multi-dimensional fitness functions evaluated by Claude Opus 4.5.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Evolvable Components](#2-evolvable-components)
3. [Fitness Functions](#3-fitness-functions)
4. [Evolution Strategies](#4-evolution-strategies)
5. [Multi-Dimensional Scoring](#5-multi-dimensional-scoring)
6. [Test Case Design](#6-test-case-design)
7. [Implementation](#7-implementation)
8. [Integration with Phases](#8-integration-with-phases)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY TESTING ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Test Cases     │     │  Population     │     │  Fitness        │       │
│  │  (Fixed)        │────▶│  (Evolving)     │────▶│  Evaluator      │       │
│  │                 │     │                 │     │  (Opus 4.5)     │       │
│  └─────────────────┘     └─────────────────┘     └────────┬────────┘       │
│                                                           │                 │
│                          ┌────────────────────────────────┘                 │
│                          │                                                  │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EVOLUTION ENGINE                                  │   │
│  │                                                                      │   │
│  │  1. SELECTION: Top performers survive                               │   │
│  │  2. CROSSOVER: Combine successful configs                           │   │
│  │  3. MUTATION: Random tweaks to parameters                           │   │
│  │  4. EVALUATION: Score new generation                                │   │
│  │  5. REPEAT until convergence                                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    OUTPUTS                                           │   │
│  │                                                                      │   │
│  │  • Best configuration per generation                                │   │
│  │  • Pareto frontier (multi-objective)                                │   │
│  │  • Evolution history and metrics                                    │   │
│  │  • Candidate configurations for next phase                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Evolvable Components

### 2.1 Component Registry

Every evolvable component is registered with:
- Parameter definitions
- Mutation strategies
- Crossover rules
- Fitness metrics

```python
@dataclass
class EvolvableComponent:
    """A component that can evolve."""

    name: str
    parameters: dict[str, ParameterDef]
    mutation_strategy: MutationStrategy
    crossover_strategy: CrossoverStrategy
    fitness_metrics: list[str]


# Registry of all evolvable components
EVOLVABLE_COMPONENTS = {
    "wsd": EvolvableComponent(
        name="Word Sense Disambiguation",
        parameters={
            "lesk_context_window": ParameterDef(type=int, min=1, max=20, default=5),
            "lesk_high_confidence": ParameterDef(type=float, min=0.5, max=1.0, default=0.8),
            "llm_fallback_threshold": ParameterDef(type=float, min=0.3, max=0.9, default=0.5),
            "wsd_prompt": ParameterDef(type=str, evolvable=True),
        },
        mutation_strategy=MutationStrategy.GAUSSIAN_NUMERIC | MutationStrategy.LLM_PROMPT,
        crossover_strategy=CrossoverStrategy.UNIFORM,
        fitness_metrics=["wsd_accuracy", "llm_call_rate", "latency"],
    ),
    "decomposition": EvolvableComponent(
        name="Decomposition",
        parameters={
            "presupposition_prompt": ParameterDef(type=str, evolvable=True),
            "inference_prompt": ParameterDef(type=str, evolvable=True),
            "max_presuppositions": ParameterDef(type=int, min=1, max=20, default=5),
            "min_confidence": ParameterDef(type=float, min=0.1, max=0.9, default=0.3),
        },
        mutation_strategy=MutationStrategy.LLM_PROMPT | MutationStrategy.GAUSSIAN_NUMERIC,
        crossover_strategy=CrossoverStrategy.SINGLE_POINT,
        fitness_metrics=["decomposition_completeness", "decomposition_precision"],
    ),
    "weighting": EvolvableComponent(
        name="Cognitive Weighting",
        parameters={
            "recency_weight": ParameterDef(type=float, min=0.0, max=0.5, default=0.2),
            "memory_weight": ParameterDef(type=float, min=0.0, max=0.5, default=0.2),
            "belief_weight": ParameterDef(type=float, min=0.0, max=0.5, default=0.15),
            "commonsense_weight": ParameterDef(type=float, min=0.0, max=0.5, default=0.1),
            # Note: weights should sum to <= 1.0 (constraint)
        },
        mutation_strategy=MutationStrategy.GAUSSIAN_NUMERIC,
        crossover_strategy=CrossoverStrategy.BLEND,
        fitness_metrics=["branch_prediction_accuracy"],
    ),
    "retrieval": EvolvableComponent(
        name="Retrieval Strategy",
        parameters={
            "context_packing_strategy": ParameterDef(
                type=str,
                choices=["triple_only", "sentence_context", "hierarchical"],
                default="triple_only"
            ),
            "max_triples": ParameterDef(type=int, min=5, max=50, default=20),
            "synset_filter_strictness": ParameterDef(type=float, min=0.0, max=1.0, default=0.5),
        },
        mutation_strategy=MutationStrategy.CATEGORICAL | MutationStrategy.GAUSSIAN_NUMERIC,
        crossover_strategy=CrossoverStrategy.UNIFORM,
        fitness_metrics=["retrieval_precision", "retrieval_recall", "context_efficiency"],
    ),
    "scoring": EvolvableComponent(
        name="Output Scoring",
        parameters={
            "scoring_prompt": ParameterDef(type=str, evolvable=True),
            "factual_weight": ParameterDef(type=float, min=0.1, max=0.5, default=0.3),
            "completeness_weight": ParameterDef(type=float, min=0.1, max=0.5, default=0.25),
            "relevance_weight": ParameterDef(type=float, min=0.1, max=0.5, default=0.25),
            "coherence_weight": ParameterDef(type=float, min=0.1, max=0.3, default=0.2),
        },
        mutation_strategy=MutationStrategy.LLM_PROMPT | MutationStrategy.GAUSSIAN_NUMERIC,
        crossover_strategy=CrossoverStrategy.BLEND,
        fitness_metrics=["correlation_with_human"],
    ),
}
```

### 2.2 Configuration Structure

```python
@dataclass
class EvolvableConfig:
    """Complete configuration for the pipeline."""

    config_id: str
    generation: int
    parent_ids: list[str]

    # Component configurations
    wsd: WSDConfig
    decomposition: DecompositionConfig
    weighting: WeightingConfig
    retrieval: RetrievalConfig
    scoring: ScoringConfig

    # Fitness tracking
    fitness_scores: dict[str, float] = field(default_factory=dict)
    evaluation_count: int = 0

    def combined_fitness(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted combination of fitness scores."""
        if not self.fitness_scores:
            return 0.0

        if weights is None:
            weights = {k: 1.0 / len(self.fitness_scores) for k in self.fitness_scores}

        return sum(
            self.fitness_scores.get(k, 0.0) * w
            for k, w in weights.items()
        )
```

---

## 3. Fitness Functions

### 3.1 Component-Level Fitness

Each component has specific fitness metrics:

```python
# WSD Fitness
async def evaluate_wsd_fitness(
    config: WSDConfig,
    test_cases: list[WSDTestCase],
    llm: LLMProvider,
) -> dict[str, float]:
    """Evaluate WSD configuration fitness."""

    correct = 0
    llm_calls = 0
    total_latency = 0.0

    for case in test_cases:
        start = time.time()
        predicted_synset, confidence, used_llm = await disambiguate(
            case.word, case.context, config, llm
        )
        total_latency += time.time() - start

        if predicted_synset == case.expected_synset:
            correct += 1
        if used_llm:
            llm_calls += 1

    return {
        "wsd_accuracy": correct / len(test_cases),
        "llm_call_rate": llm_calls / len(test_cases),
        "latency": total_latency / len(test_cases),
    }


# Decomposition Fitness
async def evaluate_decomposition_fitness(
    config: DecompositionConfig,
    test_cases: list[DecompositionTestCase],
    llm: LLMProvider,
) -> dict[str, float]:
    """Evaluate decomposition configuration fitness."""

    total_completeness = 0.0
    total_precision = 0.0

    for case in test_cases:
        decomposed = await decompose(case.text, config, llm)

        # Completeness: Did we extract expected elements?
        expected_set = set(case.expected_presuppositions + case.expected_inferences)
        extracted_set = set(
            [p.content for p in decomposed.presuppositions] +
            [i.content for i in decomposed.inferences]
        )
        completeness = len(expected_set & extracted_set) / len(expected_set) if expected_set else 1.0

        # Precision: Are extracted elements correct?
        precision = len(expected_set & extracted_set) / len(extracted_set) if extracted_set else 1.0

        total_completeness += completeness
        total_precision += precision

    return {
        "decomposition_completeness": total_completeness / len(test_cases),
        "decomposition_precision": total_precision / len(test_cases),
    }
```

### 3.2 End-to-End Fitness (Opus 4.5 Scoring)

The ultimate fitness measure is output quality as judged by Claude Opus 4.5:

```python
async def evaluate_end_to_end_fitness(
    config: EvolvableConfig,
    test_cases: list[EndToEndTestCase],
    opus_client: AnthropicClient,
) -> dict[str, float]:
    """Evaluate full pipeline using Opus 4.5 scoring."""

    scores = []

    for case in test_cases:
        # Run the full pipeline
        context = await retrieve_context(case.query, case.knowledge_base, config)
        output = await generate_response(case.query, context, config)

        # Score with Opus 4.5
        score = await opus_score_output(
            query=case.query,
            context=context,
            output=output,
            expected=case.expected,
            scoring_config=config.scoring,
            opus_client=opus_client,
        )

        scores.append(score)

    return aggregate_scores(scores)


async def opus_score_output(
    query: str,
    context: str,
    output: str,
    expected: TestExpectation,
    scoring_config: ScoringConfig,
    opus_client: AnthropicClient,
) -> OutputScore:
    """Score output using Claude Opus 4.5."""

    prompt = f"""{scoring_config.scoring_prompt}

QUERY: {query}

CONTEXT PROVIDED TO LLM:
{context}

LLM OUTPUT:
{output}

EXPECTED TOPICS: {expected.expected_topics}
EXPECTED ENTITIES: {expected.expected_entities}
{"GOLD ANSWER: " + expected.gold_answer if expected.gold_answer else ""}

Evaluate the output on these dimensions (0.0 to 1.0):

1. FACTUAL_ACCURACY: Are all stated facts correct? No hallucinations?
2. COMPLETENESS: Does it fully address the query? Cover all expected topics?
3. RELEVANCE: Is all content relevant to the query? No tangents?
4. COHERENCE: Is it well-structured, clear, and easy to understand?

Also calculate:
5. CONTEXT_EFFICIENCY: How well was the provided context used?

Respond in XML:
<evaluation>
    <factual_accuracy>0.0-1.0</factual_accuracy>
    <completeness>0.0-1.0</completeness>
    <relevance>0.0-1.0</relevance>
    <coherence>0.0-1.0</coherence>
    <context_efficiency>0.0-1.0</context_efficiency>
    <rationale>Explain your scores</rationale>
    <improvement_suggestions>How could the context be better?</improvement_suggestions>
</evaluation>
"""

    response = await opus_client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    return parse_opus_evaluation(response.content[0].text)
```

### 3.3 Multi-Objective Fitness

We optimize for multiple objectives simultaneously:

```python
@dataclass
class MultiObjectiveFitness:
    """Multi-objective fitness with Pareto ranking."""

    # Primary objectives
    output_quality: float      # From Opus 4.5 scoring
    context_efficiency: float  # Quality per token

    # Secondary objectives
    latency: float             # Processing time
    llm_cost: float            # LLM API cost

    # Derived metrics
    @property
    def pareto_rank(self) -> int:
        """Pareto rank (0 = on frontier, higher = dominated)."""
        # Computed during population evaluation
        return self._pareto_rank

    def dominates(self, other: "MultiObjectiveFitness") -> bool:
        """Does this solution dominate another?"""
        at_least_as_good = (
            self.output_quality >= other.output_quality and
            self.context_efficiency >= other.context_efficiency and
            self.latency <= other.latency and
            self.llm_cost <= other.llm_cost
        )
        strictly_better = (
            self.output_quality > other.output_quality or
            self.context_efficiency > other.context_efficiency or
            self.latency < other.latency or
            self.llm_cost < other.llm_cost
        )
        return at_least_as_good and strictly_better
```

---

## 4. Evolution Strategies

### 4.1 Selection Strategies

```python
class SelectionStrategy(str, Enum):
    TOURNAMENT = "tournament"      # K-way tournament selection
    ROULETTE = "roulette"         # Fitness-proportional
    RANK = "rank"                 # Rank-based
    ELITISM = "elitism"           # Keep top N unchanged
    NSGA2 = "nsga2"               # Multi-objective (Pareto-based)


def tournament_select(
    population: list[EvolvableConfig],
    tournament_size: int = 3,
) -> EvolvableConfig:
    """Select winner from random tournament."""
    contestants = random.sample(population, tournament_size)
    return max(contestants, key=lambda c: c.combined_fitness())


def nsga2_select(
    population: list[EvolvableConfig],
    num_select: int,
) -> list[EvolvableConfig]:
    """NSGA-II selection for multi-objective optimization."""
    # Sort by Pareto rank, then crowding distance
    ranked = compute_pareto_ranks(population)
    return sorted(ranked, key=lambda c: (c.pareto_rank, -c.crowding_distance))[:num_select]
```

### 4.2 Mutation Strategies

```python
class MutationStrategy(Flag):
    GAUSSIAN_NUMERIC = auto()     # Add Gaussian noise to numeric params
    UNIFORM_NUMERIC = auto()      # Uniform random within bounds
    CATEGORICAL = auto()          # Random choice from options
    LLM_PROMPT = auto()           # Use LLM to mutate prompts
    SWAP = auto()                 # Swap elements in sequences


def mutate_config(
    config: EvolvableConfig,
    mutation_rate: float = 0.2,
    llm: LLMProvider | None = None,
) -> EvolvableConfig:
    """Mutate a configuration."""
    new_config = deepcopy(config)
    new_config.config_id = str(uuid.uuid4())
    new_config.generation += 1
    new_config.parent_ids = [config.config_id]
    new_config.fitness_scores = {}
    new_config.evaluation_count = 0

    for component_name, component in EVOLVABLE_COMPONENTS.items():
        component_config = getattr(new_config, component_name)

        for param_name, param_def in component.parameters.items():
            if random.random() > mutation_rate:
                continue

            current_value = getattr(component_config, param_name)

            if param_def.type == float and MutationStrategy.GAUSSIAN_NUMERIC in component.mutation_strategy:
                # Gaussian mutation for floats
                std = (param_def.max - param_def.min) * 0.1
                new_value = current_value + random.gauss(0, std)
                new_value = max(param_def.min, min(param_def.max, new_value))
                setattr(component_config, param_name, new_value)

            elif param_def.type == int and MutationStrategy.GAUSSIAN_NUMERIC in component.mutation_strategy:
                # Gaussian mutation for ints
                std = (param_def.max - param_def.min) * 0.1
                new_value = int(current_value + random.gauss(0, std))
                new_value = max(param_def.min, min(param_def.max, new_value))
                setattr(component_config, param_name, new_value)

            elif param_def.type == str and param_def.evolvable and MutationStrategy.LLM_PROMPT in component.mutation_strategy:
                # LLM-based prompt mutation
                if llm:
                    new_value = await mutate_prompt_with_llm(current_value, llm)
                    setattr(component_config, param_name, new_value)

            elif param_def.choices and MutationStrategy.CATEGORICAL in component.mutation_strategy:
                # Categorical mutation
                new_value = random.choice(param_def.choices)
                setattr(component_config, param_name, new_value)

    return new_config


async def mutate_prompt_with_llm(
    current_prompt: str,
    llm: LLMProvider,
) -> str:
    """Use LLM to create a variant of a prompt."""

    mutation_prompt = f"""You are helping optimize prompts through evolution.

Here is the current prompt:
<current_prompt>
{current_prompt}
</current_prompt>

Create a VARIANT of this prompt that might perform better. You can:
1. Rephrase instructions for clarity
2. Add or remove examples
3. Change the structure
4. Adjust the output format requirements

The variant should be meaningfully different but serve the same purpose.

Respond with ONLY the new prompt, no explanation:"""

    response = await llm.chat([{"role": "user", "content": mutation_prompt}])
    return response.strip()
```

### 4.3 Crossover Strategies

```python
class CrossoverStrategy(str, Enum):
    UNIFORM = "uniform"           # Each param from random parent
    SINGLE_POINT = "single_point" # Split at one point
    BLEND = "blend"               # Blend numeric values


def crossover_configs(
    parent_a: EvolvableConfig,
    parent_b: EvolvableConfig,
) -> EvolvableConfig:
    """Create child by combining two parents."""

    child = EvolvableConfig(
        config_id=str(uuid.uuid4()),
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[parent_a.config_id, parent_b.config_id],
        wsd=WSDConfig(),
        decomposition=DecompositionConfig(),
        weighting=WeightingConfig(),
        retrieval=RetrievalConfig(),
        scoring=ScoringConfig(),
    )

    for component_name, component in EVOLVABLE_COMPONENTS.items():
        parent_a_comp = getattr(parent_a, component_name)
        parent_b_comp = getattr(parent_b, component_name)
        child_comp = getattr(child, component_name)

        for param_name, param_def in component.parameters.items():
            val_a = getattr(parent_a_comp, param_name)
            val_b = getattr(parent_b_comp, param_name)

            if component.crossover_strategy == CrossoverStrategy.UNIFORM:
                # Random parent for each param
                new_val = random.choice([val_a, val_b])

            elif component.crossover_strategy == CrossoverStrategy.BLEND:
                # Blend numeric values
                if param_def.type in (int, float):
                    alpha = random.random()
                    new_val = alpha * val_a + (1 - alpha) * val_b
                    if param_def.type == int:
                        new_val = int(round(new_val))
                else:
                    new_val = random.choice([val_a, val_b])

            else:
                new_val = random.choice([val_a, val_b])

            setattr(child_comp, param_name, new_val)

    return child
```

---

## 5. Multi-Dimensional Scoring

### 5.1 Score Dimensions

```python
@dataclass
class OutputScore:
    """Multi-dimensional quality score from Opus 4.5."""

    # Core quality dimensions (0.0 - 1.0)
    factual_accuracy: float
    completeness: float
    relevance: float
    coherence: float

    # Efficiency dimensions
    context_tokens_used: int
    context_efficiency: float  # quality / tokens

    # Scoring metadata
    scoring_rationale: str
    improvement_suggestions: str
    confidence: float  # How confident is the scorer?

    @property
    def combined_quality(self) -> float:
        """Weighted combination of quality dimensions."""
        # Default weights - these are also evolvable!
        return (
            0.30 * self.factual_accuracy +
            0.25 * self.completeness +
            0.25 * self.relevance +
            0.20 * self.coherence
        )

    @property
    def quality_per_token(self) -> float:
        """Quality efficiency metric."""
        if self.context_tokens_used == 0:
            return 0.0
        return self.combined_quality / (self.context_tokens_used / 1000)  # per 1K tokens
```

### 5.2 Baseline Comparison

Every test also runs a baseline for comparison:

```python
@dataclass
class BaselineComparison:
    """Comparison against baseline approaches."""

    # Our approach
    our_score: OutputScore
    our_context: str

    # Baseline: vanilla RAG
    baseline_rag_score: OutputScore
    baseline_rag_context: str

    # Improvement metrics
    @property
    def quality_improvement(self) -> float:
        return self.our_score.combined_quality - self.baseline_rag_score.combined_quality

    @property
    def efficiency_improvement(self) -> float:
        return self.our_score.quality_per_token - self.baseline_rag_score.quality_per_token

    @property
    def is_better(self) -> bool:
        return self.quality_improvement > 0 or (
            self.quality_improvement >= -0.05 and
            self.efficiency_improvement > 0.1
        )
```

---

## 6. Test Case Design

### 6.1 Test Case Structure

```python
@dataclass
class EndToEndTestCase:
    """A test case for end-to-end evaluation."""

    # Unique identifier
    case_id: str

    # Input
    query: str
    knowledge_base: list[str]  # Facts to store

    # Expected characteristics
    expected: TestExpectation

    # Difficulty indicators
    difficulty: TestDifficulty

    # For holdout validation
    is_holdout: bool = False


@dataclass
class TestExpectation:
    """Expected characteristics of a good answer."""

    expected_topics: list[str]      # Topics to cover
    expected_entities: list[str]    # Entities to mention
    gold_answer: str | None = None  # Optional gold standard
    required_facts: list[str] = field(default_factory=list)
    forbidden_content: list[str] = field(default_factory=list)


@dataclass
class TestDifficulty:
    """Difficulty indicators for stratified evaluation."""

    requires_multi_hop: bool = False
    has_ambiguity: bool = False
    has_temporal_reasoning: bool = False
    has_conflicting_info: bool = False
    entity_count: int = 1
    knowledge_base_size: int = 10
```

### 6.2 Test Categories

```python
TEST_CATEGORIES = {
    "simple_retrieval": {
        "description": "Single fact retrieval, no ambiguity",
        "min_cases": 20,
        "difficulty": TestDifficulty(entity_count=1),
    },
    "multi_entity": {
        "description": "Multiple entities, need to distinguish",
        "min_cases": 15,
        "difficulty": TestDifficulty(entity_count=3),
    },
    "multi_hop": {
        "description": "Answer requires combining multiple facts",
        "min_cases": 15,
        "difficulty": TestDifficulty(requires_multi_hop=True),
    },
    "ambiguous": {
        "description": "Ambiguous terms or references",
        "min_cases": 10,
        "difficulty": TestDifficulty(has_ambiguity=True),
    },
    "temporal": {
        "description": "Temporal reasoning required",
        "min_cases": 10,
        "difficulty": TestDifficulty(has_temporal_reasoning=True),
    },
    "conflicting": {
        "description": "Conflicting information in knowledge base",
        "min_cases": 10,
        "difficulty": TestDifficulty(has_conflicting_info=True),
    },
    "scale": {
        "description": "Large knowledge base (100+ facts)",
        "min_cases": 10,
        "difficulty": TestDifficulty(knowledge_base_size=100),
    },
}
```

### 6.3 Train/Holdout Split

To prevent overfitting, we maintain a strict train/holdout split:

```python
def create_train_holdout_split(
    test_cases: list[EndToEndTestCase],
    holdout_ratio: float = 0.2,
) -> tuple[list[EndToEndTestCase], list[EndToEndTestCase]]:
    """Split test cases into train and holdout sets."""

    # Stratified split by category
    by_category = defaultdict(list)
    for case in test_cases:
        category = determine_category(case)
        by_category[category].append(case)

    train = []
    holdout = []

    for category, cases in by_category.items():
        random.shuffle(cases)
        split_idx = int(len(cases) * (1 - holdout_ratio))
        train.extend(cases[:split_idx])
        holdout.extend(cases[split_idx:])

        # Mark holdout cases
        for case in cases[split_idx:]:
            case.is_holdout = True

    return train, holdout


def check_overfitting(
    train_fitness: float,
    holdout_fitness: float,
    max_gap: float = 0.10,
) -> bool:
    """Check if configuration is overfitting to train set."""
    gap = train_fitness - holdout_fitness
    return gap > max_gap
```

---

## 7. Implementation

### 7.1 Evolution Engine

```python
class EvolutionEngine:
    """Main evolution engine."""

    def __init__(
        self,
        population_size: int = 10,
        elite_count: int = 2,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
        max_generations: int = 50,
        convergence_threshold: float = 0.01,
        opus_client: AnthropicClient = None,
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.opus_client = opus_client

        self.population: list[EvolvableConfig] = []
        self.generation = 0
        self.history: list[GenerationStats] = []

    async def evolve(
        self,
        train_cases: list[EndToEndTestCase],
        holdout_cases: list[EndToEndTestCase],
        llm: LLMProvider,
    ) -> EvolutionResult:
        """Run evolution until convergence or max generations."""

        # Initialize population
        self.population = self._initialize_population()

        best_config = None
        best_fitness = 0.0
        stagnation_count = 0

        for gen in range(self.max_generations):
            self.generation = gen

            # Evaluate population on train set
            await self._evaluate_population(train_cases, llm)

            # Sort by fitness
            self.population.sort(
                key=lambda c: c.combined_fitness(),
                reverse=True
            )

            # Track best
            current_best = self.population[0]
            current_fitness = current_best.combined_fitness()

            # Check for improvement
            if current_fitness > best_fitness + self.convergence_threshold:
                best_config = current_best
                best_fitness = current_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Check holdout for overfitting
            holdout_fitness = await self._evaluate_on_holdout(
                current_best, holdout_cases, llm
            )
            if check_overfitting(current_fitness, holdout_fitness):
                print(f"Warning: Overfitting detected at generation {gen}")

            # Record stats
            self.history.append(GenerationStats(
                generation=gen,
                best_fitness=current_fitness,
                mean_fitness=sum(c.combined_fitness() for c in self.population) / len(self.population),
                holdout_fitness=holdout_fitness,
                best_config_id=current_best.config_id,
            ))

            # Check convergence
            if stagnation_count >= 10:
                print(f"Converged at generation {gen}")
                break

            # Create next generation
            self.population = self._create_next_generation(llm)

        return EvolutionResult(
            best_config=best_config,
            final_population=self.population,
            history=self.history,
            generations_run=self.generation + 1,
        )

    def _create_next_generation(
        self,
        llm: LLMProvider,
    ) -> list[EvolvableConfig]:
        """Create next generation through selection, crossover, mutation."""

        next_gen = []

        # Elitism: keep top performers unchanged
        next_gen.extend(self.population[:self.elite_count])

        # Fill rest through evolution
        while len(next_gen) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent_a = tournament_select(self.population)
                parent_b = tournament_select(self.population)
                child = crossover_configs(parent_a, parent_b)
            else:
                # Mutation only
                parent = tournament_select(self.population)
                child = mutate_config(parent, self.mutation_rate, llm)

            next_gen.append(child)

        return next_gen
```

### 7.2 File Structure

```
src/evolution/
├── __init__.py
├── engine.py           # EvolutionEngine class
├── config.py           # EvolvableConfig and component definitions
├── mutation.py         # Mutation strategies
├── crossover.py        # Crossover strategies
├── selection.py        # Selection strategies
├── fitness.py          # Fitness evaluation functions
├── scoring.py          # Opus 4.5 scoring integration
└── history.py          # Evolution history tracking

tests/evolution/
├── test_mutation.py
├── test_crossover.py
├── test_selection.py
├── test_fitness.py
├── test_engine.py
└── test_integration.py
```

---

## 8. Integration with Phases

### Phase 0: WSD Evolution

```python
# Evolve WSD configuration
wsd_evolution = EvolutionEngine(
    population_size=8,
    max_generations=20,
)

wsd_result = await wsd_evolution.evolve(
    train_cases=wsd_train_cases,
    holdout_cases=wsd_holdout_cases,
    llm=llm,
)

# Use best WSD config going forward
best_wsd_config = wsd_result.best_config.wsd
```

### Phase 1+: Full Pipeline Evolution

```python
# Evolve full pipeline
full_evolution = EvolutionEngine(
    population_size=10,
    max_generations=50,
)

# Uses Opus 4.5 for end-to-end scoring
full_result = await full_evolution.evolve(
    train_cases=e2e_train_cases,
    holdout_cases=e2e_holdout_cases,
    llm=llm,
)

# Get Pareto frontier of configs
pareto_configs = get_pareto_frontier(full_result.final_population)
```

### Carrying Configs to Next Phase

```python
# Save best configs as candidates for next phase
candidates = [
    full_result.best_config,              # Best overall
    *pareto_configs[:3],                  # Pareto frontier
]

save_candidate_configs(candidates, "phase_1_candidates.json")

# Next phase loads and continues evolution
loaded = load_candidate_configs("phase_1_candidates.json")
next_phase_evolution.initialize_population(loaded)
```

---

## Summary

The evolutionary testing framework provides:

1. **Evolvable everything** - All parameters, prompts, and strategies can evolve
2. **Multi-dimensional scoring** - Opus 4.5 evaluates output quality across dimensions
3. **Baseline comparison** - Always compare against vanilla RAG
4. **Overfitting protection** - Train/holdout split with gap monitoring
5. **Multi-objective optimization** - Pareto frontier for quality vs efficiency tradeoffs
6. **Continuous improvement** - Carry best configs between phases

---

**End of Evolutionary Testing Framework Specification**

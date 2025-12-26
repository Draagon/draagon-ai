# Behavior Architect: Go-Forward Plan

**Date:** December 2024
**Status:** Design Phase
**Priority:** High
**Depends On:** INDUSTRY_RESEARCH_2024.md

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Vision](#vision)
3. [Architecture Overview](#architecture-overview)
4. [Detailed Design](#detailed-design)
5. [Implementation Phases](#implementation-phases)
6. [Actions Specification](#actions-specification)
7. [Prompts Design](#prompts-design)
8. [Evolution System](#evolution-system)
9. [Safety and Trust](#safety-and-trust)
10. [Testing Strategy](#testing-strategy)
11. [Success Metrics](#success-metrics)
12. [Risks and Mitigations](#risks-and-mitigations)
13. [Future Extensions](#future-extensions)

---

## Executive Summary

The **Behavior Architect** is a meta-behavior that creates, tests, and evolves other behaviors. It is the culmination of draagon-ai's design philosophy: behaviors as evolvable data structures.

### Key Decisions (Based on Research)

| Decision | Rationale |
|----------|-----------|
| **Build as native Behavior** | Leverages existing infrastructure; behaviors stay as rich dataclasses |
| **Incorporate Promptbreeder's self-referential mutation** | Mutation prompts evolve too, increasing power |
| **Keep Python dataclasses** | More expressive than YAML specs; consider export later |
| **MCP as tool layer** | Don't confuse tool access with behavioral specification |
| **CORE tier, maximum trust** | Behavior creation is sensitive; needs highest trust level |

### What We're Building

A **god-level agentic AI architect** that can:
1. Research a domain (web search, user requirements, existing behaviors)
2. Design behaviors (actions, triggers, prompts, constraints)
3. Generate and run tests
4. Evolve behaviors via genetic algorithms
5. Register successful behaviors with appropriate trust levels

---

## Vision

### The Meta-Agent Dream

```
User: "Create a behavior for managing my home aquarium"

Behavior Architect:
  1. Researches aquarium management best practices
  2. Identifies core tasks: water testing, feeding, maintenance schedules
  3. Designs actions: test_water, log_feeding, schedule_maintenance, alert_issue
  4. Creates decision prompt with domain knowledge
  5. Generates 20+ test cases
  6. Runs tests, iterates on failures
  7. Evolves prompt for better performance
  8. Registers as GENERATED tier, STAGING status
  9. Monitors performance, promotes to ACTIVE if successful
```

### Hierarchy of Creation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIOR CREATION HIERARCHY                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 3: BEHAVIOR ARCHITECT (CORE tier)                                │
│           │                                                              │
│           ├── Creates → Application Behaviors (GENERATED tier)          │
│           ├── Evolves → Existing Behaviors (any tier)                   │
│           └── Tests → All Behaviors                                      │
│                                                                          │
│  Level 2: APPLICATION BEHAVIORS                                          │
│           │                                                              │
│           ├── Story Teller → Creates → Story Characters (GENERATED)     │
│           └── Voice Assistant → Uses → Existing Behaviors               │
│                                                                          │
│  Level 1: INDIVIDUAL BEHAVIORS                                           │
│           │                                                              │
│           └── Execute → Actions via Tools                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BEHAVIOR ARCHITECT LOOP                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   RESEARCH   │───▶│    DESIGN    │───▶│    BUILD     │              │
│  │    PHASE     │    │    PHASE     │    │    PHASE     │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  • Web search          • Actions           • Behavior object           │
│  • User requirements   • Triggers          • Prompts                   │
│  • Existing behaviors  • State model       • Test cases                │
│  • Domain knowledge    • Constraints       • Validation                │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │    TEST      │◀───│   EVOLVE     │◀───│   ITERATE    │              │
│  │    PHASE     │    │    PHASE     │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   ▲                       │
│         ▼                   ▼                   │                       │
│  • Run test cases      • Population        • Analyze failures          │
│  • Track pass rate     • Mutation          • Identify root causes      │
│  • Identify failures   • Selection         • Suggest fixes             │
│  • Fitness scoring     • Self-referential  • Loop until passing        │
│                          mutation                                       │
│                                                                          │
│  ┌──────────────┐                                                       │
│  │   REGISTER   │                                                       │
│  │    PHASE     │                                                       │
│  └──────────────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  • Set tier (GENERATED)                                                 │
│  • Set status (STAGING)                                                 │
│  • Add to registry                                                      │
│  • Begin monitoring                                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   BEHAVIOR ARCHITECT                             │   │
│  │                   (Behavior object)                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│          ┌───────────────────┼───────────────────┐                     │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐            │
│  │ Research      │   │ Design        │   │ Evolution     │            │
│  │ Service       │   │ Service       │   │ Service       │            │
│  │               │   │               │   │               │            │
│  │ • Web search  │   │ • Action gen  │   │ • Population  │            │
│  │ • Domain      │   │ • Prompt eng  │   │ • Mutation    │            │
│  │   analysis    │   │ • Test gen    │   │ • Selection   │            │
│  └───────────────┘   └───────────────┘   └───────────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                     │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   BEHAVIOR REGISTRY                              │   │
│  │                                                                  │   │
│  │  CORE │ ADDON │ APPLICATION │ GENERATED │ EXPERIMENTAL          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   AGENT EXECUTION                                │   │
│  │                   (Uses behaviors)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Design

### Phase 1: Research

**Goal:** Understand the domain well enough to design a useful behavior.

**Inputs:**
- User description of desired behavior
- Existing behaviors in registry (for patterns)
- Web search results
- Optional: example interactions

**Outputs:**
- `DomainResearchResult` with:
  - Core tasks identified
  - Suggested actions
  - Suggested triggers
  - Domain constraints
  - Background knowledge

**Process:**
```python
async def research_domain(
    description: str,
    search_web: bool = True,
    search_existing: bool = True,
) -> DomainResearchResult:
    """
    1. Parse user description for key concepts
    2. Search existing behaviors for similar patterns
    3. Web search for domain best practices
    4. Synthesize into structured research result
    """
```

### Phase 2: Design

**Goal:** Create the structure of the behavior.

**Inputs:**
- `DomainResearchResult`
- User constraints/preferences

**Outputs:**
- List of `Action` objects
- List of `Trigger` objects
- State model (if applicable)
- `BehaviorConstraints`

**Process:**
```python
async def design_behavior(
    research: DomainResearchResult,
    constraints: UserConstraints | None = None,
) -> BehaviorDesign:
    """
    1. Convert research tasks to actions
    2. Define parameters for each action
    3. Create trigger patterns
    4. Define any state model needed
    5. Set up constraints
    """
```

### Phase 3: Build

**Goal:** Create the complete Behavior object with prompts.

**Inputs:**
- `BehaviorDesign`
- `DomainResearchResult` (for domain context)

**Outputs:**
- Complete `Behavior` object
- Initial `BehaviorTestCase` set

**Process:**
```python
async def build_behavior(
    design: BehaviorDesign,
    research: DomainResearchResult,
) -> tuple[Behavior, list[BehaviorTestCase]]:
    """
    1. Draft decision prompt with domain knowledge
    2. Draft synthesis prompt
    3. Generate positive test cases (one per action)
    4. Generate negative test cases (common mistakes)
    5. Generate edge case tests
    6. Assemble complete Behavior object
    """
```

### Phase 4: Test

**Goal:** Validate the behavior works correctly.

**Inputs:**
- `Behavior` object
- `BehaviorTestCase` list

**Outputs:**
- `TestResults`
- `FailureAnalysis` (if failures)

**Process:**
```python
async def test_behavior(
    behavior: Behavior,
    test_cases: list[BehaviorTestCase],
) -> tuple[TestResults, FailureAnalysis | None]:
    """
    1. Run each test case against the behavior
    2. Track pass/fail for each
    3. If failures, analyze patterns
    4. Identify root causes
    5. Suggest fixes
    """
```

### Phase 5: Iterate

**Goal:** Fix failures and improve the behavior.

**Inputs:**
- `Behavior` with failures
- `FailureAnalysis`

**Outputs:**
- Improved `Behavior`
- Updated test cases

**Process:**
```python
async def iterate_behavior(
    behavior: Behavior,
    analysis: FailureAnalysis,
) -> Behavior:
    """
    1. For each root cause, apply suggested fix
    2. May modify: prompts, actions, triggers
    3. May add: new test cases
    4. Return improved behavior
    """
```

### Phase 6: Evolve

**Goal:** Use genetic algorithms to optimize the behavior.

**Inputs:**
- `Behavior` (baseline)
- `BehaviorTestCase` list (fitness function)
- `EvolutionConfig`

**Outputs:**
- `BehaviorEvolutionResult`
- Evolved `Behavior` (if improved)

**Key Innovation: Self-Referential Mutation**

Based on Promptbreeder research, we evolve not just the prompts but also the mutation prompts:

```python
@dataclass
class MutationPrompt:
    """A prompt that describes how to mutate other prompts."""
    prompt_id: str
    content: str
    fitness: float = 0.0

async def evolve_behavior(
    behavior: Behavior,
    test_cases: list[BehaviorTestCase],
    config: EvolutionConfig,
    mutation_prompts: list[MutationPrompt] | None = None,
) -> BehaviorEvolutionResult:
    """
    1. Initialize population (base + random mutations)
    2. Split test cases: 80% train, 20% holdout
    3. For each generation:
       a. Evaluate fitness on train set
       b. Tournament selection
       c. Crossover top performers
       d. Mutate using mutation prompts
       e. ALSO evolve mutation prompts (self-referential!)
    4. Validate best on holdout set
    5. Check for overfitting (train-holdout gap)
    6. Return result with approval status
    """
```

### Phase 7: Register

**Goal:** Add the behavior to the registry with appropriate settings.

**Inputs:**
- Validated `Behavior`
- Test results
- Evolution results (if evolved)

**Outputs:**
- Registered behavior ID
- Monitoring setup

**Process:**
```python
async def register_behavior(
    behavior: Behavior,
    test_results: TestResults,
    evolution_result: BehaviorEvolutionResult | None = None,
) -> str:
    """
    1. Set tier to GENERATED
    2. Set status to STAGING (shadow mode)
    3. Set is_evolvable = True
    4. Record creation metadata
    5. Add to BehaviorRegistry
    6. Set up metrics tracking
    7. Return behavior_id
    """
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic behavior creation without evolution.

**Deliverables:**
- [ ] `BehaviorArchitectService` class
- [ ] Research action: `research_domain`
- [ ] Design actions: `draft_behavior_structure`, `add_action`, `add_trigger`
- [ ] Build actions: `draft_decision_prompt`, `draft_synthesis_prompt`
- [ ] Test actions: `generate_test_cases`, `run_tests`
- [ ] Register action: `register_behavior`
- [ ] Basic prompts for each phase
- [ ] Unit tests for each action

**Success Criteria:**
- Can create a simple behavior from description
- Behavior passes basic validation
- Can register in GENERATED tier

### Phase 2: Iteration (Week 3)

**Goal:** Failure analysis and automatic fixing.

**Deliverables:**
- [ ] `analyze_failures` action
- [ ] `suggest_fixes` action
- [ ] `apply_fix` action
- [ ] `FailureAnalysis` type implementation
- [ ] Iteration loop in service
- [ ] Tests for failure patterns

**Success Criteria:**
- Can identify common failure patterns
- Can suggest and apply fixes
- Iteration converges within 5 rounds

### Phase 3: Evolution (Week 4-5)

**Goal:** Genetic algorithm optimization with self-referential mutation.

**Deliverables:**
- [ ] `initialize_population` action
- [ ] `evaluate_fitness` action
- [ ] `tournament_select` action
- [ ] `crossover` action
- [ ] `mutate_with_prompt` action
- [ ] `evolve_mutation_prompts` action (self-referential)
- [ ] `check_overfitting` action
- [ ] `MutationPrompt` type
- [ ] Evolution loop in service
- [ ] Diversity preservation (fitness sharing)
- [ ] Tests for evolution

**Success Criteria:**
- Evolution improves fitness by 10%+
- No overfitting (gap < 10%)
- Mutation prompts improve over generations

### Phase 4: Integration (Week 6)

**Goal:** Full integration with draagon-ai.

**Deliverables:**
- [ ] `BEHAVIOR_ARCHITECT_TEMPLATE` (Behavior object)
- [ ] Integration with Agent class
- [ ] CLI commands for behavior creation
- [ ] Monitoring dashboard integration
- [ ] End-to-end tests
- [ ] Documentation

**Success Criteria:**
- Can create Story Teller-quality behaviors automatically
- Behaviors perform comparably to hand-crafted ones
- Full observability into creation process

---

## Actions Specification

### Research Phase Actions

```python
Action(
    name="research_domain",
    description="""Research a domain to understand what a behavior should do.

    Uses web search, existing behavior patterns, and LLM analysis to build
    comprehensive domain knowledge.""",
    parameters={
        "description": ActionParameter(
            name="description",
            description="User's description of desired behavior",
            type="string",
            required=True,
        ),
        "search_web": ActionParameter(
            name="search_web",
            description="Whether to search web for best practices",
            type="bool",
            required=False,
        ),
        "search_existing": ActionParameter(
            name="search_existing",
            description="Whether to search existing behaviors for patterns",
            type="bool",
            required=False,
        ),
    },
)

Action(
    name="gather_requirements",
    description="Ask clarifying questions to understand behavior requirements.",
    parameters={
        "aspect": ActionParameter(
            name="aspect",
            description="What aspect to clarify: 'scope', 'actions', 'constraints', 'users'",
            type="string",
            required=True,
        ),
    },
)
```

### Design Phase Actions

```python
Action(
    name="draft_behavior_structure",
    description="Create initial behavior structure with ID, name, description.",
    parameters={
        "behavior_id": ActionParameter(...),
        "name": ActionParameter(...),
        "description": ActionParameter(...),
        "domain_context": ActionParameter(...),
    },
)

Action(
    name="add_action",
    description="Add an action to the behavior being designed.",
    parameters={
        "name": ActionParameter(...),
        "description": ActionParameter(...),
        "parameters": ActionParameter(type="dict", ...),
        "triggers": ActionParameter(type="list", ...),
        "examples": ActionParameter(type="list", ...),
    },
)

Action(
    name="add_trigger",
    description="Add a trigger pattern for behavior activation.",
    parameters={
        "name": ActionParameter(...),
        "semantic_patterns": ActionParameter(type="list", ...),
        "keyword_patterns": ActionParameter(type="list", ...),
        "priority": ActionParameter(type="int", ...),
    },
)

Action(
    name="define_state_model",
    description="Define a state model if the behavior needs to track state.",
    parameters={
        "state_class_name": ActionParameter(...),
        "fields": ActionParameter(type="dict", ...),
        "initial_values": ActionParameter(type="dict", ...),
    },
)
```

### Build Phase Actions

```python
Action(
    name="draft_decision_prompt",
    description="""Draft the decision prompt that determines which action to take.

    Should include:
    - Role definition
    - Available actions
    - Decision criteria
    - Response format""",
    parameters={
        "role_description": ActionParameter(...),
        "decision_criteria": ActionParameter(type="list", ...),
        "domain_knowledge": ActionParameter(...),
    },
)

Action(
    name="draft_synthesis_prompt",
    description="Draft the synthesis prompt that formats responses.",
    parameters={
        "style_guidelines": ActionParameter(type="list", ...),
        "tone": ActionParameter(...),
    },
)

Action(
    name="generate_test_cases",
    description="Generate test cases for the behavior.",
    parameters={
        "test_type": ActionParameter(
            description="Type: 'positive', 'negative', 'edge_case', 'all'",
            type="string",
        ),
        "count": ActionParameter(type="int", ...),
    },
)
```

### Test Phase Actions

```python
Action(
    name="run_tests",
    description="Run test cases against the behavior.",
    parameters={
        "test_ids": ActionParameter(
            description="Specific tests to run, or 'all'",
            type="list",
        ),
    },
)

Action(
    name="analyze_failures",
    description="Analyze test failures to identify patterns and root causes.",
    parameters={},  # Analyzes most recent test run
)

Action(
    name="suggest_fixes",
    description="Suggest fixes for identified failure patterns.",
    parameters={
        "failure_id": ActionParameter(...),
    },
)

Action(
    name="apply_fix",
    description="Apply a suggested fix to the behavior.",
    parameters={
        "fix_id": ActionParameter(...),
        "target": ActionParameter(
            description="What to fix: 'prompt', 'action', 'trigger'",
            type="string",
        ),
    },
)
```

### Evolution Phase Actions

```python
Action(
    name="initialize_population",
    description="Initialize evolution population from base behavior.",
    parameters={
        "population_size": ActionParameter(type="int", default=6),
        "mutation_rate": ActionParameter(type="float", default=0.3),
    },
)

Action(
    name="evolve_generation",
    description="Run one generation of evolution.",
    parameters={
        "generation": ActionParameter(type="int"),
    },
)

Action(
    name="evaluate_fitness",
    description="Evaluate fitness of a behavior variant on test cases.",
    parameters={
        "variant_id": ActionParameter(...),
        "test_set": ActionParameter(
            description="'train', 'holdout', or 'all'",
            type="string",
        ),
    },
)

Action(
    name="check_overfitting",
    description="Check if evolved behavior is overfitting to train set.",
    parameters={
        "variant_id": ActionParameter(...),
        "threshold": ActionParameter(type="float", default=0.1),
    },
)

Action(
    name="evolve_mutation_prompts",
    description="""Evolve the mutation prompts themselves (self-referential).

    This is the key innovation from Promptbreeder research.""",
    parameters={
        "generation": ActionParameter(type="int"),
    },
)
```

### Registration Phase Actions

```python
Action(
    name="register_behavior",
    description="Register the behavior in the registry.",
    parameters={
        "tier": ActionParameter(
            description="Tier: 'generated' or 'experimental'",
            type="string",
            default="generated",
        ),
        "status": ActionParameter(
            description="Status: 'staging', 'active', 'testing'",
            type="string",
            default="staging",
        ),
    },
)

Action(
    name="promote_behavior",
    description="Promote a behavior from STAGING to ACTIVE.",
    parameters={
        "behavior_id": ActionParameter(...),
        "reason": ActionParameter(...),
    },
)
```

---

## Prompts Design

### Research Prompt

```python
BEHAVIOR_ARCHITECT_RESEARCH_PROMPT = '''You are researching a domain to create an AI behavior.

USER REQUEST:
{description}

EXISTING SIMILAR BEHAVIORS:
{existing_behaviors}

WEB SEARCH RESULTS:
{search_results}

Analyze this information and produce:

1. CORE TASKS: What are the main things this behavior should do?
2. SUGGESTED ACTIONS: What specific actions should be available?
3. TRIGGERS: What user inputs should activate this behavior?
4. CONSTRAINTS: What should this behavior NOT do?
5. DOMAIN KNOWLEDGE: What background info should be in the prompt?
6. SOURCES: What sources informed this analysis?

Be thorough but focused. This research will guide behavior creation.

Respond in XML format:
<research>
  <core_tasks>
    <task>...</task>
  </core_tasks>
  <suggested_actions>
    <action name="..." description="...">
      <parameter name="..." type="..." required="..."/>
    </action>
  </suggested_actions>
  <triggers>
    <trigger>...</trigger>
  </triggers>
  <constraints>
    <constraint>...</constraint>
  </constraints>
  <domain_knowledge>...</domain_knowledge>
  <sources>
    <source>...</source>
  </sources>
</research>
'''
```

### Design Prompt

```python
BEHAVIOR_ARCHITECT_DESIGN_PROMPT = '''You are designing an AI behavior based on research.

RESEARCH RESULTS:
{research}

USER CONSTRAINTS:
{user_constraints}

Design a complete behavior structure:

1. For each core task, define an ACTION with:
   - Clear name (snake_case)
   - Helpful description
   - Required and optional parameters
   - Trigger patterns
   - Usage examples

2. Define TRIGGERS that determine when this behavior activates:
   - Semantic patterns (LLM evaluates)
   - Keyword patterns (regex)
   - Priority for conflicts

3. Define any STATE MODEL if the behavior needs to track state across turns.

4. Define CONSTRAINTS:
   - Actions requiring confirmation
   - Blocked actions
   - Rate limits
   - Style guidelines

Focus on usability and robustness. Actions should be intuitive.

Respond in XML format:
<design>
  <behavior_id>...</behavior_id>
  <name>...</name>
  <description>...</description>
  <actions>...</actions>
  <triggers>...</triggers>
  <state_model>...</state_model>
  <constraints>...</constraints>
</design>
'''
```

### Prompt Engineering Prompt

```python
BEHAVIOR_ARCHITECT_PROMPT_PROMPT = '''You are writing prompts for an AI behavior.

BEHAVIOR DESIGN:
{design}

DOMAIN KNOWLEDGE:
{domain_knowledge}

Write two prompts:

1. DECISION PROMPT: Determines which action to take
   - Define the role clearly
   - List available actions with descriptions
   - Explain when to use each action
   - Include decision criteria
   - Specify response format (XML)
   - Include domain-specific guidance

2. SYNTHESIS PROMPT: Formats the response to the user
   - Define output style
   - Include domain-appropriate tone
   - Specify any formatting requirements

The decision prompt is the "brain" - it must be thorough.
The synthesis prompt shapes the "voice" - it must match the domain.

Respond with:
<prompts>
  <decision_prompt>
    ...full prompt text...
  </decision_prompt>
  <synthesis_prompt>
    ...full prompt text...
  </synthesis_prompt>
</prompts>
'''
```

### Test Generation Prompt

```python
BEHAVIOR_ARCHITECT_TEST_PROMPT = '''You are generating test cases for an AI behavior.

BEHAVIOR:
{behavior}

Generate comprehensive test cases:

1. POSITIVE TESTS: One for each action showing correct usage
2. NEGATIVE TESTS: Common mistakes, edge cases, invalid inputs
3. EDGE CASES: Ambiguous inputs, boundary conditions

Each test should have:
- test_id: Unique identifier
- name: Human-readable name
- description: What's being tested
- user_query: Input to test
- expected_actions: Which actions should be taken
- context: Any required context
- forbidden_actions: Actions that should NOT be taken

Generate at least:
- 1 positive test per action
- 3 negative tests
- 3 edge case tests

Respond with:
<test_cases>
  <test id="..." name="...">
    <description>...</description>
    <user_query>...</user_query>
    <expected_actions>...</expected_actions>
    <context>...</context>
    <forbidden_actions>...</forbidden_actions>
  </test>
</test_cases>
'''
```

### Evolution Mutation Prompts (Self-Referential)

```python
INITIAL_MUTATION_PROMPTS = [
    MutationPrompt(
        prompt_id="expand_detail",
        content="Make this prompt more detailed and specific. Add examples.",
    ),
    MutationPrompt(
        prompt_id="simplify",
        content="Simplify this prompt. Remove redundancy. Be more direct.",
    ),
    MutationPrompt(
        prompt_id="add_constraints",
        content="Add constraints and edge case handling to this prompt.",
    ),
    MutationPrompt(
        prompt_id="improve_structure",
        content="Improve the structure and organization of this prompt.",
    ),
    MutationPrompt(
        prompt_id="domain_focus",
        content="Make this prompt more domain-specific and expert-level.",
    ),
]

MUTATION_PROMPT_EVOLUTION_PROMPT = '''You are evolving mutation prompts.

CURRENT MUTATION PROMPTS WITH FITNESS:
{mutation_prompts}

PROMPT BEING MUTATED:
{target_prompt}

MUTATION RESULTS (which mutations improved fitness):
{mutation_results}

Based on what's working, evolve the mutation prompts:
1. Keep high-fitness mutations
2. Combine elements of successful mutations
3. Create new mutations inspired by patterns
4. Remove consistently low-fitness mutations

This is META-evolution: you're improving how we improve prompts.

Respond with evolved mutation prompts:
<mutation_prompts>
  <mutation id="..." fitness="...">
    ...mutation instruction...
  </mutation>
</mutation_prompts>
'''
```

---

## Evolution System

### Algorithm: Enhanced Meta Agent Search

Based on ADAS and Promptbreeder research, with draagon-ai-specific enhancements:

```python
async def evolve_behavior(
    behavior: Behavior,
    test_cases: list[BehaviorTestCase],
    config: EvolutionConfig,
) -> BehaviorEvolutionResult:
    """
    Enhanced evolution with self-referential mutation.
    """
    # 1. Split test cases
    train_cases, holdout_cases = split_train_holdout(
        test_cases, ratio=config.train_test_split
    )

    # 2. Initialize population
    population = [behavior]  # Base
    for _ in range(config.population_size - 1):
        mutant = await mutate_behavior(behavior, random.choice(mutation_prompts))
        population.append(mutant)

    # 3. Initialize mutation prompts
    mutation_prompts = INITIAL_MUTATION_PROMPTS.copy()

    # 4. Evolution loop
    for generation in range(config.generations):
        # Evaluate fitness on train set
        for variant in population:
            variant.metrics.fitness_score = await evaluate_fitness(
                variant, train_cases
            )

        # Apply fitness sharing for diversity
        apply_fitness_sharing(population)

        # Tournament selection
        parents = tournament_select(
            population,
            config.tournament_size,
            count=config.population_size // 2,
        )

        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            if random.random() < config.crossover_rate:
                child = crossover(parents[i], parents[i+1])
                offspring.append(child)

        # Mutation (using evolved mutation prompts)
        for variant in offspring:
            if random.random() < config.mutation_rate:
                mutation = weighted_random_choice(
                    mutation_prompts,
                    weights=[m.fitness for m in mutation_prompts]
                )
                variant = await mutate_behavior(variant, mutation)

        # Elitism: keep best
        population = sorted(
            population,
            key=lambda b: b.metrics.fitness_score,
            reverse=True,
        )[:config.elitism_count] + offspring

        # SELF-REFERENTIAL: Evolve mutation prompts every N generations
        if generation % 2 == 0:
            mutation_prompts = await evolve_mutation_prompts(
                mutation_prompts,
                mutation_results,
            )

    # 5. Validate on holdout
    best = max(population, key=lambda b: b.metrics.fitness_score)
    holdout_fitness = await evaluate_fitness(best, holdout_cases)
    train_fitness = best.metrics.fitness_score

    # 6. Check overfitting
    overfitting_gap = train_fitness - holdout_fitness
    approved = overfitting_gap < config.overfitting_threshold

    return BehaviorEvolutionResult(
        original_behavior=behavior,
        evolved_behavior=best,
        original_fitness=behavior.metrics.fitness_score,
        evolved_fitness=holdout_fitness,
        overfitting_gap=overfitting_gap,
        generations_run=config.generations,
        approved=approved,
    )
```

### Fitness Function

```python
async def evaluate_fitness(
    behavior: Behavior,
    test_cases: list[BehaviorTestCase],
) -> float:
    """
    Evaluate behavior fitness on test cases.

    Fitness = (correct_action_rate * 0.5) +
              (response_quality * 0.3) +
              (latency_score * 0.2)
    """
    results = []
    for test in test_cases:
        result = await run_behavior_test(behavior, test)
        results.append(result)

    # Correct action rate
    correct = sum(1 for r in results if r.passed)
    action_rate = correct / len(results)

    # Response quality (LLM-judged)
    quality_scores = [
        await judge_response_quality(r)
        for r in results if r.passed
    ]
    quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    # Latency score (faster is better)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    latency_score = max(0, 1 - (avg_latency / 2000))  # 2s = 0 score

    return (action_rate * 0.5) + (quality * 0.3) + (latency_score * 0.2)
```

### Diversity Preservation

```python
def apply_fitness_sharing(
    population: list[Behavior],
    sigma: float = 0.3,
) -> None:
    """
    Apply fitness sharing to maintain population diversity.
    Similar individuals share fitness, preventing convergence.
    """
    for i, b1 in enumerate(population):
        niche_count = 0
        for j, b2 in enumerate(population):
            if i != j:
                similarity = compute_prompt_similarity(b1, b2)
                if similarity > sigma:
                    niche_count += 1 - (similarity - sigma) / (1 - sigma)

        # Reduce fitness by niche count
        if niche_count > 0:
            b1.metrics.fitness_score /= (1 + niche_count)
```

---

## Safety and Trust

### Trust Model

| Tier | Trust Level | Creation Method | Approval Required |
|------|-------------|-----------------|-------------------|
| CORE | Maximum | Human developer | PR review |
| ADDON | High | Official packages | Package review |
| APPLICATION | Medium | Host app | App developer |
| GENERATED | Low | Behavior Architect | Automated + manual |
| EXPERIMENTAL | Minimal | Any | None (testing only) |

### Safety Checks

```python
@dataclass
class BehaviorSafetyCheck:
    """Safety checks before registering a behavior."""

    # Prompt safety
    prompt_contains_harmful: bool = False
    prompt_enables_deception: bool = False
    prompt_bypasses_safety: bool = False

    # Action safety
    actions_require_confirmation: list[str] = field(default_factory=list)
    actions_access_sensitive: list[str] = field(default_factory=list)

    # Test coverage
    has_negative_tests: bool = True
    has_edge_case_tests: bool = True
    test_coverage_percent: float = 0.0

    # Performance
    pass_rate: float = 0.0
    min_acceptable_pass_rate: float = 0.8

    def is_safe(self) -> bool:
        return (
            not self.prompt_contains_harmful and
            not self.prompt_enables_deception and
            not self.prompt_bypasses_safety and
            self.has_negative_tests and
            self.pass_rate >= self.min_acceptable_pass_rate
        )
```

### Staged Rollout

```
STAGING (Shadow Mode)
    │
    │ Behavior executes but doesn't affect user
    │ Metrics tracked: accuracy, latency, errors
    │ Duration: 7 days minimum
    │
    ▼
REVIEW
    │
    │ Manual review of:
    │   - Metrics
    │   - Sample responses
    │   - Edge cases
    │
    ▼
ACTIVE (or DEPRECATED if issues found)
```

---

## Testing Strategy

### Unit Tests

```python
class TestBehaviorArchitectResearch:
    """Test research phase."""

    async def test_research_simple_domain(self):
        """Can research a simple domain."""
        result = await architect.research_domain(
            "A behavior for tracking daily water intake"
        )
        assert len(result.core_tasks) >= 3
        assert len(result.suggested_actions) >= 2

    async def test_research_uses_web_search(self):
        """Web search is used when enabled."""
        # ...

class TestBehaviorArchitectDesign:
    """Test design phase."""

    async def test_design_from_research(self):
        """Can design behavior from research."""
        # ...

class TestBehaviorArchitectEvolution:
    """Test evolution phase."""

    async def test_evolution_improves_fitness(self):
        """Evolution increases fitness."""
        # ...

    async def test_self_referential_mutation(self):
        """Mutation prompts evolve."""
        # ...

    async def test_overfitting_detection(self):
        """Overfitting is detected and rejected."""
        # ...
```

### Integration Tests

```python
class TestBehaviorArchitectEndToEnd:
    """End-to-end behavior creation tests."""

    async def test_create_simple_behavior(self):
        """Can create a simple behavior from scratch."""
        behavior = await architect.create_behavior(
            "A behavior for setting kitchen timers"
        )
        assert behavior.behavior_id is not None
        assert len(behavior.actions) >= 3
        assert behavior.test_results.pass_rate >= 0.8

    async def test_create_complex_behavior(self):
        """Can create a complex behavior like StoryTeller."""
        # ...

    async def test_evolve_existing_behavior(self):
        """Can evolve an existing behavior to improve it."""
        # ...
```

### Quality Tests

```python
class TestGeneratedBehaviorQuality:
    """Test quality of generated behaviors."""

    async def test_prompt_quality(self):
        """Generated prompts are well-structured."""
        # ...

    async def test_action_coverage(self):
        """Actions cover all core tasks."""
        # ...

    async def test_test_case_quality(self):
        """Generated tests are meaningful."""
        # ...
```

---

## Success Metrics

### Phase 1 (Foundation)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Basic behavior creation | Works | Manual test |
| Actions per behavior | >= 5 | Count |
| Test case generation | >= 10 per behavior | Count |
| Pass rate (first attempt) | >= 50% | Test runs |

### Phase 2 (Iteration)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Failure analysis accuracy | >= 80% | Human eval |
| Fix success rate | >= 60% | Before/after tests |
| Iterations to passing | <= 5 | Count |
| Pass rate (after iteration) | >= 80% | Test runs |

### Phase 3 (Evolution)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Fitness improvement | >= 10% | Before/after |
| Overfitting gap | < 10% | Train vs holdout |
| Mutation prompt improvement | Measurable | Generation tracking |
| Diversity maintained | > 40% | Population similarity |

### Phase 4 (Integration)

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end success rate | >= 70% | Full pipeline |
| Generated vs hand-crafted quality | >= 80% | Human eval |
| Creation time | < 5 minutes | Timer |
| User satisfaction | >= 4/5 | Survey |

---

## Risks and Mitigations

### Risk 1: Generated behaviors are low quality

**Likelihood:** Medium
**Impact:** High

**Mitigations:**
- Require minimum test pass rate (80%)
- Human review before ACTIVE promotion
- STAGING period to catch issues
- Allow user feedback to trigger review

### Risk 2: Evolution overfits

**Likelihood:** Medium
**Impact:** Medium

**Mitigations:**
- Train/holdout split (80/20)
- Overfitting gap check (< 10%)
- Diversity preservation (fitness sharing)
- Capability validation

### Risk 3: Self-referential mutation goes wrong

**Likelihood:** Low
**Impact:** High

**Mitigations:**
- Bound mutation prompt population size
- Require minimum diversity in mutation prompts
- Log all mutations for debugging
- Manual override available

### Risk 4: Malicious behavior creation

**Likelihood:** Low
**Impact:** High

**Mitigations:**
- Safety checks on all prompts
- Sensitive action flagging
- GENERATED tier has limited trust
- Manual promotion required for ACTIVE

---

## Future Extensions

### Behavior Store

A marketplace for sharing and discovering behaviors:

```python
@dataclass
class BehaviorListing:
    behavior: Behavior
    author: str
    downloads: int
    rating: float
    reviews: list[Review]
    verified: bool  # Passed extended testing
```

### Cross-Behavior Learning

Behaviors learn from each other:

```python
async def cross_pollinate(
    behavior_a: Behavior,
    behavior_b: Behavior,
) -> Behavior:
    """
    Create a new behavior combining strengths of two.
    """
```

### Real-Time Evolution

Evolution based on production metrics:

```python
async def continuous_evolution(
    behavior_id: str,
    metrics_stream: AsyncIterator[BehaviorMetrics],
) -> None:
    """
    Continuously evolve a behavior based on production performance.
    """
```

### YAML Export

Export for interoperability:

```python
async def export_to_yaml(
    behavior: Behavior,
    spec: Literal["open-agent-spec", "adl"] = "open-agent-spec",
) -> str:
    """
    Export behavior to YAML spec for portability.
    """
```

---

## Appendix: Comparison to ADAS

| Aspect | ADAS | draagon-ai Behavior Architect |
|--------|------|-------------------------------|
| **Representation** | Code strings | Rich dataclasses |
| **Searchable** | Archive list | Registry with queries |
| **Evolvable** | Code mutation | Prompt + action mutation |
| **Testable** | Task evaluation | Structured test cases |
| **Trust Levels** | None | 5-tier system |
| **Cognition** | None | Beliefs, learning, opinions |
| **Self-Referential** | No | Yes (mutation prompts evolve) |
| **MCP Integration** | No | Yes |

---

*Document created: December 2024*
*Status: Design Phase*
*Next: Implementation Phase 1*

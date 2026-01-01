# draagon-ai - Claude Context

**Last Updated:** 2025-12-31
**Version:** 0.1.0
**Project:** Agentic AI framework for building cognitive assistants

---

## 📋 Project Overview

draagon-ai is a framework for building agentic AI assistants with cognitive capabilities. It provides:

- **Agent Orchestration** - Decision engine, action execution, ReAct loops
- **Tool System** - `@tool` decorator, registry, MCP client integration
- **Memory Architecture** - 4-layer cognitive memory (working, episodic, semantic, metacognitive)
- **Cognitive Services** - Learning, belief reconciliation, curiosity, opinion formation
- **Autonomous Agent** - Self-directed background processing with safety tiers
- **Multi-Agent** - Orchestration modes for agent collaboration
- **Behaviors** - Declarative agent behavior definitions
- **Prompt Evolution** - Promptbreeder-style optimization with safety guards

### Reference Implementation

**Roxy Voice Assistant** (`../roxy-voice-assistant/`) is the primary reference implementation showing how to use draagon-ai for a production voice assistant.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Agent                                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Personality │  │   Behavior   │  │  Cognitive Services  │  │
│  │   (Persona)  │  │  (Actions)   │  │  (Learning, Beliefs) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Decision Engine                         │   │
│  │   Query → Activation → Decision → Execution → Response  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Tool Registry                         │   │
│  │         (Handler implementations via @tool)              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/draagon_ai/
├── orchestration/          # Core agent loop
│   ├── agent.py            # Agent class
│   ├── decision.py         # Decision engine
│   ├── execution.py        # Action executor
│   ├── loop.py             # ReAct loop, AgentResponse
│   ├── registry.py         # Tool, ToolParameter, ToolRegistry
│   ├── shared_memory.py    # Multi-agent shared working memory
│   └── autonomous/         # Background agent service
├── tools/                  # Tool system
│   ├── decorator.py        # @tool decorator
│   └── mcp_client.py       # MCP integration
├── memory/                 # Memory providers
│   └── providers/layered.py # 4-layer cognitive memory
├── cognition/              # Cognitive services
│   ├── learning.py         # Skill/fact extraction
│   ├── beliefs.py          # Belief reconciliation
│   ├── curiosity.py        # Knowledge gap detection
│   └── opinions.py         # Opinion formation
├── behaviors/              # Behavior definitions
│   └── templates/          # Pre-built behavior templates
├── evolution/              # Prompt evolution
│   └── promptbreeder.py    # Genetic prompt optimization
└── llm/                    # LLM provider protocols
```

---

## 🧠 Core Architectural Principles

### LLM-First Architecture (CRITICAL)

**NEVER use regex or keyword patterns for semantic understanding.** The LLM handles ALL semantic analysis:

| Task | ❌ WRONG | ✅ RIGHT |
|------|----------|----------|
| Detect user corrections | Regex: `r"actually|no,|wrong"` | LLM analyzes intent semantically |
| Identify user in speech | Regex: `r"it's\s+(\w+)"` | LLM extracts speaker identity |
| Classify intents | Keyword matching | LLM decision prompt |
| Detect learning opportunities | Pattern: `r"remember|my .* is"` | LLM semantic detection |
| Parse dates/times | Regex patterns | LLM extracts structured data |

**Why This Matters:**
1. **Robustness** - Works with any phrasing, typos, accents, speech-to-text errors
2. **Consistency** - One approach for all semantic tasks
3. **Maintainability** - No fragile regex patterns to update
4. **Accuracy** - LLM understands context, negation, sarcasm

**Exceptions (Non-Semantic Tasks):**
- Security blocklist patterns (command execution safety)
- TTS text transformations (e.g., "9:00" → "nine o'clock")
- URL/email validation
- Entity ID resolution (exact string matching)
- Parsing structured LLM output (XML element extraction)

### XML Output Format for LLM Prompts (CRITICAL)

**ALWAYS use XML format for LLM output, NOT JSON.** XML is better for LLMs because:
1. **Fewer escaping issues** - JSON requires escaping quotes, backslashes, newlines
2. **Better streaming** - XML can be parsed incrementally as tokens arrive
3. **More robust parsing** - Malformed XML is easier to recover from than malformed JSON
4. **Clearer nesting** - Element names make structure self-documenting

| ❌ WRONG (JSON) | ✅ RIGHT (XML) |
|-----------------|----------------|
| `Output JSON: {"action":"get_time","args":{}}` | `Output XML: <response><action>get_time</action></response>` |
| `{"answer":"It's 3pm","confidence":0.9}` | `<response><answer>It's 3pm</answer><confidence>0.9</confidence></response>` |

**XML Response Template:**
```xml
<response>
  <action>action_name</action>
  <reasoning>Why this action was chosen</reasoning>
  <answer>The response text (if action=answer)</answer>
  <args>
    <query>search query</query>
    <entity_id>light.bedroom</entity_id>
  </args>
  <confidence>0.9</confidence>
</response>
```

### Confidence-Based Actions

When confidence is uncertain, use graduated responses:

| Confidence | Action |
|------------|--------|
| > 0.9 | Proceed without confirmation |
| 0.7-0.9 | Proceed for normal ops, confirm for sensitive |
| 0.5-0.7 | Confirm for most operations |
| < 0.5 | Require explicit confirmation |

### Pragmatic Async

Use async when it provides real benefit. Keep sync code simple.

**Use async for:**
- External I/O (LLM calls, database, HTTP)
- Concurrent operations (parallel agents)
- Background tasks (learning, consolidation)

**Keep synchronous:**
- Pure computation and data transformation
- Configuration and initialization
- Simple utilities, getters, builders

---

## 🔧 Tool System

### @tool Decorator

The `@tool` decorator provides declarative tool registration:

```python
from draagon_ai.tools import tool

@tool(
    name="get_time",
    description="Get the current time and date",
    category="utilities",
)
async def get_time(args: dict, **context) -> dict:
    return {"time": datetime.now().isoformat()}

@tool(
    name="search_web",
    description="Search the web for information",
    parameters={
        "query": {"type": "string", "description": "Search query"},
    },
    category="search",
    tags=["web", "research"],
)
async def search_web(args: dict, **context) -> dict:
    query = args.get("query", "")
    # ... perform search
    return {"results": [...]}
```

### Tool Discovery

```python
from draagon_ai.tools import discover_tools, get_all_tools

# Discover all tools in a package
discover_tools("myapp.tools.handlers")

# Get all registered tools
tools = get_all_tools()
```

### Tool Registry

```python
from draagon_ai.tools import Tool, ToolParameter, ToolRegistry

# Create tool programmatically
tool = Tool(
    name="my_tool",
    description="Does something",
    handler=my_handler,
    parameters=[
        ToolParameter(name="arg1", type="string", description="First arg"),
    ],
)

# Register with registry
registry = ToolRegistry()
registry.register(tool)
```

---

## 🧠 Memory Architecture

### 4-Layer Cognitive Memory

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Metacognitive (Permanent)                             │
│  Self-knowledge, capabilities, limitations, learned patterns    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Semantic (6 months TTL)                               │
│  Facts, skills, preferences, consolidated knowledge             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Episodic (2 weeks TTL)                                │
│  Conversation summaries, interaction patterns                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Working (5 minutes TTL)                               │
│  Current conversation context, recent tool results              │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Types

| Type | Description | Example |
|------|-------------|---------|
| **Skill** | Procedural knowledge (how-to) | "How to restart Plex: `docker restart plex`" |
| **Fact** | Declarative knowledge | "Doug's birthday is March 15" |
| **Insight** | Meta-patterns | "Calendar queries often need location context" |
| **Preference** | User preferences | "Doug prefers Celsius for temperature" |
| **Instruction** | User directives | "Always confirm before deleting files" |
| **Episodic** | Conversation summaries | "Discussed vacation plans on Dec 15" |

### Memory Importance Weights

```python
INSTRUCTION: 1.0   # Always prioritize
PREFERENCE: 0.9
SKILL: 0.85        # Procedural knowledge
FACT: 0.8
KNOWLEDGE: 0.7
INSIGHT: 0.65
EPISODIC: 0.5
```

### Shared Cognitive Working Memory (Multi-Agent)

For multi-agent coordination, use `SharedWorkingMemory` to enable agents to share observations with attention-weighted access and conflict detection. Based on cognitive psychology research (Miller's Law: 7±2 items, Baddeley's Working Memory Model).

```python
from draagon_ai.orchestration.shared_memory import (
    SharedWorkingMemory,
    SharedWorkingMemoryConfig,
    SharedObservation,
)
from draagon_ai.orchestration.multi_agent_orchestrator import AgentRole

# Create shared memory for a task
shared_memory = SharedWorkingMemory(task_id="task_123")

# Agent A adds an observation
obs = await shared_memory.add_observation(
    content="User prefers dark mode",
    source_agent_id="agent_a",
    attention_weight=0.8,
    is_belief_candidate=True,
    belief_type="PREFERENCE",
)

# Agent B retrieves context (filtered by role, Miller's Law: 7 items max)
context = await shared_memory.get_context_for_agent(
    agent_id="agent_b",
    role=AgentRole.RESEARCHER,
    max_items=7,
)

# Apply periodic attention decay (call every N iterations)
await shared_memory.apply_attention_decay()

# Get conflicts for reconciliation
conflicts = await shared_memory.get_conflicts()

# Get belief candidates (non-conflicting observations ready for belief formation)
candidates = await shared_memory.get_belief_candidates()
```

**Key Features:**
- **Miller's Law Capacity**: 7±2 items per agent, 50 global max
- **Attention Weighting**: Decay (×0.9) and boost (+0.2) with 1.0 cap
- **Conflict Detection**: Phase 1 heuristic (same belief_type), Phase 2 embeddings (optional)
- **Role-Based Filtering**: CRITIC sees candidates, RESEARCHER sees all, EXECUTOR sees SKILL/FACT
- **Concurrent Safety**: asyncio.Lock for safe multi-agent access

**Role-Based Context Filtering:**

| Role | Sees |
|------|------|
| CRITIC | Only belief candidates (for evaluation) |
| RESEARCHER | All observations |
| EXECUTOR | Only SKILL and FACT types |

**Integration with TaskContext:**

```python
from draagon_ai.orchestration.multi_agent_orchestrator import TaskContext

# Inject shared memory into TaskContext for parallel agents
context = TaskContext(task_id="task_123", query="Analyze user preferences")
context.working_memory["__shared__"] = SharedWorkingMemory(context.task_id)

# Agents access via context.working_memory["__shared__"]
```

### Memory Reinforcement Learning

Memories learn from usage outcomes. When a memory helps produce a correct response, it gets boosted. When it leads to errors, it gets demoted. This enables memories to naturally move between layers based on their proven usefulness.

```python
from draagon_ai.memory.providers.layered import LayeredMemoryProvider

provider = LayeredMemoryProvider(...)

# After using a memory and verifying the response was helpful
await provider.record_usage(memory_id, "success")

# After using a memory that led to a wrong answer
await provider.record_usage(memory_id, "failure")

# Direct boost/demote for specific scenarios
await provider.boost_memory(memory_id, boost_amount=0.1)  # Custom boost
await provider.demote_memory(memory_id)  # Default demotion
```

**Reinforcement Constants:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| BOOST_AMOUNT | 0.05 | Default importance boost per success |
| DEMOTE_AMOUNT | 0.08 | Default importance penalty per failure |
| MAX_IMPORTANCE | 1.0 | Importance ceiling |
| MIN_IMPORTANCE | 0.1 | Importance floor |

**Layer Promotion Thresholds:**
| Transition | Importance Threshold |
|------------|---------------------|
| Working → Episodic | ≥ 0.7 |
| Episodic → Semantic | ≥ 0.8 |
| Semantic → Metacognitive | ≥ 0.9 |

**Layer Demotion Thresholds:**
| Transition | Importance Threshold |
|------------|---------------------|
| Metacognitive → Semantic | < 0.6 |
| Semantic → Episodic | < 0.4 |
| Episodic → Working | < 0.3 |

**Integration Example (Prototype):**

The semantic expansion prototype (`prototypes/semantic_expansion/`) demonstrates how memory reinforcement can be integrated with semantic processing. When graduated to core, the orchestrator would track all memories used during processing:

```python
# Example from semantic_expansion prototype (not yet in core)
# from prototypes.semantic_expansion.src.integration import TwoPassSemanticOrchestrator

orchestrator = TwoPassSemanticOrchestrator(memory=provider, llm=llm)
result = await orchestrator.process("Doug has 3 cats")

# Used memories are tracked in result.used_memories
# {'mem_1': 'supporting', 'mem_2': 'contradicting', 'mem_3': 'context'}

# After verifying response was correct
await result.record_outcome(provider, "success")

# After discovering response was wrong
await result.record_outcome(provider, "failure")
```

**Intelligent Reinforcement (planned):**
- Supporting memories get boosted on success, demoted on failure
- Contradicting memories get the OPPOSITE treatment (if response succeeded despite contradiction, the contradiction was wrong)
- Context memories follow standard reinforcement

---

## 🧠 Cognitive Architecture

### Belief System

```python
@dataclass
class UserObservation:
    """Immutable record of what a user told the agent."""
    content: str                    # "We have 6 cats"
    source_user_id: str             # "doug"
    scope: ObservationScope         # PRIVATE | PERSONAL | HOUSEHOLD
    timestamp: datetime
    confidence_expressed: float     # How certain did the user sound?

@dataclass
class AgentBelief:
    """Reconciled understanding formed from observations."""
    content: str                    # "The household has 6 cats"
    belief_type: BeliefType         # HOUSEHOLD_FACT, VERIFIED_FACT, etc.
    confidence: float               # 0.0 - 1.0
    supporting_observations: list   # Observation IDs that support this
    conflicting_observations: list  # Observations that contradict
    verified: bool                  # Has the agent verified this?
    needs_clarification: bool       # Does agent need to ask about this?
```

### Belief Reconciliation

When users provide conflicting information:
1. **Accept Latest** - User correcting themselves
2. **Weight by Credibility** - Consider track record
3. **Ask for Clarification** - Queue question for appropriate time
4. **Flag Conflict** - Mark belief as needing resolution

### Curiosity Engine

The agent identifies knowledge gaps and queues questions:
- Maximum 3 questions per day
- 30-minute minimum gap between questions
- Questions asked during natural conversation pauses

### Opinion Formation

Opinions form gradually through interaction:
- Tracked with confidence and openness to change
- Agent genuinely considers arguments before changing
- Change history is maintained

---

## 🤖 Autonomous Agent

### Action Tiers

| Tier | Scope | Examples | Approval |
|------|-------|----------|----------|
| **OBSERVE** | Read-only information gathering | Check weather, read calendar | Auto |
| **LEARN** | Store knowledge, update memories | Learn facts, store skills | Auto |
| **SUGGEST** | Queue notifications for user | "Meeting in 30 min" | Auto |
| **ACT_MINOR** | Low-impact changes | Set timer, add reminder | Auto |
| **ACT_MAJOR** | Significant changes | Delete files, modify calendar | Confirm |
| **ACT_CRITICAL** | High-risk operations | Financial, security | Passcode |

### Safety Checks

Every proposed action goes through:
1. **Harm Check** - Could this cause harm?
2. **Reversibility Check** - Can this be undone?
3. **User Intent Check** - Is this what the user would want?
4. **Scope Check** - Is this within the agent's authority?

### Self-Monitoring

The autonomous agent monitors its own behavior:
- Tracks success/failure rates
- Identifies patterns in errors
- Adjusts behavior based on feedback

---

## 🧬 Prompt Evolution

### Promptbreeder-Style Optimization

```python
from draagon_ai.evolution import PromptbreederEvolution

evolution = PromptbreederEvolution(
    base_prompt=DECISION_PROMPT,
    population_size=6,
    generations=5,
)

# Run evolution with test cases from conversation history
result = await evolution.evolve(test_cases)

# Result includes best prompt and validation metrics
if result.validated:
    # Safe to deploy
    apply_prompt(result.best_prompt)
```

### Safety Guards (Prevent Overfitting)

1. **Train/Holdout Split (80/20)** - Prompts must perform well on held-out data
2. **Overfitting Gap Check** - Rejects if train-holdout fitness gap > 10%
3. **Capability Preservation** - Validates all actions from original prompt are preserved
4. **Diversity Preservation** - Fitness sharing penalizes similar prompts
5. **Minimum Diversity Check** - Rejects if population diversity falls below 40%

---

## 🔌 Protocols

### LLMProvider

```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[dict],
        model_tier: str = "standard",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str: ...
```

### MemoryProvider

```python
class MemoryProvider(Protocol):
    async def store(self, content: str, metadata: dict) -> str: ...
    async def search(self, query: str, limit: int = 10) -> list[Memory]: ...
    async def get(self, memory_id: str) -> Memory | None: ...
    async def update(self, memory_id: str, updates: dict) -> bool: ...
    async def delete(self, memory_id: str) -> bool: ...
```

### ToolProvider

```python
class ToolProvider(Protocol):
    async def execute(self, tool_call: ToolCall, context: dict) -> ToolResult: ...
    def list_tools(self) -> list[str]: ...
    def get_tool_description(self, tool_name: str) -> str | None: ...
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/orchestration/ -v

# Run with coverage
pytest tests/ --cov=src/draagon_ai --cov-report=html
```

### Test Structure

```
tests/
├── orchestration/      # Agent, decision, execution tests
├── tools/              # Tool decorator, registry tests
├── memory/             # Memory provider tests
├── cognition/          # Learning, beliefs, curiosity tests
├── evolution/          # Prompt evolution tests
└── integration/        # End-to-end tests
```

---

## 🔒 Testing Integrity Principles (CRITICAL)

**See `CONSTITUTION.md` for full principles.** These are inviolable:

### 1. NEVER Weaken Tests to Pass (ABSOLUTE RULE)

**Tests exist to validate the system. The system must rise to meet the tests.**

| ❌ FORBIDDEN | ✅ REQUIRED |
|--------------|-------------|
| Lower threshold from 80% to 60% | Fix the underlying bug causing 60% |
| Remove failing test case | Debug why the test case fails |
| Add "skip" without root cause analysis | Document gap and create fix plan |
| Change expected value to match wrong output | Fix algorithm to produce correct output |

### 2. Always Fix Root Issues When Found

**If you discover a bug while working on something else, FIX IT IMMEDIATELY.**

Even if "out of scope" - fix wrong constants, missing edge cases, logical errors.

### 3. Tests Must Be Designed to Fail Initially

Good tests challenge the system. Include:
- **Tier 1**: Industry standard tests (must pass for production)
- **Tier 2**: Advanced tests (push boundaries, may initially fail)
- **Tier 3**: Frontier tests (represent unsolved problems)

### 4. Prevent Test Overfitting

Include novel test cases, randomized testing, out-of-domain examples, adversarial inputs.

### 5. Benchmark Against Industry Standards

Compare to published benchmarks (BioScope, CoNLL, TempEval, ATOMIC, etc.).

---

## 🧪 Prototypes

The `prototypes/` folder contains experimental code that explores new capabilities before they're integrated into the core framework. Prototypes are designed to be isolated, self-contained, and safe to experiment with.

### Philosophy

1. **Experimentation First** - Try ideas quickly without breaking production code
2. **Minimal Dependencies** - Prototypes use only stable core types (Memory, MemoryType, etc.)
3. **Explicit Integration** - Code only moves to core after validation and explicit wiring
4. **Documentation** - Each prototype documents its hypothesis, status, and findings

### Folder Structure

```
prototypes/
├── README.md                     # Overview of all prototypes
├── semantic_expansion/           # Example prototype
│   ├── CLAUDE.md                 # Prototype-specific Claude context (READ THIS!)
│   ├── README.md                 # Quick overview
│   ├── docs/
│   │   ├── research/             # Background, prior art, concepts
│   │   ├── requirements/         # FR-xxx requirement docs
│   │   ├── specs/                # Technical architecture specs
│   │   └── findings/             # Experiment results, learnings
│   ├── src/                      # Prototype code
│   │   ├── __init__.py
│   │   └── *.py                  # Implementation files
│   └── tests/                    # Prototype tests
│       ├── conftest.py           # Path setup
│       └── test_*.py             # Test files
└── [future_prototype]/           # Next experiment
```

### Prototype-Specific Claude Context

**IMPORTANT:** When working on a prototype, ALWAYS read its `CLAUDE.md` first!

Each prototype has its own `CLAUDE.md` with:
- Prototype-specific architecture and patterns
- Key files and their purposes
- Important coding conventions
- Integration status and blocking issues

```bash
# Before working on a prototype, read its context:
prototypes/semantic_expansion/CLAUDE.md
prototypes/[other_prototype]/CLAUDE.md
```

### Prototype Documentation Structure

Each prototype maintains its own documentation in `docs/`:

| Folder | Purpose | Examples |
|--------|---------|----------|
| `research/` | Background concepts, prior art | Papers, concept docs |
| `requirements/` | FR-xxx requirements | FR-006, FR-007 |
| `specs/` | Technical architecture | Architecture diagrams |
| `findings/` | Experiment results | Success/failure notes |

This keeps prototype documentation self-contained. When a prototype is deleted or archived, its docs go with it.

### Creating a New Prototype

1. Create folder structure:
```bash
mkdir -p prototypes/my_prototype/{src,tests,docs/{research,requirements,specs,findings}}
```

2. Add `CLAUDE.md` with:
   - Status (Experimental/Validated/Deprecated)
   - Hypothesis being tested
   - Key concepts and architecture
   - File structure
   - Important patterns
   - Integration readiness

3. Add `README.md` with quick overview

4. Add `tests/conftest.py` for path setup:

```python
import sys
from pathlib import Path

# Add prototype src to path
prototype_root = Path(__file__).parent.parent
src_path = prototype_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add core draagon-ai for base types
project_root = prototype_root.parent.parent
draagon_src = project_root / "src"
if str(draagon_src) not in sys.path:
    sys.path.insert(0, str(draagon_src))
```

5. Use direct imports (not relative) in prototype code
6. Only import stable core types from draagon-ai

### Running Prototype Tests

```bash
# Run specific prototype tests
cd prototypes/semantic_expansion
python3 -m pytest tests/ -v

# Run with real LLM provider
GROQ_API_KEY=your_key python3 -m pytest tests/ -v
```

### When Is a Prototype Ready for Integration?

A prototype is ready to graduate when:
1. **Validated** - Tests demonstrate the concept works
2. **Protocol Compatible** - Uses standard draagon-ai protocols (LLMProvider, MemoryProvider)
3. **Wired In** - There's a clear integration point (e.g., AgentLoop, DecisionEngine)
4. **Documented** - Integration plan is documented
5. **Safe** - Doesn't break existing functionality (regression tests pass)

### Current Prototypes

| Prototype | Status | Description |
|-----------|--------|-------------|
| `semantic_expansion` | Experimental | Two-pass semantic understanding with WSD and memory integration |

---

## 🧪 Integration Testing Framework (FR-009)

The integration testing framework provides tools for testing agentic AI behavior with real LLM providers and Neo4j databases. Core principle: **Test outcomes, not processes**.

### Key Concepts

| Concept | Purpose |
|---------|---------|
| **TestDatabase** | Neo4j lifecycle manager (initialize, clear, close) |
| **SeedItem** | Declarative test data with dependency resolution |
| **SeedSet** | Collection of seeds for a test scenario |
| **TestSequence** | Multi-step tests with shared database state |
| **AgentEvaluator** | LLM-as-judge for semantic evaluation |
| **AppProfile** | Configurable agent configurations |

### Critical Design Decision: Seeds Use Real MemoryProvider

Seeds receive the **REAL MemoryProvider**, not a wrapper. This ensures tests validate production APIs:

```python
from draagon_ai.testing import SeedItem, SeedFactory
from draagon_ai.memory import MemoryProvider

@SeedFactory.register("user_doug")
class DougUserSeed(SeedItem):
    """Seeds use REAL provider API - not wrapper methods."""

    async def create(self, provider: MemoryProvider) -> str:
        # Direct production API usage
        return await provider.store(
            content="User profile: Doug",
            metadata={"memory_type": "USER_PROFILE", "user_name": "Doug"}
        )
```

**Why?**
- Tests validate actual production interfaces
- No leaky abstractions from helper methods
- If seeds work, production will work

### Writing Integration Tests

```python
import pytest
from draagon_ai.testing import SeedSet, SeedFactory, SeedItem, AgentEvaluator
from draagon_ai.memory import MemoryProvider

# 1. Define seed items
@SeedFactory.register("user_preference")
class UserPreferenceSeed(SeedItem):
    async def create(self, provider: MemoryProvider) -> str:
        return await provider.store(
            content="User prefers dark mode",
            metadata={"memory_type": "PREFERENCE", "importance": 0.9}
        )

# 2. Create seed sets
USER_WITH_PREFS = SeedSet("user_with_prefs", ["user_preference"])

# 3. Write test using fixtures
@pytest.mark.memory_integration
async def test_preference_recall(agent, memory_provider, seed, evaluator):
    """Test agent recalls user preferences."""

    # Apply seeds using REAL provider
    await seed.apply(USER_WITH_PREFS, memory_provider)

    # Query agent
    response = await agent.process("What theme do I prefer?")

    # LLM-as-judge evaluation (NOT string matching)
    result = await evaluator.evaluate_correctness(
        query="What theme do I prefer?",
        expected_outcome="Agent should mention dark mode",
        actual_response=response.answer
    )

    assert result.correct, f"Failed: {result.reasoning}"
```

### Writing Test Sequences

For multi-step tests where database state persists between steps:

```python
from draagon_ai.testing import TestSequence, step

class TestLearningFlow(TestSequence):
    """Test agent learning across interactions."""

    @step(1)
    async def test_initial_unknown(self, agent):
        """Agent doesn't know birthday initially."""
        response = await agent.process("When is my birthday?")
        assert response.confidence < 0.5

    @step(2, depends_on="test_initial_unknown")
    async def test_learn_birthday(self, agent):
        """Agent learns birthday from user."""
        response = await agent.process("My birthday is March 15")
        assert "march 15" in response.answer.lower()

    @step(3, depends_on="test_learn_birthday")
    async def test_recall_birthday(self, agent):
        """Agent recalls learned birthday."""
        response = await agent.process("When is my birthday?")
        assert "march 15" in response.answer.lower()
```

### Available Fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `test_database` | session | Neo4j lifecycle manager |
| `clean_database` | function | Clears data before each test |
| `memory_provider` | function | REAL MemoryProvider instance |
| `seed` | function | Seed applicator |
| `evaluator` | function | LLM-as-judge evaluator |
| `agent_factory` | function | Create agents from profiles |
| `llm_provider` | session | Real LLM (Groq/OpenAI) |

### Creating New Seed Items

1. Define seed class with `@SeedFactory.register`:

```python
@SeedFactory.register("my_seed")
class MySeed(SeedItem):
    dependencies = ["other_seed"]  # Optional dependencies

    async def create(self, provider: MemoryProvider, other_seed: str = None) -> str:
        # Use REAL provider API
        return await provider.store(
            content="My test data",
            metadata={"memory_type": "FACT"}
        )
```

2. Create seed set:

```python
MY_SCENARIO = SeedSet("my_scenario", ["other_seed", "my_seed"])
```

3. Use in test:

```python
async def test_something(seed, memory_provider):
    await seed.apply(MY_SCENARIO, memory_provider)
    # Test continues with seeded data
```

### Extending with App Profiles

```python
from draagon_ai.testing import AppProfile, ToolSet

MY_PROFILE = AppProfile(
    name="custom",
    personality="You are a specialized assistant...",
    tool_set=ToolSet.BASIC,
    memory_config={"working_ttl": 300},
    llm_model_tier="fast",
)

async def test_with_profile(agent_factory):
    agent = await agent_factory.create(MY_PROFILE)
    response = await agent.process("Hello!")
```

### Test Markers

```bash
pytest -m "memory_integration"     # Memory tests only
pytest -m "smoke"                  # Critical tests
pytest -m "learning_integration"   # Learning tests
pytest -m "not slow"               # Skip slow tests
```

### Running Integration Tests

```bash
# Requires API key and Neo4j
GROQ_API_KEY=your_key pytest tests/integration/ -v

# With OpenAI instead
OPENAI_API_KEY=your_key pytest tests/integration/ -v

# Specific test file
pytest tests/integration/test_memory.py -v
```

---

## 📝 Development Notes

### Adding New Cognitive Service

1. Create service in `src/draagon_ai/cognition/`
2. Implement required protocol methods
3. Add to `cognition/__init__.py` exports
4. Write tests in `tests/cognition/`
5. Document in this file

### Adding New Tool Feature

1. Extend `Tool` dataclass in `orchestration/registry.py` if needed
2. Update `@tool` decorator in `tools/decorator.py`
3. Update exports in `tools/__init__.py`
4. Write tests in `tests/tools/`

### Version Compatibility

- Python 3.11+
- Neo4j 5.26+ (Community or Enterprise)
- Supports Groq, OpenAI, Ollama LLM providers

---

## 🔗 Related Projects

### Roxy Voice Assistant
Reference implementation for Home Assistant voice control.
Location: `../roxy-voice-assistant/`

---

**End of CLAUDE.md**

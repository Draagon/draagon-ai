# draagon-ai - Claude Context

**Last Updated:** 2025-12-30
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

### Async-First Processing

Non-blocking operations should run in background:
- Learning extraction (after response sent)
- Episode summaries (on conversation expiry)
- Memory consolidation (scheduled job)
- Reflection/quality assessment (after response)

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

- Python 3.10+
- Qdrant 1.7+
- Supports Groq, OpenAI, Ollama LLM providers

---

## 🔗 Related Projects

### Roxy Voice Assistant
Reference implementation for Home Assistant voice control.
Location: `../roxy-voice-assistant/`

---

**End of CLAUDE.md**

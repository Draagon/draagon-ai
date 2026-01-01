# draagon-ai Design Principles

## 1. LLM-First Semantic Processing

**The LLM is the brain. Let it think.**

| Task | Wrong Approach | Right Approach |
|------|---------------|----------------|
| Detect user corrections | `regex: r"actually\|no,\|wrong"` | LLM semantic analysis |
| Identify speaker | `regex: r"it's\s+(\w+)"` | LLM entity extraction |
| Classify intents | Keyword matching | LLM decision prompt |
| Detect learning opportunities | `regex: r"remember\|my .* is"` | LLM semantic detection |
| Parse dates/times | Regex patterns | LLM structured extraction |

**Exceptions (Non-Semantic Tasks):**
- Security blocklist patterns
- TTS text transformations
- URL/email validation
- Entity ID resolution (exact string matching)
- Parsing structured LLM output (XML element extraction)

## 2. Four-Layer Cognitive Memory

Inspired by cognitive psychology (Baddeley, Miller):

```yaml
working_memory:
  ttl: 5 minutes
  capacity: "7±2 items (Miller's Law)"
  features:
    - Attention weighting
    - Activation decay
    - Session isolation
  purpose: "Current conversation context"

episodic_memory:
  ttl: 2 weeks
  features:
    - Conversation summaries
    - Entity linking
    - Interaction patterns
  purpose: "What happened when"

semantic_memory:
  ttl: 6 months
  features:
    - Facts and skills
    - Preference consolidation
    - Confidence scoring
  purpose: "What is true"

metacognitive_memory:
  ttl: permanent
  features:
    - Self-knowledge
    - Learned patterns
    - Capability awareness
  purpose: "What I know about myself"
```

### Memory Promotion

Items automatically promote based on:
- **Importance threshold** (0.8 for working → episodic)
- **Access frequency** (3+ accesses)
- **Consolidation** (similar items merge)

## 3. Belief-Based Knowledge

**Observations are not facts.** Users tell agents things. Some are true, some aren't.

```yaml
observation:
  content: "We have 6 cats"
  source_user_id: "doug"
  scope: HOUSEHOLD
  confidence_expressed: 0.9

belief:
  content: "The household has 6 cats"
  belief_type: HOUSEHOLD_FACT
  confidence: 0.85
  supporting_observations: [obs_1, obs_2]
  conflicting_observations: []
  verified: false
  needs_clarification: false
```

### Belief Types

| Type | Description | Confidence Source |
|------|-------------|-------------------|
| HOUSEHOLD_FACT | Shared context (3+ users agree) | Multi-source agreement |
| VERIFIED_FACT | Externally confirmed | Verification check |
| UNVERIFIED_CLAIM | Pending verification | Single source |
| INFERRED | Agent figured out | Reasoning chain |
| USER_PREFERENCE | User stated preference | User statement |
| AGENT_PREFERENCE | Agent's own preference | Experience-based |

### Belief Reconciliation

When observations conflict:
1. **Accept Latest**: User correcting themselves
2. **Weight by Credibility**: Source track record
3. **Ask for Clarification**: Queue question
4. **Flag Conflict**: Mark for resolution

## 4. Genuine Opinion Formation

Agents should have authentic opinions, not just echo users.

```yaml
opinion:
  topic: "Best programming language for beginners"
  position: "Python is ideal due to readability and ecosystem"
  strength: MODERATE  # TENTATIVE | MODERATE | STRONG | CORE
  basis: REASONING    # EXPERIENCE | REASONING | VALUES | OBSERVATION | AESTHETIC
  confidence: 0.75
  openness_to_change: 0.6
  formed_at: "2025-12-30"
```

### Opinion Evolution

- Opinions form gradually through interaction
- Agent genuinely considers arguments before changing
- Change history is maintained
- Strong opinions require strong evidence to change

## 5. Curiosity-Driven Learning

Agents should actively seek knowledge, not just respond.

```yaml
knowledge_gap:
  topic: "User's work schedule"
  importance: HIGH
  question_type: PREFERENCE
  question: "What days do you typically work from home?"
  context: "Would help with smart home automation"
```

### Curiosity Constraints

- Maximum 3 questions per day
- 30-minute minimum gap between questions
- Questions asked during natural conversation pauses
- Never repeat recently asked questions

## 6. Transactive Memory ("Who Knows What")

For multi-agent systems, track expertise:

```yaml
expertise_map:
  agent_weather:
    weather: 0.95
    forecasting: 0.88
  agent_calendar:
    scheduling: 0.92
    time_zones: 0.85
  agent_home:
    lighting: 0.90
    climate: 0.78
```

### Expertise Routing

1. Extract topics from query
2. Score agents by topic expertise
3. Route to highest-scoring agents
4. Update expertise based on success/failure

## 7. Confidence-Based Actions

Never binary. Always graduated.

| Confidence | Action |
|------------|--------|
| > 0.9 | Proceed without confirmation |
| 0.7-0.9 | Proceed for normal ops, confirm for sensitive |
| 0.5-0.7 | Confirm for most operations |
| < 0.5 | Require explicit confirmation |

## 8. Pragmatic Async

Use async when it provides real benefit. Don't add async complexity unnecessarily.

```yaml
use_async_for:
  - External I/O (LLM calls, database, HTTP)
  - Concurrent operations (parallel agents, tool batching)
  - Background tasks (learning, consolidation, reflection)

keep_synchronous:
  - Pure computation (parsing, filtering, transforming)
  - Configuration and initialization
  - Simple utilities and helpers
  - Getters, builders, validators
  - Data structure operations
```

**Why This Matters:**
- Every `async def` forces callers to be async (viral complexity)
- Async adds debugging difficulty (stack traces, exception handling)
- Testing becomes more complex (pytest-asyncio, async fixtures)
- Only pay the complexity cost when you get real I/O or concurrency benefits

## 9. Protocol-Based Extensibility

All integrations via Python Protocols:

```python
class LLMProvider(Protocol):
    async def chat(self, messages: list[dict], ...) -> str: ...

class MemoryProvider(Protocol):
    async def store(self, content: str, ...) -> str: ...
    async def search(self, query: str, ...) -> list[Memory]: ...

class ToolProvider(Protocol):
    async def execute(self, tool_call: ToolCall, ...) -> ToolResult: ...
```

## 10. Research-Grounded Architecture

Every major feature backed by research:

| Feature | Research Basis |
|---------|---------------|
| 4-Layer Memory | Baddeley's Working Memory Model |
| 7±2 Capacity | Miller's Law (1956) |
| Belief Reconciliation | Epistemic Logic, BDI Agents |
| Transactive Memory | Wegner (1987) |
| Metacognitive Reflection | ICML 2025 Position Paper |
| Parallel Agent Coordination | Anthropic Multi-Agent Research |

---

**Document Status**: Active
**Last Updated**: 2025-12-30

# Competitive Architecture Analysis: Draagon-AI vs Mem0 vs Zep

*Deep technical comparison from an AI agentic architecture perspective*

---

> **UPDATE (December 28, 2025):** This document was written before fully understanding Draagon-AI's scope. It incorrectly frames Draagon-AI as primarily a memory system.
>
> **The correction:** Draagon-AI is a **53,117-line agentic framework** with self-evolution capabilities, not an 8,500-line memory system. Mem0 and Zep are potential **storage backends**, not competitors.
>
> **See instead:** `REVISED_MASTER_STRATEGY.md` and `FULL_SYSTEM_COMPARISON.md` for the corrected analysis.
>
> This document is retained for its technical research on Mem0 and Zep architectures.

---

## Executive Summary

~~After analyzing all three architectures, the strategic recommendation is:~~

~~**Don't align. Don't compete directly. Complement and differentiate.**~~

**REVISED:** Mem0 and Zep are memory systems (~5-10K lines). Draagon-AI is a complete agentic framework (53K lines) that includes self-evolution, multi-agent orchestration, and behavior generation. They are not comparable.

The technical research on Mem0/Zep below remains useful for understanding potential storage backends.

---

## Part 1: Architecture Comparison

### Mem0 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       Mem0                               │
├─────────────────────────────────────────────────────────┤
│  Memory Scopes: User | Agent | Session                  │
├─────────────────────────────────────────────────────────┤
│  Operations: Add → Search → Update → Delete             │
├─────────────────────────────────────────────────────────┤
│  Storage Layer:                                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │   Vector Store   │  │   Graph Store    │             │
│  │   (Embeddings)   │  │   (Relations)    │             │
│  └──────────────────┘  └──────────────────┘             │
├─────────────────────────────────────────────────────────┤
│  Features:                                               │
│  • LLM-powered fact extraction                          │
│  • Metadata filtering                                    │
│  • Reranking                                            │
│  • MCP integration                                       │
└─────────────────────────────────────────────────────────┘
```

**Core API:**
```python
memory = Memory()
memory.add(messages, user_id=...)      # Store with LLM extraction
memory.search(query, user_id=...)      # Semantic retrieval
memory.update(memory_id, data)         # Modify
memory.delete(memory_id)               # Remove
```

**Key Characteristics:**
- Simple CRUD interface
- LLM-powered fact extraction on add()
- Hybrid vector + graph storage
- No confidence tracking
- No belief reconciliation
- No curiosity/proactive learning

### Zep Architecture (Graphiti)

```
┌─────────────────────────────────────────────────────────┐
│                    Zep / Graphiti                        │
├─────────────────────────────────────────────────────────┤
│  Core: Temporal Knowledge Graph                          │
├─────────────────────────────────────────────────────────┤
│  Bi-Temporal Model:                                      │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │   Event Time     │  │  Ingestion Time  │             │
│  │ (When it happened│  │ (When we learned)│             │
│  └──────────────────┘  └──────────────────┘             │
├─────────────────────────────────────────────────────────┤
│  Data Structures:                                        │
│  • Episodes (ingestion units)                            │
│  • Nodes (entities)                                      │
│  • Edges (relationships with temporal validity)          │
├─────────────────────────────────────────────────────────┤
│  Retrieval:                                              │
│  • Semantic (embeddings)                                 │
│  • BM25 (keyword)                                        │
│  • Graph traversal                                       │
└─────────────────────────────────────────────────────────┘
```

**Core Concept:**
```
Facts = Triplets: (Entity) --[Relationship]--> (Entity)
                     │                            │
                     └── valid_at / invalid_at ───┘
```

**Key Characteristics:**
- Bi-temporal model (event time + ingestion time)
- Relationship-aware retrieval
- Temporal edge invalidation (handles contradictions)
- Context assembly for LLM prompts
- No confidence tracking
- No belief formation logic
- No curiosity engine

### Draagon-AI Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Draagon-AI                           │
├─────────────────────────────────────────────────────────┤
│  Cognitive Layer:                                        │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────┐ │
│  │    Belief      │ │   Curiosity    │ │   Learning   │ │
│  │ Reconciliation │ │    Engine      │ │   Service    │ │
│  └───────┬────────┘ └───────┬────────┘ └──────┬───────┘ │
│          │                  │                  │         │
│          └──────────────────┼──────────────────┘         │
│                             │                            │
├─────────────────────────────┼────────────────────────────┤
│  Memory Layers:             │                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Working  │→│ Episodic │→│ Semantic │→│Metacognitive│ │
│  │(seconds) │ │ (hours)  │ │ (days)   │ │ (permanent) │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
│                             │                            │
├─────────────────────────────┼────────────────────────────┤
│  Scope Hierarchy:           │                            │
│  WORLD → CONTEXT → AGENT → USER → SESSION               │
│                             │                            │
├─────────────────────────────┼────────────────────────────┤
│  Storage: Temporal Cognitive Graph                       │
│  ┌──────────────────────────┴───────────────────────────┐│
│  │  Bi-temporal nodes + edges with scope permissions    ││
│  │  (Qdrant backend for vector similarity)              ││
│  └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

**Unique Components:**

1. **Belief Reconciliation** (not in competitors):
```python
# Multi-source observation → Belief with confidence
observations = [
    UserObservation(user="alice", content="We have 6 cats"),
    UserObservation(user="bob", content="We have 5 cats"),
]
belief = await belief_system.reconcile(observations)
# Returns: AgentBelief(
#   content="The family has 5-6 cats",
#   confidence=0.6,
#   has_conflict=True,
#   needs_clarification=True
# )
```

2. **Curiosity Engine** (not in competitors):
```python
# Proactive knowledge gap detection
gaps = await curiosity.detect_gaps(conversation)
questions = await curiosity.generate_questions(gaps)
# Returns prioritized questions the agent should ask
```

3. **Layered Memory with Promotion**:
```python
# Automatic promotion based on importance + access
working → episodic → semantic → metacognitive
# Each layer has different TTL, capacity, decay
```

4. **Hierarchical Scopes**:
```python
# Fine-grained access control
WORLD (universal facts)
  └── CONTEXT (household/team)
       └── AGENT (this AI instance)
            └── USER (per-user within agent)
                 └── SESSION (current conversation)
```

---

## Part 2: Feature Comparison Matrix

| Feature | Mem0 | Zep | Draagon-AI |
|---------|------|-----|------------|
| **Storage** |
| Vector Store | Yes | Yes | Yes |
| Graph Store | Yes | Yes (Graphiti) | Yes (TCG) |
| Bi-Temporal | No | **Yes** | **Yes** |
| **Memory Model** |
| User/Session/Agent Scopes | Yes | Yes | Yes |
| Hierarchical Scopes | No | Partial (group_ids) | **Yes (5 levels)** |
| Memory Layers (Working→Semantic) | No | No | **Yes** |
| Layer Promotion | No | No | **Yes** |
| **Cognition** |
| Confidence Tracking | No | No | **Yes (0-1 scale)** |
| Belief Reconciliation | No | No | **Yes** |
| Conflict Detection | No | Temporal invalidation | **Yes (multi-source)** |
| Curiosity/Gap Detection | No | No | **Yes** |
| Proactive Questions | No | No | **Yes** |
| Source Credibility | No | No | **Yes** |
| **Retrieval** |
| Semantic Search | Yes | Yes | Yes |
| Graph Traversal | Yes | Yes | Yes |
| BM25/Keyword | Via filters | Yes | Partial |
| Reranking | Yes | Yes | Via LLM |
| **Integration** |
| MCP Support | Yes | Partial | Yes |
| Python SDK | Yes | Yes | Yes |
| TypeScript SDK | Yes | Yes | No (future) |
| Framework Integration | LangChain, CrewAI | LangChain, LlamaIndex | Roxy (custom) |

---

## Part 3: What They Have That You Don't

### Mem0 Advantages

1. **41K GitHub Stars** - Community, trust, visibility
2. **13M Python Downloads** - Proven adoption
3. **AWS Partnership** - Exclusive Agent SDK provider
4. **Simple API** - Lower learning curve
5. **MCP Already Deployed** - Claude Code integration exists
6. **Enterprise Features** - Webhooks, multimodal, custom categories

### Zep Advantages

1. **Graphiti Framework** - Best-in-class temporal graphs
2. **Hybrid Retrieval** - Semantic + BM25 + graph traversal
3. **$1M ARR** - Revenue validation
4. **LongMemEval Benchmark** - Published performance claims
5. **Framework Integrations** - AutoGen, LangChain, LlamaIndex native

### What You Should NOT Try to Replicate

| Don't Compete On | Why |
|------------------|-----|
| GitHub stars | Takes years, they have head start |
| Framework integrations | Their focus, not your differentiator |
| Pure storage/retrieval speed | Commodity, not defensible |
| Enterprise sales team | Resource-intensive, you're solo |

---

## Part 4: What You Have That They Don't

### Unique to Draagon-AI

| Capability | Business Value | Why Competitors Can't Easily Copy |
|------------|----------------|-----------------------------------|
| **Belief Reconciliation** | Handle multi-source conflicts intelligently | Requires cognitive architecture, not just storage |
| **Confidence Tracking** | Know when to trust vs. verify | Integrated into every operation |
| **Curiosity Engine** | Proactive learning, better UX | Novel concept, research-backed |
| **Source Credibility** | Weight information by trust | Requires user model integration |
| **5-Level Scope Hierarchy** | True multi-tenant isolation | More granular than competitors |
| **Memory Layer Promotion** | Automatic importance-based organization | Cognitive science foundation |
| **Healthcare Domain** | HIPAA expertise, vertical focus | Domain knowledge can't be copied |

### The "Cognitive Layer" Differentiator

Mem0 and Zep are **storage + retrieval** systems. They answer: "What do we know?"

Draagon-AI adds **cognition**. It answers:
- "How confident are we?"
- "Who told us this, and do we trust them?"
- "Does this conflict with what we already believe?"
- "What should we ask to learn more?"
- "When should we question our own beliefs?"

This is a fundamentally different layer of the stack.

---

## Part 5: Strategic Options

### Option A: Pure Competition (NOT RECOMMENDED)

Replace Mem0/Zep entirely with Draagon-AI.

**Pros:**
- Simpler story
- Full control

**Cons:**
- Fighting on their turf
- Need to replicate all their features
- Years behind on adoption/community
- Resource-intensive

**Verdict**: Don't do this.

### Option B: Complementary Layer (RECOMMENDED)

Position Draagon-AI as a **cognitive layer** that sits on top of storage providers.

```
┌─────────────────────────────────────────────────────────┐
│                     Your Application                     │
├─────────────────────────────────────────────────────────┤
│                     Draagon-AI                           │
│         (Beliefs, Curiosity, Confidence, Learning)       │
├─────────────────────────────────────────────────────────┤
│         Memory Provider (Pluggable Backend)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Mem0     │  │   Zep    │  │ Qdrant   │  │ Custom │  │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
from draagon_ai import CognitiveEngine
from draagon_ai.providers import Mem0Provider, ZepProvider, QdrantProvider

# Use Mem0 as storage backend
engine = CognitiveEngine(
    storage=Mem0Provider(api_key="..."),
    # Draagon-AI adds cognitive layer on top
)

# Or use Zep
engine = CognitiveEngine(
    storage=ZepProvider(api_key="..."),
)

# Or use raw Qdrant (your current default)
engine = CognitiveEngine(
    storage=QdrantProvider(url="..."),
)
```

**Pros:**
- Ride their adoption wave
- "Works with Mem0" is a feature, not competition
- Focus on your unique value
- Smaller scope to maintain
- Can switch/support multiple backends

**Cons:**
- Dependency on external projects
- Need to maintain adapters
- Some overlap to manage

**Verdict**: This is the strategy.

### Option C: Vertical Focus (ALSO RECOMMENDED, COMBINE WITH B)

Own a specific vertical where cognitive capabilities matter most.

**Best Verticals for Belief Reconciliation:**
1. **Healthcare** - Multiple sources, conflicting info, high stakes
2. **Legal** - Precedent reconciliation, confidence matters
3. **Enterprise Knowledge** - Multiple teams, tribal knowledge conflicts
4. **Multi-Agent Systems** - Agents need to reconcile different observations

**Implementation:**
```python
from draagon_ai.verticals import HealthcareEngine

# Pre-configured for healthcare
engine = HealthcareEngine(
    hipaa_mode=True,
    phi_detection=True,
    audit_logging=True,
)
```

---

## Part 6: Integration Architecture

### Recommended Architecture

```python
# draagon_ai/providers/base.py
class StorageProvider(Protocol):
    """Abstract storage backend - can be Mem0, Zep, or direct Qdrant."""

    async def store(self, content: str, metadata: dict) -> str: ...
    async def search(self, query: str, limit: int) -> list[Memory]: ...
    async def get(self, memory_id: str) -> Memory | None: ...
    async def update(self, memory_id: str, content: str) -> bool: ...
    async def delete(self, memory_id: str) -> bool: ...


# draagon_ai/providers/mem0.py
class Mem0Provider(StorageProvider):
    """Mem0 as storage backend."""

    def __init__(self, api_key: str):
        from mem0 import Memory
        self._memory = Memory(api_key=api_key)

    async def store(self, content: str, metadata: dict) -> str:
        result = self._memory.add(content, **metadata)
        return result["id"]

    async def search(self, query: str, limit: int) -> list[Memory]:
        results = self._memory.search(query, limit=limit)
        return [self._convert(r) for r in results]


# draagon_ai/providers/zep.py
class ZepProvider(StorageProvider):
    """Zep as storage backend."""

    def __init__(self, api_key: str):
        from zep_cloud import Zep
        self._client = Zep(api_key=api_key)

    async def store(self, content: str, metadata: dict) -> str:
        # Use Zep's memory API
        ...


# draagon_ai/cognitive/engine.py
class CognitiveEngine:
    """The cognitive layer that adds beliefs, curiosity, confidence."""

    def __init__(
        self,
        storage: StorageProvider,
        belief_system: BeliefReconciliationSystem | None = None,
        curiosity_engine: CuriosityEngine | None = None,
    ):
        self._storage = storage
        self._beliefs = belief_system or BeliefReconciliationSystem(storage)
        self._curiosity = curiosity_engine or CuriosityEngine(storage)

    async def observe(
        self,
        content: str,
        user_id: str,
        source_credibility: float = 0.8,
    ) -> AgentBelief:
        """Process an observation into a belief."""
        # 1. Store raw observation
        memory_id = await self._storage.store(content, {"user_id": user_id})

        # 2. Find related existing beliefs
        related = await self._storage.search(content, limit=10)

        # 3. Reconcile into belief
        belief = await self._beliefs.reconcile(content, related, source_credibility)

        # 4. Check for curiosity triggers
        gaps = await self._curiosity.detect_gaps(belief)

        return belief
```

### MCP Integration Strategy

Instead of replacing Mem0's MCP server, **extend it**:

```python
# MCP tools that ADD cognitive capabilities
@mcp_tool("draagon.reconcile_beliefs")
async def reconcile_beliefs(topic: str) -> BeliefSummary:
    """Get the agent's current belief about a topic, with confidence."""
    ...

@mcp_tool("draagon.check_confidence")
async def check_confidence(statement: str) -> ConfidenceAssessment:
    """How confident should the agent be about this statement?"""
    ...

@mcp_tool("draagon.get_questions")
async def get_questions(context: str) -> list[CuriousQuestion]:
    """What should the agent ask to fill knowledge gaps?"""
    ...

@mcp_tool("draagon.learn_from_correction")
async def learn_from_correction(original: str, correction: str, source: str):
    """Update beliefs based on a correction."""
    ...
```

These tools are **complementary** to Mem0's tools, not replacements.

---

## Part 7: What to Change in Your Design

### Keep (Your Differentiators)

| Component | Reason |
|-----------|--------|
| BeliefReconciliationSystem | Unique, no competitor has this |
| CuriosityEngine | Unique, proactive learning |
| Confidence tracking | Integrated throughout |
| Source credibility | Multi-source handling |
| 5-level scope hierarchy | More granular than competitors |
| Memory layer promotion | Cognitive science based |

### Adapt (Align with Market)

| Change | Current | Recommended | Why |
|--------|---------|-------------|-----|
| Storage abstraction | Qdrant-specific | Provider interface | Support Mem0/Zep backends |
| API naming | Custom | Closer to Mem0 | Lower learning curve |
| MCP tools | Full replacement | Complementary | Easier adoption |

### Add (Fill Gaps)

| Feature | Priority | Why |
|---------|----------|-----|
| Mem0Provider adapter | High | Ride their adoption |
| ZepProvider adapter | Medium | Alternative backend |
| BM25 search | Low | Zep has this, nice-to-have |
| TypeScript SDK | Low | Eventually for adoption |

### Remove/Deprioritize

| Feature | Why Deprioritize |
|---------|------------------|
| Raw Qdrant management | Let Mem0/Zep handle this |
| Basic CRUD operations | Focus on cognitive layer |
| Generic integrations | Focus on Claude/MCP first |

---

## Part 8: Positioning Statement

### For Developers

> **Draagon-AI adds cognition to your AI's memory.**
>
> Mem0 and Zep store and retrieve memories. Draagon-AI teaches your AI to *think* about them—reconciling conflicts, tracking confidence, and asking smart questions.
>
> Use it standalone with Qdrant, or add it on top of Mem0/Zep for cognitive superpowers.

### For Enterprises (Healthcare Focus)

> **The only AI memory system built for multi-source truth reconciliation.**
>
> In healthcare, you can't just store what users say—you need to know how confident you should be, when sources conflict, and what questions to ask to clarify.
>
> Draagon-AI brings belief systems to AI agents.

### For the Open Source Community

> **Cognitive memory for AI agents.**
>
> While others focus on storage, we focus on cognition:
> - Belief reconciliation when sources conflict
> - Confidence tracking for every memory
> - Curiosity engine for proactive learning
> - Works with Mem0, Zep, or standalone

---

## Part 9: Implementation Roadmap

### Phase 1: Storage Abstraction (2 weeks)

```
Tasks:
├── Define StorageProvider protocol
├── Create QdrantProvider (current implementation)
├── Create Mem0Provider adapter
├── Create ZepProvider adapter
└── Tests for all providers
```

### Phase 2: Cognitive Layer Isolation (2 weeks)

```
Tasks:
├── Extract CognitiveEngine as standalone
├── Make it storage-agnostic
├── Ensure all cognitive features work with any backend
└── Document the separation
```

### Phase 3: Complementary MCP Tools (2 weeks)

```
Tasks:
├── draagon.reconcile_beliefs
├── draagon.check_confidence
├── draagon.get_questions
├── draagon.learn_from_correction
└── Documentation showing alongside Mem0
```

### Phase 4: Open Source Launch (2 weeks)

```
Tasks:
├── Clean codebase for release
├── Write "Works with Mem0" documentation
├── Create comparison blog post
├── Launch on HN with positioning
└── Engage with Mem0/Zep communities
```

---

## Part 10: Final Recommendation

### The Strategy

1. **Don't compete on storage/retrieval** - Mem0 and Zep own this
2. **Own the cognitive layer** - Beliefs, confidence, curiosity
3. **Make competitors your distribution** - "Works with Mem0"
4. **Vertical focus** - Healthcare first
5. **Unique positioning** - "Cognition for AI memory"

### The Tagline

> **Mem0 remembers. Zep connects. Draagon-AI thinks.**

### The Moat

Your moat isn't code—it's the cognitive architecture that takes months to design and years of research to understand. Mem0 can copy your API; they can't copy your belief reconciliation logic without fundamentally rearchitecting their system.

### The Action

1. Keep building cognitive features (your unique value)
2. Add storage provider abstraction (ride their adoption)
3. Position as complementary, not competitive
4. Launch with "Works with Mem0" story
5. Own healthcare vertical

---

## Sources

- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [Mem0 Documentation](https://docs.mem0.ai)
- [Zep GitHub](https://github.com/getzep/zep)
- [Graphiti Framework](https://github.com/getzep/graphiti)
- [Zep Documentation](https://help.getzep.com)
- [Mem0 TechCrunch Announcement](https://techcrunch.com/2025/10/28/mem0-raises-24m-from-yc-peak-xv-and-basis-set-to-build-the-memory-layer-for-ai-apps/)

---

*Analysis completed: December 28, 2025*

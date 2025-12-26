# AGI-Lite Temporal Cognitive Architecture

**Version:** 1.0.0
**Last Updated:** 2025-12-26
**Status:** Design Phase - Option C (Full AGI-Lite with Self-Evolution)

---

## Executive Summary

This document defines the architecture for draagon-ai's AGI-Lite system: a temporal cognitive graph that enables agents to build persistent, evolving understanding across sessions while maintaining hierarchical memory scopes, self-referential prompt evolution, and multi-agent coordination.

**Key Innovation:** Unlike existing memory systems that treat memories as static records, our architecture treats memory as a *living temporal graph* where:
- Nodes represent evolving beliefs (not just facts)
- Edges capture semantic and temporal relationships
- The graph itself can reflect on and improve its own structure
- Behaviors are first-class graph citizens that learn and adapt

---

## Table of Contents

1. [Research Background](#1-research-background)
2. [Architecture Overview](#2-architecture-overview)
3. [Temporal Cognitive Graph](#3-temporal-cognitive-graph)
4. [Memory Layer Architecture](#4-memory-layer-architecture)
5. [Behaviors as Graph Citizens](#5-behaviors-as-graph-citizens)
6. [Self-Evolution Engine](#6-self-evolution-engine)
7. [Multi-Agent Orchestration](#7-multi-agent-orchestration)
8. [Implementation Phases](#8-implementation-phases)
9. [Data Models](#9-data-models)

---

## 1. Research Background

### 1.1 State of the Art (2025)

Our architecture draws from cutting-edge research in agentic AI memory systems:

#### Temporal Knowledge Graphs

**Zep/Graphiti Architecture** ([arxiv.org/abs/2501.13956](https://arxiv.org/abs/2501.13956)):
- Three-tier hierarchy: Episode → Semantic Entity → Community
- Bi-temporal tracking: Event time (T) when it happened + Ingestion time (T') when learned
- Incremental updates without full recomputation
- Hybrid retrieval: semantic + keyword (BM25) + graph traversal
- **P95 latency: 300ms** for retrieval

**Key Insight:** "Graphiti's bi-temporal model tracks when an event occurred and when it was ingested. Every graph edge includes explicit validity intervals."

#### Hybrid Memory Systems

**Mem0 Architecture** ([arxiv.org/abs/2504.19413](https://arxiv.org/abs/2504.19413)):
- Vector store + Key-Value store + Graph store (hybrid)
- **26% accuracy boost** over RAG-only approaches
- **91% lower p95 latency** vs full-context approaches
- **90% token savings** through selective memory extraction
- Production-scale: AWS selected as exclusive memory provider

**Key Insight:** "The consensus in 2025 is that many modern systems achieve the highest accuracy with hybrid vector-graph architectures."

#### Context Engineering (ACE Framework)

**Agentic Context Engineering** (Stanford, 2025):
- Contexts as evolving "playbooks" that grow and refine
- Write → Select → Compress → Isolate patterns
- Self-referential improvement: contexts improve themselves
- Grow-and-refine mechanism for continuous evolution

**Key Insight:** "Self-optimizing prompts that adapt based on user feedback are under development, promising further efficiency gains."

#### Graph Database Performance

**Netflix Graph Architecture** (2025):
- 8 billion nodes, 150 billion edges
- 2M reads/sec, 6M writes/sec sustained
- Emulated graph relationships in distributed key-value stores
- **Key Learning:** "Simpler to emulate graph-like relationships in existing data storage systems rather than adopting specialized graph infrastructure."

**Qdrant Benchmarks** (2024-2025):
- 4x improvement in certain scenarios over 2023
- Custom filterable HNSW for combined vector + filter queries
- Horizontal + vertical scaling support
- **Key Concern:** At moderate scale, pgvector showed 11.4x better performance in some benchmarks

### 1.2 Memory is the AGI Bottleneck

From [generalintelligencecompany.com](https://www.generalintelligencecompany.com/writing/memory-is-the-last-problem-to-solve-to-reach-agi):

> "If you wish to make an email assistant, you must first invent AGI. Why isn't there a good email assistant yet? It's memory all the way down, and memory isn't solved yet."

> "Agents can perform tasks for hours, operating across tools and self-managing their working context. But these agents don't 'feel' like AGI yet because they don't feel like people do—they're relatively stateless without memory."

**Current AI limitations:**
- Lack of persistent memory over long contexts
- No stable memory over months/years
- Catastrophic forgetting when learning new skills
- Poor retrieval compared to human recall

### 1.3 Why Hybrid Vector-Graph?

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Vector-only** | Fast similarity search, simple | No relationships, no reasoning chains |
| **Graph-only** | Rich relationships, multi-hop | Slow at scale, poor semantic matching |
| **Hybrid** | Best of both worlds | Complexity, more infrastructure |

**Research Conclusion:** GraphRAG frameworks combining vector similarity with graph traversal cut hallucination rates from 38% to 7% and achieve 70% accuracy gains on multi-hop queries.

---

## 2. Architecture Overview

### 2.1 Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DRAAGON-AI AGI-LITE SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONTEXT EVOLUTION ENGINE (ACE)                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   GENERATE  │→ │   REFLECT   │→ │   CURATE    │→ │   EVOLVE   │  │   │
│  │  │  Candidates │  │ Evaluate    │  │ Grow/Refine │  │ Meta-Prompt│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MULTI-AGENT ORCHESTRATOR                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │   │
│  │  │Sequential│  │ Parallel │  │ Handoff  │  │ Cross-Agent Learning │ │   │
│  │  │   Mode   │  │   Mode   │  │   Mode   │  │       Channel        │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TEMPORAL COGNITIVE GRAPH                          │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ LAYER 4: METACOGNITIVE                                       │   │   │
│  │   │ (Strategies, Skills, Self-Models) - Weeks to Permanent       │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ LAYER 3: SEMANTIC                                            │   │   │
│  │   │ (Entities, Relationships, Facts) - Days to Months            │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ LAYER 2: EPISODIC                                            │   │   │
│  │   │ (Episodes, Summaries, Temporal) - Hours to Days              │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ LAYER 1: WORKING                                             │   │   │
│  │   │ (Active Context, Session) - Seconds to Minutes               │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐    │   │
│  │   │  BEHAVIORS  │  │   BELIEFS   │  │ HIERARCHICAL SCOPES     │    │   │
│  │   │ (Graph Nodes)│  │ (Reconciled)│  │ World→Context→Agent→User│    │   │
│  │   └─────────────┘  └─────────────┘  └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HYBRID PERSISTENCE LAYER                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │   │
│  │  │    Qdrant    │  │   SQLite/    │  │      File System         │   │   │
│  │  │   (Vectors)  │  │   Postgres   │  │   (Behavior YAML/JSON)   │   │   │
│  │  │  + Metadata  │  │ (Relational) │  │                          │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Principles

1. **Hybrid Over Pure**: Use vector DB (Qdrant) + relational (SQLite) + graph emulation, not a pure graph DB
2. **Temporal First**: Every node has bi-temporal tracking (event time + ingestion time)
3. **Hierarchical Scopes**: World → Context → Agent → User → Session
4. **Self-Referential Evolution**: Meta-prompts that improve themselves
5. **Behaviors as Citizens**: Behaviors are graph nodes that learn and adapt
6. **Learning Channel**: Cross-agent learning via pub/sub pattern

---

## 3. Temporal Cognitive Graph

### 3.1 Core Concepts

The Temporal Cognitive Graph (TCG) is a human-inspired memory architecture:

| Layer | Human Analog | Time Scale | Purpose |
|-------|--------------|------------|---------|
| **Working** | Short-term memory | Seconds-Minutes | Active session context |
| **Episodic** | Autobiographical memory | Hours-Days | Raw experiences, events |
| **Semantic** | Factual knowledge | Days-Months | Entities, relationships, facts |
| **Metacognitive** | Procedural/strategic | Weeks-Permanent | Skills, strategies, self-knowledge |

### 3.2 Bi-Temporal Tracking

Every node tracks TWO time dimensions:

```python
@dataclass
class TemporalNode:
    # WHEN did this happen?
    event_time: datetime           # "Doug's birthday was March 15, 1985"

    # WHEN did we learn this?
    ingestion_time: datetime       # "We learned this on Dec 26, 2025"

    # Validity interval
    valid_from: datetime           # When this became true
    valid_until: datetime | None   # When this stopped being true (None = current)
```

**Why Bi-Temporal?**
- Distinguish "what was true" vs "what we knew"
- Enable point-in-time queries ("What did we know as of last week?")
- Track knowledge evolution without losing history
- Support conflict resolution based on temporal context

### 3.3 Node Types

```python
class NodeType(str, Enum):
    # Episodic Layer
    EPISODE = "episode"           # Conversation summary
    EVENT = "event"               # Specific occurrence

    # Semantic Layer
    ENTITY = "entity"             # Person, place, thing
    RELATIONSHIP = "relationship" # Connection between entities
    FACT = "fact"                 # Declarative knowledge
    BELIEF = "belief"             # Reconciled understanding

    # Metacognitive Layer
    SKILL = "skill"               # Procedural knowledge
    STRATEGY = "strategy"         # Problem-solving approach
    INSIGHT = "insight"           # Meta-learning
    BEHAVIOR = "behavior"         # Agent behavior (first-class)

    # Working Layer
    CONTEXT = "context"           # Active session context
    GOAL = "goal"                 # Current objectives
```

### 3.4 Edge Types

```python
class EdgeType(str, Enum):
    # Semantic relationships
    IS_A = "is_a"                 # Taxonomy
    HAS = "has"                   # Composition
    RELATED_TO = "related_to"    # General association

    # Temporal relationships
    BEFORE = "before"            # Temporal ordering
    DURING = "during"            # Temporal overlap
    SUPERSEDES = "supersedes"    # Newer version
    DERIVED_FROM = "derived_from" # Inference lineage

    # Causal relationships
    CAUSES = "causes"            # Causal link
    ENABLES = "enables"          # Prerequisite

    # Behavioral relationships
    TRIGGERS = "triggers"        # Behavior activation
    DELEGATES_TO = "delegates_to" # Behavior handoff
    LEARNS_FROM = "learns_from"  # Learning source
```

---

## 4. Memory Layer Architecture

### 4.1 Working Memory (Layer 1)

**Purpose:** Active context for current session

```python
@dataclass
class WorkingMemory:
    session_id: str
    active_context: dict[str, Any]
    current_goals: list[str]
    attention_weights: dict[str, float]  # What's most relevant now

    # Capacity limits (like human working memory)
    max_items: int = 7  # Miller's Law

    # Auto-decay
    ttl_seconds: int = 300  # 5 minutes without access
```

**Key Features:**
- Limited capacity (prevents context explosion)
- Attention weighting (focus on what matters)
- Auto-decay (items expire if not accessed)
- Promotion to Episodic on significance

### 4.2 Episodic Memory (Layer 2)

**Purpose:** Raw experiences with temporal context

```python
@dataclass
class EpisodicMemory(TemporalNode):
    episode_type: str  # "conversation", "task", "observation"
    summary: str
    participants: list[str]
    entities_mentioned: list[str]
    emotional_valence: float  # -1 to 1
    importance: float

    # Temporal anchoring
    duration_seconds: int
    preceding_episode_id: str | None
    following_episode_id: str | None
```

**Key Features:**
- Chronological ordering
- Rich temporal metadata
- Entity extraction for semantic linking
- Emotional tagging for significance

### 4.3 Semantic Memory (Layer 3)

**Purpose:** Structured knowledge (entities, relationships, facts)

```python
@dataclass
class SemanticNode(TemporalNode):
    # Core identity
    canonical_name: str
    aliases: list[str]
    node_type: str  # "person", "place", "concept", etc.

    # Attributes
    properties: dict[str, Any]

    # Confidence and source tracking
    confidence: float
    stated_count: int  # Times this was stated
    source_episodes: list[str]  # Episodic evidence

    # Community detection
    community_id: str | None
    community_summary: str | None
```

**Key Features:**
- Entity resolution (merge duplicates)
- Confidence propagation (stated_count boost)
- Community detection (related entities cluster)
- Source tracking (episodic evidence)

### 4.4 Metacognitive Memory (Layer 4)

**Purpose:** Skills, strategies, self-knowledge

```python
@dataclass
class MetacognitiveNode(TemporalNode):
    # Skill/Strategy definition
    skill_type: str  # "skill", "strategy", "insight", "behavior"
    description: str

    # Effectiveness tracking
    success_count: int
    failure_count: int
    effectiveness_score: float

    # Evolution
    version: int
    parent_id: str | None  # Previous version
    mutation_history: list[str]

    # Self-referential (for behaviors)
    can_self_improve: bool
    improvement_triggers: list[str]
```

**Key Features:**
- Effectiveness tracking
- Version history
- Self-improvement capability
- Cross-agent sharing potential

---

## 5. Behaviors as Graph Citizens

### 5.1 The BehaviorNode

Behaviors are first-class graph nodes that participate in the temporal cognitive graph:

```python
@dataclass
class BehaviorNode(MetacognitiveNode):
    """A behavior as a graph citizen."""

    # Core behavior reference
    behavior_id: str
    behavior: Behavior  # The actual behavior object

    # Graph relationships
    depends_on: list[str]       # Behaviors this requires
    can_delegate_to: list[str]  # Behaviors it can hand off to
    shares_context_with: list[str]  # Behaviors sharing state

    # Knowledge scopes
    learns_from_scope: str      # Where it reads knowledge
    contributes_to_scope: str   # Where it writes knowledge

    # Evolution tracking
    prompt_versions: list["PromptVersion"]
    current_fitness: float
    evolution_generation: int

    # Self-improvement
    improvement_triggers: list[str]
    last_improvement: datetime | None
    improvement_cooldown: timedelta
```

### 5.2 Behavior Learning Flow

```
User Interaction
       │
       ▼
┌──────────────────┐
│ Behavior Executes │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│ Success/Failure  │────▶│ Update Fitness   │
│   Detection      │     │   in Graph       │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│ Extract Learning │     │ Check Evolution  │
│   (Episode)      │     │   Threshold      │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│ Store Episodic   │     │ Trigger Self-    │
│   Memory         │     │   Improvement?   │
└──────────────────┘     └──────────────────┘
```

### 5.3 Cross-Behavior Learning

Behaviors can learn from each other through the graph:

```python
class BehaviorLearningChannel:
    """Pub/sub for cross-behavior learning."""

    async def broadcast_learning(
        self,
        source_behavior: str,
        learning: "Learning",
        scope: str,
    ) -> None:
        """Broadcast a learning to interested behaviors."""
        # Find behaviors that learn from this scope
        subscribers = await self.find_subscribers(scope)

        for subscriber in subscribers:
            await self.deliver_learning(subscriber, learning)

    async def find_subscribers(self, scope: str) -> list[str]:
        """Find behaviors that learn from this scope."""
        # Query graph for BehaviorNodes where learns_from_scope matches
        ...
```

---

## 6. Self-Evolution Engine

### 6.1 Context Evolution Engine (ACE-Inspired)

The heart of self-improvement is the Context Evolution Engine:

```python
class ContextEvolutionEngine:
    """ACE-inspired self-evolution engine.

    Key Innovation: The mutation prompts themselves evolve!
    """

    async def evolve_context(
        self,
        scope: str,
        feedback: list["InteractionFeedback"],
    ) -> "EvolutionResult":
        # Phase 1: Generate candidates from successful patterns
        candidates = await self._generate_context_candidates(scope, feedback)

        # Phase 2: Reflect on effectiveness
        evaluations = await self._reflect_on_candidates(candidates, feedback)

        # Phase 3: Curate (grow + refine)
        result = await self._curate_context(
            scope,
            candidates,
            evaluations,
            similarity_threshold=0.85,
            min_effectiveness=0.3,
        )

        return result
```

### 6.2 Self-Referential Meta-Prompts

**This is the key foundational change in Option C:**

```python
@dataclass
class MetaPrompt:
    """A prompt that generates/improves other prompts.

    Self-referential: MetaPrompts can improve other MetaPrompts!
    """
    prompt_id: str
    content: str
    target_type: str  # "decision_prompt", "synthesis_prompt", "meta_prompt"

    # Fitness tracking
    fitness: float
    usage_count: int
    success_count: int

    # Lineage
    parent_id: str | None
    generation: int

    # Self-reference
    can_mutate_self: bool  # Can this meta-prompt improve itself?
    mutation_history: list[str]

class SelfReferentialEvolution:
    """Evolution where the mutation prompts themselves evolve."""

    async def evolve(
        self,
        target_prompt: str,
        feedback: list["Feedback"],
    ) -> tuple[str, "MetaPrompt"]:
        # Get current best meta-prompt
        meta_prompt = await self._get_best_meta_prompt()

        # Generate improved target using meta-prompt
        improved_target = await self._apply_meta_prompt(
            meta_prompt, target_prompt, feedback
        )

        # Evolve the meta-prompt itself!
        improved_meta = await self._evolve_meta_prompt(
            meta_prompt,
            target_prompt,
            improved_target,
            feedback,
        )

        return improved_target, improved_meta
```

### 6.3 Promptbreeder-Style Evolution

Full genetic algorithm optimization:

```python
@dataclass
class EvolutionConfig:
    population_size: int = 8
    generations: int = 10
    tournament_size: int = 3
    elitism_count: int = 2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    train_test_split: float = 0.8
    overfitting_threshold: float = 0.1
    max_generations_without_improvement: int = 3

    # Self-referential settings
    evolve_meta_prompts: bool = True
    meta_evolution_frequency: int = 2  # Every N generations

class PromptbreederEvolution:
    """Full Promptbreeder-style evolution with safety guards."""

    async def evolve_behavior(
        self,
        behavior: Behavior,
        test_cases: list[TestCase],
        config: EvolutionConfig,
    ) -> "EvolutionResult":
        # Split data to prevent overfitting
        train, holdout = self._split_data(test_cases, config.train_test_split)

        # Initialize population
        population = await self._initialize_population(behavior, config)

        for generation in range(config.generations):
            # Evaluate fitness with diversity penalty
            await self._evaluate_with_fitness_sharing(population, train)

            # Tournament selection
            parents = self._tournament_select(population, config)

            # Create offspring via mutation and crossover
            offspring = await self._create_offspring(parents, config)

            # Elitism + offspring
            population = self._select_survivors(population, offspring, config)

            # Self-referential: evolve mutation prompts
            if config.evolve_meta_prompts and generation % config.meta_evolution_frequency == 0:
                await self._evolve_meta_prompts()

        # Validate on holdout (prevent overfitting)
        best = max(population, key=lambda b: b.metrics.fitness_score)
        holdout_fitness = await self._evaluate(best, holdout)

        # Safety: reject if overfitting detected
        train_fitness = best.metrics.fitness_score
        gap = train_fitness - holdout_fitness

        if gap > config.overfitting_threshold:
            return EvolutionResult(approved=False, reason="Overfitting detected")

        return EvolutionResult(approved=True, evolved_behavior=best)
```

---

## 7. Multi-Agent Orchestration

### 7.1 Orchestration Modes

```python
class OrchestrationMode(str, Enum):
    SEQUENTIAL = "sequential"  # Agents execute in order
    PARALLEL = "parallel"      # Agents execute simultaneously
    HANDOFF = "handoff"        # One agent delegates to another
    COLLABORATIVE = "collaborative"  # Agents share working memory
```

### 7.2 Agent Orchestrator

```python
class AgentOrchestrator:
    """Multi-agent coordination with learning channel."""

    def __init__(self):
        self.learning_channel = LearningChannel()
        self.shared_context = SharedContext()

    async def orchestrate(
        self,
        task: Task,
        agents: list[Agent],
        mode: OrchestrationMode,
    ) -> OrchestratorResult:
        if mode == OrchestrationMode.SEQUENTIAL:
            return await self._sequential(task, agents)
        elif mode == OrchestrationMode.PARALLEL:
            return await self._parallel(task, agents)
        elif mode == OrchestrationMode.HANDOFF:
            return await self._handoff(task, agents)
        elif mode == OrchestrationMode.COLLABORATIVE:
            return await self._collaborative(task, agents)

    async def _execute_agent(
        self,
        agent: Agent,
        task: Task,
    ) -> AgentResult:
        result = await agent.execute(task)

        # Extract and broadcast learning
        if result.learnings:
            for learning in result.learnings:
                await self.learning_channel.broadcast(
                    source=agent.id,
                    learning=learning,
                    scope=agent.contributes_to_scope,
                )

        return result
```

### 7.3 Cross-Agent Learning Channel

**This is foundational for Option C:**

```python
class LearningChannel:
    """Pub/sub for cross-agent learning.

    When one agent learns something, other agents can benefit.
    """

    def __init__(self):
        self.subscribers: dict[str, list[str]] = {}  # scope -> agent_ids
        self.pending_learnings: asyncio.Queue = asyncio.Queue()
        self.semantic_layer: SemanticLayer = None

    async def broadcast(
        self,
        source: str,
        learning: Learning,
        scope: str,
    ) -> None:
        """Broadcast learning to all subscribers of this scope."""
        # Store in semantic layer
        await self.semantic_layer.integrate(learning)

        # Notify subscribers
        for subscriber_id in self.subscribers.get(scope, []):
            await self.pending_learnings.put((subscriber_id, learning))

    async def subscribe(self, agent_id: str, scope: str) -> None:
        """Subscribe an agent to a scope's learnings."""
        if scope not in self.subscribers:
            self.subscribers[scope] = []
        self.subscribers[scope].append(agent_id)
```

---

## 8. Implementation Phases

### Phase C.1: Foundation (Week 1-2)

**Goal:** Build the core temporal graph with learning channel stubs

| Component | Description | Status |
|-----------|-------------|--------|
| `TemporalNode` | Base class with bi-temporal tracking | Design |
| `HierarchicalScope` | World→Context→Agent→User→Session | Design |
| `TemporalCognitiveGraph` | Graph container with Qdrant backend | Design |
| `LearningChannel` (stub) | Pub/sub interface with no-op impl | Design |
| `AgentOrchestrator` (basic) | Sequential mode only | Design |

**Key Deliverables:**
- `src/draagon_ai/memory/temporal_graph.py`
- `src/draagon_ai/memory/nodes.py`
- `src/draagon_ai/memory/scopes.py`
- `src/draagon_ai/orchestration/learning_channel.py` (stub)

### Phase C.2: Memory Layers (Week 3-4)

**Goal:** Implement 4-layer memory architecture

| Component | Description | Status |
|-----------|-------------|--------|
| `WorkingMemory` | Session-scoped active context | Design |
| `EpisodicMemory` | Episode storage and linking | Design |
| `SemanticMemory` | Entity/relationship extraction | Design |
| `MetacognitiveMemory` | Skills, strategies, behaviors | Design |
| `MemoryPromotion` | Auto-promotion between layers | Design |

**Key Deliverables:**
- `src/draagon_ai/memory/layers/working.py`
- `src/draagon_ai/memory/layers/episodic.py`
- `src/draagon_ai/memory/layers/semantic.py`
- `src/draagon_ai/memory/layers/metacognitive.py`
- `src/draagon_ai/memory/promotion.py`

### Phase C.3: Evolution Engine (Week 5-6)

**Goal:** Implement ACE-style evolution with self-referential meta-prompts

| Component | Description | Status |
|-----------|-------------|--------|
| `ContextEvolutionEngine` | Grow-and-refine mechanism | Design |
| `MetaPrompt` | Self-referential mutation prompts | Design |
| `SelfReferentialEvolution` | Meta-prompts evolving themselves | Design |
| `PromptbreederEvolution` | Full genetic algorithm | Design |
| `FitnessSharing` | Diversity preservation | Design |

**Key Deliverables:**
- `src/draagon_ai/evolution/context_engine.py`
- `src/draagon_ai/evolution/meta_prompts.py`
- `src/draagon_ai/evolution/promptbreeder.py`
- `src/draagon_ai/evolution/fitness.py`

### Phase C.4: Multi-Agent & Autonomous (Week 7-8)

**Goal:** Enable cross-agent learning and autonomous behaviors

| Component | Description | Status |
|-----------|-------------|--------|
| `LearningChannel` (full) | Full pub/sub implementation | Design |
| `AgentOrchestrator` (full) | All modes + collaborative | Design |
| `CuriosityAgent` | Autonomous knowledge-seeking | Design |
| `VerificationDaemon` | Background truth-seeking | Design |
| `BehaviorNode` | Behaviors as graph citizens | Design |

**Key Deliverables:**
- `src/draagon_ai/orchestration/learning_channel.py` (full)
- `src/draagon_ai/orchestration/orchestrator.py`
- `src/draagon_ai/agents/curiosity.py`
- `src/draagon_ai/agents/verification.py`
- `src/draagon_ai/behaviors/graph_integration.py`

---

## 9. Data Models

### 9.1 Core Node Model

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum

class NodeType(str, Enum):
    EPISODE = "episode"
    EVENT = "event"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    FACT = "fact"
    BELIEF = "belief"
    SKILL = "skill"
    STRATEGY = "strategy"
    INSIGHT = "insight"
    BEHAVIOR = "behavior"
    CONTEXT = "context"
    GOAL = "goal"

@dataclass
class TemporalNode:
    """Base class for all graph nodes with bi-temporal tracking."""

    node_id: str
    node_type: NodeType
    content: str

    # Vector embedding for similarity search
    embedding: list[float] | None = None

    # Bi-temporal tracking
    event_time: datetime = field(default_factory=datetime.now)
    ingestion_time: datetime = field(default_factory=datetime.now)
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None  # None = current

    # Confidence and evolution
    confidence: float = 1.0
    stated_count: int = 1
    access_count: int = 0

    # Lineage tracking
    derived_from: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    # Scope and layer
    scope: str = "agent:default"  # Hierarchical scope
    layer: str = "semantic"  # working, episodic, semantic, meta

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
```

### 9.2 Edge Model

```python
class EdgeType(str, Enum):
    IS_A = "is_a"
    HAS = "has"
    RELATED_TO = "related_to"
    BEFORE = "before"
    DURING = "during"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    CAUSES = "causes"
    ENABLES = "enables"
    TRIGGERS = "triggers"
    DELEGATES_TO = "delegates_to"
    LEARNS_FROM = "learns_from"

@dataclass
class TemporalEdge:
    """Edge between temporal nodes."""

    edge_id: str
    edge_type: EdgeType
    source_id: str
    target_id: str

    # Bi-temporal
    event_time: datetime = field(default_factory=datetime.now)
    ingestion_time: datetime = field(default_factory=datetime.now)
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None

    # Weight and confidence
    weight: float = 1.0
    confidence: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 9.3 Hierarchical Scope Model

```python
@dataclass
class HierarchicalScope:
    """Hierarchical scope with permissions and transience."""

    scope_id: str
    scope_type: str  # "world", "context", "agent", "user", "session"
    parent_scope_id: str | None = None

    # Permissions
    read_agents: list[str] = field(default_factory=list)  # Who can read
    write_agents: list[str] = field(default_factory=list)  # Who can write
    admin_agents: list[str] = field(default_factory=list)  # Who can manage

    # Transience settings
    default_ttl: timedelta | None = None  # Auto-expire items
    max_items: int | None = None  # Capacity limit
    promotion_threshold: float = 0.7  # When to promote to parent scope

    # Evolution rules (for Option C)
    evolution_enabled: bool = True
    evolution_frequency: timedelta = timedelta(days=1)
    evolution_config: dict[str, Any] = field(default_factory=dict)
```

### 9.4 Behavior Node Model

```python
@dataclass
class BehaviorNode(TemporalNode):
    """Behavior as a first-class graph citizen."""

    behavior_id: str = ""
    behavior: "Behavior" = None  # Reference to actual behavior

    # Graph relationships
    depends_on: list[str] = field(default_factory=list)
    can_delegate_to: list[str] = field(default_factory=list)
    shares_context_with: list[str] = field(default_factory=list)

    # Knowledge integration
    learns_from_scope: str = ""
    contributes_to_scope: str = ""

    # Evolution tracking
    prompt_versions: list["PromptVersion"] = field(default_factory=list)
    evolution_fitness: float = 0.0
    evolution_generation: int = 0

    # Self-improvement
    can_self_improve: bool = True
    improvement_triggers: list[str] = field(default_factory=list)
    last_improvement: datetime | None = None
    improvement_cooldown: timedelta = timedelta(hours=24)

    def __post_init__(self):
        self.node_type = NodeType.BEHAVIOR
        self.layer = "meta"
```

---

## References

### Research Papers
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [Graphiti: Knowledge Graph Memory for an Agentic World](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)

### Industry Sources
- [Netflix Real-Time Distributed Graph Architecture](https://netflixtechblog.medium.com/how-and-why-netflix-built-a-real-time-distributed-graph-part-2-building-a-scalable-storage-layer-ff4a8dbd3d1f)
- [Memory Is The Last Problem To Solve To Reach AGI](https://www.generalintelligencecompany.com/writing/memory-is-the-last-problem-to-solve-to-reach-agi)
- [Comparing Memory Systems for LLM Agents](https://www.marktechpost.com/2025/11/10/comparing-memory-systems-for-llm-agents-vector-graph-and-event-logs/)
- [Qdrant Updated Benchmarks 2024](https://qdrant.tech/blog/qdrant-benchmarks-2024/)

### Framework Documentation
- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Mem0 Research](https://mem0.ai/research)
- [Qdrant Vector Database](https://qdrant.tech/)

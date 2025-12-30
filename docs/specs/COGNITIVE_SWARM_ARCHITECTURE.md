# Cognitive Swarm Architecture (Option C)

**Version:** 1.0.0
**Status:** Specification
**Last Updated:** 2025-12-30
**Author:** draagon-ai team

---

## Executive Summary

This specification defines the **Cognitive Swarm Architecture**—a novel multi-agent orchestration system that combines draagon-ai's unique 4-layer cognitive memory with parallel execution, belief reconciliation, transactive memory, and metacognitive reflection. This architecture addresses fundamental limitations in current multi-agent systems that no existing framework solves.

### Why This Matters

Current multi-agent frameworks (LangGraph, CrewAI, AutoGen) suffer from:
- **Disconnected models**: "Models don't have continuity the way we do" - Microsoft
- **Homogeneous memory**: Same memory for all agents loses specialization
- **No belief tracking**: Agents contradict known facts
- **No expertise routing**: Wasted effort on tasks agents aren't good at
- **Coordination overhead**: Communication complexity grows exponentially

draagon-ai already has the building blocks (4-layer memory, beliefs, opinions, curiosity) that address these—they just need integration.

---

## Table of Contents

1. [Research Foundation](#1-research-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Shared Cognitive Working Memory](#3-phase-1-shared-cognitive-working-memory)
4. [Phase 2: Parallel Execution with Coordination](#4-phase-2-parallel-execution-with-coordination)
5. [Phase 3: Belief Reconciliation Across Agents](#5-phase-3-belief-reconciliation-across-agents)
6. [Phase 4: Transactive Memory System](#6-phase-4-transactive-memory-system)
7. [Phase 5: Metacognitive Reflection](#7-phase-5-metacognitive-reflection)
8. [Testing Strategy](#8-testing-strategy)
9. [Benchmark Targets](#9-benchmark-targets)
10. [Implementation Timeline](#10-implementation-timeline)

---

## 1. Research Foundation

### 1.1 Key Papers and Findings

| Paper | Key Finding | How draagon-ai Addresses It |
|-------|-------------|----------------------------|
| [Intrinsic Memory Agents (arXiv 2508.08997)](https://arxiv.org/html/2508.08997v1) | Heterogeneous agent-specific memories improve performance by 38.6% | Our 4-layer memory with attention weighting provides this naturally |
| [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system) | Multi-agent outperformed single by 90.2%; 3-5 parallel subagents optimal | Architecture supports parallel execution with shared context |
| [Why Multi-Agent LLM Systems Fail (arXiv 2503.13657)](https://arxiv.org/pdf/2503.13657) | MAST taxonomy: failures from inter-agent misalignment, not individual agent limits | Belief reconciliation directly addresses misalignment |
| [MultiAgentBench (ACL 2025)](https://arxiv.org/html/2503.01935v1) | Strategic silence over collaboration; agents refuse to share critical findings | Learning channel with scopes enables controlled sharing |
| [Memory in LLM-MAS Survey](https://www.mongodb.com/company/blog/technical/why-multi-agent-systems-need-memory-engineering) | Need transactive memory ("who knows what") like human teams | Expertise routing based on skill confidence tracking |
| [Metacognitive Learning (ICML 2025)](https://openreview.net/forum?id=4KhDd0Ozqe) | Truly self-improving agents need intrinsic metacognition | Reflection service with belief/opinion/curiosity integration |
| [MemoryAgentBench (arXiv 2507.05257)](https://arxiv.org/html/2507.05257v1) | Memory agents fail at conflict resolution and long-range consistency | Our belief system with credibility weighting handles conflicts |

### 1.2 Industry Failure Modes We Address

From the [MAST Framework](https://arxiv.org/pdf/2503.13657) analysis of multi-agent failures:

| Failure Category | % of Failures | Our Solution |
|-----------------|---------------|--------------|
| Inter-Agent Misalignment | 34.7% | Belief reconciliation + learning channel |
| Task Verification Issues | 13.48% | Skill confidence tracking + reflection |
| Coordination Breakdowns | 22.3% | Transactive memory + shared working memory |
| Goal Interpretation Drift | 18.1% | Persistent beliefs across agents |
| Memory Management | 11.4% | 4-layer memory with automatic promotion |

### 1.3 Current Benchmark Performance (What We Must Beat)

| Benchmark | Current SOTA | Key Gap | Our Advantage |
|-----------|-------------|---------|---------------|
| [MultiAgentBench](https://arxiv.org/html/2503.01935v1) Werewolf | 36.33% task score | Trust/disclosure failures | Belief-based trust with credibility |
| [GAIA](https://arxiv.org/abs/2311.12983) Level 3 | ~40% | Long-horizon reasoning | 4-layer memory persistence |
| [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) Conflict Resolution | ~45% | Conflicting information | Belief reconciliation with sources |
| [HI-TOM](https://arxiv.org/abs/2310.16755) Higher-Order ToM | Declining with depth | Recursive belief tracking | Explicit belief modeling |
| [WebArena](https://webarena.dev/) Extended Workflows | ~60% | Memory + re-planning | Working memory with attention |

---

## 2. Architecture Overview

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COGNITIVE SWARM ORCHESTRATOR                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │              SHARED COGNITIVE WORKING MEMORY (Task-Scoped)          │     │
│  │  ┌──────────────────────────────────────────────────────────────┐  │     │
│  │  │  • Observations with source attribution (agent_id, timestamp) │  │     │
│  │  │  • Attention-weighted items (Miller's Law: 7±2 per agent)     │  │     │
│  │  │  • Belief candidates flagged for reconciliation               │  │     │
│  │  │  • Knowledge gaps from curiosity engine                       │  │     │
│  │  │  • Conflict markers between observations                      │  │     │
│  │  │  • Per-item locking for concurrent access                     │  │     │
│  │  └──────────────────────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                              ↑↓ SYNC ↑↓                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    TRANSACTIVE MEMORY SYSTEM                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐    │   │
│  │   │  Expertise Map: agent_id → {topic → confidence}              │    │   │
│  │   │  Query Router: query → [ranked agents by expertise]          │    │   │
│  │   │  Updated by: skill success/failure tracking                  │    │   │
│  │   └─────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ↑↓ ROUTE ↑↓                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                          PARALLEL AGENT POOL                                 │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Agent A         │  │  Agent B         │  │  Agent C         │          │
│  │  (Researcher)    │  │  (Critic)        │  │  (Executor)      │          │
│  │                  │  │                  │  │                  │          │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │          │
│  │ │ Private WM   │ │  │ │ Private WM   │ │  │ │ Private WM   │ │          │
│  │ │ (7 items)    │ │  │ │ (7 items)    │ │  │ │ (7 items)    │ │          │
│  │ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │          │
│  │                  │  │                  │  │                  │          │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │          │
│  │ │ Beliefs      │ │  │ │ Beliefs      │ │  │ │ Beliefs      │ │          │
│  │ │ (local view) │ │  │ │ (local view) │ │  │ │ (local view) │ │          │
│  │ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │          │
│  │                  │  │                  │  │                  │          │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │          │
│  │ │ Skills/Conf  │ │  │ │ Opinions     │ │  │ │ Skills/Conf  │ │          │
│  │ │ (tracked)    │ │  │ │ (critic POV) │ │  │ │ (execution)  │ │          │
│  │ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           └─────────────────────┼─────────────────────┘                     │
│                                 ↓                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      LEARNING CHANNEL (Real-time)                     │   │
│  │   • Broadcast observations → shared working memory                    │   │
│  │   • Signal belief conflicts → trigger reconciliation                  │   │
│  │   • Share skill results → update transactive memory                   │   │
│  │   • Propagate knowledge gaps → enable curiosity routing               │   │
│  │   • Scopes: PRIVATE | CONTEXT | GLOBAL                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 ↓                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                      BELIEF RECONCILIATION SERVICE                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  • Collect observations from all agents                               │   │
│  │  • Detect conflicts (same topic, different claims)                    │   │
│  │  • Weight by agent credibility (skill success rate)                   │   │
│  │  • Form consolidated belief with uncertainty quantification           │   │
│  │  • Flag unresolvable conflicts for human clarification                │   │
│  │  • Update all agent belief caches                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                 ↓                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                      METACOGNITIVE REFLECTION SERVICE                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Post-Task Reflection:                                                │   │
│  │  • What worked? What didn't?                                          │   │
│  │  • Which agents contributed most?                                     │   │
│  │  • Were there coordination failures?                                  │   │
│  │  • Should task decomposition change next time?                        │   │
│  │  • Update expertise model based on outcomes                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
1. QUERY ARRIVES
   │
   ├─→ Transactive Memory: "Who knows about this?"
   │   └─→ Returns ranked agents by expertise match
   │
   ├─→ Shared Working Memory: Initialize task context
   │   └─→ Load relevant beliefs from semantic memory
   │
   ▼
2. PARALLEL EXECUTION (3-5 agents optimal per Anthropic research)
   │
   ├─→ Each agent:
   │   ├─→ Reads from shared working memory
   │   ├─→ Runs cognitive ReAct loop (with beliefs/opinions)
   │   ├─→ Writes observations to shared memory
   │   └─→ Broadcasts learnings to channel
   │
   ├─→ Periodic sync barriers (every N iterations)
   │   └─→ Reconcile belief conflicts
   │
   ▼
3. MERGE & RECONCILE
   │
   ├─→ Collect all agent outputs
   ├─→ Reconcile conflicting observations
   ├─→ Form consolidated response
   │
   ▼
4. METACOGNITIVE REFLECTION
   │
   ├─→ Evaluate task success
   ├─→ Update transactive memory (expertise)
   ├─→ Store learnings to semantic memory
   └─→ Flag improvements for next iteration
```

---

## 3. Phase 1: Shared Cognitive Working Memory

### 3.1 Design

Unlike simple `dict[str, Any]` in current TaskContext, this is a full cognitive working memory with:
- Source attribution (which agent contributed what)
- Attention weighting (what's most relevant now)
- Conflict detection (when observations contradict)
- Concurrent access control (safe parallel updates)

### 3.2 Data Structures

```python
@dataclass
class SharedObservation:
    """An observation in shared working memory."""

    observation_id: str
    content: str
    source_agent_id: str
    timestamp: datetime

    # Cognitive properties
    attention_weight: float = 0.5  # 0-1, how relevant is this now
    confidence: float = 1.0  # How confident is the source agent

    # Belief tracking
    is_belief_candidate: bool = False  # Should this become a belief?
    belief_type: str | None = None  # FACT, PREFERENCE, SKILL, etc.

    # Conflict tracking
    conflicts_with: list[str] = field(default_factory=list)  # observation_ids

    # Access tracking
    accessed_by: set[str] = field(default_factory=set)  # agent_ids
    access_count: int = 0


@dataclass
class SharedWorkingMemoryConfig:
    """Configuration for shared working memory."""

    max_items_per_agent: int = 7  # Miller's Law
    max_total_items: int = 50  # Total capacity
    attention_decay_factor: float = 0.9  # Decay per sync
    conflict_threshold: float = 0.7  # Semantic similarity for conflict
    sync_interval_iterations: int = 3  # Sync every N iterations


class SharedWorkingMemory:
    """Task-scoped working memory visible to all agents."""

    def __init__(
        self,
        task_id: str,
        config: SharedWorkingMemoryConfig | None = None,
    ):
        self.task_id = task_id
        self.config = config or SharedWorkingMemoryConfig()

        self._observations: dict[str, SharedObservation] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Conflict tracking
        self._conflicts: list[tuple[str, str, str]] = []  # (obs_a, obs_b, reason)

        # Per-agent views (attention-weighted subset)
        self._agent_views: dict[str, list[str]] = {}  # agent_id -> [observation_ids]

    async def add_observation(
        self,
        content: str,
        source_agent_id: str,
        *,
        attention_weight: float = 0.5,
        confidence: float = 1.0,
        is_belief_candidate: bool = False,
        belief_type: str | None = None,
    ) -> SharedObservation:
        """Add observation with automatic conflict detection."""

        async with self._global_lock:
            observation = SharedObservation(
                observation_id=str(uuid.uuid4()),
                content=content,
                source_agent_id=source_agent_id,
                timestamp=datetime.now(),
                attention_weight=attention_weight,
                confidence=confidence,
                is_belief_candidate=is_belief_candidate,
                belief_type=belief_type,
            )

            # Check for conflicts with existing observations
            conflicts = await self._detect_conflicts(observation)
            if conflicts:
                observation.conflicts_with = conflicts
                for conflict_id in conflicts:
                    self._conflicts.append(
                        (observation.observation_id, conflict_id, "semantic_conflict")
                    )

            # Manage capacity
            await self._ensure_capacity(source_agent_id)

            self._observations[observation.observation_id] = observation
            self._locks[observation.observation_id] = asyncio.Lock()

            # Update agent view
            if source_agent_id not in self._agent_views:
                self._agent_views[source_agent_id] = []
            self._agent_views[source_agent_id].append(observation.observation_id)

            return observation

    async def get_context_for_agent(
        self,
        agent_id: str,
        agent_role: AgentRole,
        max_items: int | None = None,
    ) -> list[SharedObservation]:
        """Get relevant context filtered by agent role and attention."""

        max_items = max_items or self.config.max_items_per_agent

        # Get all observations
        all_obs = list(self._observations.values())

        # Filter by role relevance (could be enhanced with embeddings)
        relevant = self._filter_by_role(all_obs, agent_role)

        # Sort by attention weight and recency
        sorted_obs = sorted(
            relevant,
            key=lambda o: (o.attention_weight, o.timestamp.timestamp()),
            reverse=True,
        )

        # Take top N
        result = sorted_obs[:max_items]

        # Track access
        for obs in result:
            obs.accessed_by.add(agent_id)
            obs.access_count += 1

        return result

    async def flag_conflict(
        self,
        observation_a_id: str,
        observation_b_id: str,
        conflict_reason: str,
    ) -> None:
        """Explicitly flag a conflict for reconciliation."""

        self._conflicts.append((observation_a_id, observation_b_id, conflict_reason))

        if observation_a_id in self._observations:
            self._observations[observation_a_id].conflicts_with.append(observation_b_id)
        if observation_b_id in self._observations:
            self._observations[observation_b_id].conflicts_with.append(observation_a_id)

    async def get_conflicts(self) -> list[tuple[SharedObservation, SharedObservation, str]]:
        """Get all unresolved conflicts for reconciliation."""

        result = []
        for obs_a_id, obs_b_id, reason in self._conflicts:
            obs_a = self._observations.get(obs_a_id)
            obs_b = self._observations.get(obs_b_id)
            if obs_a and obs_b:
                result.append((obs_a, obs_b, reason))
        return result

    async def get_belief_candidates(self) -> list[SharedObservation]:
        """Get observations that should become beliefs."""

        return [
            obs for obs in self._observations.values()
            if obs.is_belief_candidate and not obs.conflicts_with
        ]

    async def apply_attention_decay(self) -> None:
        """Decay attention weights (called periodically)."""

        for obs in self._observations.values():
            obs.attention_weight *= self.config.attention_decay_factor

    async def boost_attention(
        self,
        observation_id: str,
        boost: float = 0.2,
    ) -> None:
        """Boost attention for a specific observation."""

        if observation_id in self._observations:
            obs = self._observations[observation_id]
            obs.attention_weight = min(1.0, obs.attention_weight + boost)

    async def _detect_conflicts(
        self,
        new_observation: SharedObservation,
    ) -> list[str]:
        """Detect semantic conflicts with existing observations."""

        # TODO: Use embedding similarity for semantic conflict detection
        # For now, placeholder that checks exact contradictions

        conflicts = []
        for obs_id, obs in self._observations.items():
            # Skip same source
            if obs.source_agent_id == new_observation.source_agent_id:
                continue

            # Check for same belief type with different content
            if (
                obs.is_belief_candidate and
                new_observation.is_belief_candidate and
                obs.belief_type == new_observation.belief_type
            ):
                # Would need embedding check here
                # For now, flag if both are about same topic
                conflicts.append(obs_id)

        return conflicts

    async def _ensure_capacity(self, source_agent_id: str) -> None:
        """Evict lowest-priority items if over capacity."""

        # Per-agent capacity
        agent_obs = self._agent_views.get(source_agent_id, [])
        while len(agent_obs) >= self.config.max_items_per_agent:
            # Find lowest attention item from this agent
            lowest_id = min(
                agent_obs,
                key=lambda oid: self._observations[oid].attention_weight
                if oid in self._observations else 0
            )
            agent_obs.remove(lowest_id)
            if lowest_id in self._observations:
                del self._observations[lowest_id]

        # Total capacity
        while len(self._observations) >= self.config.max_total_items:
            # Find lowest attention item overall
            lowest = min(
                self._observations.values(),
                key=lambda o: o.attention_weight
            )
            del self._observations[lowest.observation_id]

    def _filter_by_role(
        self,
        observations: list[SharedObservation],
        role: AgentRole,
    ) -> list[SharedObservation]:
        """Filter observations by role relevance."""

        # Role-specific filtering heuristics
        if role == AgentRole.CRITIC:
            # Critics see claims and assertions
            return [o for o in observations if o.is_belief_candidate]
        elif role == AgentRole.RESEARCHER:
            # Researchers see knowledge gaps
            return observations  # All relevant
        elif role == AgentRole.EXECUTOR:
            # Executors see action-related observations
            return [o for o in observations if o.belief_type in ("SKILL", "FACT", None)]
        else:
            return observations
```

### 3.3 Integration Points

1. **AgentLoop**: Pass SharedWorkingMemory into context
2. **DecisionEngine**: Include shared observations in decision prompt
3. **ActionExecutor**: Write observations after tool execution
4. **LearningChannel**: Broadcast significant observations

---

## 4. Phase 2: Parallel Execution with Coordination

### 4.1 Design Principles

From Anthropic's research:
- 3-5 parallel subagents is optimal
- Detailed task descriptions prevent duplication
- Periodic sync prevents drift
- Lead agent coordinates, subagents execute

### 4.2 Implementation

```python
class ParallelOrchestrationMode(str, Enum):
    """How parallel agents are coordinated."""

    FORK_JOIN = "fork_join"  # All start together, sync at end
    BARRIER_SYNC = "barrier_sync"  # Sync at periodic barriers
    STREAMING = "streaming"  # Continuous sync via shared memory


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel agent execution."""

    max_concurrent_agents: int = 5  # Anthropic: 3-5 optimal
    sync_mode: ParallelOrchestrationMode = ParallelOrchestrationMode.BARRIER_SYNC
    sync_interval_iterations: int = 3
    timeout_per_agent_seconds: float = 60.0
    allow_early_termination: bool = True  # Agent can finish early


class ParallelCognitiveOrchestrator(MultiAgentOrchestrator):
    """Orchestrator with parallel execution and cognitive integration."""

    def __init__(
        self,
        config: ParallelExecutionConfig | None = None,
        belief_service: BeliefReconciliationService | None = None,
        transactive_memory: TransactiveMemory | None = None,
    ):
        super().__init__()
        self.config = config or ParallelExecutionConfig()
        self.belief_service = belief_service
        self.transactive_memory = transactive_memory

    async def orchestrate_parallel(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        agent_executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Execute agents in parallel with cognitive coordination."""

        started_at = datetime.now()

        # Initialize shared cognitive memory
        shared_memory = SharedWorkingMemory(context.task_id)

        # Route to best agents if transactive memory available
        if self.transactive_memory:
            agents = await self._route_by_expertise(agents, context.query)

        # Limit concurrent agents
        agents = agents[:self.config.max_concurrent_agents]

        # Create execution tasks with shared context
        tasks = [
            self._run_agent_with_shared_memory(
                agent=agent,
                context=context,
                shared_memory=shared_memory,
                agent_executor=agent_executor,
            )
            for agent in agents
        ]

        # Execute based on sync mode
        if self.config.sync_mode == ParallelOrchestrationMode.FORK_JOIN:
            results = await self._fork_join_execution(tasks, shared_memory)
        elif self.config.sync_mode == ParallelOrchestrationMode.BARRIER_SYNC:
            results = await self._barrier_sync_execution(tasks, shared_memory, agents)
        else:
            results = await self._streaming_execution(tasks, shared_memory)

        # Reconcile any conflicts
        if self.belief_service:
            conflicts = await shared_memory.get_conflicts()
            if conflicts:
                await self._reconcile_conflicts(conflicts, context)

        # Update transactive memory based on results
        if self.transactive_memory:
            await self._update_expertise(results, context.query)

        # Merge results
        return self._merge_results(results, context, started_at)

    async def _barrier_sync_execution(
        self,
        tasks: list[Coroutine],
        shared_memory: SharedWorkingMemory,
        agents: list[AgentSpec],
    ) -> list[AgentResult]:
        """Execute with periodic synchronization barriers."""

        results = []
        active_tasks = {
            agent.agent_id: asyncio.create_task(task)
            for agent, task in zip(agents, tasks)
        }

        sync_interval = self.config.sync_interval_iterations
        iteration = 0

        while active_tasks:
            # Wait for any task to complete or timeout for sync
            done, pending = await asyncio.wait(
                active_tasks.values(),
                timeout=5.0,  # Check every 5 seconds
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Collect completed results
            for task in done:
                result = task.result()
                results.append(result)
                # Remove from active
                for agent_id, t in list(active_tasks.items()):
                    if t == task:
                        del active_tasks[agent_id]
                        break

            iteration += 1

            # Periodic sync barrier
            if iteration % sync_interval == 0 and active_tasks:
                # Apply attention decay
                await shared_memory.apply_attention_decay()

                # Check for conflicts that need immediate resolution
                conflicts = await shared_memory.get_conflicts()
                if len(conflicts) > 3:  # Too many conflicts, pause and reconcile
                    logger.info(f"Barrier sync: {len(conflicts)} conflicts detected")
                    # Could pause agents here and reconcile

        return results

    async def _run_agent_with_shared_memory(
        self,
        agent: AgentSpec,
        context: TaskContext,
        shared_memory: SharedWorkingMemory,
        agent_executor: AgentExecutor,
    ) -> AgentResult:
        """Run a single agent with access to shared memory."""

        # Create agent-specific context with shared memory access
        agent_context = TaskContext(
            task_id=context.task_id,
            query=context.query,
            user_id=context.user_id,
            session_id=context.session_id,
            # Inject shared memory observations
            agent_outputs=context.agent_outputs,
            working_memory={
                "__shared__": shared_memory,
                **context.working_memory,
            },
            learnings=context.learnings,
        )

        try:
            result = await asyncio.wait_for(
                agent_executor(agent, agent_context),
                timeout=self.config.timeout_per_agent_seconds,
            )

            # Store agent output as observation
            if result.success and result.output:
                await shared_memory.add_observation(
                    content=str(result.output),
                    source_agent_id=agent.agent_id,
                    attention_weight=0.8,
                    confidence=0.9 if result.success else 0.5,
                    is_belief_candidate=True,
                )

            return result

        except asyncio.TimeoutError:
            return AgentResult(
                agent_id=agent.agent_id,
                success=False,
                error=f"Agent timed out after {self.config.timeout_per_agent_seconds}s",
            )

    async def _route_by_expertise(
        self,
        agents: list[AgentSpec],
        query: str,
    ) -> list[AgentSpec]:
        """Route query to agents with relevant expertise."""

        if not self.transactive_memory:
            return agents

        ranked = await self.transactive_memory.route_query(query, agents)
        return ranked

    async def _update_expertise(
        self,
        results: list[AgentResult],
        query: str,
    ) -> None:
        """Update transactive memory based on execution results."""

        if not self.transactive_memory:
            return

        # Extract topic from query (simple version)
        topic = query[:50]  # Would use LLM extraction in production

        for result in results:
            await self.transactive_memory.update_expertise(
                agent_id=result.agent_id,
                topic=topic,
                success=result.success,
            )
```

---

## 5. Phase 3: Belief Reconciliation Across Agents

### 5.1 The Problem

When Agent A says "The meeting is at 3pm" and Agent B says "The meeting is at 4pm":
- Current systems: Last writer wins, or undefined behavior
- Our system: Detect conflict, weight by credibility, flag for resolution

### 5.2 Multi-Agent Belief Reconciliation

```python
@dataclass
class ReconciliationResult:
    """Result of reconciling conflicting beliefs."""

    resolved: bool
    consolidated_belief: AgentBelief | None
    confidence: float

    # If unresolved
    needs_human_clarification: bool = False
    clarification_question: str | None = None

    # Audit trail
    observations_considered: list[str] = field(default_factory=list)
    reasoning: str = ""


class MultiAgentBeliefReconciliation:
    """Reconcile beliefs when multiple agents observe different things."""

    def __init__(
        self,
        base_belief_service: BeliefReconciliationService,
        llm: LLMProvider,
    ):
        self.base_service = base_belief_service
        self.llm = llm

    async def reconcile_multi_agent(
        self,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> ReconciliationResult:
        """
        Reconcile conflicting observations from multiple agents.

        Strategy:
        1. Group by topic/entity
        2. Detect contradictions
        3. Weight by agent credibility
        4. Use LLM to synthesize or flag for human
        """

        # Group observations by topic
        grouped = self._group_by_topic(observations)

        results = []
        for topic, topic_observations in grouped.items():
            if len(topic_observations) == 1:
                # No conflict
                result = await self._single_observation_to_belief(topic_observations[0])
            else:
                # Potential conflict - analyze
                result = await self._reconcile_topic(
                    topic,
                    topic_observations,
                    agent_credibilities,
                )
            results.append(result)

        # Combine results
        return self._combine_results(results)

    async def _reconcile_topic(
        self,
        topic: str,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> ReconciliationResult:
        """Reconcile potentially conflicting observations on a topic."""

        # Build context for LLM
        obs_text = "\n".join([
            f"- Agent {o.source_agent_id} (credibility: {agent_credibilities.get(o.source_agent_id, 0.5):.2f}): {o.content}"
            for o in observations
        ])

        prompt = f"""Analyze these observations from different agents about "{topic}":

{obs_text}

Determine:
1. Do these observations conflict? (yes/no)
2. If yes, which is most likely correct based on credibility?
3. Can you synthesize a consolidated understanding?
4. If unresolvable, what clarifying question would help?

Respond in XML:
<reconciliation>
    <conflicts>yes/no</conflicts>
    <consolidated_belief>The reconciled understanding, or null if unresolvable</consolidated_belief>
    <confidence>0.0-1.0</confidence>
    <needs_clarification>true/false</needs_clarification>
    <clarification_question>Question to ask if needed</clarification_question>
    <reasoning>Why you reached this conclusion</reasoning>
</reconciliation>
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_reconciliation_response(response, observations)

    async def _single_observation_to_belief(
        self,
        observation: SharedObservation,
    ) -> ReconciliationResult:
        """Convert single observation to belief."""

        belief = AgentBelief(
            content=observation.content,
            belief_type=BeliefType.UNVERIFIED_CLAIM,
            confidence=observation.confidence * 0.8,  # Slight uncertainty
            supporting_observations=[observation.observation_id],
        )

        return ReconciliationResult(
            resolved=True,
            consolidated_belief=belief,
            confidence=belief.confidence,
            observations_considered=[observation.observation_id],
            reasoning="Single source, converted with reduced confidence",
        )

    def _group_by_topic(
        self,
        observations: list[SharedObservation],
    ) -> dict[str, list[SharedObservation]]:
        """Group observations by topic (would use embedding clustering)."""

        # Simple grouping by belief_type for now
        # TODO: Use embedding similarity for semantic grouping
        groups: dict[str, list[SharedObservation]] = {}
        for obs in observations:
            key = obs.belief_type or "general"
            if key not in groups:
                groups[key] = []
            groups[key].append(obs)
        return groups
```

---

## 6. Phase 4: Transactive Memory System

### 6.1 The Concept

Human teams develop "transactive memory"—knowing who knows what. This enables:
- Efficient query routing (ask the expert)
- Reduced redundant work
- Better task decomposition

### 6.2 Implementation

```python
@dataclass
class ExpertiseEntry:
    """An agent's expertise in a topic."""

    topic: str
    confidence: float  # 0-1
    success_count: int = 0
    failure_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Prior
        return self.success_count / total


class TransactiveMemory:
    """Track which agents are experts at what."""

    def __init__(
        self,
        embedding_provider: Any = None,  # For semantic topic matching
    ):
        # agent_id -> topic -> ExpertiseEntry
        self._expertise: dict[str, dict[str, ExpertiseEntry]] = {}
        self._embedding_provider = embedding_provider

        # Topic hierarchy (for generalization)
        self._topic_hierarchy: dict[str, list[str]] = {}  # topic -> parent_topics

    async def route_query(
        self,
        query: str,
        available_agents: list[AgentSpec],
    ) -> list[AgentSpec]:
        """Route query to agents ranked by expertise."""

        # Extract topics from query
        topics = await self._extract_topics(query)

        # Score each agent
        agent_scores: dict[str, float] = {}
        for agent in available_agents:
            score = await self._compute_expertise_score(agent.agent_id, topics)
            agent_scores[agent.agent_id] = score

        # Sort by score (highest first)
        sorted_agents = sorted(
            available_agents,
            key=lambda a: agent_scores.get(a.agent_id, 0),
            reverse=True,
        )

        return sorted_agents

    async def update_expertise(
        self,
        agent_id: str,
        topic: str,
        success: bool,
    ) -> None:
        """Update expertise based on execution outcome."""

        if agent_id not in self._expertise:
            self._expertise[agent_id] = {}

        if topic not in self._expertise[agent_id]:
            self._expertise[agent_id][topic] = ExpertiseEntry(
                topic=topic,
                confidence=0.5,
            )

        entry = self._expertise[agent_id][topic]

        if success:
            entry.success_count += 1
            entry.confidence = min(1.0, entry.confidence + 0.1)
        else:
            entry.failure_count += 1
            entry.confidence = max(0.0, entry.confidence - 0.15)

        entry.last_updated = datetime.now()

        # Update parent topics too
        if topic in self._topic_hierarchy:
            for parent in self._topic_hierarchy[topic]:
                await self.update_expertise(agent_id, parent, success)

    async def get_experts(
        self,
        topic: str,
        min_confidence: float = 0.6,
    ) -> list[tuple[str, float]]:
        """Get agents who are experts in a topic."""

        experts = []
        for agent_id, topics in self._expertise.items():
            if topic in topics:
                entry = topics[topic]
                if entry.confidence >= min_confidence:
                    experts.append((agent_id, entry.confidence))

        return sorted(experts, key=lambda x: x[1], reverse=True)

    async def who_knows_about(self, topic: str) -> str:
        """Natural language answer to 'who knows about X?'"""

        experts = await self.get_experts(topic)

        if not experts:
            return f"No agents have demonstrated expertise in '{topic}' yet."

        expert_strs = [
            f"{agent_id} (confidence: {conf:.0%})"
            for agent_id, conf in experts[:3]
        ]
        return f"For '{topic}': {', '.join(expert_strs)}"

    async def _extract_topics(self, query: str) -> list[str]:
        """Extract relevant topics from query."""

        # Simple keyword extraction
        # TODO: Use LLM or embedding-based topic extraction
        words = query.lower().split()
        # Filter stopwords and return significant terms
        stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "when", "where"}
        return [w for w in words if w not in stopwords and len(w) > 3]

    async def _compute_expertise_score(
        self,
        agent_id: str,
        topics: list[str],
    ) -> float:
        """Compute agent's expertise score for given topics."""

        if agent_id not in self._expertise:
            return 0.5  # Neutral prior

        agent_expertise = self._expertise[agent_id]

        if not topics:
            return 0.5

        # Average confidence across matching topics
        scores = []
        for topic in topics:
            if topic in agent_expertise:
                scores.append(agent_expertise[topic].confidence)
            else:
                # Check for partial matches
                for known_topic, entry in agent_expertise.items():
                    if topic in known_topic or known_topic in topic:
                        scores.append(entry.confidence * 0.7)  # Partial match

        if not scores:
            return 0.5

        return sum(scores) / len(scores)

    def get_expertise_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all agent expertise."""

        return {
            agent_id: {
                topic: entry.confidence
                for topic, entry in topics.items()
            }
            for agent_id, topics in self._expertise.items()
        }
```

---

## 7. Phase 5: Metacognitive Reflection

### 7.1 The Research

From [ICML 2025](https://openreview.net/forum?id=4KhDd0Ozqe):
> "Truly self-improving agents require intrinsic metacognitive learning—an agent's intrinsic ability to actively evaluate, reflect on, and adapt its own learning processes."

Three components:
1. **Metacognitive Knowledge**: Self-assessment of capabilities
2. **Metacognitive Planning**: Deciding what and how to learn
3. **Metacognitive Evaluation**: Reflecting on learning experiences

### 7.2 Implementation

```python
@dataclass
class ReflectionResult:
    """Result of metacognitive reflection on a task."""

    task_id: str

    # What happened
    overall_success: bool
    task_duration_ms: float
    agents_involved: list[str]

    # What worked
    successful_patterns: list[str]
    effective_agents: list[tuple[str, str]]  # (agent_id, why_effective)

    # What didn't work
    failure_patterns: list[str]
    ineffective_agents: list[tuple[str, str]]  # (agent_id, why_ineffective)
    coordination_issues: list[str]

    # Improvements
    suggested_changes: list[str]
    expertise_updates: dict[str, dict[str, float]]  # agent -> topic -> new_confidence

    # Learnings to store
    new_insights: list[str]


class MetacognitiveReflectionService:
    """Reflect on task execution to improve future performance."""

    def __init__(
        self,
        llm: LLMProvider,
        transactive_memory: TransactiveMemory,
        learning_service: LearningService,
    ):
        self.llm = llm
        self.transactive_memory = transactive_memory
        self.learning_service = learning_service

    async def reflect_on_task(
        self,
        task_context: TaskContext,
        orchestration_result: OrchestratorResult,
    ) -> ReflectionResult:
        """Perform metacognitive reflection after task completion."""

        # Build reflection context
        context = self._build_reflection_context(task_context, orchestration_result)

        prompt = f"""Reflect on this multi-agent task execution:

## Task
Query: {task_context.query}
Duration: {orchestration_result.duration_ms:.0f}ms
Overall Success: {orchestration_result.success}

## Agent Results
{context['agent_results_text']}

## Coordination Events
{context['coordination_text']}

## Conflicts Encountered
{context['conflicts_text']}

Analyze:
1. Which agents contributed most effectively? Why?
2. Which agents struggled? What were the issues?
3. Were there coordination failures? What caused them?
4. What patterns led to success or failure?
5. How should we adjust for similar future tasks?
6. What did we learn that should be stored?

Respond in XML:
<reflection>
    <successful_patterns>
        <pattern>Description of what worked</pattern>
    </successful_patterns>
    <failure_patterns>
        <pattern>Description of what didn't work</pattern>
    </failure_patterns>
    <effective_agents>
        <agent id="agent_id">Why they were effective</agent>
    </effective_agents>
    <ineffective_agents>
        <agent id="agent_id">Why they struggled</agent>
    </ineffective_agents>
    <coordination_issues>
        <issue>Description of coordination problem</issue>
    </coordination_issues>
    <suggested_changes>
        <change>What to do differently next time</change>
    </suggested_changes>
    <expertise_updates>
        <update agent_id="agent_id" topic="topic" confidence_delta="+0.1"/>
    </expertise_updates>
    <new_insights>
        <insight>Something we learned that should be remembered</insight>
    </new_insights>
</reflection>
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        result = self._parse_reflection_response(
            response,
            task_context,
            orchestration_result,
        )

        # Apply learnings
        await self._apply_reflection(result)

        return result

    async def _apply_reflection(self, result: ReflectionResult) -> None:
        """Apply reflection learnings to the system."""

        # Update transactive memory
        for agent_id, updates in result.expertise_updates.items():
            for topic, delta in updates.items():
                current = await self.transactive_memory.get_expertise(agent_id, topic)
                new_confidence = max(0, min(1, (current or 0.5) + delta))
                # Direct update (not through success/failure)
                if agent_id not in self.transactive_memory._expertise:
                    self.transactive_memory._expertise[agent_id] = {}
                self.transactive_memory._expertise[agent_id][topic] = ExpertiseEntry(
                    topic=topic,
                    confidence=new_confidence,
                )

        # Store new insights
        for insight in result.new_insights:
            await self.learning_service.store_learning(
                content=insight,
                learning_type=LearningType.INSIGHT,
                source="metacognitive_reflection",
            )

    def _build_reflection_context(
        self,
        task_context: TaskContext,
        result: OrchestratorResult,
    ) -> dict[str, str]:
        """Build context for reflection prompt."""

        agent_results_text = "\n".join([
            f"- {r.agent_id}: {'SUCCESS' if r.success else 'FAILED'} "
            f"({r.duration_ms:.0f}ms) - {r.error or 'OK'}"
            for r in result.agent_results
        ])

        # Would extract from shared memory in real implementation
        coordination_text = "Agents shared observations via shared working memory."
        conflicts_text = "No major conflicts." if result.success else "Conflicts detected."

        return {
            "agent_results_text": agent_results_text,
            "coordination_text": coordination_text,
            "conflicts_text": conflicts_text,
        }
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

Each component tested in isolation:

```python
# tests/orchestration/test_shared_working_memory.py

class TestSharedWorkingMemory:
    """Unit tests for SharedWorkingMemory."""

    @pytest.mark.asyncio
    async def test_capacity_management(self):
        """Test Miller's Law capacity enforcement."""
        memory = SharedWorkingMemory("task_1", SharedWorkingMemoryConfig(max_items_per_agent=7))

        # Add 10 items from same agent
        for i in range(10):
            await memory.add_observation(
                content=f"Observation {i}",
                source_agent_id="agent_a",
            )

        # Should only keep 7
        assert len(memory._observations) == 7
        # Highest attention items kept
        remaining = [o.content for o in memory._observations.values()]
        assert "Observation 9" in remaining  # Most recent

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        """Test automatic conflict detection."""
        memory = SharedWorkingMemory("task_1")

        await memory.add_observation(
            content="The meeting is at 3pm",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        obs2 = await memory.add_observation(
            content="The meeting is at 4pm",
            source_agent_id="agent_b",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        # Should detect conflict
        assert len(obs2.conflicts_with) > 0
        conflicts = await memory.get_conflicts()
        assert len(conflicts) == 1

    @pytest.mark.asyncio
    async def test_attention_decay(self):
        """Test attention weight decay."""
        memory = SharedWorkingMemory("task_1", SharedWorkingMemoryConfig(attention_decay_factor=0.9))

        obs = await memory.add_observation(
            content="Test",
            source_agent_id="agent_a",
            attention_weight=1.0,
        )

        initial_weight = obs.attention_weight
        await memory.apply_attention_decay()

        assert memory._observations[obs.observation_id].attention_weight == initial_weight * 0.9

    @pytest.mark.asyncio
    async def test_role_filtered_context(self):
        """Test context filtering by agent role."""
        memory = SharedWorkingMemory("task_1")

        await memory.add_observation(
            content="Claim to verify",
            source_agent_id="agent_a",
            is_belief_candidate=True,
        )
        await memory.add_observation(
            content="General observation",
            source_agent_id="agent_b",
            is_belief_candidate=False,
        )

        # Critic should see only belief candidates
        critic_context = await memory.get_context_for_agent("critic", AgentRole.CRITIC)
        assert len(critic_context) == 1
        assert critic_context[0].is_belief_candidate
```

### 8.2 Integration Tests

```python
# tests/integration/test_parallel_orchestration.py

class TestParallelOrchestration:
    """Integration tests for parallel cognitive orchestration."""

    @pytest.mark.asyncio
    async def test_parallel_execution_no_race_conditions(self):
        """Verify parallel execution doesn't cause race conditions."""
        orchestrator = ParallelCognitiveOrchestrator()

        agents = [
            AgentSpec(agent_id=f"agent_{i}", role=AgentRole.RESEARCHER)
            for i in range(5)
        ]

        context = TaskContext(query="Test parallel execution")

        # Mock executor that writes to shared memory
        async def mock_executor(agent: AgentSpec, ctx: TaskContext) -> AgentResult:
            await asyncio.sleep(random.uniform(0.1, 0.5))
            return AgentResult(
                agent_id=agent.agent_id,
                success=True,
                output=f"Result from {agent.agent_id}",
            )

        result = await orchestrator.orchestrate_parallel(
            agents=agents,
            context=context,
            agent_executor=mock_executor,
        )

        assert result.success
        assert len(result.agent_results) == 5
        # No duplicates
        agent_ids = [r.agent_id for r in result.agent_results]
        assert len(set(agent_ids)) == 5

    @pytest.mark.asyncio
    async def test_belief_reconciliation_in_parallel(self):
        """Test that conflicting beliefs are reconciled."""
        belief_service = MockBeliefService()
        orchestrator = ParallelCognitiveOrchestrator(
            belief_service=belief_service,
        )

        # Agents that will produce conflicting observations
        agents = [
            AgentSpec(agent_id="optimist", role=AgentRole.RESEARCHER),
            AgentSpec(agent_id="pessimist", role=AgentRole.RESEARCHER),
        ]

        context = TaskContext(query="What's the weather forecast?")

        async def conflicting_executor(agent: AgentSpec, ctx: TaskContext) -> AgentResult:
            if agent.agent_id == "optimist":
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output="Weather will be sunny tomorrow",
                )
            else:
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output="Weather will be rainy tomorrow",
                )

        result = await orchestrator.orchestrate_parallel(
            agents=agents,
            context=context,
            agent_executor=conflicting_executor,
        )

        # Should have triggered reconciliation
        assert belief_service.reconcile_called
        assert len(belief_service.reconciled_beliefs) > 0
```

### 8.3 Stress Tests

```python
# tests/stress/test_cognitive_swarm_stress.py

class TestCognitiveSwarmStress:
    """Stress tests for cognitive swarm architecture."""

    @pytest.mark.asyncio
    async def test_high_concurrency(self):
        """Test with maximum concurrent agents."""
        orchestrator = ParallelCognitiveOrchestrator(
            config=ParallelExecutionConfig(max_concurrent_agents=10),
        )

        agents = [
            AgentSpec(agent_id=f"agent_{i}", role=AgentRole.RESEARCHER)
            for i in range(10)
        ]

        # Each agent produces many observations
        observation_count = 0

        async def heavy_executor(agent: AgentSpec, ctx: TaskContext) -> AgentResult:
            nonlocal observation_count
            shared_memory = ctx.working_memory.get("__shared__")

            for j in range(20):
                await shared_memory.add_observation(
                    content=f"Observation {j} from {agent.agent_id}",
                    source_agent_id=agent.agent_id,
                    attention_weight=random.random(),
                )
                observation_count += 1

            return AgentResult(agent_id=agent.agent_id, success=True)

        context = TaskContext(query="Stress test")
        result = await orchestrator.orchestrate_parallel(
            agents=agents,
            context=context,
            agent_executor=heavy_executor,
        )

        assert result.success
        # Some observations evicted due to capacity
        shared_memory = context.working_memory.get("__shared__")
        assert len(shared_memory._observations) <= 50  # Total capacity

    @pytest.mark.asyncio
    async def test_many_conflicts(self):
        """Test handling many simultaneous conflicts."""
        belief_service = MockBeliefService()
        orchestrator = ParallelCognitiveOrchestrator(
            belief_service=belief_service,
        )

        # 20 agents, each with different "facts"
        agents = [
            AgentSpec(agent_id=f"agent_{i}", role=AgentRole.RESEARCHER)
            for i in range(20)
        ]

        async def conflicting_executor(agent: AgentSpec, ctx: TaskContext) -> AgentResult:
            shared_memory = ctx.working_memory.get("__shared__")

            # Each agent has a different opinion
            await shared_memory.add_observation(
                content=f"The answer is {agent.agent_id}",
                source_agent_id=agent.agent_id,
                is_belief_candidate=True,
                belief_type="FACT",
            )

            return AgentResult(agent_id=agent.agent_id, success=True)

        context = TaskContext(query="What is the answer?")
        result = await orchestrator.orchestrate_parallel(
            agents=agents,
            context=context,
            agent_executor=conflicting_executor,
        )

        # Should handle gracefully
        assert result.success
        # Many conflicts should be detected
        shared_memory = context.working_memory.get("__shared__")
        conflicts = await shared_memory.get_conflicts()
        assert len(conflicts) > 10
```

---

## 9. Benchmark Targets

### 9.1 Existing Benchmarks to Conquer

| Benchmark | Current SOTA | Our Target | Why We Can Beat It |
|-----------|-------------|------------|-------------------|
| **[MultiAgentBench](https://arxiv.org/html/2503.01935v1) Werewolf** | 36.33% | 55%+ | Trust-based belief sharing solves disclosure failure |
| **[MultiAgentBench](https://arxiv.org/html/2503.01935v1) Research** | 84.13% | 90%+ | Transactive memory routes to experts |
| **[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) FactConsolidation** | ~45% | 70%+ | Belief reconciliation with credibility |
| **[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) Conflict Resolution** | ~50% | 75%+ | Multi-source belief reconciliation |
| **[GAIA](https://arxiv.org/abs/2311.12983) Level 3** | ~40% | 55%+ | 4-layer memory for long-horizon |
| **[HI-TOM](https://arxiv.org/abs/2310.16755) 3rd Order** | ~40% | 60%+ | Explicit belief tracking with recursion |

### 9.2 Novel Benchmarks We Should Create

These don't exist yet—we should create them to showcase our unique capabilities:

#### 9.2.1 Multi-Agent Belief Conflict Benchmark

```python
# Test scenarios where agents receive contradictory information

BELIEF_CONFLICT_SCENARIOS = [
    {
        "name": "Simple Contradiction",
        "agent_a_info": "The meeting is at 3pm in Room A",
        "agent_b_info": "The meeting is at 4pm in Room B",
        "ground_truth": "The meeting is at 3pm in Room A",
        "difficulty": "easy",
    },
    {
        "name": "Partial Overlap",
        "agent_a_info": "John has 3 cats and 2 dogs",
        "agent_b_info": "John has several pets including cats",
        "ground_truth": "John has 3 cats and 2 dogs",
        "difficulty": "medium",
        "note": "B's info is compatible but less specific",
    },
    {
        "name": "Temporal Conflict",
        "agent_a_info": "As of Monday, the project was on track",
        "agent_b_info": "As of Wednesday, the project is delayed",
        "ground_truth": "The project was on track Monday but is now delayed",
        "difficulty": "hard",
        "note": "Both are correct for their time context",
    },
    {
        "name": "Source Credibility",
        "agent_a_info": "The CEO said revenue is up 10% (from earnings call)",
        "agent_b_info": "Revenue might be up 15% (from Reddit rumor)",
        "ground_truth": "Revenue is up 10%",
        "difficulty": "hard",
        "note": "Must weight by source credibility",
    },
]
```

#### 9.2.2 Transactive Memory Benchmark

```python
# Test whether agents learn "who knows what"

TRANSACTIVE_MEMORY_SCENARIOS = [
    {
        "name": "Expertise Routing",
        "setup": [
            ("agent_weather", "weather", "success"),
            ("agent_weather", "weather", "success"),
            ("agent_calendar", "scheduling", "success"),
        ],
        "query": "What's the weather for my meeting tomorrow?",
        "expected_route": ["agent_weather", "agent_calendar"],  # Weather first
    },
    {
        "name": "Expertise Learning",
        "setup": [
            ("agent_a", "coding", "success"),
            ("agent_a", "coding", "success"),
            ("agent_a", "coding", "failure"),
            ("agent_b", "coding", "success"),
            ("agent_b", "coding", "success"),
        ],
        "query": "Help me debug this code",
        "expected_route": ["agent_b", "agent_a"],  # B has higher success rate
    },
    {
        "name": "Cross-Domain Generalization",
        "setup": [
            ("agent_x", "python", "success"),
            ("agent_x", "javascript", "success"),
        ],
        "query": "Help me with TypeScript",
        "expected_route": ["agent_x"],  # Should generalize to programming
    },
]
```

#### 9.2.3 Cognitive Curiosity Benchmark

```python
# Test whether agents identify and fill knowledge gaps

CURIOSITY_SCENARIOS = [
    {
        "name": "Missing Context Detection",
        "user_statement": "Schedule a meeting with the team",
        "expected_gaps": ["which team?", "what time?", "what topic?"],
        "expected_question_priority": "HIGH",
    },
    {
        "name": "Implicit Preference Learning",
        "conversation_history": [
            "User: Turn on the lights",
            "Agent: Which lights?",
            "User: The living room ones",
            "User: Turn on the lights",
            "Agent: Done, living room lights are on",
        ],
        "expected_learning": "User's default lights are living room",
        "expected_question": None,  # Should not ask again
    },
]
```

### 9.3 Metrics We'll Track

| Metric | Description | Target |
|--------|-------------|--------|
| **Coordination Score** | How well agents share information (from MultiAgentBench) | > 80% |
| **Conflict Resolution Rate** | % of belief conflicts successfully resolved | > 90% |
| **Expertise Routing Accuracy** | % of queries routed to correct expert | > 85% |
| **Time-to-Synchronization** | How quickly new info propagates to all agents | < 500ms |
| **Belief Consistency** | % of agent beliefs that are mutually consistent | > 95% |
| **Memory Efficiency** | Task completion per token consumed | > 1.5x baseline |
| **Curiosity Question Quality** | % of questions that yield useful information | > 80% |

---

## 10. Implementation Timeline

### Phase 1: Shared Cognitive Working Memory (Weeks 1-2)

**Week 1:**
- [ ] Implement `SharedWorkingMemory` class
- [ ] Add conflict detection logic
- [ ] Implement attention weighting and decay
- [ ] Add capacity management (Miller's Law)
- [ ] Write unit tests

**Week 2:**
- [ ] Integrate with `AgentLoop`
- [ ] Update `DecisionEngine` to include shared observations
- [ ] Connect to existing `LearningChannel`
- [ ] Integration tests

### Phase 2: Parallel Execution (Weeks 3-4)

**Week 3:**
- [ ] Implement `ParallelCognitiveOrchestrator`
- [ ] Add barrier sync mode
- [ ] Implement fork-join mode
- [ ] Handle timeouts and failures
- [ ] Write unit tests

**Week 4:**
- [ ] Stress testing with many concurrent agents
- [ ] Performance optimization
- [ ] Integration with Phase 1

### Phase 3: Belief Reconciliation (Weeks 5-6)

**Week 5:**
- [ ] Implement `MultiAgentBeliefReconciliation`
- [ ] Add credibility weighting
- [ ] LLM-based conflict analysis
- [ ] Human clarification flagging
- [ ] Unit tests

**Week 6:**
- [ ] Integration with orchestrator
- [ ] Create belief conflict benchmark
- [ ] Tune reconciliation prompts
- [ ] Integration tests

### Phase 4: Transactive Memory (Weeks 7-8)

**Week 7:**
- [ ] Implement `TransactiveMemory` class
- [ ] Add expertise tracking
- [ ] Implement query routing
- [ ] Topic extraction
- [ ] Unit tests

**Week 8:**
- [ ] Integration with orchestrator
- [ ] Create transactive memory benchmark
- [ ] Tune routing algorithms
- [ ] Integration tests

### Phase 5: Metacognitive Reflection (Weeks 9-10)

**Week 9:**
- [ ] Implement `MetacognitiveReflectionService`
- [ ] Post-task analysis
- [ ] Expertise updates from reflection
- [ ] Learning storage
- [ ] Unit tests

**Week 10:**
- [ ] Full system integration
- [ ] End-to-end testing
- [ ] Benchmark runs
- [ ] Documentation

### Deliverables

1. **Code**: Full implementation of all 5 phases
2. **Tests**: Unit, integration, and stress tests
3. **Benchmarks**: Novel benchmark suite for cognitive multi-agent systems
4. **Documentation**: API docs, architecture guide, benchmark results
5. **Paper**: Technical writeup suitable for arXiv/conference submission

---

## Appendix A: Comparison with Other Frameworks

| Capability | LangGraph | CrewAI | AutoGen | draagon-ai (This Spec) |
|------------|-----------|--------|---------|------------------------|
| Parallel Execution | ✅ | ✅ | ✅ | ✅ |
| Shared Memory | Dict only | Basic | Blackboard | **Cognitive 4-layer** |
| Attention Weighting | ❌ | ❌ | ❌ | **✅** |
| Belief Tracking | ❌ | ❌ | ❌ | **✅** |
| Belief Reconciliation | ❌ | ❌ | ❌ | **✅** |
| Transactive Memory | ❌ | ❌ | ❌ | **✅** |
| Curiosity Engine | ❌ | ❌ | ❌ | **✅** |
| Opinion Formation | ❌ | ❌ | ❌ | **✅** |
| Metacognitive Reflection | ❌ | ❌ | Partial | **✅** |
| Memory Promotion | ❌ | ❌ | ❌ | **✅ (4-layer)** |
| Capacity Management | ❌ | ❌ | ❌ | **✅ (Miller's Law)** |

---

## Appendix B: Research References

1. [Intrinsic Memory Agents](https://arxiv.org/html/2508.08997v1) - Heterogeneous agent-specific memory (2025)
2. [Anthropic Multi-Agent System](https://www.anthropic.com/engineering/multi-agent-research-system) - 90% improvement with parallel agents
3. [Why Multi-Agent LLM Systems Fail](https://arxiv.org/pdf/2503.13657) - MAST taxonomy of failures
4. [MultiAgentBench](https://arxiv.org/html/2503.01935v1) - Collaboration/competition benchmark (ACL 2025)
5. [Metacognitive Learning](https://openreview.net/forum?id=4KhDd0Ozqe) - ICML 2025 position paper
6. [MemoryAgentBench](https://arxiv.org/html/2507.05257v1) - Memory evaluation benchmark
7. [Memory in LLM-MAS Survey](https://www.mongodb.com/company/blog/technical/why-multi-agent-systems-need-memory-engineering) - MongoDB engineering
8. [GAIA Benchmark](https://arxiv.org/abs/2311.12983) - General AI assistant evaluation
9. [HI-TOM](https://arxiv.org/abs/2310.16755) - Higher-order Theory of Mind benchmark
10. [Theory of Mind in LLMs](https://aclanthology.org/2025.acl-long.1522.pdf) - ToM assessment (ACL 2025)

---

**End of Specification**

# FR-002, FR-003, FR-004 Implementation Plan

**Features:**
- FR-002: Parallel Multi-Agent Orchestration
- FR-003: Multi-Agent Belief Reconciliation
- FR-004: Transactive Memory System

**Date:** 2025-12-30
**Status:** Ready for Implementation
**Total Estimate:** 38 days (FR-002: 15 days, FR-003: 13 days, FR-004: 10 days)

---

## Executive Summary

This plan implements the complete cognitive swarm architecture for draagon-ai, building on FR-001 (Shared Cognitive Working Memory). The three features are tightly integrated:

1. **FR-002** provides the parallel execution infrastructure
2. **FR-003** handles conflict reconciliation during parallel execution
3. **FR-004** enables intelligent agent routing based on expertise

All three will be implemented together to ensure seamless integration, with FR-002 as the foundation.

---

## Implementation Strategy

### Phase 1: FR-002 Core (Days 1-10)
Build the parallel orchestration foundation with stubs for FR-003 and FR-004.

### Phase 2: FR-004 Transactive Memory (Days 11-20)
Implement expertise tracking and routing, integrate with FR-002.

### Phase 3: FR-003 Belief Reconciliation (Days 21-33)
Implement conflict resolution, integrate with FR-002 barrier sync.

### Phase 4: Integration & Testing (Days 34-38)
End-to-end testing, benchmarks, documentation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 ParallelCognitiveOrchestrator                    │
│                                                                  │
│  ┌────────────────┐   ┌─────────────────┐   ┌────────────────┐ │
│  │  FR-004        │   │  FR-002         │   │  FR-003        │ │
│  │  Transactive   │──▶│  Orchestration  │──▶│  Belief        │ │
│  │  Memory        │   │  Modes          │   │  Reconciliation│ │
│  │                │   │                 │   │                │ │
│  │ route_query()  │   │ orchestrate()   │   │ reconcile()    │ │
│  └────────────────┘   └─────────────────┘   └────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│                   ┌─────────────────────┐                        │
│                   │  FR-001             │                        │
│                   │  SharedWorkingMemory│                        │
│                   │                     │                        │
│                   │  add_observation()  │                        │
│                   │  get_conflicts()    │                        │
│                   └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

### New Files to Create

```
src/draagon_ai/
├── orchestration/
│   ├── parallel_orchestrator.py        # FR-002 main implementation
│   ├── result_merger.py                # FR-002 result merging strategies
│   └── transactive_memory.py           # FR-004 expertise tracking
├── cognition/
│   └── multi_agent_reconciliation.py   # FR-003 belief reconciliation
└── llm/
    └── providers.py                     # LLMProvider protocol (if not exists)

tests/
├── orchestration/
│   ├── test_parallel_orchestrator.py   # FR-002 tests
│   ├── test_result_merger.py           # FR-002 result merging tests
│   └── test_transactive_memory.py      # FR-004 tests
└── cognition/
    └── test_multi_agent_reconciliation.py  # FR-003 tests
```

---

## Phase 1: FR-002 Core Implementation (Days 1-10)

### Day 1-2: Module Skeleton & Data Structures

**File:** `src/draagon_ai/orchestration/parallel_orchestrator.py`

```python
"""Parallel multi-agent orchestration with cognitive coordination.

Based on research:
- Anthropic Multi-Agent Research (90.2% improvement with 3-5 agents)
- MAST Framework (34.7% failures from misalignment)
- MultiAgentBench (coordination prevents failures)
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Any
import asyncio
import logging

from .multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    AgentSpec,
    TaskContext,
    AgentResult,
    OrchestratorResult,
    AgentExecutor,
)
from .shared_memory import SharedWorkingMemory, SharedObservation
from .transactive_memory import TransactiveMemory  # FR-004
from ..cognition.multi_agent_reconciliation import (  # FR-003
    MultiAgentBeliefReconciliation,
    ReconciliationResult,
)

logger = logging.getLogger(__name__)


class ParallelOrchestrationMode(str, Enum):
    """Synchronization modes for parallel execution."""

    FORK_JOIN = "fork_join"          # All agents start together, sync at end only
    BARRIER_SYNC = "barrier_sync"    # Periodic sync barriers every N iterations
    STREAMING = "streaming"          # Continuous sync via shared memory


class ResultMergeStrategy(str, Enum):
    """How to merge results from multiple agents."""

    ALL_OUTPUTS = "all_outputs"              # Return all, host app merges
    HIGHEST_CONFIDENCE = "highest_confidence"  # Use result with highest confidence
    CONCATENATE = "concatenate"              # Join text outputs with attribution
    CUSTOM = "custom"                        # User-provided merger function


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel agent execution.

    Defaults based on Anthropic research (3-5 agents optimal).
    """

    # Execution control
    max_concurrent_agents: int = 5
    sync_mode: ParallelOrchestrationMode = ParallelOrchestrationMode.BARRIER_SYNC
    timeout_per_agent_seconds: float = 60.0
    allow_early_termination: bool = True

    # Synchronization (BARRIER_SYNC mode)
    sync_interval_iterations: int = 3  # Sync every N iterations

    # Result merging
    merge_strategy: ResultMergeStrategy = ResultMergeStrategy.ALL_OUTPUTS
    custom_merger: Any = None  # Callable[[list[AgentResult]], Any]

    # Agent state isolation
    agents_read_only_observations: bool = True  # Agents can only ADD, not modify


class ParallelCognitiveOrchestrator(MultiAgentOrchestrator):
    """Parallel multi-agent orchestrator with cognitive coordination.

    Extends MultiAgentOrchestrator to provide:
    - Parallel agent execution (FORK_JOIN, BARRIER_SYNC, STREAMING)
    - Shared cognitive working memory (FR-001)
    - Belief reconciliation on conflicts (FR-003)
    - Expertise-based agent routing (FR-004)

    Example:
        ```python
        from draagon_ai.orchestration import (
            ParallelCognitiveOrchestrator,
            ParallelExecutionConfig,
            ParallelOrchestrationMode,
        )
        from draagon_ai.orchestration.transactive_memory import TransactiveMemory
        from draagon_ai.cognition import MultiAgentBeliefReconciliation

        # Create services
        transactive_memory = TransactiveMemory()
        belief_service = MultiAgentBeliefReconciliation(llm)

        # Configure orchestrator
        config = ParallelExecutionConfig(
            max_concurrent_agents=5,
            sync_mode=ParallelOrchestrationMode.BARRIER_SYNC,
        )

        orchestrator = ParallelCognitiveOrchestrator(
            config=config,
            transactive_memory=transactive_memory,
            belief_service=belief_service,
        )

        # Execute agents
        agents = [
            AgentSpec(agent_id="researcher", role=AgentRole.RESEARCHER),
            AgentSpec(agent_id="critic", role=AgentRole.CRITIC),
            AgentSpec(agent_id="executor", role=AgentRole.EXECUTOR),
        ]

        result = await orchestrator.orchestrate_parallel(
            agents=agents,
            context=TaskContext(query="Analyze market trends"),
            agent_executor=my_executor,
        )
        ```

    Research Foundation:
        - Anthropic Multi-Agent Research: 3-5 agents optimal, 90.2% improvement
        - MAST Framework: 34.7% failures from inter-agent misalignment
        - MultiAgentBench: Coordination prevents 22.3% of failures
    """

    def __init__(
        self,
        config: ParallelExecutionConfig | None = None,
        transactive_memory: TransactiveMemory | None = None,
        belief_service: MultiAgentBeliefReconciliation | None = None,
        learning_channel: Any = None,
    ):
        """Initialize parallel orchestrator.

        Args:
            config: Execution configuration (uses defaults if None)
            transactive_memory: Expertise routing service (FR-004, optional)
            belief_service: Conflict reconciliation service (FR-003, optional)
            learning_channel: Learning broadcast channel
        """
        super().__init__(learning_channel=learning_channel)

        self.config = config or ParallelExecutionConfig()
        self.transactive_memory = transactive_memory
        self.belief_service = belief_service

        logger.info(
            f"Initialized ParallelCognitiveOrchestrator: "
            f"mode={self.config.sync_mode.value}, "
            f"max_agents={self.config.max_concurrent_agents}, "
            f"has_transactive={transactive_memory is not None}, "
            f"has_beliefs={belief_service is not None}"
        )
```

### Day 3-4: FORK_JOIN Mode (Simplest)

```python
    async def orchestrate_parallel(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        agent_executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Execute agents in parallel with cognitive coordination.

        Steps:
        1. Create shared working memory for task
        2. Route agents by expertise (if TransactiveMemory available)
        3. Select top N agents (max_concurrent_agents)
        4. Execute based on sync_mode (FORK_JOIN, BARRIER_SYNC, or STREAMING)
        5. Reconcile conflicts (if BeliefReconciliation available)
        6. Update expertise based on results (if TransactiveMemory available)
        7. Merge results based on merge_strategy

        Args:
            agents: List of agents to orchestrate
            context: Task context (will inject shared memory)
            agent_executor: Function to execute each agent

        Returns:
            Orchestration result with all agent outputs
        """
        logger.info(
            f"Starting parallel orchestration: task={context.task_id}, "
            f"mode={self.config.sync_mode.value}, agents={len(agents)}"
        )

        result = OrchestratorResult(
            task_id=context.task_id,
            success=True,
            started_at=datetime.now(),
        )

        # Step 1: Create shared working memory
        shared_memory = SharedWorkingMemory(task_id=context.task_id)
        context.working_memory["__shared__"] = shared_memory

        # Step 2: Route agents by expertise (FR-004)
        if self.transactive_memory:
            agents = await self._route_by_expertise(context.query, agents)

        # Step 3: Select top N agents
        selected_agents = agents[: self.config.max_concurrent_agents]
        logger.info(f"Selected {len(selected_agents)} agents for parallel execution")

        # Step 4: Execute based on sync mode
        if self.config.sync_mode == ParallelOrchestrationMode.FORK_JOIN:
            agent_results = await self._fork_join_execution(
                selected_agents, context, shared_memory, agent_executor
            )
        elif self.config.sync_mode == ParallelOrchestrationMode.BARRIER_SYNC:
            agent_results = await self._barrier_sync_execution(
                selected_agents, context, shared_memory, agent_executor
            )
        elif self.config.sync_mode == ParallelOrchestrationMode.STREAMING:
            agent_results = await self._streaming_execution(
                selected_agents, context, shared_memory, agent_executor
            )
        else:
            raise ValueError(f"Unknown sync mode: {self.config.sync_mode}")

        result.agent_results = agent_results

        # Step 5: Reconcile conflicts (FR-003)
        conflicts = await shared_memory.get_conflicts()
        if conflicts and self.belief_service:
            reconciliation_result = await self._reconcile_conflicts(
                shared_memory, conflicts
            )
            # Store reconciliation result for reflection
            context.working_memory["__reconciliation__"] = reconciliation_result

        # Step 6: Update expertise (FR-004)
        if self.transactive_memory:
            await self._update_expertise(context.query, agent_results)

        # Step 7: Merge results
        result.final_output = await self._merge_results(agent_results)

        # Finalize
        result.completed_at = datetime.now()
        result.success = any(r.success for r in agent_results)
        context.completed_at = datetime.now()
        result.final_context = context

        # Collect learnings
        for agent_result in agent_results:
            result.learnings.extend(agent_result.learnings)

        # Broadcast learnings
        await self._broadcast_learnings(result.learnings)

        logger.info(
            f"Parallel orchestration complete: task={context.task_id}, "
            f"success={result.success}, duration={result.duration_ms:.1f}ms, "
            f"successful_agents={sum(1 for r in agent_results if r.success)}/{len(agent_results)}"
        )

        return result

    async def _fork_join_execution(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        shared_memory: SharedWorkingMemory,
        agent_executor: AgentExecutor,
    ) -> list[AgentResult]:
        """Execute all agents concurrently, sync at end only (simplest mode).

        FORK_JOIN is the simplest parallel mode:
        - All agents start simultaneously
        - Each runs independently
        - No mid-execution synchronization
        - Conflict reconciliation happens after all complete

        Args:
            agents: Agents to execute
            context: Task context with shared memory
            shared_memory: Shared working memory instance
            agent_executor: Execution function

        Returns:
            List of agent results
        """
        logger.info(f"FORK_JOIN: Launching {len(agents)} agents in parallel")

        # Create tasks for all agents
        tasks = [
            self._run_agent_with_shared_memory(
                agent, context, shared_memory, agent_executor
            )
            for agent in agents
        ]

        # Run all concurrently (asyncio.gather collects all results)
        agent_results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            f"FORK_JOIN: All {len(agents)} agents completed "
            f"(successful: {sum(1 for r in agent_results if r.success)})"
        )

        return agent_results
```

### Day 5-6: BARRIER_SYNC Mode

```python
    async def _barrier_sync_execution(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        shared_memory: SharedWorkingMemory,
        agent_executor: AgentExecutor,
    ) -> list[AgentResult]:
        """Execute with periodic synchronization barriers.

        BARRIER_SYNC provides periodic coordination:
        - Agents run in parallel
        - Every sync_interval_iterations, perform sync:
          * Apply attention decay to shared memory
          * Check for conflicts
          * Log warning if >3 conflicts (coordination issue)
        - Conflicts reconciled asynchronously (don't pause agents)
        - Final reconciliation after all agents complete

        This prevents inter-agent drift without blocking parallelism.

        Args:
            agents: Agents to execute
            context: Task context with shared memory
            shared_memory: Shared working memory instance
            agent_executor: Execution function

        Returns:
            List of agent results
        """
        logger.info(
            f"BARRIER_SYNC: Launching {len(agents)} agents with sync every "
            f"{self.config.sync_interval_iterations} iterations"
        )

        # Create tasks for all agents
        tasks = [
            asyncio.create_task(
                self._run_agent_with_shared_memory(
                    agent, context, shared_memory, agent_executor
                )
            )
            for agent in agents
        ]

        # Track pending tasks
        pending = set(tasks)
        completed_results = []
        iteration = 0

        # Run with periodic synchronization
        while pending:
            # Wait for sync interval or agent completion
            done, pending = await asyncio.wait(
                pending,
                timeout=None if iteration == 0 else 1.0,  # Check every second after first
                return_when=asyncio.FIRST_COMPLETED if iteration == 0 else asyncio.ALL_COMPLETED,
            )

            # Collect completed results
            for task in done:
                try:
                    result = await task
                    completed_results.append(result)
                except Exception as e:
                    logger.error(f"Agent task failed: {e}")
                    # Create failed result
                    completed_results.append(
                        AgentResult(
                            agent_id="unknown",
                            success=False,
                            error=str(e),
                        )
                    )

            iteration += 1

            # Periodic sync barrier
            if iteration % self.config.sync_interval_iterations == 0 and pending:
                logger.debug(f"BARRIER_SYNC: Iteration {iteration} - sync barrier")

                # Apply attention decay (non-blocking for agents)
                await shared_memory.apply_attention_decay()

                # Check conflicts (log warning if many)
                conflicts = await shared_memory.get_conflicts()
                if len(conflicts) > 3:
                    logger.warning(
                        f"BARRIER_SYNC: {len(conflicts)} conflicts detected at iteration {iteration}. "
                        f"Agents may be producing contradictory observations."
                    )

                # Async reconciliation (don't block agents)
                if conflicts and self.belief_service:
                    asyncio.create_task(
                        self._reconcile_conflicts(shared_memory, conflicts)
                    )

        logger.info(
            f"BARRIER_SYNC: All {len(agents)} agents completed after {iteration} iterations "
            f"(successful: {sum(1 for r in completed_results if r.success)})"
        )

        return completed_results
```

### Day 7: STREAMING Mode

```python
    async def _streaming_execution(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        shared_memory: SharedWorkingMemory,
        agent_executor: AgentExecutor,
    ) -> list[AgentResult]:
        """Execute with continuous shared memory synchronization.

        STREAMING mode provides real-time coordination:
        - Agents continuously read latest observations
        - No explicit barriers
        - Attention decay on every observation add
        - Best for highly coupled tasks where agents need immediate context

        Implementation:
        - Uses SharedWorkingMemory.add_observation() which is already concurrent-safe
        - Agents naturally see latest observations via get_context_for_agent()
        - Attention decay happens automatically per observation

        Args:
            agents: Agents to execute
            context: Task context with shared memory
            shared_memory: Shared working memory instance
            agent_executor: Execution function

        Returns:
            List of agent results
        """
        logger.info(
            f"STREAMING: Launching {len(agents)} agents with continuous sync"
        )

        # In STREAMING mode, agents continuously read from shared memory
        # The SharedWorkingMemory class already provides this via:
        # - add_observation() adds new observations immediately
        # - get_context_for_agent() retrieves latest observations
        # - Concurrent access is safe via asyncio.Lock

        # Create tasks for all agents
        tasks = [
            self._run_agent_with_shared_memory(
                agent, context, shared_memory, agent_executor
            )
            for agent in agents
        ]

        # Run all concurrently
        agent_results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            f"STREAMING: All {len(agents)} agents completed "
            f"(successful: {sum(1 for r in agent_results if r.success)})"
        )

        return agent_results
```

### Day 8: Agent Execution with Shared Memory

```python
    async def _run_agent_with_shared_memory(
        self,
        agent: AgentSpec,
        context: TaskContext,
        shared_memory: SharedWorkingMemory,
        agent_executor: AgentExecutor,
    ) -> AgentResult:
        """Run a single agent with shared working memory integration.

        Steps:
        1. Execute agent with timeout (reuse parent class logic)
        2. Capture agent output as observation
        3. Add observation to shared memory
        4. Return result

        Observation creation:
        - content: agent output (stringified)
        - source_agent_id: agent.agent_id
        - attention_weight: 0.8 if success, 0.3 if failure
        - confidence: 1.0 if success, 0.0 if failure
        - is_belief_candidate: True if agent succeeded
        - belief_type: Based on agent role (RESEARCHER=FACT, CRITIC=INSIGHT, etc.)

        Args:
            agent: Agent specification
            context: Task context with shared memory
            shared_memory: Shared working memory instance
            agent_executor: Execution function

        Returns:
            Agent result
        """
        # Execute with timeout and retry (reuse parent class logic)
        result = await self._execute_with_retry(agent, context, agent_executor)

        # Capture output as observation
        if result.output is not None:
            # Determine belief type from agent role
            belief_type_map = {
                AgentRole.RESEARCHER: "FACT",
                AgentRole.CRITIC: "INSIGHT",
                AgentRole.EXECUTOR: "SKILL",
                AgentRole.PLANNER: "INSTRUCTION",
                AgentRole.SPECIALIST: "FACT",
                AgentRole.PRIMARY: None,  # Primary agent doesn't add beliefs
            }

            belief_type = belief_type_map.get(agent.role)

            # Add observation to shared memory
            observation = await shared_memory.add_observation(
                content=str(result.output),
                source_agent_id=agent.agent_id,
                attention_weight=0.8 if result.success else 0.3,
                confidence=1.0 if result.success else 0.0,
                is_belief_candidate=result.success and belief_type is not None,
                belief_type=belief_type,
            )

            logger.debug(
                f"Added observation from {agent.agent_id}: "
                f"success={result.success}, belief_candidate={observation.is_belief_candidate}"
            )

        return result
```

### Day 9-10: Result Merging & Conflict Reconciliation

**File:** `src/draagon_ai/orchestration/result_merger.py`

```python
"""Result merging strategies for parallel agent execution.

Addresses FR-002 review gap: "Result merging strategy not specified"
"""

from dataclasses import dataclass
from typing import Any, Callable
import logging

from .multi_agent_orchestrator import AgentResult

logger = logging.getLogger(__name__)


class ResultMerger:
    """Merge results from multiple parallel agents.

    Strategies:
    - ALL_OUTPUTS: Return all agent results (host app merges)
    - HIGHEST_CONFIDENCE: Select result with highest confidence
    - CONCATENATE: Join text outputs with agent attribution
    - CUSTOM: User-provided merger function
    """

    @staticmethod
    async def merge_all_outputs(results: list[AgentResult]) -> dict[str, Any]:
        """Return all results, let host application merge.

        This is the default strategy. It preserves maximum information
        and allows the host application to decide how to combine outputs.

        Args:
            results: Agent results to merge

        Returns:
            Dict with all agent outputs keyed by agent_id
        """
        return {
            result.agent_id: result.output
            for result in results
            if result.success and result.output is not None
        }

    @staticmethod
    async def merge_highest_confidence(results: list[AgentResult]) -> Any:
        """Select output from result with highest confidence.

        Assumes agent results have a 'confidence' field in output dict.
        Falls back to first successful result if no confidence found.

        Args:
            results: Agent results to merge

        Returns:
            Output from highest-confidence agent
        """
        successful = [r for r in results if r.success and r.output is not None]

        if not successful:
            return None

        # Try to extract confidence from output
        def get_confidence(result: AgentResult) -> float:
            if isinstance(result.output, dict):
                return result.output.get("confidence", 0.5)
            return 0.5  # Default neutral confidence

        best = max(successful, key=get_confidence)
        logger.info(
            f"Selected output from {best.agent_id} "
            f"(confidence: {get_confidence(best)})"
        )

        return best.output

    @staticmethod
    async def merge_concatenate(results: list[AgentResult]) -> str:
        """Concatenate text outputs with agent attribution.

        Useful for research/analysis tasks where multiple perspectives
        are valuable.

        Args:
            results: Agent results to merge

        Returns:
            Concatenated string with agent attribution
        """
        successful = [r for r in results if r.success and r.output is not None]

        if not successful:
            return ""

        lines = []
        for result in successful:
            lines.append(f"[{result.agent_id}]")
            lines.append(str(result.output))
            lines.append("")  # Blank line

        return "\n".join(lines)

    @staticmethod
    async def merge_custom(
        results: list[AgentResult],
        merger_fn: Callable[[list[AgentResult]], Any],
    ) -> Any:
        """Use custom merger function provided by user.

        Args:
            results: Agent results to merge
            merger_fn: User-provided merger function

        Returns:
            Merged output from custom function
        """
        return merger_fn(results)
```

Back in `parallel_orchestrator.py`:

```python
    async def _merge_results(self, agent_results: list[AgentResult]) -> Any:
        """Merge agent results based on configured strategy.

        Args:
            agent_results: Results from all agents

        Returns:
            Merged output
        """
        from .result_merger import ResultMerger

        merger = ResultMerger()

        if self.config.merge_strategy == ResultMergeStrategy.ALL_OUTPUTS:
            return await merger.merge_all_outputs(agent_results)
        elif self.config.merge_strategy == ResultMergeStrategy.HIGHEST_CONFIDENCE:
            return await merger.merge_highest_confidence(agent_results)
        elif self.config.merge_strategy == ResultMergeStrategy.CONCATENATE:
            return await merger.merge_concatenate(agent_results)
        elif self.config.merge_strategy == ResultMergeStrategy.CUSTOM:
            if self.config.custom_merger is None:
                raise ValueError("CUSTOM merge strategy requires custom_merger function")
            return await merger.merge_custom(agent_results, self.config.custom_merger)
        else:
            raise ValueError(f"Unknown merge strategy: {self.config.merge_strategy}")

    async def _reconcile_conflicts(
        self,
        shared_memory: SharedWorkingMemory,
        conflicts: list[tuple[Any, Any, str]],
    ) -> Any:
        """Trigger belief reconciliation for detected conflicts.

        This is a stub for FR-003 integration. Will be implemented in Phase 3.

        Args:
            shared_memory: Shared working memory with conflicts
            conflicts: List of (obs_a, obs_b, reason) tuples

        Returns:
            ReconciliationResult (or None if no belief service)
        """
        if not self.belief_service:
            logger.info(f"No belief service configured, skipping reconciliation of {len(conflicts)} conflicts")
            return None

        logger.info(f"Triggering belief reconciliation for {len(conflicts)} conflicts")

        # Get all belief candidates from shared memory
        candidates = await shared_memory.get_belief_candidates()

        # Get agent credibilities from transactive memory (if available)
        agent_credibilities = {}
        if self.transactive_memory:
            # TODO: Extract credibilities from transactive memory
            pass

        # Call FR-003 reconciliation service
        reconciliation_result = await self.belief_service.reconcile_multi_agent(
            observations=candidates,
            agent_credibilities=agent_credibilities,
        )

        logger.info(
            f"Reconciliation complete: resolved={reconciliation_result.resolved}, "
            f"confidence={reconciliation_result.confidence}"
        )

        return reconciliation_result
```

---

## Phase 2: FR-004 Transactive Memory (Days 11-20)

### Day 11-12: Core Data Structures

**File:** `src/draagon_ai/orchestration/transactive_memory.py`

```python
"""Transactive memory system for expertise tracking.

Based on research:
- Wegner (1987): Transactive memory in human groups
- Memory in LLM-MAS Survey: "Who knows what" critical for coordination
- MultiAgentBench: Expertise routing reduces strategic silence failures

FR-004: Transactive Memory System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import logging

from .multi_agent_orchestrator import AgentSpec

logger = logging.getLogger(__name__)


@dataclass
class ExpertiseEntry:
    """Track agent expertise in a specific topic.

    Attributes:
        topic: Topic name (e.g., "weather", "scheduling", "python")
        confidence: Expertise confidence (0-1, default 0.5 neutral prior)
        success_count: Number of successful executions
        failure_count: Number of failed executions
        last_updated: Timestamp of last update
    """

    topic: str
    confidence: float = 0.5  # Neutral prior
    success_count: int = 0
    failure_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Success rate (0-1).

        Returns:
            success_count / (success_count + failure_count)
            or 0.5 if no data
        """
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Prior
        return self.success_count / total


class TransactiveMemory:
    """Track "who knows what" for intelligent agent routing.

    This implements Wegner's (1987) transactive memory theory for
    multi-agent systems: agents know which other agents are experts
    in which topics, enabling efficient task routing.

    Example:
        ```python
        from draagon_ai.orchestration.transactive_memory import TransactiveMemory

        # Create transactive memory
        tm = TransactiveMemory()

        # Update expertise based on task outcomes
        await tm.update_expertise("agent_weather", "weather", success=True)
        await tm.update_expertise("agent_calendar", "scheduling", success=True)

        # Route query to experts
        agents = [weather_agent, calendar_agent, generic_agent]
        routed = await tm.route_query(
            "What's the weather for my meeting tomorrow?",
            agents
        )
        # Returns: [weather_agent, calendar_agent, generic_agent]
        #          (sorted by expertise score)

        # Query experts
        experts = await tm.get_experts("weather", min_confidence=0.6)
        # Returns: [("agent_weather", 0.85), ...]

        # Natural language query
        summary = await tm.who_knows_about("weather")
        # Returns: "For 'weather': agent_weather (confidence: 85%)"
        ```

    Research Foundation:
        - Wegner (1987): Transactive memory in human groups
        - Anthropic Multi-Agent Research: Detailed task descriptions prevent duplication
        - MultiAgentBench: Expertise routing reduces strategic silence
    """

    def __init__(
        self,
        embedding_provider: Any = None,
        llm_provider: Any = None,
    ):
        """Initialize transactive memory.

        Args:
            embedding_provider: For semantic topic matching (Phase 2, optional)
            llm_provider: For LLM-based topic extraction (Phase 2, optional)
        """
        # Storage: agent_id -> {topic -> ExpertiseEntry}
        self._expertise: dict[str, dict[str, ExpertiseEntry]] = {}

        # Topic hierarchy for generalization
        # e.g., "python" -> ["programming", "software"]
        self._topic_hierarchy: dict[str, list[str]] = {}

        # Optional providers
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider

        logger.info(
            f"Initialized TransactiveMemory "
            f"(embeddings={embedding_provider is not None}, "
            f"llm={llm_provider is not None})"
        )
```

### Day 13-14: Expertise Tracking

```python
    async def update_expertise(
        self,
        agent_id: str,
        topic: str,
        success: bool,
    ) -> None:
        """Update agent expertise based on task outcome.

        On success:
        - Increment success_count
        - Increase confidence by 0.1 (capped at 1.0)

        On failure:
        - Increment failure_count
        - Decrease confidence by 0.15 (floored at 0.0)

        Also updates parent topics in hierarchy with decay factor (0.5x).

        Args:
            agent_id: Agent whose expertise to update
            topic: Topic of the task
            success: Whether the agent succeeded
        """
        if agent_id not in self._expertise:
            self._expertise[agent_id] = {}

        # Update direct topic
        if topic not in self._expertise[agent_id]:
            self._expertise[agent_id][topic] = ExpertiseEntry(topic=topic)

        entry = self._expertise[agent_id][topic]

        if success:
            entry.success_count += 1
            entry.confidence = min(1.0, entry.confidence + 0.1)
        else:
            entry.failure_count += 1
            entry.confidence = max(0.0, entry.confidence - 0.15)

        entry.last_updated = datetime.now()

        logger.debug(
            f"Updated expertise: {agent_id}/{topic} "
            f"success={success}, confidence={entry.confidence:.2f}, "
            f"success_rate={entry.success_rate:.2f}"
        )

        # Update parent topics with decay
        if topic in self._topic_hierarchy:
            for parent_topic in self._topic_hierarchy[topic]:
                if parent_topic not in self._expertise[agent_id]:
                    self._expertise[agent_id][parent_topic] = ExpertiseEntry(
                        topic=parent_topic
                    )

                parent_entry = self._expertise[agent_id][parent_topic]

                if success:
                    parent_entry.success_count += 1
                    parent_entry.confidence = min(
                        1.0, parent_entry.confidence + 0.05  # Half of direct update
                    )
                else:
                    parent_entry.failure_count += 1
                    parent_entry.confidence = max(
                        0.0, parent_entry.confidence - 0.075  # Half of direct update
                    )

                parent_entry.last_updated = datetime.now()

                logger.debug(
                    f"Updated parent topic: {agent_id}/{parent_topic} "
                    f"(from {topic}) confidence={parent_entry.confidence:.2f}"
                )
```

### Day 15-16: Query Routing

```python
    async def route_query(
        self,
        query: str,
        available_agents: list[AgentSpec],
    ) -> list[AgentSpec]:
        """Route query to agents ranked by expertise.

        Steps:
        1. Extract topics from query
        2. Compute expertise score for each agent
        3. Sort agents by score (highest first)
        4. Return sorted list

        Args:
            query: Natural language query
            available_agents: Agents to rank

        Returns:
            Agents sorted by expertise (highest first)
        """
        # Extract topics from query
        topics = await self._extract_topics(query)
        logger.debug(f"Extracted topics from query: {topics}")

        if not topics:
            # No topics extracted, return agents unchanged
            return available_agents

        # Score each agent
        agent_scores = []
        for agent in available_agents:
            score = await self._compute_expertise_score(agent.agent_id, topics)
            agent_scores.append((agent, score))
            logger.debug(f"Agent {agent.agent_id} expertise score: {score:.3f}")

        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        sorted_agents = [agent for agent, score in agent_scores]

        logger.info(
            f"Routed query by expertise: "
            f"top_agent={sorted_agents[0].agent_id if sorted_agents else 'none'}"
        )

        return sorted_agents

    async def _extract_topics(self, query: str) -> list[str]:
        """Extract relevant topics from natural language query.

        Phase 1 (simple): Keyword extraction
        - Split on whitespace
        - Filter stopwords
        - Keep words with length >= 4

        Phase 2 (LLM): Structured topic extraction
        - Use LLM to identify topics
        - Returns semantic topics (e.g., "weather" + "scheduling")

        Args:
            query: Natural language query

        Returns:
            List of relevant topics
        """
        # Phase 1: Simple keyword extraction
        stopwords = {
            "what", "when", "where", "who", "why", "how",
            "the", "is", "are", "was", "were", "been",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "should", "could", "may", "might",
            "can", "for", "with", "about", "from", "this", "that",
        }

        words = query.lower().split()
        topics = [
            word.strip(",.?!")
            for word in words
            if len(word) >= 4 and word.lower() not in stopwords
        ]

        logger.debug(f"Extracted {len(topics)} topics (Phase 1): {topics}")

        # TODO Phase 2: LLM-based topic extraction
        if self.llm_provider:
            # Use LLM to extract semantic topics
            pass

        return topics[:5]  # Limit to top 5 topics

    async def _compute_expertise_score(
        self,
        agent_id: str,
        topics: list[str],
    ) -> float:
        """Compute agent's expertise score for given topics.

        Scoring:
        - Exact match: full confidence
        - Partial match (substring): 0.7x confidence
        - Parent topic match: 0.7x parent confidence
        - No match: 0.5 (neutral prior)

        Average across all topics.

        Args:
            agent_id: Agent to score
            topics: Topics from query

        Returns:
            Expertise score (0-1)
        """
        if agent_id not in self._expertise or not topics:
            return 0.5  # Neutral prior

        agent_expertise = self._expertise[agent_id]
        scores = []

        for topic in topics:
            # Check for exact match
            if topic in agent_expertise:
                scores.append(agent_expertise[topic].confidence)
                continue

            # Check for partial match (substring)
            partial_match = None
            for expert_topic, entry in agent_expertise.items():
                if topic in expert_topic or expert_topic in topic:
                    if partial_match is None or entry.confidence > partial_match.confidence:
                        partial_match = entry

            if partial_match:
                scores.append(partial_match.confidence * 0.7)
                continue

            # Check for parent topic match
            if topic in self._topic_hierarchy:
                for parent_topic in self._topic_hierarchy[topic]:
                    if parent_topic in agent_expertise:
                        scores.append(agent_expertise[parent_topic].confidence * 0.7)
                        break
                else:
                    scores.append(0.5)  # No match
            else:
                scores.append(0.5)  # No match

        # Average across topics
        return sum(scores) / len(scores) if scores else 0.5
```

### Day 17-18: Expert Queries & Natural Language

```python
    async def get_experts(
        self,
        topic: str,
        min_confidence: float = 0.6,
    ) -> list[tuple[str, float]]:
        """Get agents who are experts in a topic.

        Args:
            topic: Topic to query
            min_confidence: Minimum confidence threshold

        Returns:
            List of (agent_id, confidence) sorted by confidence descending
        """
        experts = []

        for agent_id, agent_expertise in self._expertise.items():
            if topic in agent_expertise:
                entry = agent_expertise[topic]
                if entry.confidence >= min_confidence:
                    experts.append((agent_id, entry.confidence))

        # Sort by confidence descending
        experts.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Found {len(experts)} experts for '{topic}' (min={min_confidence})")

        return experts

    async def who_knows_about(self, topic: str) -> str:
        """Natural language response to "who knows about X?"

        Args:
            topic: Topic to query

        Returns:
            Natural language string listing experts
        """
        experts = await self.get_experts(topic, min_confidence=0.0)  # Get all

        if not experts:
            return f"No agents have demonstrated expertise in '{topic}' yet."

        # Take top 3
        top_experts = experts[:3]

        expert_strs = [
            f"{agent_id} (confidence: {int(conf * 100)}%)"
            for agent_id, conf in top_experts
        ]

        return f"For '{topic}': {', '.join(expert_strs)}"

    def get_expertise_summary(self) -> dict[str, dict[str, float]]:
        """Get full expertise map for debugging/visualization.

        Returns:
            Dict of agent_id -> {topic -> confidence}
        """
        summary = {}
        for agent_id, agent_expertise in self._expertise.items():
            summary[agent_id] = {
                topic: entry.confidence
                for topic, entry in agent_expertise.items()
            }
        return summary
```

### Day 19-20: Integration with FR-002

Back in `parallel_orchestrator.py`:

```python
    async def _route_by_expertise(
        self,
        query: str,
        agents: list[AgentSpec],
    ) -> list[AgentSpec]:
        """Route agents by expertise using TransactiveMemory.

        Args:
            query: Task query
            agents: Available agents

        Returns:
            Agents sorted by expertise (highest first)
        """
        if not self.transactive_memory:
            return agents

        routed_agents = await self.transactive_memory.route_query(query, agents)

        logger.info(
            f"Routed {len(agents)} agents by expertise. "
            f"Top agent: {routed_agents[0].agent_id if routed_agents else 'none'}"
        )

        return routed_agents

    async def _update_expertise(
        self,
        query: str,
        agent_results: list[AgentResult],
    ) -> None:
        """Update transactive memory based on agent execution results.

        Args:
            query: Task query (for topic extraction)
            agent_results: Results from all agents
        """
        if not self.transactive_memory:
            return

        # Extract topics from query
        topics = await self.transactive_memory._extract_topics(query)

        if not topics:
            logger.debug("No topics extracted, skipping expertise update")
            return

        # Update expertise for each agent
        for result in agent_results:
            for topic in topics:
                await self.transactive_memory.update_expertise(
                    agent_id=result.agent_id,
                    topic=topic,
                    success=result.success,
                )

        logger.info(
            f"Updated expertise for {len(agent_results)} agents on topics: {topics}"
        )
```

---

## Phase 3: FR-003 Belief Reconciliation (Days 21-33)

### Day 21-23: Core Reconciliation Logic

**File:** `src/draagon_ai/cognition/multi_agent_reconciliation.py`

```python
"""Multi-agent belief reconciliation service.

Based on research:
- MAST Framework: Inter-agent misalignment causes 34.7% of failures
- MemoryAgentBench: Current systems fail at conflict resolution (45%)
- Epistemic Logic: BDI (Beliefs, Desires, Intentions) architectures

FR-003: Multi-Agent Belief Reconciliation
"""

from dataclasses import dataclass, field
from typing import Any
import logging

from ..orchestration.shared_memory import SharedObservation
from .beliefs import AgentBelief, BeliefType  # Existing belief structures

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result from multi-agent belief reconciliation.

    Attributes:
        resolved: Was the conflict resolved?
        consolidated_belief: Synthesized belief (if resolved)
        confidence: Confidence in consolidated belief (0-1)
        needs_human_clarification: Should we ask the user?
        clarification_question: Question to ask user (if needed)
        observations_considered: Observation IDs used (audit trail)
        reasoning: Why this conclusion was reached
    """

    resolved: bool
    consolidated_belief: AgentBelief | None = None
    confidence: float = 0.0
    needs_human_clarification: bool = False
    clarification_question: str | None = None
    observations_considered: list[str] = field(default_factory=list)
    reasoning: str = ""


class MultiAgentBeliefReconciliation:
    """Reconcile conflicting beliefs from multiple agents.

    This service takes observations from different agents (stored in
    SharedWorkingMemory) and synthesizes consolidated beliefs or
    flags unresolvable conflicts for human clarification.

    Example:
        ```python
        from draagon_ai.cognition import MultiAgentBeliefReconciliation

        # Create reconciliation service
        reconciler = MultiAgentBeliefReconciliation(llm=llm_provider)

        # Get observations from shared memory
        observations = await shared_memory.get_belief_candidates()

        # Get agent credibilities from transactive memory
        agent_credibilities = {
            "agent_a": 0.9,  # High credibility
            "agent_b": 0.4,  # Low credibility
        }

        # Reconcile conflicts
        result = await reconciler.reconcile_multi_agent(
            observations=observations,
            agent_credibilities=agent_credibilities,
        )

        if result.resolved:
            print(f"Resolved: {result.consolidated_belief}")
        elif result.needs_human_clarification:
            print(f"Need clarification: {result.clarification_question}")
        ```

    Research Foundation:
        - MAST Framework: 34.7% failures from inter-agent misalignment
        - MemoryAgentBench: Target 75% conflict resolution (vs 50% SOTA)
        - Epistemic Logic: BDI agent architectures for belief tracking
    """

    def __init__(
        self,
        llm: Any,  # LLMProvider protocol
        embedding_provider: Any = None,
    ):
        """Initialize reconciliation service.

        Args:
            llm: LLM provider for conflict analysis
            embedding_provider: For semantic topic grouping (optional)
        """
        self.llm = llm
        self.embedding_provider = embedding_provider

        logger.info(
            f"Initialized MultiAgentBeliefReconciliation "
            f"(embeddings={embedding_provider is not None})"
        )
```

### Day 24-25: LLM Prompt Design

```python
    async def reconcile_multi_agent(
        self,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> ReconciliationResult:
        """Reconcile conflicting observations from multiple agents.

        Steps:
        1. Group observations by topic (semantic clustering)
        2. For each topic with conflicts, reconcile
        3. Combine results into final ReconciliationResult

        Args:
            observations: Observations to reconcile
            agent_credibilities: Credibility per agent (0-1)

        Returns:
            Reconciliation result
        """
        logger.info(f"Reconciling {len(observations)} observations")

        if not observations:
            return ReconciliationResult(resolved=True, reasoning="No observations to reconcile")

        # Group by topic
        topic_groups = self._group_by_topic(observations)
        logger.debug(f"Grouped into {len(topic_groups)} topics")

        # Reconcile each topic
        results = []
        for topic, topic_obs in topic_groups.items():
            if len(topic_obs) == 1:
                # Single observation, no conflict
                result = await self._single_observation_to_belief(topic_obs[0])
            else:
                # Multiple observations, reconcile
                result = await self._reconcile_topic(topic, topic_obs, agent_credibilities)

            results.append(result)

        # Combine results
        final_result = self._combine_results(results)

        logger.info(
            f"Reconciliation complete: resolved={final_result.resolved}, "
            f"confidence={final_result.confidence:.2f}"
        )

        return final_result

    async def _reconcile_topic(
        self,
        topic: str,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> ReconciliationResult:
        """Reconcile observations about a single topic using LLM.

        Args:
            topic: Topic being reconciled
            observations: All observations about this topic
            agent_credibilities: Agent credibility scores

        Returns:
            Reconciliation result for this topic
        """
        # Build LLM prompt (XML format)
        prompt = self._build_reconciliation_prompt(topic, observations, agent_credibilities)

        # Call LLM
        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for consistent reconciliation
        )

        # Parse XML response
        result = self._parse_reconciliation_response(response, observations)

        return result

    def _build_reconciliation_prompt(
        self,
        topic: str,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> str:
        """Build XML prompt for LLM conflict reconciliation.

        Constitution requirement: XML output format (MANDATORY)

        Args:
            topic: Topic being reconciled
            observations: Observations to reconcile
            agent_credibilities: Agent credibility scores

        Returns:
            XML-formatted prompt
        """
        # Build observations XML
        obs_xml_parts = []
        for obs in observations:
            credibility = agent_credibilities.get(obs.source_agent_id, 0.5)
            obs_xml_parts.append(
                f'  <observation agent_id="{obs.source_agent_id}" '
                f'credibility="{credibility:.2f}" confidence="{obs.confidence:.2f}">\n'
                f'    {obs.content}\n'
                f'  </observation>'
            )

        obs_xml = "\n".join(obs_xml_parts)

        prompt = f"""Analyze these observations from different agents about "{topic}":

<observations>
{obs_xml}
</observations>

Your task is to reconcile these observations into a consolidated belief.
Consider:
1. Agent credibility (higher credibility agents are more trustworthy)
2. Observation confidence (agent's confidence in their observation)
3. Temporal context (observations may be true at different times)
4. Semantic compatibility (observations may overlap without conflicting)

Determine:
1. Do these observations truly conflict, or are they compatible?
2. If conflicting, which is most likely correct based on credibility?
3. Can you synthesize a consolidated understanding?
4. If unresolvable, what clarifying question would help?

Respond in XML:
<reconciliation>
    <conflicts>yes/no</conflicts>
    <consolidated_belief>The reconciled understanding, or null if unresolvable</consolidated_belief>
    <confidence>0.0-1.0</confidence>
    <needs_clarification>true/false</needs_clarification>
    <clarification_question>Question to ask if needed, or null</clarification_question>
    <reasoning>Explain your reasoning (2-3 sentences)</reasoning>
</reconciliation>"""

        return prompt

    def _parse_reconciliation_response(
        self,
        response: str,
        observations: list[SharedObservation],
    ) -> ReconciliationResult:
        """Parse LLM XML response into ReconciliationResult.

        Args:
            response: XML response from LLM
            observations: Original observations (for audit trail)

        Returns:
            Parsed reconciliation result
        """
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(f"<root>{response}</root>")
            reconciliation = root.find("reconciliation")

            if reconciliation is None:
                raise ValueError("No <reconciliation> element found")

            # Extract fields
            conflicts = reconciliation.findtext("conflicts", "yes").lower() == "yes"
            belief_text = reconciliation.findtext("consolidated_belief")
            confidence = float(reconciliation.findtext("confidence", "0.0"))
            needs_clarification = reconciliation.findtext("needs_clarification", "false").lower() == "true"
            clarification_question = reconciliation.findtext("clarification_question")
            reasoning = reconciliation.findtext("reasoning", "")

            # Create consolidated belief if resolved
            consolidated_belief = None
            resolved = False

            if belief_text and belief_text.lower() != "null" and not needs_clarification:
                consolidated_belief = AgentBelief(
                    content=belief_text,
                    belief_type=BeliefType.HOUSEHOLD_FACT,  # TODO: Determine from observations
                    confidence=confidence,
                    supporting_observations=[obs.observation_id for obs in observations],
                    conflicting_observations=[],
                    verified=False,
                    needs_clarification=False,
                )
                resolved = True

            return ReconciliationResult(
                resolved=resolved,
                consolidated_belief=consolidated_belief,
                confidence=confidence,
                needs_human_clarification=needs_clarification,
                clarification_question=clarification_question if needs_clarification else None,
                observations_considered=[obs.observation_id for obs in observations],
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Failed to parse reconciliation response: {e}")
            logger.debug(f"Response was: {response}")

            # Return unresolved result
            return ReconciliationResult(
                resolved=False,
                needs_human_clarification=True,
                clarification_question="Unable to automatically reconcile. Please clarify.",
                observations_considered=[obs.observation_id for obs in observations],
                reasoning=f"XML parsing failed: {str(e)}",
            )
```

### Day 26-28: Topic Grouping & Edge Cases

```python
    def _group_by_topic(
        self,
        observations: list[SharedObservation],
    ) -> dict[str, list[SharedObservation]]:
        """Group observations by topic using semantic clustering.

        Phase 1 (simple): Group by belief_type
        Phase 2 (embeddings): Semantic similarity clustering

        Args:
            observations: Observations to group

        Returns:
            Dict of topic -> observations
        """
        # Phase 1: Group by belief_type
        groups: dict[str, list[SharedObservation]] = {}

        for obs in observations:
            # Use belief_type as topic for now
            # TODO Phase 2: Use embeddings for semantic clustering
            topic = obs.belief_type or "unknown"

            if topic not in groups:
                groups[topic] = []

            groups[topic].append(obs)

        logger.debug(f"Grouped {len(observations)} observations into {len(groups)} topics")

        return groups

    async def _single_observation_to_belief(
        self,
        observation: SharedObservation,
    ) -> ReconciliationResult:
        """Convert single observation to belief (no conflict).

        Args:
            observation: Single observation

        Returns:
            Reconciliation result with belief
        """
        belief = AgentBelief(
            content=observation.content,
            belief_type=BeliefType.UNVERIFIED_CLAIM,  # TODO: Map from observation.belief_type
            confidence=observation.confidence,
            supporting_observations=[observation.observation_id],
            conflicting_observations=[],
            verified=False,
            needs_clarification=False,
        )

        return ReconciliationResult(
            resolved=True,
            consolidated_belief=belief,
            confidence=observation.confidence,
            observations_considered=[observation.observation_id],
            reasoning="Single observation, no conflict",
        )

    def _combine_results(
        self,
        results: list[ReconciliationResult],
    ) -> ReconciliationResult:
        """Combine multiple topic reconciliation results.

        Args:
            results: Results for each topic

        Returns:
            Combined result
        """
        if not results:
            return ReconciliationResult(
                resolved=True,
                reasoning="No results to combine",
            )

        # Check if all resolved
        all_resolved = all(r.resolved for r in results)

        # Collect observations
        all_observations = []
        for r in results:
            all_observations.extend(r.observations_considered)

        # Collect clarification questions
        clarification_questions = [
            r.clarification_question
            for r in results
            if r.needs_human_clarification and r.clarification_question
        ]

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Combine reasoning
        reasoning_parts = [r.reasoning for r in results if r.reasoning]
        combined_reasoning = " | ".join(reasoning_parts)

        return ReconciliationResult(
            resolved=all_resolved,
            consolidated_belief=None,  # Multiple topics, can't merge into single belief
            confidence=avg_confidence,
            needs_human_clarification=len(clarification_questions) > 0,
            clarification_question=clarification_questions[0] if clarification_questions else None,
            observations_considered=all_observations,
            reasoning=combined_reasoning,
        )
```

### Day 29-31: Integration & Credibility Weighting

Back in `parallel_orchestrator.py`, update `_reconcile_conflicts`:

```python
    async def _reconcile_conflicts(
        self,
        shared_memory: SharedWorkingMemory,
        conflicts: list[tuple[Any, Any, str]],
    ) -> ReconciliationResult:
        """Trigger belief reconciliation for detected conflicts.

        Args:
            shared_memory: Shared working memory with conflicts
            conflicts: List of (obs_a, obs_b, reason) tuples

        Returns:
            ReconciliationResult
        """
        if not self.belief_service:
            logger.info(f"No belief service, skipping reconciliation of {len(conflicts)} conflicts")
            return ReconciliationResult(
                resolved=False,
                reasoning="No belief service configured",
            )

        logger.info(f"Reconciling {len(conflicts)} conflicts")

        # Get all belief candidates from shared memory
        candidates = await shared_memory.get_belief_candidates()

        # Get agent credibilities from transactive memory
        agent_credibilities = {}
        if self.transactive_memory:
            # Extract unique agent IDs from observations
            agent_ids = {obs.source_agent_id for obs in candidates}

            for agent_id in agent_ids:
                # Get agent's overall success rate as credibility
                # Average success rate across all topics
                expertise_summary = self.transactive_memory.get_expertise_summary()

                if agent_id in expertise_summary:
                    # Calculate average success rate
                    topic_success_rates = []
                    for topic in expertise_summary[agent_id]:
                        if agent_id in self.transactive_memory._expertise:
                            if topic in self.transactive_memory._expertise[agent_id]:
                                entry = self.transactive_memory._expertise[agent_id][topic]
                                topic_success_rates.append(entry.success_rate)

                    if topic_success_rates:
                        agent_credibilities[agent_id] = sum(topic_success_rates) / len(topic_success_rates)
                    else:
                        agent_credibilities[agent_id] = 0.5  # Neutral prior
                else:
                    agent_credibilities[agent_id] = 0.5  # Neutral prior
        else:
            # No transactive memory, use neutral credibility for all
            agent_ids = {obs.source_agent_id for obs in candidates}
            agent_credibilities = {agent_id: 0.5 for agent_id in agent_ids}

        logger.debug(f"Agent credibilities: {agent_credibilities}")

        # Call FR-003 reconciliation service
        reconciliation_result = await self.belief_service.reconcile_multi_agent(
            observations=candidates,
            agent_credibilities=agent_credibilities,
        )

        logger.info(
            f"Reconciliation complete: resolved={reconciliation_result.resolved}, "
            f"confidence={reconciliation_result.confidence:.2f}, "
            f"needs_clarification={reconciliation_result.needs_human_clarification}"
        )

        return reconciliation_result
```

### Day 32-33: Testing FR-003

**File:** `tests/cognition/test_multi_agent_reconciliation.py`

```python
"""Tests for multi-agent belief reconciliation (FR-003)."""

import pytest
from unittest.mock import AsyncMock, Mock

from draagon_ai.cognition.multi_agent_reconciliation import (
    MultiAgentBeliefReconciliation,
    ReconciliationResult,
)
from draagon_ai.orchestration.shared_memory import SharedObservation
from draagon_ai.cognition.beliefs import AgentBelief, BeliefType
from datetime import datetime


@pytest.fixture
def mock_llm():
    """Mock LLM provider."""
    llm = AsyncMock()
    return llm


@pytest.fixture
def reconciler(mock_llm):
    """Create reconciliation service."""
    return MultiAgentBeliefReconciliation(llm=mock_llm)


@pytest.mark.asyncio
async def test_simple_contradiction_high_credibility_wins(reconciler, mock_llm):
    """Test that high credibility agent wins simple contradiction."""
    # Setup
    obs_a = SharedObservation(
        observation_id="obs_a",
        content="The household has 6 cats",
        source_agent_id="agent_a",
        timestamp=datetime.now(),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    obs_b = SharedObservation(
        observation_id="obs_b",
        content="The household has 5 cats",
        source_agent_id="agent_b",
        timestamp=datetime.now(),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    agent_credibilities = {
        "agent_a": 0.9,  # High credibility
        "agent_b": 0.3,  # Low credibility
    }

    # Mock LLM response
    mock_llm.chat.return_value = """
<reconciliation>
    <conflicts>yes</conflicts>
    <consolidated_belief>The household has 6 cats</consolidated_belief>
    <confidence>0.85</confidence>
    <needs_clarification>false</needs_clarification>
    <clarification_question>null</clarification_question>
    <reasoning>Agent A has higher credibility (0.9) than Agent B (0.3), so we trust Agent A's observation.</reasoning>
</reconciliation>
"""

    # Execute
    result = await reconciler.reconcile_multi_agent(
        observations=[obs_a, obs_b],
        agent_credibilities=agent_credibilities,
    )

    # Assert
    assert result.resolved is True
    assert result.consolidated_belief is not None
    assert "6 cats" in result.consolidated_belief.content
    assert result.confidence > 0.7
    assert result.needs_human_clarification is False


@pytest.mark.asyncio
async def test_unresolvable_equal_credibility(reconciler, mock_llm):
    """Test that equal credibility triggers clarification."""
    # Setup
    obs_a = SharedObservation(
        observation_id="obs_a",
        content="Meeting at 3pm",
        source_agent_id="agent_a",
        timestamp=datetime.now(),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    obs_b = SharedObservation(
        observation_id="obs_b",
        content="Meeting at 4pm",
        source_agent_id="agent_b",
        timestamp=datetime.now(),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    agent_credibilities = {
        "agent_a": 0.7,
        "agent_b": 0.7,  # Equal credibility
    }

    # Mock LLM response
    mock_llm.chat.return_value = """
<reconciliation>
    <conflicts>yes</conflicts>
    <consolidated_belief>null</consolidated_belief>
    <confidence>0.5</confidence>
    <needs_clarification>true</needs_clarification>
    <clarification_question>What time is the meeting scheduled?</clarification_question>
    <reasoning>Both agents have equal credibility, and the observations directly contradict. Human clarification needed.</reasoning>
</reconciliation>
"""

    # Execute
    result = await reconciler.reconcile_multi_agent(
        observations=[obs_a, obs_b],
        agent_credibilities=agent_credibilities,
    )

    # Assert
    assert result.resolved is False
    assert result.needs_human_clarification is True
    assert result.clarification_question is not None
    assert "meeting" in result.clarification_question.lower()


@pytest.mark.asyncio
async def test_temporal_conflict_resolution(reconciler, mock_llm):
    """Test that temporal conflicts can be resolved."""
    # Setup
    obs_a = SharedObservation(
        observation_id="obs_a",
        content="Project on track (as of Monday)",
        source_agent_id="agent_a",
        timestamp=datetime(2025, 12, 28),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    obs_b = SharedObservation(
        observation_id="obs_b",
        content="Project delayed (as of Wednesday)",
        source_agent_id="agent_b",
        timestamp=datetime(2025, 12, 30),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    agent_credibilities = {"agent_a": 0.8, "agent_b": 0.8}

    # Mock LLM response
    mock_llm.chat.return_value = """
<reconciliation>
    <conflicts>no</conflicts>
    <consolidated_belief>Project was on track Monday but became delayed by Wednesday</consolidated_belief>
    <confidence>0.9</confidence>
    <needs_clarification>false</needs_clarification>
    <clarification_question>null</clarification_question>
    <reasoning>These observations are from different times. Both can be true - the project status changed over time.</reasoning>
</reconciliation>
"""

    # Execute
    result = await reconciler.reconcile_multi_agent(
        observations=[obs_a, obs_b],
        agent_credibilities=agent_credibilities,
    )

    # Assert
    assert result.resolved is True
    assert "Monday" in result.consolidated_belief.content
    assert "Wednesday" in result.consolidated_belief.content


@pytest.mark.asyncio
async def test_partial_overlap_merging(reconciler, mock_llm):
    """Test that compatible observations merge."""
    # Setup
    obs_a = SharedObservation(
        observation_id="obs_a",
        content="User works from home Mon-Wed",
        source_agent_id="agent_a",
        timestamp=datetime.now(),
        confidence=1.0,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    obs_b = SharedObservation(
        observation_id="obs_b",
        content="User sometimes works from home",
        source_agent_id="agent_b",
        timestamp=datetime.now(),
        confidence=0.7,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    agent_credibilities = {"agent_a": 0.8, "agent_b": 0.6}

    # Mock LLM response
    mock_llm.chat.return_value = """
<reconciliation>
    <conflicts>no</conflicts>
    <consolidated_belief>User works from home Mon-Wed</consolidated_belief>
    <confidence>0.85</confidence>
    <needs_clarification>false</needs_clarification>
    <clarification_question>null</clarification_question>
    <reasoning>Agent B's observation is a subset of Agent A's more specific information. They are compatible, not conflicting.</reasoning>
</reconciliation>
"""

    # Execute
    result = await reconciler.reconcile_multi_agent(
        observations=[obs_a, obs_b],
        agent_credibilities=agent_credibilities,
    )

    # Assert
    assert result.resolved is True
    assert "Mon-Wed" in result.consolidated_belief.content


# Add more tests for:
# - Source authority override
# - Audit trail completeness
# - Multi-topic reconciliation
# - XML parsing errors
```

---

## Phase 4: Integration & Testing (Days 34-38)

### Day 34: End-to-End Integration Tests

**File:** `tests/orchestration/test_parallel_integration.py`

```python
"""End-to-end integration tests for FR-002, FR-003, FR-004."""

import pytest
from unittest.mock import AsyncMock, Mock
import asyncio

from draagon_ai.orchestration.parallel_orchestrator import (
    ParallelCognitiveOrchestrator,
    ParallelExecutionConfig,
    ParallelOrchestrationMode,
)
from draagon_ai.orchestration.multi_agent_orchestrator import (
    AgentSpec,
    AgentRole,
    TaskContext,
    AgentResult,
)
from draagon_ai.orchestration.transactive_memory import TransactiveMemory
from draagon_ai.cognition.multi_agent_reconciliation import MultiAgentBeliefReconciliation


@pytest.mark.asyncio
async def test_full_pipeline_with_all_features():
    """Test complete pipeline: routing -> parallel exec -> reconciliation."""
    # Setup
    transactive_memory = TransactiveMemory()
    mock_llm = AsyncMock()
    belief_service = MultiAgentBeliefReconciliation(llm=mock_llm)

    config = ParallelExecutionConfig(
        max_concurrent_agents=3,
        sync_mode=ParallelOrchestrationMode.FORK_JOIN,
    )

    orchestrator = ParallelCognitiveOrchestrator(
        config=config,
        transactive_memory=transactive_memory,
        belief_service=belief_service,
    )

    # Pre-populate expertise
    await transactive_memory.update_expertise("agent_weather", "weather", True)
    await transactive_memory.update_expertise("agent_weather", "weather", True)
    await transactive_memory.update_expertise("agent_calendar", "scheduling", True)

    # Create agents
    agents = [
        AgentSpec(agent_id="agent_weather", role=AgentRole.RESEARCHER),
        AgentSpec(agent_id="agent_calendar", role=AgentRole.RESEARCHER),
        AgentSpec(agent_id="agent_generic", role=AgentRole.RESEARCHER),
    ]

    # Mock executor
    async def mock_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
        await asyncio.sleep(0.1)  # Simulate work
        return AgentResult(
            agent_id=agent.agent_id,
            success=True,
            output=f"{agent.agent_id} completed task",
        )

    # Mock LLM for reconciliation
    mock_llm.chat.return_value = """
<reconciliation>
    <conflicts>no</conflicts>
    <consolidated_belief>All agents provided compatible information</consolidated_belief>
    <confidence>0.9</confidence>
    <needs_clarification>false</needs_clarification>
    <clarification_question>null</clarification_question>
    <reasoning>No conflicts detected</reasoning>
</reconciliation>
"""

    # Execute
    context = TaskContext(
        query="What's the weather for my meeting tomorrow?",
        user_id="test_user",
    )

    result = await orchestrator.orchestrate_parallel(
        agents=agents,
        context=context,
        agent_executor=mock_executor,
    )

    # Assert
    assert result.success is True
    assert len(result.agent_results) == 3
    assert all(r.success for r in result.agent_results)

    # Verify expertise routing worked (weather agent should be first)
    routed_ids = [r.agent_id for r in result.agent_results]
    assert routed_ids[0] == "agent_weather"  # Highest expertise in "weather"


@pytest.mark.asyncio
async def test_conflict_reconciliation_triggered():
    """Test that conflicts trigger reconciliation."""
    # Setup with conflicting agents
    transactive_memory = TransactiveMemory()
    mock_llm = AsyncMock()
    belief_service = MultiAgentBeliefReconciliation(llm=mock_llm)

    config = ParallelExecutionConfig(
        max_concurrent_agents=2,
        sync_mode=ParallelOrchestrationMode.FORK_JOIN,
    )

    orchestrator = ParallelCognitiveOrchestrator(
        config=config,
        transactive_memory=transactive_memory,
        belief_service=belief_service,
    )

    agents = [
        AgentSpec(agent_id="agent_a", role=AgentRole.RESEARCHER),
        AgentSpec(agent_id="agent_b", role=AgentRole.RESEARCHER),
    ]

    # Mock executor that produces conflicting outputs
    async def mock_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
        await asyncio.sleep(0.1)

        if agent.agent_id == "agent_a":
            output = "Meeting at 3pm"
        else:
            output = "Meeting at 4pm"

        return AgentResult(
            agent_id=agent.agent_id,
            success=True,
            output=output,
        )

    # Mock LLM for reconciliation
    mock_llm.chat.return_value = """
<reconciliation>
    <conflicts>yes</conflicts>
    <consolidated_belief>null</consolidated_belief>
    <confidence>0.5</confidence>
    <needs_clarification>true</needs_clarification>
    <clarification_question>What time is the meeting?</clarification_question>
    <reasoning>Conflicting times from agents with equal credibility</reasoning>
</reconciliation>
"""

    # Execute
    context = TaskContext(query="When is the meeting?", user_id="test_user")

    result = await orchestrator.orchestrate_parallel(
        agents=agents,
        context=context,
        agent_executor=mock_executor,
    )

    # Assert
    assert result.success is True  # Orchestration succeeded despite conflict

    # Check that reconciliation was triggered
    assert mock_llm.chat.called

    # Check reconciliation result in context
    assert "__reconciliation__" in context.working_memory
    reconciliation = context.working_memory["__reconciliation__"]
    assert reconciliation.needs_human_clarification is True


# Add more integration tests for:
# - BARRIER_SYNC mode with conflicts
# - STREAMING mode
# - Expertise updates after execution
# - All result merging strategies
```

### Day 35-36: Documentation & Examples

Update `CLAUDE.md` with comprehensive examples, add `examples/` directory:

**File:** `examples/parallel_orchestration_example.py`

```python
"""Example: Parallel multi-agent orchestration with cognitive swarm.

This example demonstrates the full cognitive swarm architecture:
- FR-001: Shared working memory
- FR-002: Parallel orchestration
- FR-003: Belief reconciliation
- FR-004: Transactive memory
"""

import asyncio
from draagon_ai.orchestration.parallel_orchestrator import (
    ParallelCognitiveOrchestrator,
    ParallelExecutionConfig,
    ParallelOrchestrationMode,
)
from draagon_ai.orchestration.multi_agent_orchestrator import (
    AgentSpec,
    AgentRole,
    TaskContext,
    AgentResult,
)
from draagon_ai.orchestration.transactive_memory import TransactiveMemory
from draagon_ai.cognition.multi_agent_reconciliation import MultiAgentBeliefReconciliation


async def example_agent_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
    """Mock agent executor for demonstration."""
    # Simulate agent work
    await asyncio.sleep(1)

    # Simulate different outputs based on agent role
    if agent.role == AgentRole.RESEARCHER:
        output = f"{agent.agent_id} found relevant research data"
    elif agent.role == AgentRole.CRITIC:
        output = f"{agent.agent_id} identified potential issues"
    elif agent.role == AgentRole.EXECUTOR:
        output = f"{agent.agent_id} completed action"
    else:
        output = f"{agent.agent_id} completed task"

    return AgentResult(
        agent_id=agent.agent_id,
        success=True,
        output=output,
    )


async def main():
    """Run example orchestration."""
    print("=== Cognitive Swarm Example ===\n")

    # Step 1: Create transactive memory and populate with expertise
    print("Step 1: Initialize transactive memory")
    transactive_memory = TransactiveMemory()

    # Simulate past task outcomes to build expertise
    await transactive_memory.update_expertise("agent_research", "data_analysis", True)
    await transactive_memory.update_expertise("agent_research", "data_analysis", True)
    await transactive_memory.update_expertise("agent_critic", "quality_review", True)
    await transactive_memory.update_expertise("agent_action", "task_execution", True)

    print(f"  Expertise summary: {transactive_memory.get_expertise_summary()}\n")

    # Step 2: Create belief reconciliation service (mock LLM)
    print("Step 2: Initialize belief reconciliation service")
    class MockLLM:
        async def chat(self, messages, **kwargs):
            return """
<reconciliation>
    <conflicts>no</conflicts>
    <consolidated_belief>All agents provided compatible insights</consolidated_belief>
    <confidence>0.9</confidence>
    <needs_clarification>false</needs_clarification>
    <clarification_question>null</clarification_question>
    <reasoning>Agent outputs are complementary, not conflicting</reasoning>
</reconciliation>
"""

    belief_service = MultiAgentBeliefReconciliation(llm=MockLLM())

    # Step 3: Configure parallel orchestrator
    print("Step 3: Configure parallel orchestrator")
    config = ParallelExecutionConfig(
        max_concurrent_agents=5,
        sync_mode=ParallelOrchestrationMode.BARRIER_SYNC,
        sync_interval_iterations=3,
    )

    orchestrator = ParallelCognitiveOrchestrator(
        config=config,
        transactive_memory=transactive_memory,
        belief_service=belief_service,
    )

    print(f"  Mode: {config.sync_mode.value}")
    print(f"  Max agents: {config.max_concurrent_agents}\n")

    # Step 4: Define agents
    print("Step 4: Define agents")
    agents = [
        AgentSpec(agent_id="agent_research", role=AgentRole.RESEARCHER),
        AgentSpec(agent_id="agent_critic", role=AgentRole.CRITIC),
        AgentSpec(agent_id="agent_action", role=AgentRole.EXECUTOR),
    ]

    for agent in agents:
        print(f"  - {agent.agent_id} ({agent.role.value})")
    print()

    # Step 5: Execute parallel orchestration
    print("Step 5: Execute parallel orchestration")
    context = TaskContext(
        query="Analyze market trends and recommend actions",
        user_id="demo_user",
    )

    result = await orchestrator.orchestrate_parallel(
        agents=agents,
        context=context,
        agent_executor=example_agent_executor,
    )

    # Step 6: Display results
    print("\n=== Results ===")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_ms:.1f}ms")
    print(f"Agents executed: {len(result.agent_results)}")
    print(f"\nAgent results:")
    for agent_result in result.agent_results:
        status = "✓" if agent_result.success else "✗"
        print(f"  {status} {agent_result.agent_id}: {agent_result.output}")

    print(f"\nMerged output:")
    print(f"  {result.final_output}")

    # Step 7: Check reconciliation
    if "__reconciliation__" in context.working_memory:
        reconciliation = context.working_memory["__reconciliation__"]
        print(f"\nReconciliation:")
        print(f"  Resolved: {reconciliation.resolved}")
        print(f"  Confidence: {reconciliation.confidence:.2f}")
        print(f"  Reasoning: {reconciliation.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Day 37-38: Final Testing & Cleanup

- Run full test suite
- Verify constitution compliance
- Check test coverage (target 90%+)
- Update `__init__.py` exports
- Final code review

---

## Testing Strategy Summary

### FR-002 Tests (15 tests)
- `test_fork_join_execution()` - Simple parallel mode
- `test_barrier_sync_execution()` - Periodic sync
- `test_streaming_execution()` - Continuous sync
- `test_shared_memory_injection()` - Memory integration
- `test_timeout_handling()` - Agent timeouts
- `test_failure_handling()` - Agent failures
- `test_result_merge_all_outputs()` - Merging strategy
- `test_result_merge_highest_confidence()` - Merging strategy
- `test_result_merge_concatenate()` - Merging strategy
- `test_expertise_routing()` - FR-004 integration
- `test_conflict_reconciliation_triggered()` - FR-003 integration
- `test_parallel_speedup()` - Performance metric
- `test_max_concurrency()` - Stress test
- `test_agent_state_isolation()` - Read-only enforcement
- `test_early_termination()` - Config option

### FR-003 Tests (10 tests)
- `test_simple_contradiction()` - High credibility wins
- `test_temporal_conflict()` - Time-aware reconciliation
- `test_partial_overlap()` - Compatible observations
- `test_source_authority()` - CEO vs Reddit
- `test_unresolvable()` - Clarification triggered
- `test_audit_trail()` - Full reasoning
- `test_xml_parsing_error()` - Malformed response
- `test_single_observation()` - No conflict case
- `test_multi_topic_reconciliation()` - Multiple topics
- `test_credibility_weighting()` - Agent credibility

### FR-004 Tests (12 tests)
- `test_expertise_entry()` - Success/failure tracking
- `test_update_expertise()` - Confidence updates
- `test_route_query()` - Agent ranking
- `test_topic_extraction_simple()` - Keyword extraction
- `test_topic_hierarchy()` - Generalization
- `test_get_experts()` - Filtering/sorting
- `test_who_knows_about()` - Natural language
- `test_expertise_score_exact_match()` - Scoring
- `test_expertise_score_partial_match()` - Scoring
- `test_expertise_score_parent_match()` - Hierarchy
- `test_no_expertise_neutral_prior()` - Default behavior
- `test_expertise_summary()` - Debugging output

### Integration Tests (5 tests)
- `test_full_pipeline_all_features()` - FR-002 + FR-003 + FR-004
- `test_barrier_sync_with_conflicts()` - Mode integration
- `test_expertise_updates_after_execution()` - FR-004 integration
- `test_concurrent_access_safety()` - Race conditions
- `test_long_running_agents()` - Stress test

**Total:** 42 tests covering all three features

---

## Constitution Compliance Checklist

### ✅ LLM-First Architecture
- [ ] No semantic regex patterns (except security blocklist, entity IDs)
- [ ] Topic extraction via LLM (FR-004 Phase 2)
- [ ] Conflict analysis via LLM (FR-003)

### ✅ XML Output Format
- [ ] All LLM prompts use XML (FR-003 reconciliation prompt)
- [ ] Reconciliation response parsed from XML

### ✅ Protocol-Based Design
- [ ] Uses existing protocols (TaskContext, AgentSpec, AgentResult)
- [ ] LLMProvider protocol for FR-003
- [ ] EmbeddingProvider protocol (optional)

### ✅ Async-First Processing
- [ ] All orchestration methods are async
- [ ] Non-blocking reconciliation (BARRIER_SYNC mode)
- [ ] Async expertise updates

### ✅ Research-Grounded Development
- [ ] Anthropic research (3-5 agents, 90.2% improvement)
- [ ] MAST framework (34.7% misalignment failures)
- [ ] Wegner 1987 (transactive memory)
- [ ] MemoryAgentBench target (75% vs 50%)

### ✅ Test Outcomes, Not Processes
- [ ] Tests verify parallelism outcomes (speedup, success rate)
- [ ] Tests verify reconciliation outcomes (resolved, confidence)
- [ ] Tests verify routing outcomes (correct agent order)

---

## Dependencies & Integration

### FR-001 (Completed)
- ✅ `SharedWorkingMemory` - Core dependency
- ✅ `SharedObservation` - Used by FR-003
- ✅ `get_context_for_agent()` - Used by orchestrator
- ✅ `apply_attention_decay()` - Used by BARRIER_SYNC
- ✅ `get_conflicts()` - Triggers FR-003

### Existing Code (Reuse)
- ✅ `MultiAgentOrchestrator` - Base class
- ✅ `AgentSpec`, `TaskContext`, `AgentResult` - Data structures
- ✅ `_execute_with_retry()` - Timeout/retry logic
- ✅ `_broadcast_learnings()` - Learning propagation

### New Protocols Needed
- [ ] `LLMProvider` - For FR-003 (may already exist)
- [ ] `EmbeddingProvider` - For FR-003/FR-004 Phase 2

---

## Implementation Checklist

### Phase 1: FR-002 Core (Days 1-10)
- [ ] Day 1-2: Module skeleton, data structures
- [ ] Day 3-4: FORK_JOIN mode implementation
- [ ] Day 5-6: BARRIER_SYNC mode implementation
- [ ] Day 7: STREAMING mode implementation
- [ ] Day 8: Agent execution with shared memory
- [ ] Day 9-10: Result merging & conflict stubs

### Phase 2: FR-004 Transactive Memory (Days 11-20)
- [ ] Day 11-12: Core data structures (ExpertiseEntry, TransactiveMemory)
- [ ] Day 13-14: Expertise tracking (update_expertise)
- [ ] Day 15-16: Query routing (route_query, _extract_topics)
- [ ] Day 17-18: Expert queries (_get_experts, who_knows_about)
- [ ] Day 19-20: FR-002 integration (_route_by_expertise, _update_expertise)

### Phase 3: FR-003 Belief Reconciliation (Days 21-33)
- [ ] Day 21-23: Core reconciliation logic (MultiAgentBeliefReconciliation)
- [ ] Day 24-25: LLM prompt design (_build_reconciliation_prompt)
- [ ] Day 26-28: Topic grouping & edge cases
- [ ] Day 29-31: Credibility weighting & FR-002 integration
- [ ] Day 32-33: FR-003 unit tests

### Phase 4: Integration & Testing (Days 34-38)
- [ ] Day 34: End-to-end integration tests
- [ ] Day 35-36: Documentation & examples
- [ ] Day 37-38: Final testing & cleanup

---

## Success Metrics

### FR-002 Metrics
- **Speedup**: 3-agent parallel ≥ 2.5x faster than sequential ✓
- **Throughput**: 5 concurrent agents without latency increase ✓
- **Reliability**: 99%+ orchestration success rate ✓

### FR-003 Metrics
- **Conflict Resolution**: >90% resolved or flagged ✓
- **Resolution Accuracy**: >85% align with ground truth ✓
- **Audit Completeness**: 100% have reasoning trail ✓

### FR-004 Metrics
- **Routing Accuracy**: 85%+ to highest-expertise agent ✓
- **Expertise Learning**: Converge within 10 tasks ✓
- **Generalization**: 70%+ parent topic matches work ✓

---

## Risk Mitigation

### Risk: BARRIER_SYNC Complexity
**Mitigation:** Implement FORK_JOIN first (simplest), then add BARRIER_SYNC.

### Risk: LLM Latency (FR-003)
**Mitigation:** Async reconciliation, cache results, batch reconciliations.

### Risk: Expertise Cold Start (FR-004)
**Mitigation:** Default to neutral prior (0.5), allow manual expertise initialization.

### Risk: Result Merging Ambiguity
**Mitigation:** Default to ALL_OUTPUTS (maximum information preserved).

---

**Implementation Plan Status:** Ready for Execution
**Estimated Completion:** 38 days from start
**Dependencies:** FR-001 (✅ Complete)
**Approval:** Pending user sign-off

---

**End of Implementation Plan**

# God-Level Orchestrator Architecture

*A self-evolving, opinion-reconciling, memory-sharing multi-agent orchestrator*

*December 28, 2025*

---

## The Vision

**Don't just orchestrate agents. Let them think together.**

Current orchestrator: Pass a dict between sequential method calls.
Target orchestrator: Agents share semantic memory, reconcile opinions, learn failure patterns, and evolve their own coordination strategies.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLVABLE ORCHESTRATOR                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  ORCHESTRATION   │    │   ORCHESTRATION  │    │   ORCHESTRATION  │      │
│  │     PROMPTS      │───▶│    DECISIONS     │───▶│     ACTIONS      │      │
│  │   (evolvable)    │    │   (LLM-driven)   │    │   (execution)    │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    QDRANT SHARED MEMORY                          │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │      │
│  │  │ Agent A │  │ Agent B │  │ Agent C │  │ Agent D │             │      │
│  │  │ Output  │  │ Output  │  │ Output  │  │ Output  │             │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘             │      │
│  │       │            │            │            │                   │      │
│  │       └────────────┴─────┬──────┴────────────┘                   │      │
│  │                          ▼                                       │      │
│  │                 SEMANTIC RETRIEVAL                               │      │
│  │            "What does the researcher know?"                      │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │    FAILURE       │    │     OPINION      │    │    LEARNING      │      │
│  │   ASSESSMENT     │    │   RECONCILIATION │    │    CHANNEL       │      │
│  │  (LLM + history) │    │  (merge outputs) │    │ (share insights) │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Qdrant-Based Shared Memory

### Why Not Just Pass Dicts?

Current approach:
```python
# Brittle, no semantic access, lost on failure
context.working_memory["research"] = research_output
design = await self.design_behavior(context.working_memory["research"])
```

Problems:
1. **No semantic access** - Designer can only get "research" by exact key
2. **Lost on failure** - If process crashes, everything is gone
3. **No cross-task learning** - Can't learn from previous orchestrations
4. **Tight coupling** - Every agent must know exact key names

### The Qdrant Approach

```python
class OrchestrationMemory:
    """Shared memory for agent orchestration backed by Qdrant.

    Agents publish outputs and retrieve context semantically.
    Persists across failures. Enables cross-task learning.
    """

    # Memory type for orchestration
    ORCHESTRATION_TYPE = MemoryType.WORKING  # or new MemoryType.ORCHESTRATION

    def __init__(
        self,
        memory: MemoryProvider,
        task_id: str,
        agent_id: str = "orchestrator",
    ):
        self.memory = memory
        self.task_id = task_id
        self.agent_id = agent_id

        # Use a task-specific scope
        # This allows cleanup after task completion
        self._task_prefix = f"orch:{task_id}"

    # =========================================================================
    # Publishing (Agent → Memory)
    # =========================================================================

    async def publish(
        self,
        producer_agent: str,
        key: str,
        content: Any,
        content_type: str = "output",  # output | error | insight | question
        importance: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Agent publishes output for other agents to consume.

        Args:
            producer_agent: ID of the producing agent
            key: Semantic key (e.g., "domain_research", "behavior_design")
            content: The content to store (will be serialized)
            content_type: Type of content
            importance: How important is this (affects retrieval ranking)
            metadata: Additional metadata

        Returns:
            Memory ID for direct retrieval
        """
        # Serialize complex objects
        if not isinstance(content, str):
            content_str = json.dumps(content, default=str)
        else:
            content_str = content

        # Build rich content for semantic search
        searchable_content = f"""
        Task: {self.task_id}
        Agent: {producer_agent}
        Key: {key}
        Type: {content_type}

        Content:
        {content_str[:2000]}  # Truncate for embedding
        """

        memory_id = await self.memory.store(
            content=searchable_content,
            memory_type=self.ORCHESTRATION_TYPE,
            scope=MemoryScope.CONTEXT,  # Task-level scope
            agent_id=self.agent_id,
            importance=importance,
            entities=[
                f"task:{self.task_id}",
                f"agent:{producer_agent}",
                f"key:{key}",
                f"type:{content_type}",
            ],
            metadata={
                "task_id": self.task_id,
                "producer_agent": producer_agent,
                "key": key,
                "content_type": content_type,
                "full_content": content_str,  # Store full content in metadata
                "produced_at": datetime.now().isoformat(),
                **(metadata or {}),
            },
        )

        logger.info(f"Published {key} from {producer_agent} to orchestration memory")
        return memory_id

    # =========================================================================
    # Retrieval (Memory → Agent)
    # =========================================================================

    async def retrieve_by_key(
        self,
        key: str,
        producer_agent: str | None = None,
    ) -> Any | None:
        """Retrieve content by exact key.

        Args:
            key: The key to retrieve
            producer_agent: Optional filter by producer

        Returns:
            The content, or None if not found
        """
        query = f"task:{self.task_id} key:{key}"
        if producer_agent:
            query += f" agent:{producer_agent}"

        results = await self.memory.search(
            query=query,
            scopes=[MemoryScope.CONTEXT],
            limit=1,
        )

        if not results:
            return None

        # Extract full content from metadata
        result = results[0]
        if hasattr(result, 'memory') and hasattr(result.memory, 'metadata'):
            full_content = result.memory.metadata.get("full_content")
            if full_content:
                try:
                    return json.loads(full_content)
                except json.JSONDecodeError:
                    return full_content

        return None

    async def semantic_retrieve(
        self,
        query: str,
        content_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantically search for relevant context.

        This is the power move - agents can find relevant context
        without knowing exact keys.

        Args:
            query: Natural language query
            content_types: Filter by content types
            limit: Max results

        Returns:
            List of relevant content items
        """
        # Scope the search to this task
        full_query = f"task:{self.task_id} {query}"

        results = await self.memory.search(
            query=full_query,
            scopes=[MemoryScope.CONTEXT],
            limit=limit,
        )

        items = []
        for r in results:
            metadata = r.memory.metadata if hasattr(r, 'memory') else {}

            # Filter by content type if specified
            if content_types and metadata.get("content_type") not in content_types:
                continue

            items.append({
                "key": metadata.get("key"),
                "producer_agent": metadata.get("producer_agent"),
                "content_type": metadata.get("content_type"),
                "content": metadata.get("full_content"),
                "score": r.score if hasattr(r, 'score') else 1.0,
            })

        return items

    async def get_all_outputs(self) -> dict[str, Any]:
        """Get all outputs for this task (for final aggregation)."""
        results = await self.memory.search(
            query=f"task:{self.task_id} type:output",
            scopes=[MemoryScope.CONTEXT],
            limit=100,
        )

        outputs = {}
        for r in results:
            metadata = r.memory.metadata if hasattr(r, 'memory') else {}
            key = metadata.get("key")
            if key:
                outputs[key] = metadata.get("full_content")

        return outputs

    # =========================================================================
    # Cross-Agent Communication
    # =========================================================================

    async def ask_question(
        self,
        asking_agent: str,
        question: str,
        target_agent: str | None = None,
    ) -> str:
        """Agent asks a question for other agents (or humans) to answer.

        This enables agents to request clarification mid-orchestration.
        """
        return await self.publish(
            producer_agent=asking_agent,
            key=f"question:{uuid.uuid4().hex[:8]}",
            content=question,
            content_type="question",
            metadata={"target_agent": target_agent},
        )

    async def share_insight(
        self,
        agent: str,
        insight: str,
        related_to: str | None = None,
    ) -> str:
        """Agent shares an insight for the learning channel."""
        return await self.publish(
            producer_agent=agent,
            key=f"insight:{uuid.uuid4().hex[:8]}",
            content=insight,
            content_type="insight",
            metadata={"related_to": related_to},
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self, keep_insights: bool = True) -> None:
        """Clean up task memory after completion.

        Args:
            keep_insights: Whether to preserve insights for learning
        """
        # In a real implementation, delete memories with this task_id
        # except insights if keep_insights is True
        pass
```

### How Agents Use It

```python
# Researcher agent
async def research_agent(context: TaskContext, memory: OrchestrationMemory):
    research = await do_research(context.query)

    # Publish to shared memory
    await memory.publish(
        producer_agent="researcher",
        key="domain_research",
        content=research,
    )

    # Share an insight for future tasks
    await memory.share_insight(
        agent="researcher",
        insight=f"Domain {research.domain} has {len(research.competitors)} competitors",
    )

# Designer agent - doesn't need to know researcher's exact output format
async def designer_agent(context: TaskContext, memory: OrchestrationMemory):
    # Semantic retrieval - finds relevant research even with fuzzy query
    research_items = await memory.semantic_retrieve(
        query="what do we know about this domain and existing solutions?",
        content_types=["output"],
    )

    # Or exact key if known
    research = await memory.retrieve_by_key("domain_research")

    design = await create_design(research)
    await memory.publish(
        producer_agent="designer",
        key="behavior_design",
        content=design,
    )
```

---

## Component 2: Intelligent Failure Handling

### The Problem with Binary Policies

Current thinking:
- **Fail fast**: Stop on first error (too aggressive)
- **Collect all**: Continue always (too permissive)

Neither is intelligent. We need context-aware decisions.

### Failure Assessment Framework

```python
class FailurePolicy(str, Enum):
    """How to handle agent failures."""

    FAIL_FAST = "fail_fast"           # Stop immediately (critical agents)
    CONTINUE = "continue"              # Continue, mark as failed
    ASSESS = "assess"                  # LLM assesses recoverability
    RETRY_WITH_CONTEXT = "retry"       # Retry with failure context
    DELEGATE = "delegate"              # Hand off to a different agent
    LEARNED = "learned"                # Use historical patterns


@dataclass
class FailureContext:
    """Rich context about a failure for assessment."""

    agent: AgentSpec
    error: str
    error_type: str  # timeout | exception | validation | llm_error
    attempt_number: int
    task_context: TaskContext

    # What other agents have produced
    available_outputs: dict[str, Any]

    # Historical data
    past_failures: list[dict]  # Similar failures and outcomes
    agent_success_rate: float  # This agent's historical success rate


class FailureAssessor:
    """Intelligently assesses failures and recommends actions."""

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        learning: LearningService,
    ):
        self.llm = llm
        self.memory = memory
        self.learning = learning

        # This prompt can evolve!
        self.assessment_prompt = FAILURE_ASSESSMENT_PROMPT

    async def assess(
        self,
        failure: FailureContext,
    ) -> FailureDecision:
        """Assess a failure and decide what to do.

        Uses LLM reasoning + historical patterns.
        """
        # First, check historical patterns
        historical = await self._check_historical_patterns(failure)
        if historical.confidence > 0.8:
            return historical.decision

        # LLM assessment
        prompt = self.assessment_prompt.format(
            agent_name=failure.agent.name,
            agent_role=failure.agent.role.value,
            error=failure.error,
            error_type=failure.error_type,
            task=failure.task_context.query,
            available_outputs=json.dumps(failure.available_outputs, indent=2),
            historical_patterns=self._format_historical(failure.past_failures),
            agent_success_rate=failure.agent_success_rate,
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Assess this failure and recommend action."},
            ],
            response_format={"type": "json"},
        )

        decision = self._parse_decision(response.content)

        # Store this assessment for future learning
        await self._record_assessment(failure, decision)

        return decision

    async def _check_historical_patterns(
        self,
        failure: FailureContext,
    ) -> HistoricalMatch:
        """Check if we've seen this failure pattern before."""

        # Search for similar failures
        results = await self.memory.search(
            query=f"failure {failure.agent.role} {failure.error_type} {failure.error[:100]}",
            memory_type=MemoryType.INSIGHT,
            limit=5,
        )

        for r in results:
            metadata = r.memory.metadata
            if metadata.get("record_type") == "failure_assessment":
                # Check similarity
                if self._is_similar_failure(failure, metadata):
                    outcome = metadata.get("outcome")
                    if outcome == "recovered":
                        return HistoricalMatch(
                            confidence=0.85,
                            decision=FailureDecision(
                                action=metadata.get("action_taken"),
                                reason="Similar failure recovered with this action",
                            ),
                        )

        return HistoricalMatch(confidence=0.0, decision=None)


# The assessment prompt - this evolves!
FAILURE_ASSESSMENT_PROMPT = """You are assessing a failure in a multi-agent orchestration.

FAILED AGENT:
- Name: {agent_name}
- Role: {agent_role}
- Historical success rate: {agent_success_rate:.0%}

ERROR:
- Type: {error_type}
- Message: {error}

TASK CONTEXT:
{task}

ALREADY COMPLETED (available from other agents):
{available_outputs}

HISTORICAL PATTERNS:
{historical_patterns}

Based on this context, decide:

1. **Can we continue without this agent?**
   - What would we lose?
   - Can another agent compensate?

2. **Should we retry?**
   - Is this a transient error (timeout, rate limit)?
   - Would more context help?

3. **Can we delegate to another agent?**
   - Is there an agent with overlapping capabilities?

4. **What's the risk of continuing vs stopping?**

Output JSON:
{{
    "action": "continue" | "retry" | "delegate" | "fail",
    "reason": "Explanation",
    "retry_with_context": "Additional context for retry" or null,
    "delegate_to": "Agent ID to delegate to" or null,
    "can_proceed_without": true/false,
    "risk_level": "low" | "medium" | "high",
    "confidence": 0.0-1.0
}}
"""
```

### Behavior-Specified Failure Policies

Behaviors can specify their own failure tolerances that **evolve with their prompts**:

```python
@dataclass
class OrchestrationHints:
    """Orchestration hints embedded in behaviors - can evolve."""

    # Failure handling
    critical_phases: list[str] = field(default_factory=list)
    optional_phases: list[str] = field(default_factory=list)
    retry_phases: list[str] = field(default_factory=list)

    # Parallelization
    parallel_safe: list[tuple[str, str]] = field(default_factory=list)
    requires_sequential: list[tuple[str, str]] = field(default_factory=list)

    # Merging
    merge_strategy: str = "opinion"  # opinion | last_wins | aggregate | vote

    # Resource hints
    optimize_for: str = "quality"  # quality | speed | cost


# Extracted from behavior prompts or specified explicitly
BEHAVIOR_ARCHITECT_HINTS = OrchestrationHints(
    critical_phases=["research", "design"],  # Can't continue without these
    optional_phases=["evolve"],               # Nice to have
    retry_phases=["test"],                    # Often fails transiently

    parallel_safe=[
        ("prompt_generator", "test_generator"),  # Can run together
        ("quality_check", "capability_check"),   # Independent validations
    ],

    requires_sequential=[
        ("research", "design"),   # Design needs research
        ("design", "build"),      # Build needs design
        ("build", "test"),        # Test needs built behavior
    ],

    merge_strategy="opinion",
    optimize_for="quality",
)
```

---

## Component 3: Opinion-Based Output Merging

### The Problem

When agents run in parallel, they may produce:
1. **Complementary outputs** - Different aspects, should combine
2. **Competing outputs** - Same thing, different approaches
3. **Conflicting outputs** - Contradictions that need resolution

### Using the Opinion Framework

```python
class OpinionBasedMerger:
    """Merge parallel agent outputs using the opinion framework."""

    def __init__(
        self,
        opinion_service: OpinionService,
        belief_service: BeliefReconciliationService,
        llm: LLMProvider,
    ):
        self.opinions = opinion_service
        self.beliefs = belief_service
        self.llm = llm

    async def merge(
        self,
        outputs: list[AgentResult],
        merge_strategy: str,
        context: TaskContext,
    ) -> MergeResult:
        """Merge outputs from parallel agents.

        Strategies:
        - opinion: Use opinion reconciliation for conflicts
        - aggregate: Combine all outputs
        - vote: Agents vote on best output
        - last_wins: Just use last successful output
        """
        valid_outputs = [o for o in outputs if o.success and o.output]

        if not valid_outputs:
            return MergeResult(success=False, output=None)

        if len(valid_outputs) == 1:
            return MergeResult(success=True, output=valid_outputs[0].output)

        if merge_strategy == "opinion":
            return await self._opinion_merge(valid_outputs, context)
        elif merge_strategy == "aggregate":
            return await self._aggregate_merge(valid_outputs, context)
        elif merge_strategy == "vote":
            return await self._vote_merge(valid_outputs, context)
        else:
            return MergeResult(success=True, output=valid_outputs[-1].output)

    async def _opinion_merge(
        self,
        outputs: list[AgentResult],
        context: TaskContext,
    ) -> MergeResult:
        """Use opinion reconciliation for competing outputs."""

        # Have each output form an opinion
        opinions = []
        for result in outputs:
            # Create observation from this agent's output
            observation = await self.beliefs.create_observation(
                statement=f"The best approach is: {self._summarize(result.output)}",
                source_user=result.agent_id,
                source_type="agent",
            )

            # Form opinion with confidence based on agent's role
            confidence = self._get_agent_confidence(result.agent_id, context)

            opinion = await self.opinions.form_opinion(
                topic=f"Best output for: {context.query}",
                observation=observation,
                initial_confidence=confidence,
            )
            opinions.append((opinion, result))

        # Reconcile opinions
        if len(opinions) >= 2:
            # Check for conflicts
            conflicts = await self._detect_conflicts(
                [o[0] for o in opinions],
                context,
            )

            if conflicts:
                # Use belief reconciliation
                reconciled = await self.beliefs.reconcile_conflicting_observations(
                    observations=[o[0].observation for o in opinions],
                    topic=context.query,
                )

                # Find which output aligns with reconciled belief
                best_output = await self._find_aligned_output(
                    reconciled,
                    opinions,
                )
                return MergeResult(
                    success=True,
                    output=best_output,
                    reconciliation_notes=reconciled.reasoning,
                )

        # No conflicts - aggregate or pick highest confidence
        if self._are_complementary(outputs):
            return await self._aggregate_merge(outputs, context)
        else:
            best = max(opinions, key=lambda x: x[0].confidence)
            return MergeResult(success=True, output=best[1].output)

    async def _vote_merge(
        self,
        outputs: list[AgentResult],
        context: TaskContext,
    ) -> MergeResult:
        """Agents vote on the best output."""

        # Each agent evaluates all outputs
        votes = {r.agent_id: {} for r in outputs}

        for voter in outputs:
            for candidate in outputs:
                if voter.agent_id == candidate.agent_id:
                    continue

                score = await self._evaluate_output(
                    voter_context=voter,
                    candidate=candidate,
                    task_context=context,
                )
                votes[voter.agent_id][candidate.agent_id] = score

        # Tally votes
        scores = {}
        for candidate in outputs:
            scores[candidate.agent_id] = sum(
                votes[voter][candidate.agent_id]
                for voter in votes
                if candidate.agent_id in votes[voter]
            )

        winner_id = max(scores, key=scores.get)
        winner = next(r for r in outputs if r.agent_id == winner_id)

        return MergeResult(
            success=True,
            output=winner.output,
            vote_scores=scores,
        )

    def _get_agent_confidence(
        self,
        agent_id: str,
        context: TaskContext,
    ) -> float:
        """Get confidence weight for an agent based on role and history."""

        # Role-based base confidence
        role_confidence = {
            AgentRole.SPECIALIST: 0.9,
            AgentRole.RESEARCHER: 0.8,
            AgentRole.CRITIC: 0.85,
            AgentRole.PLANNER: 0.75,
            AgentRole.EXECUTOR: 0.7,
            AgentRole.PRIMARY: 0.75,
        }

        # Could also factor in historical accuracy
        # agent_history = await self.get_agent_accuracy(agent_id)

        return role_confidence.get(agent_id, 0.7)
```

---

## Component 4: Evolvable Orchestration Strategies

### The Meta-Level: Evolving How We Orchestrate

The orchestration decisions themselves can be driven by prompts that evolve:

```python
class EvolvableOrchestrator(MultiAgentOrchestrator):
    """Orchestrator with evolvable decision-making.

    The prompts that decide WHEN to parallelize, HOW to handle failures,
    and WHAT to merge can themselves evolve via Promptbreeder.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        promptbreeder: Promptbreeder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.memory = memory
        self.promptbreeder = promptbreeder

        # These prompts evolve
        self.decision_prompt = ORCHESTRATION_DECISION_PROMPT
        self.parallelization_prompt = PARALLELIZATION_DECISION_PROMPT
        self.failure_prompt = FAILURE_ASSESSMENT_PROMPT
        self.merge_prompt = MERGE_DECISION_PROMPT

    async def orchestrate_dynamic(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Dynamic orchestration - LLM decides mode at each step.

        Instead of fixed sequential/parallel, the orchestrator decides
        dynamically based on the current state.
        """
        remaining_agents = list(agents)
        completed_agents = []

        while remaining_agents:
            # Decide what to do next
            decision = await self._make_orchestration_decision(
                remaining=remaining_agents,
                completed=completed_agents,
                context=context,
            )

            if decision.mode == "parallel":
                # Run selected agents in parallel
                results = await self._run_parallel(
                    decision.agents_to_run,
                    context,
                    executor,
                )
            else:
                # Run single agent
                results = [await self._execute_with_retry(
                    decision.agents_to_run[0],
                    context,
                    executor,
                )]

            # Handle results
            for result in results:
                agent = next(a for a in remaining_agents if a.agent_id == result.agent_id)
                remaining_agents.remove(agent)
                completed_agents.append((agent, result))

                # Handle failures dynamically
                if not result.success:
                    failure_decision = await self._assess_failure_dynamic(
                        agent, result, context, remaining_agents,
                    )
                    if failure_decision.action == "fail":
                        return self._create_failure_result(context, result)
                    elif failure_decision.action == "delegate":
                        # Add delegate agent to remaining
                        remaining_agents.insert(0, failure_decision.delegate_agent)

        # Merge all results
        return await self._create_final_result(completed_agents, context)

    async def _make_orchestration_decision(
        self,
        remaining: list[AgentSpec],
        completed: list[tuple[AgentSpec, AgentResult]],
        context: TaskContext,
    ) -> OrchestrationDecision:
        """LLM decides what to do next."""

        prompt = self.decision_prompt.format(
            remaining_agents=self._format_agents(remaining),
            completed_agents=self._format_completed(completed),
            task=context.query,
            working_memory=json.dumps(dict(context.working_memory), indent=2),
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Decide the next orchestration step."},
            ],
            response_format={"type": "json"},
        )

        return self._parse_decision(response.content)

    # =========================================================================
    # Evolution
    # =========================================================================

    async def evolve_orchestration_strategy(
        self,
        test_cases: list[OrchestrationTestCase],
    ) -> EvolutionResult:
        """Evolve the orchestration prompts based on performance.

        Test cases measure:
        - Task success rate
        - Total execution time
        - LLM token usage (cost)
        - Failure recovery rate
        """

        # Evolve the main decision prompt
        result = await self.promptbreeder.evolve(
            base_prompt=self.decision_prompt,
            test_cases=self._convert_to_prompt_tests(test_cases),
            config=EvolutionConfig(
                generations=5,
                population_size=8,
                optimize_for="balanced",  # Balance quality and speed
            ),
        )

        if result.improvement > 0.1:  # 10% improvement threshold
            self.decision_prompt = result.best_prompt

            # Store the evolution
            await self.memory.store(
                content=f"Evolved orchestration prompt: {result.improvement:.0%} improvement",
                memory_type=MemoryType.INSIGHT,
                entities=["evolution", "orchestration", "prompt"],
                metadata={
                    "old_prompt_hash": hash(ORCHESTRATION_DECISION_PROMPT),
                    "new_prompt_hash": hash(result.best_prompt),
                    "improvement": result.improvement,
                },
            )

        return result


# The orchestration decision prompt - THIS EVOLVES
ORCHESTRATION_DECISION_PROMPT = """You are an orchestration controller deciding the next step.

REMAINING AGENTS:
{remaining_agents}

COMPLETED AGENTS AND OUTPUTS:
{completed_agents}

TASK:
{task}

CURRENT WORKING MEMORY:
{working_memory}

Decide:
1. Which agent(s) should run next?
2. Can any of them run in parallel?
3. What context should be passed?

Consider:
- Dependencies between agents
- Whether outputs are already available for parallel agents
- Cost vs speed tradeoffs

Output JSON:
{{
    "mode": "sequential" | "parallel",
    "agents_to_run": ["agent_id1", ...],
    "reasoning": "Why this decision",
    "context_to_pass": {{"key": "value", ...}}
}}
"""
```

---

## Component 5: Behavior-Specified Orchestration

### Behaviors Define Their Own Coordination

Instead of hard-coding orchestration logic, behaviors include hints that **evolve with their prompts**:

```python
@dataclass
class Behavior:
    """A behavior with orchestration configuration."""

    name: str
    actions: list[Action]
    prompts: dict[str, str]

    # Orchestration configuration - extracted or evolved
    orchestration: OrchestrationConfig | None = None


@dataclass
class OrchestrationConfig:
    """How this behavior prefers to be orchestrated."""

    # Phase definitions
    phases: list[PhaseConfig] = field(default_factory=list)

    # Parallelization rules
    parallelizable: list[set[str]] = field(default_factory=list)

    # Failure handling
    failure_policies: dict[str, FailurePolicy] = field(default_factory=dict)

    # Merge strategies per phase
    merge_strategies: dict[str, str] = field(default_factory=dict)

    # Optimization target
    optimize_for: str = "quality"  # quality | speed | cost | balanced

    # Evolution hints
    evolvable_phases: list[str] = field(default_factory=list)


@dataclass
class PhaseConfig:
    """Configuration for a single orchestration phase."""

    name: str
    agent_type: str  # researcher | designer | executor | critic | etc
    required: bool = True
    timeout_seconds: float = 60.0

    # Dependencies
    depends_on: list[str] = field(default_factory=list)

    # Prompt to use (from behavior's prompts dict)
    prompt_key: str | None = None

    # Failure handling for this phase
    failure_policy: FailurePolicy = FailurePolicy.ASSESS


# Behavior Architect's orchestration config
BEHAVIOR_ARCHITECT_CONFIG = OrchestrationConfig(
    phases=[
        PhaseConfig(
            name="research",
            agent_type="researcher",
            required=True,
            timeout_seconds=60.0,
            depends_on=[],
        ),
        PhaseConfig(
            name="design",
            agent_type="planner",
            required=True,
            timeout_seconds=45.0,
            depends_on=["research"],
        ),
        PhaseConfig(
            name="generate_prompts",
            agent_type="executor",
            required=True,
            timeout_seconds=45.0,
            depends_on=["design"],
        ),
        PhaseConfig(
            name="generate_tests",
            agent_type="executor",
            required=True,
            timeout_seconds=45.0,
            depends_on=["design"],  # Same dependency as prompts!
        ),
        PhaseConfig(
            name="test",
            agent_type="critic",
            required=True,
            timeout_seconds=120.0,
            depends_on=["generate_prompts", "generate_tests"],
            failure_policy=FailurePolicy.RETRY_WITH_CONTEXT,
        ),
        PhaseConfig(
            name="evolve",
            agent_type="specialist",
            required=False,
            timeout_seconds=300.0,
            depends_on=["test"],
        ),
    ],

    # These can run in parallel (same dependencies)
    parallelizable=[
        {"generate_prompts", "generate_tests"},
    ],

    failure_policies={
        "research": FailurePolicy.FAIL_FAST,
        "design": FailurePolicy.FAIL_FAST,
        "generate_prompts": FailurePolicy.RETRY_WITH_CONTEXT,
        "generate_tests": FailurePolicy.RETRY_WITH_CONTEXT,
        "test": FailurePolicy.ASSESS,
        "evolve": FailurePolicy.CONTINUE,
    },

    merge_strategies={
        "parallel_generation": "aggregate",
        "test_results": "opinion",
    },

    optimize_for="quality",

    # Phases whose prompts can evolve
    evolvable_phases=["design", "generate_prompts", "test"],
)
```

### Automatic Orchestration from Config

```python
class ConfigDrivenOrchestrator:
    """Orchestrates based on behavior configuration."""

    async def orchestrate_behavior(
        self,
        config: OrchestrationConfig,
        context: TaskContext,
        executor: AgentExecutor,
        memory: OrchestrationMemory,
    ) -> OrchestratorResult:
        """Execute a behavior using its orchestration config."""

        # Build dependency graph
        graph = self._build_dependency_graph(config.phases)

        # Find parallelizable groups
        parallel_groups = self._find_parallel_groups(graph, config.parallelizable)

        # Execute in topological order, parallelizing where possible
        completed = {}

        for group in self._topological_groups(graph, parallel_groups):
            if len(group) > 1:
                # Parallel execution
                agents = [self._phase_to_agent(config.phases[p]) for p in group]
                results = await self._run_parallel(agents, context, executor)

                # Merge results if needed
                merge_strategy = config.merge_strategies.get(
                    f"parallel_{group[0]}",
                    "aggregate",
                )
                merged = await self.merger.merge(results, merge_strategy, context)

                for result in results:
                    completed[result.agent_id] = result
                    await memory.publish(
                        producer_agent=result.agent_id,
                        key=result.agent_id,
                        content=result.output,
                    )
            else:
                # Sequential execution
                phase_name = group[0]
                phase = next(p for p in config.phases if p.name == phase_name)
                agent = self._phase_to_agent(phase)

                result = await self._execute_with_policy(
                    agent,
                    context,
                    executor,
                    config.failure_policies.get(phase_name, FailurePolicy.ASSESS),
                )

                completed[phase_name] = result
                await memory.publish(
                    producer_agent=phase_name,
                    key=phase_name,
                    content=result.output,
                )

                if not result.success and phase.required:
                    return self._create_failure_result(context, result)

        return self._create_success_result(completed, context)
```

---

## Putting It All Together

### The Refactored Behavior Architect

```python
class BehaviorArchitectService:
    """Creates behaviors using the god-level orchestrator."""

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        orchestrator: EvolvableOrchestrator,
        opinion_service: OpinionService,
        belief_service: BeliefReconciliationService,
        promptbreeder: Promptbreeder,
    ):
        self.llm = llm
        self.memory = memory
        self.orchestrator = orchestrator
        self.opinions = opinion_service
        self.beliefs = belief_service
        self.promptbreeder = promptbreeder

        # The behavior's orchestration config
        self.config = BEHAVIOR_ARCHITECT_CONFIG

    async def create_behavior(
        self,
        description: str,
        user_constraints: dict[str, Any] | None = None,
        evolve: bool = False,
    ) -> Behavior:
        """Create a behavior using orchestrated agents."""

        # Create task context
        context = TaskContext(
            query=description,
            user_id="system",
            working_memory={"constraints": user_constraints},
        )

        # Create shared memory for this task
        orch_memory = OrchestrationMemory(
            memory=self.memory,
            task_id=context.task_id,
        )

        # Optionally modify config based on request
        config = self.config
        if not evolve:
            config = self._remove_phase(config, "evolve")

        # Orchestrate!
        result = await self.orchestrator.orchestrate_from_config(
            config=config,
            context=context,
            executor=self._create_executor(orch_memory),
            memory=orch_memory,
        )

        if not result.success:
            raise BehaviorCreationError(result.error)

        # Extract behavior from memory
        behavior = await orch_memory.retrieve_by_key("final_behavior")

        # Cleanup (keep insights for learning)
        await orch_memory.cleanup(keep_insights=True)

        return behavior

    def _create_executor(
        self,
        memory: OrchestrationMemory,
    ) -> AgentExecutor:
        """Create the agent executor with shared memory access."""

        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            """Execute a phase, using shared memory for I/O."""

            try:
                if agent.agent_id == "research":
                    # Semantic retrieval - get any relevant prior context
                    prior = await memory.semantic_retrieve(
                        "previous research or domain knowledge",
                        limit=3,
                    )

                    research = await self._do_research(
                        context.query,
                        prior_context=prior,
                    )

                    await memory.publish(
                        producer_agent="research",
                        key="domain_research",
                        content=research,
                    )

                    return AgentResult(
                        agent_id="research",
                        success=True,
                        output=research,
                    )

                elif agent.agent_id == "design":
                    # Get research from memory
                    research = await memory.retrieve_by_key("domain_research")

                    design = await self._do_design(research, context)

                    await memory.publish(
                        producer_agent="design",
                        key="behavior_design",
                        content=design,
                    )

                    return AgentResult(
                        agent_id="design",
                        success=True,
                        output=design,
                    )

                elif agent.agent_id == "generate_prompts":
                    design = await memory.retrieve_by_key("behavior_design")

                    prompts = await self._generate_prompts(design)

                    await memory.publish(
                        producer_agent="generate_prompts",
                        key="behavior_prompts",
                        content=prompts,
                    )

                    return AgentResult(
                        agent_id="generate_prompts",
                        success=True,
                        output=prompts,
                    )

                elif agent.agent_id == "generate_tests":
                    # RUNS IN PARALLEL with generate_prompts
                    design = await memory.retrieve_by_key("behavior_design")

                    tests = await self._generate_tests(design)

                    await memory.publish(
                        producer_agent="generate_tests",
                        key="test_cases",
                        content=tests,
                    )

                    return AgentResult(
                        agent_id="generate_tests",
                        success=True,
                        output=tests,
                    )

                # ... etc

            except Exception as e:
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=False,
                    error=str(e),
                )

        return executor
```

---

## Implementation Plan (Revised)

### Phase 1: Foundation (Days 1-5)

| Day | Task |
|-----|------|
| 1 | Implement `OrchestrationMemory` with Qdrant backend |
| 2 | Add `semantic_retrieve()` and cross-agent communication |
| 3 | Implement parallel mode in `MultiAgentOrchestrator` |
| 4 | Add `FailureAssessor` with LLM-based assessment |
| 5 | Add historical pattern matching for failures |

### Phase 2: Opinion Integration (Days 6-8)

| Day | Task |
|-----|------|
| 6 | Implement `OpinionBasedMerger` |
| 7 | Add voting and conflict detection |
| 8 | Integrate with existing opinion/belief services |

### Phase 3: Evolvable Orchestration (Days 9-11)

| Day | Task |
|-----|------|
| 9 | Create evolvable orchestration prompts |
| 10 | Implement `EvolvableOrchestrator` |
| 11 | Add orchestration evolution test harness |

### Phase 4: Behavior Architect Refactor (Days 12-15)

| Day | Task |
|-----|------|
| 12 | Define `OrchestrationConfig` for Behavior Architect |
| 13 | Create agent executor with memory integration |
| 14 | Refactor `create_behavior()` to use orchestrator |
| 15 | Add E2E tests proving parallel execution |

### Total: 15 days

---

## Key Design Decisions

### 1. Shared Memory via Qdrant

**Decision**: Use Qdrant for inter-agent communication

**Rationale**:
- Follows existing patterns (MemoryProvider)
- Enables semantic retrieval ("find relevant research")
- Persists across failures
- Enables cross-task learning
- Vector search for fuzzy matching

### 2. LLM-Assessed Failure Handling

**Decision**: Use LLM + historical patterns to decide on failures

**Rationale**:
- Context-aware decisions
- Learns from past failures
- Can evolve with Promptbreeder
- More intelligent than binary rules

### 3. Opinion Framework for Merging

**Decision**: Use existing opinion/belief services for conflict resolution

**Rationale**:
- Already built for multi-source reconciliation
- Has confidence tracking
- Can handle contradictions
- Consistent with cognitive architecture

### 4. Evolvable Orchestration Strategies

**Decision**: Make orchestration prompts evolvable

**Rationale**:
- The meta-level: "how we coordinate" can improve
- Test cases measure real outcomes (success, speed, cost)
- Self-improving coordination over time

### 5. Behavior-Specified Orchestration

**Decision**: Behaviors include orchestration hints that evolve

**Rationale**:
- Different behaviors have different needs
- Hints evolve alongside prompts
- Reduces hard-coding in orchestrator
- Enables behavior-specific optimization

---

*"Don't just coordinate agents. Let them think together, learn together, and evolve together."*

# draagon-ai Testing Principles

## Philosophy

Testing cognitive AI systems requires different approaches than traditional software. We must verify not just functionality, but cognitive behavior, belief consistency, and emergent properties.

## Testing Hierarchy

### 1. Unit Tests (Foundation)

Test individual components in isolation:

```python
# Memory layer tests
async def test_working_memory_capacity():
    """Verify Miller's Law enforcement."""
    memory = WorkingMemory(capacity=7)
    for i in range(10):
        await memory.add(f"item_{i}")
    assert len(memory) == 7  # Capacity enforced

# Belief tests
async def test_belief_reconciliation():
    """Verify conflicting observations reconcile."""
    service = BeliefReconciliationService()
    obs1 = Observation("The meeting is at 3pm", confidence=0.9)
    obs2 = Observation("The meeting is at 4pm", confidence=0.7)
    belief = await service.reconcile([obs1, obs2])
    assert belief.content == "The meeting is at 3pm"  # Higher confidence wins
```

### 2. Integration Tests (Critical)

Test component interactions with real implementations:

```python
# Agent loop with memory
async def test_react_loop_uses_memory():
    """Verify ReAct loop retrieves relevant memories."""
    memory = InMemoryProvider()
    await memory.store("User prefers Celsius", type="PREFERENCE")

    agent = Agent(memory=memory, llm=MockLLM())
    response = await agent.run("What's the temperature?")

    # Should have retrieved preference
    assert "celsius" in response.context_used.lower()

# Multi-agent coordination
async def test_parallel_agents_share_observations():
    """Verify observations propagate between parallel agents."""
    orchestrator = ParallelOrchestrator()

    result = await orchestrator.run(
        agents=[researcher, critic],
        query="Analyze this topic"
    )

    # Critic should see researcher's findings
    assert researcher.output in result.shared_context
```

### 3. Cognitive Tests (Unique to draagon-ai)

Test cognitive behaviors that define the framework:

```python
# Belief consistency
async def test_beliefs_remain_consistent():
    """Agent should not contradict its own beliefs."""
    agent = Agent()

    # Establish belief
    await agent.run("My name is Doug")

    # Query belief
    response = await agent.run("What's my name?")
    assert "Doug" in response.answer

    # Should not accept contradictory info without reconciliation
    response = await agent.run("Actually, my name is Steve")
    beliefs = await agent.get_beliefs()
    # Should flag conflict, not blindly accept
    assert beliefs.has_conflict("user_name")

# Curiosity behavior
async def test_curiosity_identifies_gaps():
    """Agent should identify missing information."""
    agent = Agent(curiosity_intensity=0.8)

    await agent.run("Schedule a meeting")
    gaps = await agent.curiosity.get_knowledge_gaps()

    # Should identify missing context
    assert any("when" in g.topic.lower() for g in gaps)
    assert any("who" in g.topic.lower() for g in gaps)

# Opinion evolution
async def test_opinions_evolve_with_evidence():
    """Opinions should change given strong evidence."""
    agent = Agent()

    # Form initial opinion
    opinion = await agent.form_opinion("Is Python good for beginners?")
    initial_confidence = opinion.confidence

    # Present strong counter-evidence
    await agent.run("Studies show Python's error messages confuse beginners")

    # Opinion should shift
    new_opinion = await agent.get_opinion("Python for beginners")
    assert new_opinion.confidence < initial_confidence
```

### 4. Stress Tests

Test system behavior under load:

```python
async def test_parallel_agents_no_race_conditions():
    """10 concurrent agents should not corrupt shared state."""
    orchestrator = ParallelOrchestrator()

    results = await orchestrator.run(
        agents=[Agent() for _ in range(10)],
        query="Analyze this data"
    )

    # All agents completed
    assert len(results.agent_results) == 10
    # No duplicate observations
    observations = results.shared_memory.get_all()
    assert len(set(o.id for o in observations)) == len(observations)

async def test_memory_under_load():
    """Memory should handle high throughput."""
    memory = WorkingMemory()

    # Rapid insertions
    for i in range(1000):
        await memory.add(f"observation_{i}")

    # Should maintain capacity
    assert len(memory) <= memory.capacity
    # Should retain highest-priority items
    recent = await memory.get_active_context()
    assert len(recent) > 0
```

### 5. Benchmark Tests

Test against established benchmarks:

```python
# MultiAgentBench scenarios
async def test_multiagentbench_research():
    """Should achieve >90% on research collaboration."""
    orchestrator = CognitiveSwarmOrchestrator()

    result = await orchestrator.run(
        scenario=MULTIAGENTBENCH_RESEARCH_PROPOSAL,
    )

    assert result.task_score >= 0.90

# Memory benchmark
async def test_memoryagentbench_conflict():
    """Should resolve conflicting information correctly."""
    agent = Agent()

    score = await run_memoryagentbench(
        agent,
        scenario="FactConsolidation"
    )

    assert score >= 0.70  # Target: 70%+ (vs 45% SOTA)

# Theory of Mind
async def test_hitom_second_order():
    """Should handle 2nd-order belief reasoning."""
    agent = Agent()

    score = await run_hitom_benchmark(agent, order=2)

    assert score >= 0.60  # Target: 60%+
```

## Test Categories by Priority

### P0 - Critical (Block Release)

- Constitution violations (regex for semantics, JSON output)
- Memory corruption under concurrency
- Belief consistency failures
- Decision engine crashes

### P1 - Important (Should Fix)

- Performance regressions (>20% slowdown)
- Memory capacity violations
- Incomplete belief reconciliation
- Curiosity over-questioning

### P2 - Nice to Have

- Edge case handling
- Documentation tests
- Style consistency

## Testing Anti-Patterns

### Avoid

1. **Mocking LLMs for semantic tests**
   - Use real LLMs or deterministic test doubles
   - Mocks hide semantic failures

2. **Testing implementation, not behavior**
   - Don't test private methods
   - Test observable outcomes

3. **Ignoring cognitive properties**
   - Don't just test "it works"
   - Test "it thinks correctly"

4. **Serial-only multi-agent tests**
   - Test actual parallelism
   - Race conditions hide in serial tests

### Prefer

1. **Real integrations over mocks**
   ```python
   # Prefer
   llm = GroqProvider(model="llama-3.1-8b")

   # Avoid
   llm = Mock(return_value="yes")
   ```

2. **Behavioral assertions**
   ```python
   # Prefer
   assert agent.believes("user prefers celsius")

   # Avoid
   assert memory._internal_store["pref_123"].value == "celsius"
   ```

3. **Property-based testing for cognitive systems**
   ```python
   @given(observations=lists(observations()))
   async def test_beliefs_always_consistent(observations):
       service = BeliefService()
       for obs in observations:
           await service.add_observation(obs)
       beliefs = await service.get_all_beliefs()
       assert not beliefs.has_contradictions()
   ```

## Test Infrastructure

### Required

- **pytest** with async support
- **pytest-asyncio** for async tests
- **hypothesis** for property-based tests
- **pytest-benchmark** for performance
- **coverage.py** for coverage tracking

### Test Organization

```
tests/
├── unit/
│   ├── orchestration/
│   │   ├── test_loop.py
│   │   ├── test_decision.py
│   │   └── test_execution.py
│   ├── memory/
│   │   ├── test_working.py
│   │   ├── test_episodic.py
│   │   └── test_semantic.py
│   └── cognition/
│       ├── test_beliefs.py
│       ├── test_curiosity.py
│       └── test_opinions.py
├── integration/
│   ├── test_agent_memory.py
│   ├── test_multi_agent.py
│   └── test_cognitive_loop.py
├── cognitive/
│   ├── test_belief_consistency.py
│   ├── test_opinion_evolution.py
│   └── test_knowledge_gaps.py
├── stress/
│   ├── test_concurrent_agents.py
│   └── test_memory_load.py
└── benchmarks/
    ├── test_multiagentbench.py
    ├── test_memoryagentbench.py
    └── test_hitom.py
```

## Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| Orchestration | 90% |
| Memory Layers | 85% |
| Cognitive Services | 80% |
| Multi-Agent | 80% |
| Tool System | 90% |

---

**Document Status**: Active
**Last Updated**: 2025-12-30

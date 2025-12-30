---
name: test
description: Generate and execute tests for cognitive AI components
type: workflow
tools: [Read, Write, Edit, Bash, Glob, Grep, TodoWrite]
model: claude-sonnet-4-5-20250929
---

# /test - Cognitive Test Generator and Executor

## Purpose
Generate and execute tests for cognitive AI components, including unit tests, integration tests, cognitive behavior tests, and benchmark tests.

## Usage
```
/test [area: unit|integration|cognitive|benchmark|all] [component]
```

## Process

When this command is invoked:

1. **Identify Test Scope**
   - Parse test area and component from arguments
   - Locate relevant source files in `src/draagon_ai/`
   - Review testing principles in `.specify/constitution/testing-principles.md`

2. **Analyze Component**
   - Read source code for component
   - Identify cognitive aspects:
     - Memory layer interactions
     - Belief handling
     - Multi-agent coordination
   - Determine test categories needed

3. **Generate Tests**

   **Unit Tests** (`tests/unit/`):
   - Individual function/method tests
   - Dataclass validation
   - Protocol compliance
   - Edge cases

   **Integration Tests** (`tests/integration/`):
   - Component interactions
   - Memory + orchestration
   - Cognitive service + agent loop

   **Cognitive Tests** (`tests/cognitive/`):
   - Belief consistency
   - Memory capacity (Miller's Law)
   - Opinion evolution
   - Curiosity behavior

   **Benchmark Tests** (`tests/benchmarks/`):
   - MultiAgentBench scenarios
   - MemoryAgentBench scenarios
   - HI-TOM scenarios

4. **Execute Tests**
   - Run pytest with appropriate markers
   - Collect coverage data
   - Report failures and coverage

5. **Update Documentation**
   - Add test files
   - Update test coverage tracking

6. **Stage Changes**
   - Use `git add .` to stage test files
   - DO NOT commit
   - Provide test results summary

## Test Categories

### Unit Tests (/test unit)
```python
class TestDecisionEngine:
    @pytest.mark.asyncio
    async def test_decide_returns_action(self):
        engine = DecisionEngine(mock_llm)
        result = await engine.decide(context)
        assert result.action is not None

    @pytest.mark.asyncio
    async def test_decide_with_empty_context(self):
        engine = DecisionEngine(mock_llm)
        with pytest.raises(ValueError):
            await engine.decide(None)
```

### Integration Tests (/test integration)
```python
class TestAgentMemoryIntegration:
    @pytest.mark.asyncio
    async def test_agent_uses_memory(self):
        memory = InMemoryProvider()
        await memory.store("User prefers Celsius", type="PREFERENCE")

        agent = Agent(memory=memory, llm=real_llm)
        response = await agent.run("What temperature format?")

        assert "celsius" in response.lower()
```

### Cognitive Tests (/test cognitive)
```python
class TestBeliefConsistency:
    @pytest.mark.asyncio
    async def test_beliefs_dont_contradict(self):
        agent = Agent()

        await agent.run("My name is Doug")
        response = await agent.run("What's my name?")

        assert "Doug" in response

    @pytest.mark.asyncio
    async def test_conflicting_info_flags_conflict(self):
        agent = Agent()

        await agent.run("My birthday is March 15")
        await agent.run("My birthday is April 20")

        beliefs = await agent.get_beliefs("birthday")
        assert beliefs.has_conflict


class TestMemoryCapacity:
    @pytest.mark.asyncio
    async def test_working_memory_millers_law(self):
        memory = WorkingMemory(capacity=7)

        for i in range(10):
            await memory.add(f"item_{i}")

        assert len(memory) == 7  # Capacity enforced
```

### Benchmark Tests (/test benchmark)
```python
class TestMultiAgentBench:
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_research_collaboration(self):
        orchestrator = CognitiveSwarmOrchestrator()

        result = await orchestrator.run(
            scenario=RESEARCH_PROPOSAL_SCENARIO
        )

        assert result.task_score >= 0.90  # Target: 90%+


class TestMemoryAgentBench:
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_fact_consolidation(self):
        agent = Agent()

        score = await run_fact_consolidation(agent)

        assert score >= 0.70  # Target: 70%+
```

## Test Fixtures

### Common Fixtures
```python
@pytest.fixture
def mock_llm():
    """Deterministic LLM for unit tests."""
    return DeterministicLLM(responses={
        "decide": "<response><action>answer</action></response>"
    })

@pytest.fixture
def real_llm():
    """Real LLM for integration tests."""
    return GroqProvider(model="llama-3.1-8b-instant")

@pytest.fixture
def memory_provider():
    """In-memory provider for tests."""
    return InMemoryMemoryProvider()

@pytest.fixture
def cognitive_agent(memory_provider, real_llm):
    """Fully configured agent for cognitive tests."""
    return Agent(
        memory=memory_provider,
        llm=real_llm,
        belief_service=BeliefService(real_llm),
        curiosity=CuriosityEngine(real_llm),
    )
```

## Output Format
Provide:
- Test execution results
- Coverage report
- Failed tests with details
- Cognitive behavior verification
- Recommendations for additional tests

## pytest Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/unit/ -v
pytest tests/cognitive/ -v
pytest tests/benchmarks/ -v --benchmark

# Run with coverage
pytest tests/ --cov=src/draagon_ai --cov-report=html

# Run specific test file
pytest tests/cognitive/test_beliefs.py -v
```

Remember: Test cognitive behavior, not just functionality. Verify beliefs remain consistent, memory capacity is enforced, and multi-agent coordination works correctly.

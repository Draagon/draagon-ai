# FR-010: Real Agent Integration Tests

**Status:** Draft
**Priority:** High
**Complexity:** High
**Created:** 2026-01-01
**Updated:** 2026-01-01
**Depends On:** FR-009 (Integration Testing Framework)

---

## Overview

Comprehensive end-to-end integration tests that validate actual agent behavior using real LLM providers, Neo4j memory, and cognitive services. These tests extend the FR-009 testing framework from mock agents to real production components, ensuring the entire cognitive architecture works correctly in practice.

**Core Principle:** Test real agent behavior with real components. Validate that the cognitive architecture (memory, learning, beliefs, reasoning) works correctly when integrated, not just in isolation.

---

## Motivation

### Current State

FR-009 provides testing infrastructure (seeds, evaluators, sequences) but uses **mock agents** with canned responses. This validates framework mechanics but doesn't test actual agent capabilities:

- ✅ **Framework validated**: SeedFactory, TestDatabase, LLM-as-judge work correctly
- ❌ **Agent untested**: Real agent query → decision → action → response flow untested
- ❌ **Memory untested**: Neo4jMemoryProvider storage, search, reinforcement untested
- ❌ **Learning untested**: Skill extraction, belief reconciliation untested
- ❌ **Cognitive untested**: Multi-step reasoning, tool execution untested

### Problems

1. **Integration Bugs Escape**: Bugs that only appear when components integrate aren't caught
2. **Memory Behavior Unknown**: Don't validate that memories actually persist/retrieve correctly
3. **Learning Unvalidated**: Can't confirm agent actually learns from interactions
4. **Confidence Untested**: Don't validate that confidence-based actions work in practice
5. **Performance Unknown**: No baseline for real-world latency/throughput

### Research Basis

| Aspect | Research | Application |
|--------|----------|-------------|
| **End-to-End Testing** | "Testing Machine Learning Systems" (Google) | Validate full pipeline, not just units |
| **Cognitive Benchmarks** | MultiAgentBench, AgentBench | Compare against industry standards |
| **Memory Validation** | Baddeley's Working Memory Model | Test capacity limits (7±2), decay |
| **LLM-as-Judge** | "Judging LLM-as-a-Judge" (2023) | Semantic evaluation reliability |
| **ReAct Reasoning** | "ReAct: Synergizing Reasoning and Acting" (2022) | Validate multi-step traces |

---

## Requirements

### FR-010.1: Core Agent Processing Tests

**Description:** Validate the complete agent query processing pipeline from input to response.

**What It Tests:**
- `AgentLoop.process()` - Full query → decision → action → response
- `DecisionEngine.decide()` - LLM-based action selection
- `ActionExecutor.execute()` - Tool invocation and result handling
- Session management and context tracking

**Test Scenarios:**

```python
from draagon_ai.testing import SeedFactory, SeedItem, AgentEvaluator
from draagon_ai.orchestration import AgentLoop, AgentLoopConfig
from draagon_ai.memory import Neo4jMemoryProvider, MemoryType, MemoryScope

@pytest.mark.agent_integration
class TestAgentCore:
    """Test core agent query processing."""

    @pytest.mark.asyncio
    async def test_simple_query_direct_answer(self, agent, evaluator):
        """Agent answers simple query without tools."""
        response = await agent.process("What is 2+2?")

        result = await evaluator.evaluate_correctness(
            query="What is 2+2?",
            expected_outcome="Agent says 4",
            actual_response=response.answer,
        )

        assert result.correct
        assert response.confidence > 0.9

    @pytest.mark.asyncio
    async def test_query_requires_tool(self, agent, tool_registry, evaluator):
        """Agent uses tool when needed."""

        # Register test tool
        @tool_registry.register
        async def get_weather(location: str) -> dict:
            return {"temp": 72, "condition": "sunny"}

        response = await agent.process("What's the weather in Portland?")

        result = await evaluator.evaluate_correctness(
            query="What's the weather in Portland?",
            expected_outcome="Mentions temperature and sunny condition",
            actual_response=response.answer,
        )

        assert result.correct
        # Don't assert WHICH tool was used - test outcome, not process

    @pytest.mark.asyncio
    async def test_confidence_affects_response(self, agent):
        """Low confidence leads to hedging."""
        response = await agent.process("What's the population of Xanadu?")

        # Fictional place should have low confidence
        assert response.confidence < 0.5
        # Should hedge (but don't regex - use evaluator)
```

**Acceptance Criteria:**
- ✅ Test covers simple query → answer
- ✅ Test covers query → tool → answer
- ✅ Test covers confidence-based responses
- ✅ Test covers error recovery
- ✅ Test covers session persistence

**Success Metrics:**
- Agent responds within 2s for simple queries
- Agent responds within 5s for tool-requiring queries
- Confidence calibration: <0.5 when uncertain, >0.8 when certain
- 95%+ success rate on test suite

---

### FR-010.2: Memory Integration Tests

**Description:** Validate memory storage, retrieval, reinforcement, and layer promotion with real Neo4jMemoryProvider.

**What It Tests:**
- Memory persistence across agent restarts
- Semantic search correctness
- Memory reinforcement (boost on success, demote on failure)
- Layer promotion (working → episodic → semantic → metacognitive)
- TTL enforcement and expiration

**Test Scenarios:**

```python
@pytest.mark.memory_integration
class TestAgentMemory:
    """Test memory persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_store_and_recall_fact(self, agent, memory_provider, evaluator):
        """Agent stores fact and recalls it later."""

        # Teach agent a fact
        response1 = await agent.process("My birthday is March 15")

        # Verify storage
        memories = await memory_provider.search(
            query="birthday",
            user_id="test_user",
            limit=5,
        )
        assert len(memories) > 0
        assert any("March 15" in m.content for m in memories)

        # Verify recall
        response2 = await agent.process("When is my birthday?")

        result = await evaluator.evaluate_correctness(
            query="When is my birthday?",
            expected_outcome="Agent says March 15",
            actual_response=response2.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_memory_reinforcement_boost(self, agent, memory_provider):
        """Using a memory successfully boosts its importance."""

        # Store initial memory
        memory_id = await memory_provider.store(
            content="Doug prefers Celsius",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.5,
        )

        # Agent uses memory successfully
        response = await agent.process("What temperature unit do I prefer?")

        # Record success (simulated - real implementation would track this)
        await memory_provider.record_usage(memory_id, outcome="success")

        # Check importance increased
        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance > 0.5  # Boosted by 0.05

    @pytest.mark.asyncio
    async def test_layer_promotion(self, agent, memory_provider):
        """Repeated successful use promotes memory through layers."""

        memory_id = await memory_provider.store(
            content="Restart command: sudo systemctl restart myservice",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            importance=0.6,  # Starts in working memory
        )

        # Use skill multiple times successfully
        for _ in range(10):
            await memory_provider.record_usage(memory_id, outcome="success")

        # Check promoted to higher layer
        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance >= 0.7  # Promoted to episodic

        # More uses
        for _ in range(20):
            await memory_provider.record_usage(memory_id, outcome="success")

        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance >= 0.8  # Promoted to semantic

    @pytest.mark.asyncio
    async def test_memory_expiration(self, agent, memory_provider):
        """Memories expire based on TTL and layer."""

        # Store short-lived working memory
        memory_id = await memory_provider.store(
            content="Temporary note: Call back at 3pm",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.SESSION,
            importance=0.3,  # Working memory (5 min TTL)
        )

        # Fast-forward time (test utility)
        await advance_time(minutes=6)

        # Memory should be expired/deleted
        memory = await memory_provider.get(memory_id)
        assert memory is None or memory.expired
```

**Acceptance Criteria:**
- ✅ Memories persist across agent sessions
- ✅ Semantic search returns relevant memories
- ✅ Reinforcement boost increases importance
- ✅ Layer promotion happens at correct thresholds
- ✅ TTL expiration works correctly

**Success Metrics:**
- Search recall@5: >80% for relevant memories
- Reinforcement: +0.05 importance per success
- Promotion: working→episodic at 0.7, episodic→semantic at 0.8
- TTL accuracy: ±1 minute

---

### FR-010.3: Learning Integration Tests

**Description:** Validate that agents extract skills, learn facts, and apply corrections from interactions.

**What It Tests:**
- Autonomous skill extraction from successful interactions
- Fact learning from user statements
- Correction acceptance and belief updates
- Skill verification after learning
- Multi-user knowledge integration

**Test Scenarios:**

```python
from draagon_ai.cognition.learning import LearningService

@pytest.mark.learning_integration
class TestAgentLearning:
    """Test agent learning from interactions."""

    @pytest.mark.asyncio
    async def test_extract_skill_from_success(self, agent, memory_provider):
        """Agent learns skill from successful execution."""

        # Manually perform task successfully
        response = await agent.process(
            "Restart the web server",
            tool_result="sudo systemctl restart nginx - Success"
        )

        # Learning service should extract skill
        skills = await memory_provider.search(
            query="restart nginx",
            memory_types=[MemoryType.SKILL],
            limit=5,
        )

        assert len(skills) > 0
        assert any("systemctl restart nginx" in s.content for s in skills)

    @pytest.mark.asyncio
    async def test_learn_fact_from_statement(self, agent, memory_provider, evaluator):
        """Agent learns and stores facts from user statements."""

        # User states fact
        response = await agent.process("I have 3 cats named Whiskers, Mittens, and Shadow")

        # Fact should be stored
        facts = await memory_provider.search(
            query="cats",
            memory_types=[MemoryType.FACT],
            user_id="test_user",
        )

        assert len(facts) > 0

        # Agent should recall
        response2 = await agent.process("What are my cats' names?")
        result = await evaluator.evaluate_correctness(
            query="What are my cats' names?",
            expected_outcome="Lists Whiskers, Mittens, Shadow",
            actual_response=response2.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_apply_correction(self, agent, memory_provider):
        """Agent updates beliefs when corrected."""

        # Initial wrong belief
        await agent.process("I have 3 cats")

        # User corrects
        await agent.process("Actually, I have 4 cats now. Got a new one!")

        # Check belief updated
        beliefs = await memory_provider.search(
            query="how many cats",
            memory_types=[MemoryType.FACT, MemoryType.BELIEF],
        )

        # Should reflect correction
        assert any("4" in b.content for b in beliefs)

    @pytest.mark.asyncio
    async def test_skill_verification(self, agent, memory_provider):
        """Agent verifies learned skills actually work."""

        # Learn potentially broken skill
        memory_id = await memory_provider.store(
            content="To restart: sudo reboot-everything",
            memory_type=MemoryType.SKILL,
            importance=0.7,
        )

        # Agent tries to use it
        response = await agent.process("Restart the system")

        # If skill fails, confidence should drop
        updated = await memory_provider.get(memory_id)
        if response.success is False:
            assert updated.confidence < 0.7  # Demoted
```

**Acceptance Criteria:**
- ✅ Skills extracted from successful interactions
- ✅ Facts learned from user statements
- ✅ Corrections update existing beliefs
- ✅ Skill verification detects broken skills
- ✅ Multi-user knowledge properly scoped

**Success Metrics:**
- Skill extraction accuracy: >70%
- Fact recall after learning: >90%
- Correction acceptance: >95%
- Broken skill detection: >80%

---

### FR-010.4: Belief Reconciliation Tests

**Description:** Validate that agents reconcile conflicting observations into coherent beliefs.

**What It Tests:**
- Conflict detection between observations
- Credibility-weighted belief formation
- Clarification detection and queueing
- Multi-user observation handling

**Test Scenarios:**

```python
from draagon_ai.cognition.beliefs import BeliefReconciliationService

@pytest.mark.belief_integration
class TestBeliefReconciliation:
    """Test belief formation and conflict resolution."""

    @pytest.mark.asyncio
    async def test_reconcile_conflicting_observations(self, agent, belief_service):
        """Agent reconciles contradictory information."""

        # User 1 says 3 cats
        await agent.process("I have 3 cats", user_id="doug")

        # User 2 says 4 cats (different user in same household)
        await agent.process("We have 4 cats", user_id="sarah")

        # Belief service should detect conflict
        beliefs = await belief_service.get_beliefs(
            scope="household",
            belief_type="HOUSEHOLD_FACT"
        )

        # Should either:
        # 1. Form tentative belief with low confidence
        # 2. Queue clarification question
        belief = next((b for b in beliefs if "cat" in b.content.lower()), None)
        assert belief is not None
        assert belief.confidence < 0.8 or belief.needs_clarification

    @pytest.mark.asyncio
    async def test_credibility_weighting(self, agent, belief_service):
        """More credible sources get higher weight."""

        # Expert source
        await agent.process(
            "The capital of France is Paris",
            source_credibility=0.95  # Wikipedia, verified source
        )

        # Unreliable source
        await agent.process(
            "The capital of France is Lyon",
            source_credibility=0.3  # Random forum post
        )

        # Belief should favor expert
        belief = await belief_service.get_belief("capital of France")
        assert "Paris" in belief.content
        assert belief.confidence > 0.8

    @pytest.mark.asyncio
    async def test_clarification_queueing(self, agent, curiosity_service):
        """Agent queues questions for unclear situations."""

        # Ambiguous statement
        await agent.process("I might have mentioned my cats before...")

        # Curiosity service should queue clarification
        questions = await curiosity_service.get_pending_questions()

        assert len(questions) > 0
        assert any("cat" in q.content.lower() for q in questions)
        assert any(q.question_type == "CLARIFICATION" for q in questions)
```

**Acceptance Criteria:**
- ✅ Conflicts detected between observations
- ✅ Credibility weights influence beliefs
- ✅ Clarifications queued appropriately
- ✅ Multi-user observations scoped correctly
- ✅ Belief confidence calibrated accurately

**Success Metrics:**
- Conflict detection: >85%
- Credibility weighting correlation: >0.7
- Clarification queue precision: >80%

---

### FR-010.5: ReAct Reasoning Tests

**Description:** Validate multi-step reasoning with thought, action, observation loops.

**What It Tests:**
- `LoopMode.REACT` multi-step reasoning
- Thought → Action → Observation traces
- Tool usage within reasoning loop
- Final answer synthesis

**Test Scenarios:**

```python
from draagon_ai.orchestration import LoopMode, ReActStep

@pytest.mark.react_integration
class TestReActReasoning:
    """Test multi-step ReAct reasoning."""

    @pytest.mark.asyncio
    async def test_react_multi_step_trace(self, agent, evaluator):
        """Agent uses ReAct for complex query."""

        # Configure for ReAct mode
        agent.config.loop_mode = LoopMode.REACT

        response = await agent.process(
            "What's the weather like in Portland today, and should I bring an umbrella?"
        )

        # Check trace contains expected steps
        assert len(response.react_trace) > 0

        # Should have THOUGHT steps
        thoughts = [s for s in response.react_trace if s.step_type == "THOUGHT"]
        assert len(thoughts) > 0

        # Should have ACTION steps (get_weather)
        actions = [s for s in response.react_trace if s.step_type == "ACTION"]
        assert len(actions) > 0

        # Should have OBSERVATION steps
        observations = [s for s in response.react_trace if s.step_type == "OBSERVATION"]
        assert len(observations) > 0

        # Final answer should address both questions
        result = await evaluator.evaluate_correctness(
            query="What's the weather and should I bring umbrella?",
            expected_outcome="Mentions current weather and umbrella recommendation",
            actual_response=response.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_react_tool_usage(self, agent):
        """ReAct mode correctly invokes tools during reasoning."""

        agent.config.loop_mode = LoopMode.REACT

        response = await agent.process(
            "Search for recent papers on transformer models and summarize findings"
        )

        # Should have used search tool
        actions = [s for s in response.react_trace if s.step_type == "ACTION"]
        assert any("search" in s.content.lower() for s in actions)
```

**Acceptance Criteria:**
- ✅ ReAct traces contain THOUGHT/ACTION/OBSERVATION steps
- ✅ Tools invoked correctly within reasoning loop
- ✅ Observations integrated into reasoning
- ✅ Final answer synthesizes all steps
- ✅ Trace correctness validated

**Success Metrics:**
- Average steps per complex query: 3-7
- Tool invocation accuracy: >90%
- Final answer relevance: >85% (LLM-as-judge)

---

### FR-010.6: Tool Execution Tests

**Description:** Validate tool discovery, registration, execution, timeout, and metrics.

**What It Tests:**
- Tool discovery from `@tool` decorator
- Parameter validation
- Timeout enforcement
- Error handling
- Metrics collection (invocation count, success rate, latency)

**Test Scenarios:**

```python
from draagon_ai.tools import tool, ToolRegistry

@pytest.mark.tool_integration
class TestToolExecution:
    """Test tool discovery and execution."""

    @pytest.mark.asyncio
    async def test_tool_discovery(self, tool_registry):
        """Agent discovers registered tools."""

        @tool(name="test_tool", description="A test tool")
        async def my_tool(arg: str) -> str:
            return f"Result: {arg}"

        # Tool should be in registry
        tools = tool_registry.list_tools()
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self, agent):
        """Agent validates tool parameters before execution."""

        @tool(name="add", description="Add two numbers")
        async def add_numbers(a: int, b: int) -> int:
            return a + b

        # Valid parameters
        result = await agent.execute_tool("add", {"a": 2, "b": 3})
        assert result.success
        assert result.data == 5

        # Invalid parameters (missing required)
        result = await agent.execute_tool("add", {"a": 2})
        assert not result.success
        assert "missing" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_timeout(self, agent):
        """Tools that exceed timeout are terminated."""

        @tool(name="slow_tool", timeout=1.0)
        async def slow_operation() -> str:
            await asyncio.sleep(5)  # Exceeds timeout
            return "done"

        result = await agent.execute_tool("slow_tool", {})

        assert not result.success
        assert result.timeout
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_metrics(self, tool_registry):
        """Tool metrics track invocations and success rate."""

        @tool(name="flaky_tool")
        async def flaky() -> str:
            import random
            if random.random() < 0.5:
                raise Exception("Random failure")
            return "success"

        # Execute multiple times
        for _ in range(10):
            try:
                await agent.execute_tool("flaky_tool", {})
            except:
                pass

        # Check metrics
        metrics = tool_registry.get_metrics("flaky_tool")
        assert metrics.invocation_count == 10
        assert 0.3 < metrics.success_rate < 0.7  # ~50% success
        assert metrics.average_latency > 0
```

**Acceptance Criteria:**
- ✅ Tools discovered from decorator
- ✅ Parameters validated before execution
- ✅ Timeouts enforced correctly
- ✅ Errors handled gracefully
- ✅ Metrics collected accurately

**Success Metrics:**
- Tool discovery: 100% of registered tools
- Parameter validation accuracy: >95%
- Timeout enforcement: ±100ms
- Metrics accuracy: 100%

---

### FR-010.7: Multi-Agent Coordination Tests

**Description:** Validate shared working memory, role-based filtering, and agent coordination.

**What It Tests:**
- SharedWorkingMemory observation flow
- Role-based context filtering (CRITIC, RESEARCHER, EXECUTOR)
- Attention weighting and decay
- Belief candidate identification
- Concurrent access safety

**Test Scenarios:**

```python
from draagon_ai.orchestration.shared_memory import SharedWorkingMemory, AgentRole

@pytest.mark.multiagent_integration
class TestMultiAgentCoordination:
    """Test multi-agent coordination with shared memory."""

    @pytest.mark.asyncio
    async def test_shared_memory_observation_flow(self, agent_factory):
        """Agents share observations via shared memory."""

        researcher = await agent_factory.create(RESEARCHER_PROFILE)
        executor = await agent_factory.create(EXECUTOR_PROFILE)

        shared_memory = SharedWorkingMemory(task_id="test_task")

        # Researcher gathers info
        await shared_memory.add_observation(
            content="User prefers dark mode",
            source_agent_id="researcher",
            attention_weight=0.9,
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )

        # Executor retrieves context
        context = await shared_memory.get_context_for_agent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
        )

        # Executor should see the observation
        assert len(context) > 0
        assert any("dark mode" in obs.content for obs in context)

    @pytest.mark.asyncio
    async def test_role_based_filtering(self, shared_memory):
        """Different roles see different context."""

        # Add various observation types
        await shared_memory.add_observation(
            content="User likes sci-fi",
            belief_type="PREFERENCE",
            is_belief_candidate=True,
        )
        await shared_memory.add_observation(
            content="API endpoint: /preferences",
            belief_type="FACT",
            is_belief_candidate=False,
        )
        await shared_memory.add_observation(
            content="How to update: PUT /preferences",
            belief_type="SKILL",
            is_belief_candidate=False,
        )

        # CRITIC sees only belief candidates
        critic_context = await shared_memory.get_context_for_agent(
            agent_id="critic",
            role=AgentRole.CRITIC,
        )
        assert len(critic_context) == 1
        assert critic_context[0].is_belief_candidate

        # EXECUTOR sees FACT and SKILL
        executor_context = await shared_memory.get_context_for_agent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
        )
        assert len(executor_context) == 2

        # RESEARCHER sees all
        researcher_context = await shared_memory.get_context_for_agent(
            agent_id="researcher",
            role=AgentRole.RESEARCHER,
        )
        assert len(researcher_context) == 3

    @pytest.mark.asyncio
    async def test_attention_decay(self, shared_memory):
        """Attention weights decay over time."""

        obs_id = await shared_memory.add_observation(
            content="Important info",
            attention_weight=1.0,
        )

        # Apply decay
        await shared_memory.apply_attention_decay()

        obs = await shared_memory.get_observation(obs_id)
        assert obs.attention_weight == 0.9  # Decayed by 0.1

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, shared_memory):
        """Shared memory handles concurrent access safely."""

        async def add_observation(i):
            await shared_memory.add_observation(
                content=f"Observation {i}",
                source_agent_id=f"agent_{i}",
            )

        # Concurrent additions
        await asyncio.gather(*[add_observation(i) for i in range(100)])

        # All observations should be stored
        observations = await shared_memory.get_all_observations()
        assert len(observations) == 100
```

**Acceptance Criteria:**
- ✅ Observations shared between agents
- ✅ Role-based filtering works correctly
- ✅ Attention decay applies properly
- ✅ Belief candidates identified
- ✅ Concurrent access safe

**Success Metrics:**
- Observation propagation: 100%
- Role filtering accuracy: 100%
- Concurrent access: No data loss
- Attention decay precision: ±0.01

---

## Test Fixtures Architecture

### Required Fixtures

```python
# tests/integration/agents/conftest.py

import pytest
from draagon_ai.orchestration import AgentLoop, AgentLoopConfig
from draagon_ai.memory import Neo4jMemoryProvider, Neo4jMemoryConfig
from draagon_ai.tools import ToolRegistry
from draagon_ai.testing import AgentEvaluator

@pytest.fixture(scope="session")
async def embedding_provider():
    """Real embedding provider for vector operations."""
    # Implementation depends on chosen provider (Qdrant, etc.)
    pass

@pytest.fixture(scope="session")
async def real_llm():
    """Real LLM provider (Groq/OpenAI based on env vars)."""
    import os
    if os.getenv("GROQ_API_KEY"):
        from draagon_ai.llm import GroqProvider
        return GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
    elif os.getenv("OPENAI_API_KEY"):
        from draagon_ai.llm import OpenAIProvider
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        pytest.skip("No LLM provider configured (need GROQ_API_KEY or OPENAI_API_KEY)")

@pytest.fixture
async def memory_provider(clean_database, embedding_provider, real_llm):
    """Real Neo4jMemoryProvider with semantic decomposition."""
    config = Neo4jMemoryConfig(
        uri=clean_database.get_config()["uri"],
        username=clean_database.get_config()["username"],
        password=clean_database.get_config()["password"],
    )

    provider = Neo4jMemoryProvider(
        config=config,
        embedding_provider=embedding_provider,
        llm=real_llm,
    )
    await provider.initialize()

    yield provider

    await provider.close()

@pytest.fixture
def tool_registry():
    """Fresh tool registry for each test."""
    return ToolRegistry()

@pytest.fixture
async def agent(memory_provider, real_llm, tool_registry):
    """Fully configured real agent."""
    config = AgentLoopConfig(
        personality="You are a helpful AI assistant for testing.",
        loop_mode=LoopMode.AUTO,
    )

    agent = AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        tool_registry=tool_registry,
        config=config,
    )

    return agent

@pytest.fixture
def evaluator(real_llm):
    """LLM-as-judge evaluator with real LLM."""
    return AgentEvaluator(llm=real_llm, max_retries=3)
```

---

## Test Organization

```
tests/integration/agents/
├── conftest.py                    # Real agent fixtures
├── test_agent_core.py             # FR-010.1: Query processing
├── test_agent_memory.py           # FR-010.2: Memory integration
├── test_agent_learning.py         # FR-010.3: Learning
├── test_agent_beliefs.py          # FR-010.4: Belief reconciliation
├── test_agent_react.py            # FR-010.5: ReAct reasoning
├── test_agent_tools.py            # FR-010.6: Tool execution
└── test_agent_multiagent.py       # FR-010.7: Multi-agent coordination
```

---

## Dependencies

### External Services

1. **Neo4j 5.26+** (with vector indexes)
   - Test database: `bolt://localhost:7687`
   - Credentials: Configured via environment

2. **LLM Provider** (one required):
   - Groq: `GROQ_API_KEY` environment variable
   - OpenAI: `OPENAI_API_KEY` environment variable
   - Cost: ~$0.01-0.10 per test run (LLM-as-judge calls)

3. **Embedding Provider**:
   - For vector similarity search
   - Qdrant, Pinecone, or built-in

### Python Packages

```
neo4j>=5.0
groq>=0.4.0
openai>=1.0
pytest>=7.0
pytest-asyncio>=0.21.0
```

---

## Performance Requirements

| Test Category | Max Latency | Success Rate |
|---------------|-------------|--------------|
| Core Agent | 2s (simple), 5s (tool) | >95% |
| Memory | 1s (store/retrieve) | >99% |
| Learning | 3s (extraction) | >70% |
| Beliefs | 2s (reconciliation) | >85% |
| ReAct | 10s (multi-step) | >80% |
| Tools | 500ms (execution) | >90% |
| Multi-Agent | 5s (coordination) | >85% |

---

## Success Criteria

### Quantitative Metrics

- **Test Coverage**: >80% of agent features
- **Pass Rate**: >95% on stable tests
- **Flakiness**: <5% flake rate
- **Performance**: All tests within latency requirements
- **LLM Costs**: <$1 per full test suite run

### Qualitative Metrics

- **Cognitive Validation**: Memory, learning, beliefs work in practice
- **Integration Confidence**: Full pipeline tested end-to-end
- **Regression Detection**: Catches integration bugs before production
- **Documentation**: Tests serve as usage examples

---

## Constitution Compliance

✅ **LLM-First Architecture**: All semantic evaluation uses LLM (AgentEvaluator), not regex
✅ **XML Output Format**: N/A (tests validate agent behavior, not output format)
✅ **Protocol-Based Design**: Tests use MemoryProvider protocol, not concrete classes
✅ **Pragmatic Async**: All async operations are I/O (LLM, DB, tools)
✅ **Test Outcomes, Not Processes**: Validate agent gets correct result, not specific method
✅ **Research-Grounded**: Follows established agent benchmarking patterns

---

## Implementation Phases

### Phase 1: Core Infrastructure (2 days)
- [ ] Create `tests/integration/agents/conftest.py` with real fixtures
- [ ] Implement `real_llm`, `memory_provider`, `agent` fixtures
- [ ] Test fixture initialization and cleanup

### Phase 2: Core Agent Tests (2 days)
- [ ] Implement `test_agent_core.py` (FR-010.1)
- [ ] Test simple queries, tool usage, confidence
- [ ] Validate session persistence

### Phase 3: Memory Tests (3 days)
- [ ] Implement `test_agent_memory.py` (FR-010.2)
- [ ] Test storage, retrieval, reinforcement
- [ ] Test layer promotion and TTL

### Phase 4: Learning & Beliefs (3 days)
- [ ] Implement `test_agent_learning.py` (FR-010.3)
- [ ] Implement `test_agent_beliefs.py` (FR-010.4)
- [ ] Test skill extraction, corrections, reconciliation

### Phase 5: Advanced Features (3 days)
- [ ] Implement `test_agent_react.py` (FR-010.5)
- [ ] Implement `test_agent_tools.py` (FR-010.6)
- [ ] Implement `test_agent_multiagent.py` (FR-010.7)

### Phase 6: CI/CD Integration (1 day)
- [ ] Add GitHub Actions workflow
- [ ] Configure test database
- [ ] Set up LLM API secrets
- [ ] Add cost monitoring

**Total Effort:** ~14 days

---

## Design Decisions

### Real Embeddings Required

**Decision:** Tests use **real embedding providers**, not mocks.

**Rationale:** We want real tests that validate actual behavior. Mock embeddings could hide bugs in:
- Vector similarity search
- Semantic decomposition
- Memory retrieval accuracy

**Implementation:**
```python
@pytest.fixture(scope="session")
async def embedding_provider():
    """Real embedding provider - required for integration tests."""
    from draagon_ai.llm import get_embedding_provider

    provider = get_embedding_provider()  # Uses configured provider
    await provider.initialize()
    return provider
```

### CI/CD Disabled Initially

**Decision:** GitHub Actions workflows created but **disabled by default**.

**Rationale:** Focus on getting tests working locally first. Enable CI when:
- All tests pass reliably
- Cost controls validated
- Team ready for automated runs

**Implementation:**
```yaml
# Workflows created but triggered only manually
on:
  workflow_dispatch:  # Manual trigger only
  # push:  # Disabled until ready
  #   branches: [main]
```

### LLM Tier Validation Required

**Decision:** Tests **must validate** LLM tier selection (local/complex/deep).

**Rationale:** Tier selection affects:
- Response quality
- Latency
- Cost

**Implementation:**
```python
@pytest.mark.tier_integration
class TestLLMTierSelection:
    """Validate LLM tier selection works correctly."""

    @pytest.mark.asyncio
    async def test_simple_query_uses_local_tier(self, agent):
        """Simple queries should use fast local tier."""
        response = await agent.process("What is 2+2?")
        assert response.model_tier == "local"

    @pytest.mark.asyncio
    async def test_complex_query_uses_complex_tier(self, agent):
        """Complex reasoning should use complex tier."""
        response = await agent.process(
            "Analyze the trade-offs between microservices and monoliths"
        )
        assert response.model_tier in ["complex", "deep"]

    @pytest.mark.asyncio
    async def test_deep_analysis_uses_deep_tier(self, agent):
        """Deep analysis explicitly requests deep tier."""
        response = await agent.process(
            "Provide a comprehensive analysis of climate change impacts",
            require_tier="deep"
        )
        assert response.model_tier == "deep"
```

---

## Related Work

- **FR-009**: Integration Testing Framework (provides infrastructure)
- **FR-001**: Shared Cognitive Working Memory (tested in FR-010.7)
- **FR-003**: Multi-Agent Belief Reconciliation (tested in FR-010.4)
- **FR-006**: Word Sense Disambiguation (used in memory tests)
- **FR-007**: Semantic Expansion Service (used in memory tests)

---

**Status:** Ready for implementation planning
**Next Steps:** Create task breakdown and begin Phase 1

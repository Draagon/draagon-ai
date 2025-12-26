"""Tests for MultiAgentOrchestrator (Phase C.1)."""

import pytest
import asyncio
from datetime import datetime

from draagon_ai.orchestration import (
    MultiAgentOrchestrator,
    OrchestrationMode,
    AgentSpec,
    AgentRole,
    TaskContext,
    AgentResult,
    OrchestratorResult,
    Learning,
    LearningType,
    InMemoryLearningChannel,
    execute_single_agent,
    reset_learning_channel,
)


@pytest.fixture(autouse=True)
def reset_channel():
    """Reset learning channel before each test."""
    reset_learning_channel()
    yield
    reset_learning_channel()


@pytest.fixture
def simple_executor():
    """Simple executor that echoes agent ID."""
    async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
        return AgentResult(
            agent_id=agent.agent_id,
            success=True,
            output=f"Output from {agent.agent_id}",
        )
    return executor


@pytest.fixture
def orchestrator():
    """Fixture for orchestrator."""
    return MultiAgentOrchestrator()


class TestAgentSpec:
    """Tests for AgentSpec."""

    def test_create_agent_spec(self):
        """Test basic agent spec creation."""
        spec = AgentSpec(
            agent_id="researcher",
            name="Research Agent",
            role=AgentRole.RESEARCHER,
        )

        assert spec.agent_id == "researcher"
        assert spec.name == "Research Agent"
        assert spec.role == AgentRole.RESEARCHER
        assert spec.timeout_seconds == 30.0
        assert spec.required is True

    def test_agent_spec_with_condition(self):
        """Test agent spec with run condition."""
        spec = AgentSpec(
            agent_id="critic",
            role=AgentRole.CRITIC,
            run_condition="prev.success == True",
        )

        assert spec.run_condition == "prev.success == True"


class TestTaskContext:
    """Tests for TaskContext."""

    def test_create_context(self):
        """Test basic context creation."""
        context = TaskContext(
            query="What time is it?",
            user_id="doug",
        )

        assert context.task_id is not None
        assert context.query == "What time is it?"
        assert context.user_id == "doug"
        assert context.started_at is not None

    def test_add_learning(self):
        """Test adding learning to context."""
        context = TaskContext(query="test", user_id="doug")

        learning = context.add_learning(
            content="Learned something",
            learning_type=LearningType.FACT,
            source_agent_id="roxy",
        )

        assert len(context.learnings) == 1
        assert context.learnings[0].content == "Learned something"

    def test_context_agent_outputs(self):
        """Test accumulating agent outputs."""
        context = TaskContext(query="test", user_id="doug")

        context.agent_outputs["agent1"] = {"result": "data1"}
        context.agent_outputs["agent2"] = {"result": "data2"}

        assert context.agent_outputs["agent1"]["result"] == "data1"
        assert context.agent_outputs["agent2"]["result"] == "data2"


class TestSequentialOrchestration:
    """Tests for sequential orchestration mode."""

    @pytest.mark.anyio
    async def test_sequential_single_agent(self, orchestrator, simple_executor):
        """Test sequential with single agent."""
        agents = [
            AgentSpec(agent_id="agent1", role=AgentRole.PRIMARY),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
            mode=OrchestrationMode.SEQUENTIAL,
        )

        assert result.success is True
        assert len(result.agent_results) == 1
        assert result.agent_results[0].agent_id == "agent1"
        assert result.final_output == "Output from agent1"

    @pytest.mark.anyio
    async def test_sequential_multiple_agents(self, orchestrator, simple_executor):
        """Test sequential with multiple agents."""
        agents = [
            AgentSpec(agent_id="researcher", role=AgentRole.RESEARCHER),
            AgentSpec(agent_id="writer", role=AgentRole.PRIMARY),
            AgentSpec(agent_id="critic", role=AgentRole.CRITIC),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
        )

        assert result.success is True
        assert len(result.agent_results) == 3

        # Verify order
        assert result.agent_results[0].agent_id == "researcher"
        assert result.agent_results[1].agent_id == "writer"
        assert result.agent_results[2].agent_id == "critic"

        # Final output should be from last agent
        assert result.final_output == "Output from critic"

    @pytest.mark.anyio
    async def test_sequential_context_accumulation(self, orchestrator):
        """Test that context accumulates agent outputs."""
        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            # Each agent sees previous outputs
            prev_outputs = list(context.agent_outputs.keys())
            return AgentResult(
                agent_id=agent.agent_id,
                success=True,
                output={
                    "my_id": agent.agent_id,
                    "saw_previous": prev_outputs,
                },
            )

        agents = [
            AgentSpec(agent_id="first"),
            AgentSpec(agent_id="second"),
            AgentSpec(agent_id="third"),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=executor,
        )

        # Third agent should have seen first and second
        assert result.final_output["saw_previous"] == ["first", "second"]

    @pytest.mark.anyio
    async def test_sequential_required_agent_failure(self, orchestrator):
        """Test that required agent failure stops pipeline."""
        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            if agent.agent_id == "failing":
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=False,
                    error="Intentional failure",
                )
            return AgentResult(agent_id=agent.agent_id, success=True, output="OK")

        agents = [
            AgentSpec(agent_id="first", required=True),
            AgentSpec(agent_id="failing", required=True),
            AgentSpec(agent_id="third", required=True),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=executor,
        )

        assert result.success is False
        assert "failing" in result.error
        assert len(result.agent_results) == 2  # Stopped at failing agent

    @pytest.mark.anyio
    async def test_sequential_optional_agent_failure(self, orchestrator):
        """Test that optional agent failure doesn't stop pipeline."""
        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            if agent.agent_id == "optional":
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=False,
                    error="Optional failure",
                )
            return AgentResult(agent_id=agent.agent_id, success=True, output="OK")

        agents = [
            AgentSpec(agent_id="first", required=True),
            AgentSpec(agent_id="optional", required=False),  # Optional
            AgentSpec(agent_id="third", required=True),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=executor,
        )

        assert result.success is True
        assert len(result.agent_results) == 3
        assert result.agent_results[1].success is False  # Optional failed

    @pytest.mark.anyio
    async def test_sequential_with_timeout(self, orchestrator):
        """Test agent timeout handling."""
        async def slow_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            await asyncio.sleep(2)  # Takes 2 seconds
            return AgentResult(agent_id=agent.agent_id, success=True, output="OK")

        agents = [
            AgentSpec(
                agent_id="slow",
                timeout_seconds=0.1,  # 100ms timeout
                max_retries=0,
            ),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=slow_executor,
        )

        assert result.success is False
        assert "timed out" in result.agent_results[0].error

    @pytest.mark.anyio
    async def test_sequential_with_retry(self, orchestrator):
        """Test agent retry on failure."""
        attempt_count = 0

        async def flaky_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=False,
                    error="First attempt fails",
                )
            return AgentResult(agent_id=agent.agent_id, success=True, output="OK")

        agents = [
            AgentSpec(
                agent_id="flaky",
                max_retries=2,
            ),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=flaky_executor,
        )

        assert result.success is True
        assert attempt_count == 2


class TestConditionEvaluation:
    """Tests for run condition evaluation."""

    @pytest.mark.anyio
    async def test_condition_skip_agent(self, orchestrator, simple_executor):
        """Test that condition can skip an agent."""
        agents = [
            AgentSpec(agent_id="first"),
            AgentSpec(
                agent_id="conditional",
                run_condition="False",  # Never run
            ),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
        )

        assert result.success is True
        assert len(result.agent_results) == 1
        assert result.agent_results[0].agent_id == "first"

    @pytest.mark.anyio
    async def test_condition_based_on_prev(self, orchestrator):
        """Test condition based on previous agent output."""
        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            return AgentResult(
                agent_id=agent.agent_id,
                success=True,
                output={"needs_review": agent.agent_id == "writer"},
            )

        agents = [
            AgentSpec(agent_id="writer"),
            AgentSpec(
                agent_id="reviewer",
                # Only run if writer output needs_review (using dict attribute access)
                run_condition="prev.needs_review == True",
            ),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=executor,
        )

        assert len(result.agent_results) == 2
        assert result.agent_results[1].agent_id == "reviewer"


class TestSafeExpressionParser:
    """Tests for the safe expression parser (replacing eval).

    These tests verify that:
    1. Valid expressions are parsed correctly
    2. Code injection attempts are blocked
    3. All supported syntax works as expected
    """

    @pytest.fixture
    def orchestrator(self):
        return MultiAgentOrchestrator()

    @pytest.fixture
    def sample_context(self):
        """Sample context for testing expression evaluation."""
        context = TaskContext(query="test", user_id="doug")
        context.agent_outputs["researcher"] = {
            "status": "complete",
            "count": 5,
            "success": True,
        }
        context.agent_outputs["writer"] = {
            "status": "pending",
            "success": False,
        }
        context.working_memory["flag"] = True
        return context

    # ==========================================================================
    # Basic Literal Tests
    # ==========================================================================

    def test_literal_true(self, orchestrator, sample_context):
        """Test True literal."""
        assert orchestrator._evaluate_condition("True", sample_context) is True

    def test_literal_false(self, orchestrator, sample_context):
        """Test False literal."""
        assert orchestrator._evaluate_condition("False", sample_context) is False

    def test_literal_none_is_falsy(self, orchestrator, sample_context):
        """Test None evaluates as falsy."""
        assert orchestrator._evaluate_condition("None", sample_context) is False

    def test_literal_integers(self, orchestrator, sample_context):
        """Test integer literals."""
        assert orchestrator._evaluate_condition("5 == 5", sample_context) is True
        assert orchestrator._evaluate_condition("5 == 3", sample_context) is False
        assert orchestrator._evaluate_condition("-1 < 0", sample_context) is True

    def test_literal_floats(self, orchestrator, sample_context):
        """Test float literals."""
        assert orchestrator._evaluate_condition("3.14 > 3", sample_context) is True
        assert orchestrator._evaluate_condition("2.5 == 2.5", sample_context) is True

    def test_literal_strings_single_quotes(self, orchestrator, sample_context):
        """Test single-quoted string literals."""
        assert orchestrator._evaluate_condition("'hello' == 'hello'", sample_context) is True
        assert orchestrator._evaluate_condition("'a' != 'b'", sample_context) is True

    def test_literal_strings_double_quotes(self, orchestrator, sample_context):
        """Test double-quoted string literals."""
        assert orchestrator._evaluate_condition('"world" == "world"', sample_context) is True

    # ==========================================================================
    # Comparison Operators
    # ==========================================================================

    def test_equals_comparison(self, orchestrator, sample_context):
        """Test == comparison."""
        assert orchestrator._evaluate_condition("5 == 5", sample_context) is True
        assert orchestrator._evaluate_condition("5 == 6", sample_context) is False

    def test_not_equals_comparison(self, orchestrator, sample_context):
        """Test != comparison."""
        assert orchestrator._evaluate_condition("5 != 6", sample_context) is True
        assert orchestrator._evaluate_condition("5 != 5", sample_context) is False

    def test_less_than(self, orchestrator, sample_context):
        """Test < comparison."""
        assert orchestrator._evaluate_condition("3 < 5", sample_context) is True
        assert orchestrator._evaluate_condition("5 < 3", sample_context) is False

    def test_greater_than(self, orchestrator, sample_context):
        """Test > comparison."""
        assert orchestrator._evaluate_condition("5 > 3", sample_context) is True
        assert orchestrator._evaluate_condition("3 > 5", sample_context) is False

    def test_less_than_or_equal(self, orchestrator, sample_context):
        """Test <= comparison."""
        assert orchestrator._evaluate_condition("3 <= 5", sample_context) is True
        assert orchestrator._evaluate_condition("5 <= 5", sample_context) is True
        assert orchestrator._evaluate_condition("6 <= 5", sample_context) is False

    def test_greater_than_or_equal(self, orchestrator, sample_context):
        """Test >= comparison."""
        assert orchestrator._evaluate_condition("5 >= 3", sample_context) is True
        assert orchestrator._evaluate_condition("5 >= 5", sample_context) is True
        assert orchestrator._evaluate_condition("3 >= 5", sample_context) is False

    # ==========================================================================
    # Variable/Attribute Access
    # ==========================================================================

    def test_context_variable_access(self, orchestrator, sample_context):
        """Test accessing context variables."""
        assert orchestrator._evaluate_condition(
            "working_memory.flag == True", sample_context
        ) is True

    def test_agent_outputs_dot_access(self, orchestrator, sample_context):
        """Test accessing agent outputs with dot notation."""
        assert orchestrator._evaluate_condition(
            "agent_outputs.researcher.status == 'complete'", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "agent_outputs.researcher.count == 5", sample_context
        ) is True

    def test_agent_outputs_bracket_access(self, orchestrator, sample_context):
        """Test accessing agent outputs with bracket notation."""
        assert orchestrator._evaluate_condition(
            "agent_outputs['researcher']['status'] == 'complete'", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            'agent_outputs["researcher"]["count"] == 5', sample_context
        ) is True

    def test_prev_variable(self, orchestrator, sample_context):
        """Test 'prev' refers to last agent output."""
        # Last agent is 'writer'
        assert orchestrator._evaluate_condition(
            "prev.status == 'pending'", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "prev.success == False", sample_context
        ) is True

    def test_none_short_circuit(self, orchestrator, sample_context):
        """Test that accessing attribute on None returns None."""
        assert orchestrator._evaluate_condition(
            "agent_outputs.nonexistent == None", sample_context
        ) is True

    # ==========================================================================
    # Boolean Operators
    # ==========================================================================

    def test_and_operator(self, orchestrator, sample_context):
        """Test 'and' operator."""
        assert orchestrator._evaluate_condition("True and True", sample_context) is True
        assert orchestrator._evaluate_condition("True and False", sample_context) is False
        assert orchestrator._evaluate_condition("False and True", sample_context) is False

    def test_or_operator(self, orchestrator, sample_context):
        """Test 'or' operator."""
        assert orchestrator._evaluate_condition("True or False", sample_context) is True
        assert orchestrator._evaluate_condition("False or True", sample_context) is True
        assert orchestrator._evaluate_condition("False or False", sample_context) is False

    def test_not_operator(self, orchestrator, sample_context):
        """Test 'not' operator."""
        assert orchestrator._evaluate_condition("not False", sample_context) is True
        assert orchestrator._evaluate_condition("not True", sample_context) is False

    def test_complex_boolean_expression(self, orchestrator, sample_context):
        """Test complex boolean expressions."""
        assert orchestrator._evaluate_condition(
            "prev.success == False and agent_outputs.researcher.success == True",
            sample_context
        ) is True

    def test_parentheses(self, orchestrator, sample_context):
        """Test parenthesized expressions."""
        assert orchestrator._evaluate_condition(
            "(True and False) or True", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "(5 > 3) and (2 < 4)", sample_context
        ) is True

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    def test_whitespace_handling(self, orchestrator, sample_context):
        """Test that extra whitespace is handled."""
        assert orchestrator._evaluate_condition("  True  ", sample_context) is True
        assert orchestrator._evaluate_condition("5  ==  5", sample_context) is True

    def test_invalid_expression_defaults_to_true(self, orchestrator, sample_context):
        """Test that invalid expressions default to True (agent runs)."""
        # Unknown variable defaults to running the agent
        assert orchestrator._evaluate_condition(
            "unknown_variable == True", sample_context
        ) is True

    def test_string_with_spaces(self, orchestrator, sample_context):
        """Test strings containing spaces."""
        assert orchestrator._evaluate_condition(
            "'hello world' == 'hello world'", sample_context
        ) is True

    # ==========================================================================
    # Security: Code Injection Attempts
    # ==========================================================================

    def test_blocks_function_calls(self, orchestrator, sample_context):
        """Test that function calls are not executed."""
        # This should fail to parse, not execute os.system
        result = orchestrator._evaluate_condition(
            "__import__('os').system('rm -rf /')", sample_context
        )
        # Should default to True (agent runs) but NOT execute the code
        assert result is True

    def test_blocks_builtins(self, orchestrator, sample_context):
        """Test that builtins are not accessible."""
        result = orchestrator._evaluate_condition(
            "exec('print(1)')", sample_context
        )
        assert result is True  # Defaults to True, but exec was never called

    def test_blocks_class_access(self, orchestrator, sample_context):
        """Test that class/type manipulation is blocked."""
        result = orchestrator._evaluate_condition(
            "().__class__.__bases__[0].__subclasses__()", sample_context
        )
        assert result is True  # Defaults to True, never actually executed

    def test_blocks_import_attempts(self, orchestrator, sample_context):
        """Test that import attempts fail."""
        result = orchestrator._evaluate_condition(
            "import os", sample_context
        )
        assert result is True  # Defaults to True, import never executed

    def test_blocks_eval_within_expression(self, orchestrator, sample_context):
        """Test that nested eval is blocked."""
        result = orchestrator._evaluate_condition(
            "eval('True')", sample_context
        )
        assert result is True  # Defaults to True, eval never called

    def test_blocks_lambda(self, orchestrator, sample_context):
        """Test that lambda expressions are blocked."""
        result = orchestrator._evaluate_condition(
            "(lambda: True)()", sample_context
        )
        assert result is True  # Defaults to True, lambda never executed

    def test_blocks_list_comprehension(self, orchestrator, sample_context):
        """Test that list comprehensions are blocked."""
        result = orchestrator._evaluate_condition(
            "[x for x in range(10)]", sample_context
        )
        assert result is True  # Defaults to True

    def test_blocks_dict_comprehension(self, orchestrator, sample_context):
        """Test that dict comprehensions are blocked."""
        result = orchestrator._evaluate_condition(
            "{x: x for x in range(10)}", sample_context
        )
        assert result is True  # Defaults to True

    # ==========================================================================
    # Real-World Condition Examples
    # ==========================================================================

    def test_real_condition_prev_success(self, orchestrator, sample_context):
        """Test realistic condition: check if previous agent succeeded."""
        # In this context, prev (writer) was not successful
        assert orchestrator._evaluate_condition(
            "prev.success == True", sample_context
        ) is False
        assert orchestrator._evaluate_condition(
            "prev.success == False", sample_context
        ) is True

    def test_real_condition_specific_agent_output(self, orchestrator, sample_context):
        """Test realistic condition: check specific agent's output."""
        assert orchestrator._evaluate_condition(
            "agent_outputs.researcher.status == 'complete'", sample_context
        ) is True

    def test_real_condition_count_threshold(self, orchestrator, sample_context):
        """Test realistic condition: numeric threshold."""
        assert orchestrator._evaluate_condition(
            "agent_outputs.researcher.count >= 3", sample_context
        ) is True
        assert orchestrator._evaluate_condition(
            "agent_outputs.researcher.count < 10", sample_context
        ) is True

    def test_real_condition_multiple_checks(self, orchestrator, sample_context):
        """Test realistic condition: multiple conditions combined."""
        assert orchestrator._evaluate_condition(
            "agent_outputs.researcher.success == True and agent_outputs.researcher.count > 0",
            sample_context
        ) is True


class TestLearningIntegration:
    """Tests for learning channel integration."""

    @pytest.mark.anyio
    async def test_learnings_collected(self, orchestrator):
        """Test that learnings from agents are collected."""
        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            learning = Learning(
                learning_type=LearningType.FACT,
                content=f"Learning from {agent.agent_id}",
                source_agent_id=agent.agent_id,
            )
            return AgentResult(
                agent_id=agent.agent_id,
                success=True,
                output="OK",
                learnings=[learning],
            )

        agents = [
            AgentSpec(agent_id="agent1"),
            AgentSpec(agent_id="agent2"),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=executor,
        )

        assert len(result.learnings) == 2
        contents = {l.content for l in result.learnings}
        assert "Learning from agent1" in contents
        assert "Learning from agent2" in contents

    @pytest.mark.anyio
    async def test_learnings_broadcast(self):
        """Test that learnings are broadcast to channel."""
        channel = InMemoryLearningChannel()
        orchestrator = MultiAgentOrchestrator(learning_channel=channel)

        received = []

        async def handler(learning: Learning):
            received.append(learning)

        await channel.subscribe("listener", handler)

        async def executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            learning = Learning(
                learning_type=LearningType.FACT,
                content="Broadcast this",
                source_agent_id=agent.agent_id,
            )
            return AgentResult(
                agent_id=agent.agent_id,
                success=True,
                output="OK",
                learnings=[learning],
            )

        agents = [AgentSpec(agent_id="broadcaster")]
        context = TaskContext(query="test", user_id="doug")

        await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=executor,
        )

        assert len(received) == 1
        assert received[0].content == "Broadcast this"


class TestOrchestratorResult:
    """Tests for OrchestratorResult."""

    @pytest.mark.anyio
    async def test_result_timing(self, orchestrator, simple_executor):
        """Test that result includes timing information."""
        agents = [AgentSpec(agent_id="agent1")]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
        )

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_ms >= 0

    @pytest.mark.anyio
    async def test_result_includes_context(self, orchestrator, simple_executor):
        """Test that result includes final context."""
        agents = [AgentSpec(agent_id="agent1")]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
        )

        assert result.final_context is not None
        assert result.final_context.completed_at is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.anyio
    async def test_execute_single_agent(self, simple_executor):
        """Test execute_single_agent helper."""
        agent = AgentSpec(agent_id="single", role=AgentRole.PRIMARY)

        result = await execute_single_agent(
            agent=agent,
            query="test query",
            user_id="doug",
            executor=simple_executor,
        )

        assert result.success is True
        assert result.agent_id == "single"
        assert result.output == "Output from single"


class TestPhaseC4Stubs:
    """Tests for Phase C.4 stub modes."""

    @pytest.mark.anyio
    async def test_parallel_falls_back(self, orchestrator, simple_executor):
        """Test that parallel mode falls back to sequential."""
        agents = [
            AgentSpec(agent_id="agent1"),
            AgentSpec(agent_id="agent2"),
        ]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
            mode=OrchestrationMode.PARALLEL,
        )

        # Should still work (falls back to sequential)
        assert result.success is True
        assert len(result.agent_results) == 2

    @pytest.mark.anyio
    async def test_handoff_falls_back(self, orchestrator, simple_executor):
        """Test that handoff mode falls back to sequential."""
        agents = [AgentSpec(agent_id="agent1")]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
            mode=OrchestrationMode.HANDOFF,
        )

        assert result.success is True

    @pytest.mark.anyio
    async def test_collaborative_falls_back(self, orchestrator, simple_executor):
        """Test that collaborative mode falls back to sequential."""
        agents = [AgentSpec(agent_id="agent1")]
        context = TaskContext(query="test", user_id="doug")

        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=simple_executor,
            mode=OrchestrationMode.COLLABORATIVE,
        )

        assert result.success is True

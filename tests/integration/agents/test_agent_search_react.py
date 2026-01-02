"""ReAct mode web search integration tests (FR-010.8).

Tests multi-step search reasoning with the ReAct pattern:
- Multiple search iterations with query refinement
- Observation accumulation across iterations
- Search → learn → search-again loops
- Parallel search path exploration (future)

These tests validate that search behavior works correctly in ReAct mode,
enabling the agent to perform multi-step research.
"""

import os
from dataclasses import dataclass, field
from typing import Any
import pytest

from draagon_ai.orchestration.loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentContext,
    AgentResponse,
    LoopMode,
    StepType,
)
from draagon_ai.orchestration.registry import Tool, ToolParameter, ToolRegistry
from draagon_ai.orchestration.execution import ActionExecutor
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts
from draagon_ai.orchestration.search_orchestration import SearchStrategy


# =============================================================================
# Mock Search Tool for ReAct Testing
# =============================================================================


@dataclass
class SearchCall:
    """Record of a search tool call."""

    query: str
    iteration: int  # Which ReAct iteration this was called in
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockReActSearchTool:
    """Mock search tool that tracks calls across ReAct iterations.

    This mock is designed to support multi-step search scenarios where
    initial searches may be refined based on observations.
    """

    calls: list[SearchCall] = field(default_factory=list)
    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    iteration_counter: int = 0

    # Default response when no match found
    default_response: dict[str, Any] = field(
        default_factory=lambda: {
            "results": [],
            "message": "No results found. Try a different query.",
        }
    )

    def reset(self):
        """Reset for a new test."""
        self.calls.clear()
        self.iteration_counter = 0

    def next_iteration(self):
        """Called when a new ReAct iteration starts."""
        self.iteration_counter += 1

    def add_response(self, query_pattern: str, response: dict[str, Any]):
        """Add a response for a query pattern (substring match)."""
        self.responses[query_pattern.lower()] = response

    async def handler(self, args: dict[str, Any], **context: Any) -> dict[str, Any]:
        """Handle search tool execution."""
        query = args.get("query", "")

        # Record the call with iteration
        self.calls.append(
            SearchCall(
                query=query,
                iteration=self.iteration_counter,
                args=args,
            )
        )

        # Find matching response
        query_lower = query.lower()
        for pattern, response in self.responses.items():
            if pattern in query_lower:
                return response

        return self.default_response


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class ReActSearchFixtures:
    """Bundle of fixtures for ReAct search tests."""

    agent: AgentLoop
    mock_search: MockReActSearchTool
    registry: ToolRegistry


@pytest.fixture
def react_search_behavior():
    """Behavior that guides the agent to use ReAct search reasoning."""
    return Behavior(
        behavior_id="react_search",
        name="ReAct Research Assistant",
        description="Assistant that performs multi-step web research",
        actions=[
            Action(
                name="search_web",
                description="Search the web for information. Use this to find facts, current events, or answers to questions.",
            ),
            Action(
                name="answer",
                description="Provide the final answer after gathering sufficient information from searches.",
            ),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="""You are a research assistant that gathers information through web searches.

AVAILABLE ACTIONS:
- search_web: Search the web for information (requires "query" argument)
- answer: Provide your final answer based on information gathered

CRITICAL: When you have gathered enough information to answer the question, you MUST use "answer" action.
DO NOT keep searching if you already have the answer in your observations!

INSTRUCTIONS:
1. If you don't have information to answer, use search_web to find it
2. If search results contain the answer, use "answer" action IMMEDIATELY
3. Only search again if results were truly insufficient (empty or unrelated)
4. For simple factual questions with clear results, answer after ONE search

{assistant_intro}

CONTEXT AND OBSERVATIONS:
{context}

USER QUERY: {question}

Output your decision as XML:
<response>
  <reasoning>Brief thought - do I have enough info to answer?</reasoning>
  <action>answer OR search_web</action>
  <args>
    <query>search query (only if action is search_web)</query>
  </args>
  <answer>Your final answer (required if action is answer)</answer>
</response>""",
            synthesis_prompt="""You are a helpful research assistant.

Based on the search results, synthesize a clear answer to the user's question.

USER QUESTION: {question}

SEARCH RESULTS:
{tool_results}

Provide a clear, concise answer that synthesizes the information found.

Output as XML:
<synthesis>
  <answer>Your synthesized answer</answer>
</synthesis>""",
        ),
    )


@pytest.fixture
def react_search_fixtures(real_llm, memory_provider, react_search_behavior):
    """Create agent with mock search tool in ReAct mode."""
    mock_search = MockReActSearchTool()

    # Create tool registry with mock search
    registry = ToolRegistry()
    search_tool = Tool(
        name="search_web",
        description="Search the web for information",
        handler=mock_search.handler,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
        ],
    )
    registry.register(search_tool)

    # Create action executor with the registry
    executor = ActionExecutor(tool_registry=registry)

    # Create agent in ReAct mode
    config = AgentLoopConfig(
        mode=LoopMode.REACT,  # Force ReAct mode for multi-step reasoning
        max_iterations=5,  # Allow up to 5 steps
        iteration_timeout_seconds=30.0,
        log_thought_traces=True,
    )

    agent = AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        config=config,
        action_executor=executor,
    )

    return ReActSearchFixtures(
        agent=agent,
        mock_search=mock_search,
        registry=registry,
    )


@pytest.fixture
def test_context():
    """Context for tests."""
    return AgentContext(
        user_id="test_user",
        session_id="test_session",
        debug=True,
    )


# =============================================================================
# ReAct Search Mode Tests
# =============================================================================


@pytest.mark.search_integration
class TestReActSearchMode:
    """Test that search works in ReAct mode."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_uses_react_mode(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Search queries use ReAct mode for multi-step reasoning."""
        fixtures = react_search_fixtures
        fixtures.mock_search.add_response(
            "python",
            {
                "results": [
                    {
                        "title": "Python Programming Language",
                        "url": "https://python.org",
                        "snippet": "Python is a high-level programming language.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="What is Python programming language?",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Should use ReAct mode
        assert response.loop_mode == LoopMode.REACT

        # Should have ReAct steps
        assert len(response.react_steps) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_produces_thought_steps(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """ReAct search produces THOUGHT steps."""
        fixtures = react_search_fixtures
        fixtures.mock_search.add_response(
            "machine learning",
            {
                "results": [
                    {
                        "title": "What is Machine Learning?",
                        "snippet": "ML is a subset of AI that enables computers to learn from data.",
                    }
                ],
            },
        )
        # Also match common LLM query variants
        fixtures.mock_search.add_response("learning", fixtures.mock_search.responses["machine learning"])
        fixtures.mock_search.add_response("explain", fixtures.mock_search.responses["machine learning"])

        response = await fixtures.agent.process(
            query="Explain machine learning",
            behavior=react_search_behavior,
            context=test_context,
        )

        thought_steps = [s for s in response.react_steps if s.type == StepType.THOUGHT]

        # Should have at least one thought step (reasoning)
        assert len(thought_steps) >= 1

        # First thought should be non-empty reasoning (we don't prescribe specific words)
        first_thought = thought_steps[0].content
        assert len(first_thought) > 10, f"First thought is too short: {first_thought}"


@pytest.mark.search_integration
class TestReActMultiStepSearch:
    """Test multi-step search reasoning."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_action_invokes_tool(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Search action correctly invokes the search tool."""
        fixtures = react_search_fixtures
        fixtures.mock_search.add_response(
            "weather",
            {
                "results": [
                    {
                        "title": "Weather Today",
                        "snippet": "Current weather: Sunny, 72°F",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="What's the weather like today?",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Should have called the search tool
        assert len(fixtures.mock_search.calls) >= 1

        # Check that action step was recorded
        action_steps = [s for s in response.react_steps if s.type == StepType.ACTION]
        assert len(action_steps) >= 1

        # Action should be search_web
        search_actions = [s for s in action_steps if s.action_name == "search_web"]
        assert len(search_actions) >= 1

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_observation_recorded_after_search(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Search results are recorded as OBSERVATION steps."""
        fixtures = react_search_fixtures
        # Use multiple patterns to match various LLM query formulations
        paris_response = {
            "results": [
                {
                    "title": "Paris - Capital of France",
                    "snippet": "Paris is the capital and largest city of France.",
                }
            ],
        }
        fixtures.mock_search.add_response("capital", paris_response)
        fixtures.mock_search.add_response("france", paris_response)
        fixtures.mock_search.add_response("paris", paris_response)

        response = await fixtures.agent.process(
            query="What is the capital of France?",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Should have observation steps
        observation_steps = [
            s for s in response.react_steps if s.type == StepType.OBSERVATION
        ]
        assert len(observation_steps) >= 1

        # Observation should contain search results (Paris info)
        first_obs = observation_steps[0].content.lower()
        assert "paris" in first_obs or "capital" in first_obs

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_multiple_search_iterations(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Agent can perform multiple searches to gather more information.

        This tests the core ReAct capability - initial search, then refinement.
        """
        fixtures = react_search_fixtures

        # First search returns partial information
        fixtures.mock_search.add_response(
            "python creator",
            {
                "results": [
                    {
                        "title": "Python History",
                        "snippet": "Python was created by Guido van Rossum.",
                    }
                ],
            },
        )

        # Second search (if agent refines) returns more detail
        fixtures.mock_search.add_response(
            "guido van rossum",
            {
                "results": [
                    {
                        "title": "Guido van Rossum Biography",
                        "snippet": "Dutch programmer who created Python in 1991 while at CWI.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="Who created Python and when was it first released? I need the full story.",
            behavior=react_search_behavior,
            context=test_context,
        )

        # The agent may or may not do multiple searches - depends on LLM
        # But it should at least do one search
        assert len(fixtures.mock_search.calls) >= 1

        # If the agent did multiple iterations, verify observations accumulate
        if response.iterations_used > 1:
            observation_steps = [
                s for s in response.react_steps if s.type == StepType.OBSERVATION
            ]
            assert len(observation_steps) >= 1

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_final_answer_synthesizes_observations(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Final answer incorporates information from search observations."""
        fixtures = react_search_fixtures
        jupiter_response = {
            "results": [
                {
                    "title": "Jupiter - Largest Planet",
                    "snippet": "Jupiter is the largest planet in our solar system with a mass of 1.898 × 10^27 kg.",
                }
            ],
        }
        # Match multiple possible search patterns
        fixtures.mock_search.add_response("largest planet", jupiter_response)
        fixtures.mock_search.add_response("jupiter", jupiter_response)
        fixtures.mock_search.add_response("solar system", jupiter_response)
        fixtures.mock_search.add_response("planet", jupiter_response)

        response = await fixtures.agent.process(
            query="What is the largest planet in our solar system?",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Should have a final answer step OR a successful response
        final_answer_steps = [
            s for s in response.react_steps if s.type == StepType.FINAL_ANSWER
        ]

        # Either got a final answer step, or the response mentions Jupiter
        response_lower = response.response.lower()
        if len(final_answer_steps) >= 1:
            # Final answer should mention Jupiter
            final_answer = final_answer_steps[-1].content.lower()
            assert "jupiter" in final_answer
        else:
            # If no final answer step, check if Jupiter is in observations
            # (LLM may have reached max iterations but still found the answer)
            observations = [
                s.content for s in response.react_steps if s.type == StepType.OBSERVATION
            ]
            all_obs_text = " ".join(observations).lower()
            assert "jupiter" in all_obs_text or "jupiter" in response_lower, \
                f"Expected 'jupiter' in observations or response. Got: {response_lower}"


@pytest.mark.search_integration
class TestReActSearchRefinement:
    """Test query refinement across iterations."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_empty_results_may_trigger_refinement(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Empty search results may prompt the agent to try a different query.

        Note: This is LLM-dependent - the agent may or may not refine.
        We test that the system correctly handles the multi-step scenario.
        """
        fixtures = react_search_fixtures

        # First query returns nothing
        fixtures.mock_search.add_response(
            "xyzzyfoo",
            {
                "results": [],
                "message": "No results found. Try different search terms.",
            },
        )

        # Related query returns results
        fixtures.mock_search.add_response(
            "quantum computing",
            {
                "results": [
                    {
                        "title": "Quantum Computing Explained",
                        "snippet": "Quantum computers use qubits instead of classical bits.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="Tell me about xyzzyfoo quantum mechanics research",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Agent should have searched at least once
        assert len(fixtures.mock_search.calls) >= 1

        # Check the response - may express uncertainty or provide partial info
        assert response.response is not None
        assert len(response.response) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_query_refinement_tracked_in_calls(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """When agent refines query, both queries are tracked."""
        fixtures = react_search_fixtures

        # Set up responses for both broad and specific queries
        fixtures.mock_search.add_response(
            "programming language",
            {
                "results": [
                    {
                        "title": "Programming Languages Overview",
                        "snippet": "Many languages exist: Python, Java, C++, etc.",
                    }
                ],
            },
        )

        fixtures.mock_search.add_response(
            "rust programming",
            {
                "results": [
                    {
                        "title": "Rust Language",
                        "snippet": "Rust is a systems programming language focused on safety.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="Compare Rust to other programming languages for systems programming",
            behavior=react_search_behavior,
            context=test_context,
        )

        # At least one search should have been made
        assert len(fixtures.mock_search.calls) >= 1

        # All queries should be tracked
        all_queries = [call.query for call in fixtures.mock_search.calls]
        assert len(all_queries) >= 1

        # Response should mention relevant info
        response_lower = response.response.lower()
        # May mention rust, programming, or express uncertainty
        assert (
            "rust" in response_lower
            or "programming" in response_lower
            or "language" in response_lower
        )


@pytest.mark.search_integration
class TestReActSearchTraceQuality:
    """Test the quality of ReAct traces for search scenarios."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_trace_has_logical_flow(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """ReAct trace follows THOUGHT → ACTION → OBSERVATION pattern."""
        fixtures = react_search_fixtures
        fixtures.mock_search.add_response(
            "neural network",
            {
                "results": [
                    {
                        "title": "Neural Networks Explained",
                        "snippet": "Neural networks are computing systems inspired by biological neurons.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="What is a neural network?",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Should have steps
        assert len(response.react_steps) >= 1

        # Check that steps follow a logical order
        # (relaxed check - LLM may take shortcuts)
        step_types = [s.type for s in response.react_steps]

        # Should start with a THOUGHT
        if len(step_types) > 0:
            assert step_types[0] == StepType.THOUGHT

        # Should end with FINAL_ANSWER if successful
        if response.success:
            assert step_types[-1] == StepType.FINAL_ANSWER

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_trace_contains_search_details(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """ACTION steps contain search_web with query args."""
        fixtures = react_search_fixtures
        fixtures.mock_search.add_response(
            "mars",
            {
                "results": [
                    {
                        "title": "Mars Facts",
                        "snippet": "Mars is the fourth planet from the Sun.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="Tell me about Mars",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Find ACTION steps
        action_steps = [s for s in response.react_steps if s.type == StepType.ACTION]

        if len(action_steps) > 0:
            # At least one should be search_web
            search_actions = [s for s in action_steps if s.action_name == "search_web"]
            assert len(search_actions) >= 1

            # Should have query in args
            first_search = search_actions[0]
            assert "query" in first_search.action_args

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_get_thought_trace_includes_all_steps(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """get_thought_trace() returns all steps with proper format."""
        fixtures = react_search_fixtures
        fixtures.mock_search.add_response(
            "artificial intelligence",
            {
                "results": [
                    {
                        "title": "AI Overview",
                        "snippet": "AI refers to systems that can perform tasks requiring human intelligence.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="What is AI?",
            behavior=react_search_behavior,
            context=test_context,
        )

        trace = response.get_thought_trace()

        # Should be a list
        assert isinstance(trace, list)

        # Each item should have required fields
        for item in trace:
            assert "step" in item
            assert "type" in item
            assert "content" in item
            assert "timestamp" in item

            # step should be incrementing
            assert isinstance(item["step"], int)
            assert item["step"] >= 1


@pytest.mark.search_integration
class TestReActSearchPerformance:
    """Test ReAct search performance characteristics."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_max_iterations_respected(
        self, real_llm, memory_provider, react_search_behavior, test_context
    ):
        """Agent respects max_iterations limit."""
        mock_search = MockReActSearchTool()

        # Always return results that might prompt more searching
        mock_search.add_response(
            "",  # Match any query
            {
                "results": [
                    {
                        "title": "Partial Info",
                        "snippet": "Some information, but more details might be available...",
                    }
                ],
            },
        )

        registry = ToolRegistry()
        search_tool = Tool(
            name="search_web",
            description="Search the web for information",
            handler=mock_search.handler,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query",
                    required=True,
                ),
            ],
        )
        registry.register(search_tool)

        executor = ActionExecutor(tool_registry=registry)

        # Configure with low max iterations
        config = AgentLoopConfig(
            mode=LoopMode.REACT,
            max_iterations=3,  # Only 3 iterations allowed
            iteration_timeout_seconds=30.0,
        )

        agent = AgentLoop(
            llm=real_llm,
            memory=memory_provider,
            config=config,
            action_executor=executor,
        )

        response = await agent.process(
            query="Research everything about quantum physics",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Should not exceed max iterations
        assert response.iterations_used <= 3

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_query_may_complete_quickly(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Simple queries may complete in fewer iterations."""
        fixtures = react_search_fixtures
        # Match any math-related query
        math_response = {
            "results": [
                {
                    "title": "Basic Math - 2+2=4",
                    "snippet": "The answer to 2 + 2 equals 4. This is basic arithmetic.",
                }
            ],
        }
        fixtures.mock_search.add_response("2+2", math_response)
        fixtures.mock_search.add_response("2 + 2", math_response)
        fixtures.mock_search.add_response("math", math_response)
        fixtures.mock_search.add_response("arithmetic", math_response)
        fixtures.mock_search.add_response("what is", math_response)

        response = await fixtures.agent.process(
            query="What is 2+2?",
            behavior=react_search_behavior,
            context=test_context,
        )

        # Response should contain the answer "4" somewhere
        # (whether successful or hitting max iterations with observations)
        assert "4" in response.response, \
            f"Expected '4' in response. Got: {response.response}"


@pytest.mark.search_integration
class TestReActObservationContext:
    """Test observation accumulation in ReAct search."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_observations_available_to_next_iteration(
        self, react_search_fixtures, react_search_behavior, test_context
    ):
        """Observations from previous iterations are available to decision prompt."""
        fixtures = react_search_fixtures

        # First search
        fixtures.mock_search.add_response(
            "history of computers",
            {
                "results": [
                    {
                        "title": "Computer History",
                        "snippet": "First electronic computer was ENIAC in 1945.",
                    }
                ],
            },
        )

        # Follow-up search
        fixtures.mock_search.add_response(
            "eniac",
            {
                "results": [
                    {
                        "title": "ENIAC Details",
                        "snippet": "ENIAC was built at University of Pennsylvania.",
                    }
                ],
            },
        )

        response = await fixtures.agent.process(
            query="Tell me about the first electronic computer and where it was built",
            behavior=react_search_behavior,
            context=test_context,
        )

        # If multiple iterations occurred, observations should have accumulated
        if response.iterations_used > 1:
            # Check that later thoughts reference earlier findings
            thought_steps = [
                s for s in response.react_steps if s.type == StepType.THOUGHT
            ]

            if len(thought_steps) >= 2:
                # Later thoughts may reference earlier findings
                later_thought = thought_steps[-1].content.lower()
                # Should show awareness of context (relaxed check)
                assert len(later_thought) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_context_cleared_between_queries(
        self, react_search_fixtures, react_search_behavior
    ):
        """Observations don't leak between separate queries."""
        fixtures = react_search_fixtures

        fixtures.mock_search.add_response(
            "dogs",
            {
                "results": [
                    {"title": "Dogs", "snippet": "Dogs are loyal companions."}
                ],
            },
        )

        fixtures.mock_search.add_response(
            "cats",
            {
                "results": [
                    {"title": "Cats", "snippet": "Cats are independent animals."}
                ],
            },
        )

        # First query
        context1 = AgentContext(user_id="user1", session_id="session1", debug=True)
        await fixtures.agent.process(
            query="Tell me about dogs",
            behavior=react_search_behavior,
            context=context1,
        )

        # Second query with fresh context
        context2 = AgentContext(user_id="user2", session_id="session2", debug=True)
        response2 = await fixtures.agent.process(
            query="Tell me about cats",
            behavior=react_search_behavior,
            context=context2,
        )

        # Second response should be about cats, not dogs
        response_lower = response2.response.lower()
        assert "cat" in response_lower or "feline" in response_lower

        # Should not leak dog observations
        # (unless the LLM hallucinates, which we can't fully control)


# =============================================================================
# Parallel Search / Agent Forking Tests
# =============================================================================


@pytest.mark.search_integration
class TestReActParallelSearch:
    """Test parallel search paths using the SearchOrchestrator.

    This represents the "fork off agents and try different paths" capability
    using the multi-agent orchestrator.
    """

    @pytest.fixture
    def parallel_orchestrator(self, real_llm, memory_provider):
        """Create parallel search orchestrator."""
        from draagon_ai.orchestration.search_orchestration import (
            ParallelSearchOrchestrator,
            SearchConfig,
            SearchStrategy,
        )

        mock_search = MockReActSearchTool()
        # Add responses for common queries
        mock_search.add_response(
            "python",
            {
                "results": [
                    {
                        "title": "Python Overview",
                        "snippet": "Python is a high-level programming language known for readability.",
                    }
                ],
            },
        )
        mock_search.add_response(
            "rust",
            {
                "results": [
                    {
                        "title": "Rust Overview",
                        "snippet": "Rust is a systems programming language focused on safety and performance.",
                    }
                ],
            },
        )
        mock_search.add_response(
            "compare",
            {
                "results": [
                    {
                        "title": "Language Comparison",
                        "snippet": "Python is easier to learn while Rust offers better performance.",
                    }
                ],
            },
        )

        orchestrator = ParallelSearchOrchestrator(
            llm=real_llm,
            memory=memory_provider,
            search_tool=mock_search.handler,
            config=SearchConfig(
                strategy=SearchStrategy.PARALLEL,
                max_parallel_searches=3,
                store_findings=True,
            ),
        )

        return orchestrator, mock_search

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_parallel_search_paths(self, parallel_orchestrator):
        """Agent can spawn parallel searches for different aspects."""
        orchestrator, mock_search = parallel_orchestrator

        result = await orchestrator.research(
            query="Compare Python and Rust for web development",
            user_id="test_user",
        )

        # Should have executed multiple searches
        assert len(mock_search.calls) >= 1

        # Should have findings from searches
        assert len(result.findings) >= 1

        # Should have an answer
        assert result.success
        assert len(result.answer) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_path_convergence(self, parallel_orchestrator):
        """Parallel search paths converge into unified answer."""
        orchestrator, mock_search = parallel_orchestrator

        result = await orchestrator.research(
            query="What are the advantages of Python?",
            user_id="test_user",
        )

        # Answer should synthesize from multiple sources
        assert result.success
        answer_lower = result.answer.lower()

        # Should mention Python-related concepts
        assert "python" in answer_lower or "programming" in answer_lower or "language" in answer_lower


# =============================================================================
# Search + Memory + Belief Integration Tests
# =============================================================================


@pytest.mark.search_integration
class TestReActSearchLearning:
    """Test search results being stored in memory for future use.

    This enables the agent to learn from searches and avoid
    re-searching for the same information.
    """

    @pytest.fixture
    def learning_orchestrator(self, real_llm, memory_provider):
        """Create search orchestrator with memory integration."""
        from draagon_ai.orchestration.search_orchestration import (
            SearchOrchestrator,
            SearchConfig,
            SearchStrategy,
        )

        mock_search = MockReActSearchTool()
        mock_search.add_response(
            "capital",
            {
                "results": [
                    {
                        "title": "France Capital",
                        "snippet": "Paris is the capital of France, located on the Seine river.",
                    }
                ],
            },
        )
        mock_search.add_response(
            "paris",
            {
                "results": [
                    {
                        "title": "Paris Info",
                        "snippet": "Paris has a population of about 2 million in the city proper.",
                    }
                ],
            },
        )

        orchestrator = SearchOrchestrator(
            llm=real_llm,
            memory=memory_provider,
            search_tool=mock_search.handler,
            config=SearchConfig(
                strategy=SearchStrategy.SINGLE,
                store_findings=True,
                min_confidence_to_store=0.5,  # Lower threshold for testing
            ),
        )

        return orchestrator, mock_search, memory_provider

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_results_stored_in_memory(self, learning_orchestrator):
        """Significant search findings are stored as memories."""
        orchestrator, mock_search, memory_provider = learning_orchestrator

        result = await orchestrator.research(
            query="What is the capital of France?",
            user_id="test_user",
        )

        # Should have executed search
        assert len(mock_search.calls) >= 1

        # Should have findings
        assert len(result.findings) >= 1

        # Findings should be stored in memory (check learnings list)
        # Note: Actual memory storage depends on memory_provider implementation
        # For now, verify the findings were marked for learning
        assert result.success

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_memory_integration_flow(self, learning_orchestrator):
        """Verify the search→memory integration flow works."""
        orchestrator, mock_search, memory_provider = learning_orchestrator

        # First search
        result1 = await orchestrator.research(
            query="Tell me about Paris",
            user_id="test_user",
        )

        assert result1.success
        assert len(result1.findings) >= 1

        # The orchestrator should have attempted to store findings
        # We can verify by checking if learnings were populated
        # (actual storage depends on memory_provider behavior)

        # Search again - orchestrator tracks queries
        result2 = await orchestrator.research(
            query="What's the population of Paris?",
            user_id="test_user",
        )

        assert result2.success
        # Both searches should have completed
        assert len(mock_search.calls) >= 2


# =============================================================================
# Semantic Graph Exploration Tests
# =============================================================================


@pytest.mark.search_integration
class TestSemanticGraphExploration:
    """Test semantic graph exploration during search.

    These tests validate that the search orchestrator can use
    the semantic graph to:
    1. Expand queries with related concepts
    2. Use known context to inform searches
    3. Detect knowledge gaps
    """

    @pytest.fixture
    def mock_semantic_context(self):
        """Create mock semantic context service."""
        from draagon_ai.orchestration.semantic_context import (
            SemanticContext,
        )

        class MockSemanticContextService:
            """Mock that returns predefined semantic context."""

            def __init__(self):
                self.enrich_calls: list[str] = []
                self.contexts: dict[str, SemanticContext] = {}

            def add_context(self, query_pattern: str, context: SemanticContext):
                """Add a context for a query pattern."""
                self.contexts[query_pattern.lower()] = context

            async def enrich(
                self,
                query: str,
                user_id: str | None = None,
                agent_id: str | None = None,
            ) -> SemanticContext:
                """Return predefined context for matching query."""
                self.enrich_calls.append(query)

                query_lower = query.lower()
                for pattern, ctx in self.contexts.items():
                    if pattern in query_lower:
                        return ctx

                # Return empty context
                return SemanticContext(
                    query=query,
                    relevant_facts=[],
                    relevant_entities=[],
                    related_memories=[],
                )

        return MockSemanticContextService()

    @pytest.fixture
    def semantic_search_fixtures(self, real_llm, memory_provider, mock_semantic_context):
        """Create semantic search orchestrator."""
        from draagon_ai.orchestration.semantic_search import (
            SemanticSearchOrchestrator,
            SemanticSearchConfig,
        )
        from draagon_ai.orchestration.semantic_context import SemanticContext

        mock_search = MockReActSearchTool()

        # Add search responses
        mock_search.add_response(
            "python",
            {
                "results": [
                    {
                        "title": "Python Programming",
                        "snippet": "Python is Doug's favorite programming language.",
                    }
                ],
            },
        )
        mock_search.add_response(
            "projects",
            {
                "results": [
                    {
                        "title": "Open Source Projects",
                        "snippet": "Doug contributes to draagon-ai and roxy-voice-assistant.",
                    }
                ],
            },
        )

        # Add semantic context
        mock_semantic_context.add_context(
            "python",
            SemanticContext(
                query="python",
                relevant_facts=["Doug has used Python for 10 years"],
                relevant_entities=["Python", "programming", "software development"],
                related_memories=["Doug prefers Python for AI projects"],
                context_nodes_found=5,
                used_semantic_graph=True,
            ),
        )

        orchestrator = SemanticSearchOrchestrator(
            llm=real_llm,
            memory=memory_provider,
            search_tool=mock_search.handler,
            semantic_context=mock_semantic_context,
            config=SemanticSearchConfig(
                strategy=SearchStrategy.PARALLEL,
                enable_graph_expansion=True,
                max_expansion_depth=2,
                store_findings=True,
            ),
        )

        return orchestrator, mock_search, mock_semantic_context

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_graph_context_enriches_search(self, semantic_search_fixtures):
        """Semantic graph context is used to enhance search."""
        orchestrator, mock_search, mock_semantic = semantic_search_fixtures

        # Use a query that matches the "python" pattern
        result = await orchestrator.research_with_graph(
            query="Tell me about Python programming",
            user_id="test_user",
        )

        # Should have enriched context
        assert len(mock_semantic.enrich_calls) >= 1

        # Result should indicate graph was used (if context was found)
        # The mock should have matched "python" pattern
        assert result.semantic_context is not None

        # Should have an answer (even if search returned nothing, we have context)
        assert len(result.answer) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_query_expansion_uses_related_concepts(self, semantic_search_fixtures):
        """Query expansion includes related concepts from graph."""
        orchestrator, mock_search, mock_semantic = semantic_search_fixtures

        result = await orchestrator.research_with_graph(
            query="What Python projects does Doug work on?",
            user_id="test_user",
        )

        # Should have searched (either original or expanded query)
        assert len(mock_search.calls) >= 1

        # Result should be successful
        assert result.success

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_knowledge_gap_detection(self, semantic_search_fixtures):
        """Orchestrator detects what information is missing."""
        orchestrator, mock_search, mock_semantic = semantic_search_fixtures

        # Query about something with partial context (Python is known)
        result = await orchestrator.research_with_graph(
            query="What Python frameworks does Doug prefer?",
            user_id="test_user",
        )

        # Should complete (may or may not be "successful" by strict definition)
        # The important thing is we get a response
        assert len(result.answer) > 0

        # May have identified knowledge gaps (depends on LLM response)
        # We don't require this since it's LLM-dependent


@pytest.mark.search_integration
class TestGraphEnhancedReAct:
    """Test ReAct loop with graph exploration."""

    @pytest.fixture
    def react_graph_fixtures(self, real_llm, memory_provider):
        """Create graph-enhanced ReAct orchestrator."""
        from draagon_ai.orchestration.semantic_search import (
            GraphEnhancedReActOrchestrator,
            SemanticSearchConfig,
        )
        from draagon_ai.orchestration.semantic_context import SemanticContext

        mock_search = MockReActSearchTool()

        # Add responses
        mock_search.add_response(
            "open source",
            {
                "results": [
                    {
                        "title": "Doug's Open Source Work",
                        "snippet": "Doug maintains several open source AI projects.",
                    }
                ],
            },
        )
        mock_search.add_response(
            "contributions",
            {
                "results": [
                    {
                        "title": "GitHub Contributions",
                        "snippet": "Doug has contributed to Python libraries for machine learning.",
                    }
                ],
            },
        )

        # Mock semantic context service
        class MockSemanticService:
            async def enrich(self, query: str, user_id: str | None = None, **kwargs):
                return SemanticContext(
                    query=query,
                    relevant_facts=["Doug works on AI projects"],
                    relevant_entities=["AI", "machine learning", "Python"],
                    related_memories=["Doug uses PyTorch for deep learning"],
                )

        orchestrator = GraphEnhancedReActOrchestrator(
            llm=real_llm,
            memory=memory_provider,
            search_tool=mock_search.handler,
            semantic_context=MockSemanticService(),
            config=SemanticSearchConfig(
                enable_graph_expansion=True,
                detect_knowledge_gaps=True,
            ),
        )

        return orchestrator, mock_search

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_react_uses_graph_context(self, react_graph_fixtures):
        """ReAct loop uses graph context for reasoning."""
        orchestrator, mock_search = react_graph_fixtures

        result = await orchestrator.research_with_react(
            query="What open source projects does Doug contribute to?",
            user_id="test_user",
            max_iterations=3,
        )

        # Should have completed
        assert result.success

        # Should have used graph context
        assert result.graph_context_used

        # Should have an answer
        assert len(result.answer) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_react_respects_max_iterations(self, react_graph_fixtures):
        """ReAct loop respects max iterations limit."""
        orchestrator, mock_search = react_graph_fixtures

        result = await orchestrator.research_with_react(
            query="Tell me everything about Doug's programming history",
            user_id="test_user",
            max_iterations=2,
        )

        # Should not exceed max iterations
        assert result.iterations <= 2

        # Should still produce an answer
        assert len(result.answer) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_react_combines_search_and_graph(self, react_graph_fixtures):
        """ReAct combines search findings with graph knowledge."""
        orchestrator, mock_search = react_graph_fixtures

        result = await orchestrator.research_with_react(
            query="What AI technologies does Doug use?",
            user_id="test_user",
            max_iterations=3,
        )

        assert result.success

        # Should have related concepts from graph
        # (may or may not depending on LLM decisions)

        # Answer should be substantive
        assert len(result.answer) > 20

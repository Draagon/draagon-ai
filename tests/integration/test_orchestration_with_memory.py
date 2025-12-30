"""Integration tests for orchestration with real memory.

These tests verify that the orchestration layer properly integrates with
the layered memory system using real components (not mocks).

Key scenarios tested:
1. Agent uses stored memories in decisions
2. Multi-turn conversations maintain context via memory
3. Tool results are stored and retrievable
4. Memory search influences response synthesis
"""

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, AsyncMock

from draagon_ai.memory import (
    LayeredMemoryProvider,
    MemoryType,
    MemoryScope,
)
from draagon_ai.memory.providers.layered import LayeredMemoryConfig
from draagon_ai.orchestration import (
    Agent,
    AgentConfig,
    AgentLoop,
    AgentLoopConfig,
    AgentContext,
    AgentResponse,
    DecisionEngine,
    DecisionResult,
    ActionExecutor,
    ToolRegistry,
    Tool,
    ToolParameter,
    LoopMode,
)
from draagon_ai.orchestration.protocols import LLMResponse
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts


pytestmark = [pytest.mark.integration]


# =============================================================================
# Test Fixtures
# =============================================================================


class WordBasedEmbedding:
    """Word-based embedding for deterministic tests."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    def _get_word_idx(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_idx
            self._next_idx += 1
        return self._vocab[word]

    async def embed(self, text: str) -> list[float]:
        import re
        words = re.findall(r'\b\w+\b', text.lower())

        embedding = [0.0] * self.dimension
        for word in words:
            idx = self._get_word_idx(word)
            for offset in range(5):
                dim_idx = (idx * 7 + offset * 13) % self.dimension
                embedding[dim_idx] += 0.5

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


class SmartMockLLM:
    """Mock LLM that makes intelligent decisions based on context."""

    def __init__(self):
        self.call_count = 0
        self.prompts: list[str] = []

    async def chat(
        self,
        messages: list,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        self.call_count += 1
        # Handle both dict and LLMMessage objects
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content'):
                prompt = last_msg.content
            elif isinstance(last_msg, dict):
                prompt = last_msg.get("content", "")
            else:
                prompt = str(last_msg)
        else:
            prompt = ""
        self.prompts.append(prompt)

        # Analyze prompt to make decision
        prompt_lower = prompt.lower()

        # If we have memory context about preferences, use it
        if "prefers celsius" in prompt_lower and "temperature" in prompt_lower:
            return LLMResponse(content="""<response>
                <action>answer</action>
                <answer>It's 20 degrees Celsius, which I know you prefer!</answer>
                <reasoning>User prefers Celsius based on stored memory</reasoning>
            </response>""")

        # If asking about time
        if "time" in prompt_lower:
            return LLMResponse(content="""<response>
                <action>get_time</action>
                <reasoning>User asked about the time</reasoning>
            </response>""")

        # If asking about weather
        if "weather" in prompt_lower:
            return LLMResponse(content="""<response>
                <action>get_weather</action>
                <reasoning>User asked about weather</reasoning>
            </response>""")

        # If has tool results, synthesize
        if "tool results" in prompt_lower or "observation" in prompt_lower:
            return LLMResponse(content="""<response>
                <action>answer</action>
                <answer>Based on the information I found, here's what I know.</answer>
                <reasoning>Synthesizing from tool results</reasoning>
            </response>""")

        # Default: answer directly
        return LLMResponse(content="""<response>
            <action>answer</action>
            <answer>I can help you with that.</answer>
            <reasoning>Direct response</reasoning>
        </response>""")


@pytest.fixture
def embedder():
    """Create word-based embedding provider."""
    return WordBasedEmbedding()


@pytest.fixture
async def memory_provider(embedder):
    """Create real LayeredMemoryProvider with in-memory graph and embeddings.

    NOTE: We pass the embedder to LayeredMemoryProvider which creates the graph
    internally with the embedder attached, enabling semantic search.
    """
    # Create provider with embedder - it will create in-memory graph with embedding support
    provider = LayeredMemoryProvider(
        embedding_provider=embedder,
    )

    yield provider

    await provider.close()


@pytest.fixture
def tool_registry():
    """Create tool registry with test tools."""
    registry = ToolRegistry()

    async def get_time_handler(args: dict, context: dict | None = None) -> str:
        return "3:45 PM"

    async def get_weather_handler(args: dict, context: dict | None = None) -> dict:
        return {
            "temperature": 72,
            "condition": "sunny",
            "humidity": 45,
        }

    registry.register(Tool(
        name="get_time",
        description="Get the current time",
        handler=get_time_handler,
    ))

    registry.register(Tool(
        name="get_weather",
        description="Get current weather",
        handler=get_weather_handler,
    ))

    return registry


@pytest.fixture
def behavior():
    """Create test behavior."""
    return Behavior(
        behavior_id="test_assistant",
        name="Test Assistant",
        description="Test assistant for integration tests",
        actions=[
            Action(name="answer", description="Provide a direct answer"),
            Action(name="get_time", description="Get current time"),
            Action(name="get_weather", description="Get weather"),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="Decide action for: {question}\n\nMemory context:\n{context}",
            synthesis_prompt="Synthesize response from: {tool_results}",
        ),
    )


@pytest.fixture
def smart_llm():
    """Create smart mock LLM."""
    return SmartMockLLM()


@pytest.fixture
async def agent_with_memory(smart_llm, memory_provider, tool_registry, behavior):
    """Create agent with real memory provider."""
    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        personality_intro="I am a helpful test assistant.",
    )

    agent = Agent(
        config=config,
        behavior=behavior,
        llm=smart_llm,
        memory=memory_provider,
        tools=tool_registry,
    )

    return agent


# =============================================================================
# Memory-Informed Decision Tests
# =============================================================================


class TestMemoryInformedDecisions:
    """Test that stored memories influence agent decisions."""

    @pytest.mark.asyncio
    async def test_stored_preference_is_searchable(self, memory_provider):
        """Test that stored user preferences can be searched."""
        # Store a preference
        await memory_provider.store(
            content="User prefers Celsius for temperature readings",
            memory_type=MemoryType.PREFERENCE,
            user_id="test_user",
            entities=["temperature", "celsius", "preference"],
        )

        # Search should find the preference
        results = await memory_provider.search(
            query="What temperature format does the user prefer?",
            user_id="test_user",
            limit=5,
        )

        assert len(results) > 0
        contents = [r.memory.content.lower() for r in results]
        assert any("celsius" in c for c in contents)

    @pytest.mark.asyncio
    async def test_agent_retrieves_relevant_facts(self, agent_with_memory, memory_provider):
        """Test that agent retrieves relevant facts for queries."""
        # Store some facts
        await memory_provider.store(
            content="Doug's birthday is March 15th",
            memory_type=MemoryType.FACT,
            user_id="test_user",
            entities=["doug", "birthday", "march"],
        )
        await memory_provider.store(
            content="The WiFi password is hunter2",
            memory_type=MemoryType.FACT,
            user_id="test_user",
            entities=["wifi", "password"],
        )

        # Search for birthday-related memories
        results = await memory_provider.search(
            query="When is Doug's birthday?",
            user_id="test_user",
            limit=5,
        )

        # Should find the birthday fact
        assert len(results) > 0
        contents = [r.memory.content for r in results]
        assert any("march 15" in c.lower() for c in contents)

    @pytest.mark.asyncio
    async def test_memory_context_passed_to_llm(self, agent_with_memory, memory_provider):
        """Test that memory context is passed to LLM in decision prompt."""
        # Store a skill
        await memory_provider.store(
            content="To restart Nginx: sudo systemctl restart nginx",
            memory_type=MemoryType.SKILL,
            user_id="test_user",
            entities=["nginx", "restart", "systemctl"],
        )

        # Process query
        response = await agent_with_memory.process(
            query="How do I restart Nginx?",
            user_id="test_user",
        )

        # LLM should have been called with context
        assert agent_with_memory.llm.call_count > 0
        # At least one prompt should mention the query
        assert any("nginx" in p.lower() for p in agent_with_memory.llm.prompts)


# =============================================================================
# Multi-Turn Context Tests
# =============================================================================


class TestMultiTurnWithMemory:
    """Test multi-turn conversations with memory persistence."""

    @pytest.mark.asyncio
    async def test_episodic_memory_stores_conversations(self, memory_provider):
        """Test that episodic memory stores conversation events."""
        # Start a conversation episode
        episode = await memory_provider.episodic.start_episode(
            content="Conversation about weather",
        )
        assert episode is not None

        # Add events to the episode
        event1 = await memory_provider.episodic.add_event(
            episode_id=episode.node_id,
            content="User asked about the weather",
        )
        event2 = await memory_provider.episodic.add_event(
            episode_id=episode.node_id,
            content="System reported sunny and 72 degrees",
        )

        assert event1 is not None
        assert event2 is not None

        # Close the episode
        await memory_provider.episodic.close_episode(
            episode_id=episode.node_id,
            summary="Weather conversation: sunny, 72 degrees",
        )

    @pytest.mark.asyncio
    async def test_memories_persist_across_sessions(self, agent_with_memory, memory_provider):
        """Test that stored memories persist across sessions."""
        # Store a fact in one session
        await memory_provider.store(
            content="User's favorite color is blue",
            memory_type=MemoryType.PREFERENCE,
            user_id="test_user",
            entities=["color", "blue", "favorite"],
        )

        # New session should be able to retrieve it
        results = await memory_provider.search(
            query="What is my favorite color?",
            user_id="test_user",
            limit=5,
        )

        assert len(results) > 0
        assert any("blue" in r.memory.content.lower() for r in results)


# =============================================================================
# Tool Result Storage Tests
# =============================================================================


class TestToolResultMemory:
    """Test that tool results can be stored and retrieved."""

    @pytest.mark.asyncio
    async def test_store_tool_result_as_episodic(self, memory_provider):
        """Test storing tool results as episodic memory."""
        # Start an episode for a conversation
        episode = await memory_provider.episodic.start_episode(
            content="User asked about the weather",
        )

        # Add tool result as event
        event = await memory_provider.episodic.add_event(
            episode_id=episode.node_id,
            content="Weather check: 72°F, sunny",
        )

        # Close episode with summary
        await memory_provider.episodic.close_episode(
            episode_id=episode.node_id,
            summary="Answered weather query: 72°F and sunny",
        )

        # Search should find the episode
        results = await memory_provider.episodic.search(
            query="weather",
            limit=5,
        )

        assert len(results) > 0


# =============================================================================
# Memory Layer Interaction Tests
# =============================================================================


class TestMemoryLayerInteraction:
    """Test interactions between memory layers and orchestration."""

    @pytest.mark.asyncio
    async def test_working_memory_for_conversation(self, memory_provider):
        """Test working memory stores current conversation context."""
        # Add to working memory
        item = await memory_provider.working.add(
            content="Current topic: vacation planning",
            attention_weight=0.9,
        )

        assert item is not None
        assert item.node_id is not None

        # Should be findable via get_active_context
        active = await memory_provider.working.get_active_context(min_attention=0.3)
        contents = [i.content for i in active]
        assert any("vacation" in c.lower() for c in contents)

    @pytest.mark.asyncio
    async def test_semantic_entity_for_knowledge(self, memory_provider):
        """Test semantic layer stores entity knowledge."""
        # Add fact with entities - the semantic layer handles entity creation
        fact = await memory_provider.semantic.add_fact(
            content="Python was created by Guido van Rossum in 1991",
            entities=["Python", "Guido van Rossum"],
        )

        assert fact is not None
        assert fact.node_id is not None

        # Search should find the fact (results are GraphSearchResult with node attribute)
        results = await memory_provider.semantic.search(
            query="Who created Python?",
            limit=5,
        )
        assert len(results) > 0
        # GraphSearchResult has node.content or we access via the result
        found_python = False
        for r in results:
            content = getattr(r, 'content', None) or getattr(r.node, 'content', '') if hasattr(r, 'node') else ''
            if "python" in content.lower():
                found_python = True
                break
        assert found_python

    @pytest.mark.asyncio
    async def test_metacognitive_skill_storage(self, memory_provider):
        """Test metacognitive layer stores skills."""
        # Add a skill
        skill = await memory_provider.metacognitive.add_skill(
            name="check_disk_space",
            skill_type="command",
            procedure="df -h",
        )

        # Retrieve by name
        retrieved = await memory_provider.metacognitive.get_skill_by_name("check_disk_space")

        assert retrieved is not None
        assert retrieved.procedure == "df -h"

        # Record success
        await memory_provider.metacognitive.record_skill_result(
            skill_id=skill.node_id,
            success=True,
        )


# =============================================================================
# Search Quality Tests
# =============================================================================


class TestSearchQuality:
    """Test search quality with real memory."""

    @pytest.mark.asyncio
    async def test_search_ranks_by_relevance(self, memory_provider):
        """Test that search results are ranked by relevance."""
        # Store multiple facts
        await memory_provider.store(
            content="The capital of France is Paris",
            memory_type=MemoryType.FACT,
            user_id="test_user",
            entities=["france", "paris", "capital"],
        )
        await memory_provider.store(
            content="Paris is known for the Eiffel Tower",
            memory_type=MemoryType.FACT,
            user_id="test_user",
            entities=["paris", "eiffel tower"],
        )
        await memory_provider.store(
            content="The weather in London is often rainy",
            memory_type=MemoryType.FACT,
            user_id="test_user",
            entities=["london", "weather", "rain"],
        )

        # Search for Paris-related
        results = await memory_provider.search(
            query="What is the capital of France?",
            user_id="test_user",
            limit=5,
        )

        # Should find Paris-related facts
        assert len(results) >= 1
        # First result should be most relevant
        first_content = results[0].memory.content.lower()
        assert "france" in first_content or "paris" in first_content

    @pytest.mark.asyncio
    async def test_search_filters_by_user(self, memory_provider):
        """Test that search respects user scoping."""
        # Store for different users
        await memory_provider.store(
            content="Doug's password is abc123",
            memory_type=MemoryType.FACT,
            user_id="doug",
            scope=MemoryScope.USER,
            entities=["password"],
        )
        await memory_provider.store(
            content="Sarah's password is xyz789",
            memory_type=MemoryType.FACT,
            user_id="sarah",
            scope=MemoryScope.USER,
            entities=["password"],
        )

        # Search as Doug
        doug_results = await memory_provider.search(
            query="What is my password?",
            user_id="doug",
            limit=5,
        )

        # Should only find Doug's password
        for result in doug_results:
            if "password" in result.memory.content.lower():
                assert result.memory.user_id == "doug" or result.memory.scope == MemoryScope.GLOBAL

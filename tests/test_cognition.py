"""Tests for draagon_ai cognition services.

These tests verify that the cognition services work correctly with
the protocol interfaces.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.llm import ChatResponse, ModelTier
from draagon_ai.memory import Memory, MemoryScope, MemoryType, SearchResult
from draagon_ai.cognition import (
    BeliefReconciliationService,
    OpinionFormationService,
    OpinionRequest,
    CuriosityEngine,
    LearningService,
    LearningType,
    IdentityManager,
)


# =============================================================================
# Mock Providers
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    mock = MagicMock()
    mock.chat = AsyncMock(return_value=ChatResponse(
        content='{"reconciled": true, "belief": "The household has cats"}',
        role="assistant",
    ))
    return mock


@pytest.fixture
def mock_memory():
    """Create a mock memory provider."""
    mock = MagicMock()
    mock.store = AsyncMock(return_value=Memory(
        id="mem_123",
        content="Test memory",
        memory_type=MemoryType.FACT,
        scope=MemoryScope.USER,
    ))
    mock.search = AsyncMock(return_value=[])
    mock.get = AsyncMock(return_value=None)
    mock.update = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_identity_manager():
    """Create mock identity manager that implements the IdentityManager protocol."""
    from draagon_ai.core import AgentIdentity

    # Create a minimal AgentIdentity
    identity = MagicMock(spec=AgentIdentity)
    identity.values = {}
    identity.worldview = {}
    identity.principles = {}
    identity.opinions = {}
    identity.preferences = {}
    identity.traits = {}

    mock = MagicMock()
    mock.load = AsyncMock(return_value=identity)
    mock.mark_dirty = MagicMock()
    mock.save_if_dirty = AsyncMock(return_value=True)
    return mock


# =============================================================================
# Belief Reconciliation Tests
# =============================================================================


class TestBeliefReconciliationService:
    """Tests for BeliefReconciliationService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_llm, mock_memory):
        """Test that service can be initialized with protocol implementations."""
        service = BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        assert service.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_create_observation_calls_llm(self, mock_llm, mock_memory):
        """Test that creating observation calls the LLM provider."""
        mock_llm.chat.return_value = ChatResponse(
            content='''{
                "content": "The household has 5 cats",
                "scope": "household",
                "confidence": 0.9
            }''',
            role="assistant",
        )

        service = BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        # This should call the LLM
        result = await service.create_observation(
            statement="We have 5 cats",
            user_id="doug",
            context="Discussing pets",
        )

        mock_llm.chat.assert_called()


# =============================================================================
# Opinion Formation Tests
# =============================================================================


class TestOpinionFormationService:
    """Tests for OpinionFormationService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_llm, mock_memory, mock_identity_manager):
        """Test that service can be initialized."""
        service = OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=mock_identity_manager,
            agent_name="TestBot",
        )

        assert service.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_form_opinion_calls_llm(self, mock_llm, mock_memory, mock_identity_manager):
        """Test that opinion formation calls the LLM."""
        mock_llm.chat.return_value = ChatResponse(
            content='''{
                "have_opinion": true,
                "stance": "I think it can be good in moderation",
                "basis": "aesthetic",
                "strength": "tentative",
                "confidence": 0.7,
                "reasoning": "Sweet and savory can work well together",
                "caveats": [],
                "could_be_wrong": true
            }''',
            role="assistant",
        )

        service = OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=mock_identity_manager,
            agent_name="TestBot",
        )

        # form_opinion takes an OpinionRequest object
        request = OpinionRequest(
            topic="pineapple on pizza",
            user_id="test_user",
            context="User asked for my opinion on Hawaiian pizza",
        )
        result = await service.form_opinion(request)

        mock_llm.chat.assert_called()


# =============================================================================
# Curiosity Engine Tests
# =============================================================================


class TestCuriosityEngine:
    """Tests for CuriosityEngine."""

    @pytest.fixture
    def mock_trait_provider(self):
        """Create mock trait provider that implements TraitProvider protocol."""
        mock = MagicMock()
        # TraitProvider protocol uses get_trait_value, not get_personality_trait
        mock.get_trait_value = MagicMock(return_value=0.7)
        return mock

    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_llm, mock_memory, mock_trait_provider):
        """Test that service can be initialized."""
        service = CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
        )

        assert service.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_analyze_for_curiosity_calls_llm(self, mock_llm, mock_memory, mock_trait_provider):
        """Test that curiosity analysis calls the LLM."""
        mock_llm.chat.return_value = ChatResponse(
            content='''{
                "analysis": "The conversation is about calendar events",
                "knowledge_gaps": [
                    {
                        "topic": "user's birthday",
                        "importance": 0.7,
                        "why_important": "Useful for reminders"
                    }
                ],
                "potential_questions": [
                    {
                        "question": "When is your birthday?",
                        "purpose": "learn_about_user",
                        "type": "knowledge_gap",
                        "priority": "medium",
                        "target_user": "doug",
                        "why_this_question": "For reminders",
                        "what_roxy_will_do_with_answer": "Store for future reminders",
                        "context_to_remember": "Discussing calendar events",
                        "interesting_to_user": true
                    }
                ],
                "should_explore": false,
                "exploration_topic": null
            }''',
            role="assistant",
        )

        service = CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
        )

        # The method is analyze_for_curiosity, not analyze_for_questions
        result = await service.analyze_for_curiosity(
            conversation="User: Discussed calendar events\nAssistant: Here are your events...",
            user_id="doug",
        )

        mock_llm.chat.assert_called()


# =============================================================================
# Learning Service Tests
# =============================================================================


class TestLearningService:
    """Tests for LearningService."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, mock_llm, mock_memory):
        """Test that service can be initialized."""
        service = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        assert service.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_process_interaction_calls_llm(self, mock_llm, mock_memory):
        """Test learning opportunity detection."""
        # LearningService uses chat_json, not chat
        mock_llm.chat_json = AsyncMock(return_value={
            "parsed": {
                "learned_something": False,
                "learning_type": None,
                "confidence": 0.3,
                "reasoning": "Simple greeting",
            }
        })

        service = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        # process_interaction takes: user_query, response, tool_calls, user_id, conversation_id
        result = await service.process_interaction(
            user_query="My birthday is March 15th",
            response="Got it! I'll remember your birthday is March 15.",
            tool_calls=[],
            user_id="doug",
            conversation_id="conv_123",
        )

        mock_llm.chat_json.assert_called()


# Note: Tests for Roxy-specific adapters are in the roxy-voice-assistant repo
# in tests/suites/draagon_ai/test_adapters.py

"""Integration tests for draagon_ai cognition services.

These tests verify the full service behavior including:
- Prompt construction
- Response parsing
- Error handling
- Service interactions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from draagon_ai.llm import ChatResponse, ModelTier
from draagon_ai.memory import Memory, MemoryScope, MemoryType, SearchResult
from draagon_ai.cognition import (
    BeliefReconciliationService,
    ReconciliationResult,
    OpinionFormationService,
    OpinionRequest,
    FormedOpinion,
    OpinionStrength,
    OpinionBasis,
    CuriosityEngine,
    CuriousQuestion,
    QuestionType,
    QuestionPriority,
    LearningService,
    LearningResult,
    LearningType,
    LearningExtension,
    VerificationResult,
    ProactiveQuestionTimingService,
    QuestionOpportunity,
    ConversationMoment,
)
from draagon_ai.core import AgentIdentity


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider with both chat and chat_json."""
    mock = MagicMock()
    mock.chat = AsyncMock()
    mock.chat_json = AsyncMock()
    return mock


@pytest.fixture
def mock_memory():
    """Create a mock memory provider.

    Note: The LearningService expects store() to return a dict with
    'success' and 'memory_id' keys.
    """
    mock = MagicMock()
    mock.store = AsyncMock(return_value={
        "success": True,
        "memory_id": "mem_123",
    })
    mock.search = AsyncMock(return_value=[])
    mock.get = AsyncMock(return_value=None)
    mock.update = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_identity_manager():
    """Create mock identity manager with proper AgentIdentity structure."""
    # Create a proper AgentIdentity-like mock
    # The values dict should contain AgentValue-like objects
    class MockAgentValue:
        def __init__(self, desc, strength):
            self.description = desc
            self.strength = strength

    identity = MagicMock()
    identity.values = {
        "truth_seeking": MockAgentValue("Seeking truth", 0.95),
    }
    identity.worldview = {}
    identity.principles = {}
    identity.opinions = {}
    identity.preferences = {}
    identity.traits = {"curiosity_intensity": 0.7}

    mock = MagicMock()
    mock.load = AsyncMock(return_value=identity)
    mock.mark_dirty = MagicMock()
    mock.save_if_dirty = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_trait_provider():
    """Create mock trait provider."""
    mock = MagicMock()
    mock.get_trait_value = MagicMock(return_value=0.7)
    return mock


@pytest.fixture
def mock_user_preferences_provider():
    """Create mock user preferences provider."""
    mock = MagicMock()
    mock.get_question_tolerance = MagicMock(return_value=0.8)
    return mock


# =============================================================================
# Belief Reconciliation Integration Tests
# =============================================================================


class TestBeliefReconciliationIntegration:
    """Integration tests for BeliefReconciliationService."""

    @pytest.mark.asyncio
    async def test_observation_creation_calls_llm(self, mock_llm, mock_memory):
        """Test that observation creation calls the LLM."""
        mock_llm.chat.return_value = ChatResponse(
            content='{"content": "User has 3 cats", "scope": "household", "confidence_expressed": 0.9, "entities": ["cats"]}',
            role="assistant",
        )

        service = BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        await service.create_observation(
            statement="We have 3 cats",
            user_id="doug",
            context="Discussing household pets",
        )

        # Verify LLM was called
        mock_llm.chat.assert_called()

    @pytest.mark.asyncio
    async def test_reconciliation_with_conflicting_observations(self, mock_llm, mock_memory):
        """Test belief reconciliation when observations conflict."""
        # First return observations, then reconciled belief
        mock_llm.chat.side_effect = [
            ChatResponse(
                content='{"content": "User has 5 cats", "scope": "household", "confidence_expressed": 0.9, "entities": ["cats"]}',
                role="assistant",
            ),
            ChatResponse(
                content='{"content": "User has 5 cats", "confidence": 0.85, "needs_clarification": false}',
                role="assistant",
            ),
        ]

        # Return existing conflicting observation
        mock_memory.search.return_value = [
            SearchResult(
                memory=Memory(
                    id="obs_old",
                    content='{"content": "User has 3 cats", "scope": "household"}',
                    memory_type=MemoryType.OBSERVATION,
                    scope=MemoryScope.USER,
                ),
                score=0.9,
            )
        ]

        service = BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        await service.create_observation(
            statement="Actually we have 5 cats now",
            user_id="doug",
            context="Updating pet count",
        )

        # Should have called search for existing observations
        mock_memory.search.assert_called()

    @pytest.mark.asyncio
    async def test_service_with_credibility_provider(self, mock_llm, mock_memory):
        """Test that service can be initialized with credibility provider."""
        mock_llm.chat.return_value = ChatResponse(
            content='{"content": "Sky is green", "scope": "general", "confidence_expressed": 0.9, "entities": []}',
            role="assistant",
        )

        # Credibility provider
        cred_provider = MagicMock()
        cred_provider.get_credibility = MagicMock(return_value=0.3)

        service = BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
            credibility_provider=cred_provider,
        )

        # Just verify the service can be created with credibility provider
        assert service is not None


# =============================================================================
# Opinion Formation Integration Tests
# =============================================================================


class TestOpinionFormationIntegration:
    """Integration tests for OpinionFormationService."""

    @pytest.mark.asyncio
    async def test_opinion_formation_uses_identity(self, mock_llm, mock_memory, mock_identity_manager):
        """Test that opinion formation incorporates agent identity."""
        mock_llm.chat.return_value = ChatResponse(
            content="""{
                "have_opinion": true,
                "stance": "I think it's a matter of personal preference",
                "basis": "aesthetic",
                "strength": "tentative",
                "confidence": 0.6,
                "reasoning": "Taste is subjective",
                "caveats": ["Everyone's palate is different"],
                "could_be_wrong": true
            }""",
            role="assistant",
        )

        service = OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=mock_identity_manager,
            agent_name="TestBot",
        )

        request = OpinionRequest(
            topic="pineapple on pizza",
            user_id="doug",
            context="Casual conversation about food",
        )

        result = await service.form_opinion(request)

        # Should load identity for context
        mock_identity_manager.load.assert_called()

        # Result should have formed opinion
        assert result is not None

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_llm_failure(self, mock_llm, mock_memory, mock_identity_manager):
        """Test graceful handling when LLM returns invalid JSON."""
        mock_llm.chat.return_value = ChatResponse(
            content="I don't know, that's a tough question!",  # Invalid JSON
            role="assistant",
        )

        service = OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=mock_identity_manager,
            agent_name="TestBot",
        )

        request = OpinionRequest(
            topic="best programming language",
            user_id="doug",
            context="Casual discussion",
        )

        # Should not raise, should return a fallback opinion
        result = await service.form_opinion(request)
        # Service returns fallback opinion when parsing fails
        assert result is not None
        # Fallback should have low confidence
        assert result.confidence < 0.5
        assert "haven't formed an opinion" in result.stance.lower() or "need to think" in result.stance.lower()


# =============================================================================
# Curiosity Engine Integration Tests
# =============================================================================


class TestCuriosityEngineIntegration:
    """Integration tests for CuriosityEngine."""

    @pytest.mark.asyncio
    async def test_curiosity_analysis_extracts_questions(self, mock_llm, mock_memory, mock_trait_provider):
        """Test that curiosity analysis extracts potential questions.

        Note: analyze_for_curiosity returns List[CuriousQuestion], not an object.
        """
        mock_llm.chat.return_value = ChatResponse(
            content="""{
                "analysis": "User mentioned a deadline but didn't specify date",
                "knowledge_gaps": [
                    {
                        "topic": "project deadline",
                        "importance": 0.8,
                        "why_important": "Helps with scheduling"
                    }
                ],
                "potential_questions": [
                    {
                        "question": "When is the project deadline?",
                        "purpose": "complete_knowledge",
                        "type": "knowledge_gap",
                        "priority": "high",
                        "target_user": "doug",
                        "why_this_question": "Deadline was mentioned but not specified",
                        "what_agent_will_do_with_answer": "Add calendar reminder",
                        "context_to_remember": "Discussing project work",
                        "interesting_to_user": true
                    }
                ],
                "should_explore": false,
                "exploration_topic": null
            }""",
            role="assistant",
        )

        service = CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
        )

        # analyze_for_curiosity returns List[CuriousQuestion]
        result = await service.analyze_for_curiosity(
            conversation="User: I need to finish the project before the deadline.\nAssistant: Got it!",
            user_id="doug",
        )

        # Result is a list of questions
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].question == "When is the project deadline?"

    @pytest.mark.asyncio
    async def test_curiosity_respects_trait_level(self, mock_llm, mock_memory):
        """Test that curiosity respects personality trait level."""
        low_curiosity_provider = MagicMock()
        low_curiosity_provider.get_trait_value = MagicMock(return_value=0.2)  # Low curiosity

        mock_llm.chat.return_value = ChatResponse(
            content="""{
                "analysis": "Simple greeting",
                "knowledge_gaps": [],
                "potential_questions": [],
                "should_explore": false,
                "exploration_topic": null
            }""",
            role="assistant",
        )

        service = CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=low_curiosity_provider,
            agent_name="TestBot",
        )

        result = await service.analyze_for_curiosity(
            conversation="User: Hello!\nAssistant: Hi!",
            user_id="doug",
        )

        # With low curiosity, should still work but be less proactive
        low_curiosity_provider.get_trait_value.assert_called()
        assert isinstance(result, list)


# =============================================================================
# Learning Service Integration Tests
# =============================================================================


class TestLearningServiceIntegration:
    """Integration tests for LearningService."""

    @pytest.mark.asyncio
    async def test_learning_detection_for_facts(self, mock_llm, mock_memory):
        """Test that facts are properly detected and stored.

        Note: LearningResult uses 'learned' not 'learned_something'.
        """
        mock_llm.chat_json.return_value = {
            "parsed": {
                "learned_something": True,
                "learning_type": "fact",
                "content": "Doug's birthday is March 15",
                "confidence": 0.95,
                "entities": ["doug", "birthday", "march 15"],
                "reasoning": "User explicitly stated their birthday",
            }
        }

        service = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        result = await service.process_interaction(
            user_query="My birthday is March 15th",
            response="I'll remember that your birthday is March 15th!",
            tool_calls=[],
            user_id="doug",
            conversation_id="conv_123",
        )

        # Should store the memory
        mock_memory.store.assert_called()
        # LearningResult uses 'learned' attribute
        assert result.learned is True

    @pytest.mark.asyncio
    async def test_learning_detection_for_skills(self, mock_llm, mock_memory):
        """Test that skills/procedures are properly detected."""
        mock_llm.chat_json.return_value = {
            "parsed": {
                "learned_something": True,
                "learning_type": "skill",
                "content": "To restart Plex: docker restart plex",
                "confidence": 0.9,
                "entities": ["plex", "docker", "restart"],
                "reasoning": "Successful execution of restart command",
            }
        }

        service = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        result = await service.process_interaction(
            user_query="Restart plex",
            response="Done! I restarted Plex.",
            tool_calls=[{
                "tool": "execute_command",
                "args": {"command": "docker restart plex"},
                "result": {"success": True},
            }],
            user_id="doug",
            conversation_id="conv_123",
        )

        assert result.learned is True

    @pytest.mark.asyncio
    async def test_no_learning_when_detection_fails(self, mock_llm, mock_memory):
        """Test that no learning occurs when LLM doesn't detect anything."""
        mock_llm.chat_json.return_value = {
            "parsed": {
                "learned_something": False,
                "learning_type": None,
                "confidence": 0.3,
                "reasoning": "Simple greeting, nothing to learn",
            }
        }

        service = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        result = await service.process_interaction(
            user_query="Hello!",
            response="Hi there!",
            tool_calls=[],
            user_id="doug",
            conversation_id="conv_123",
        )

        # Should not store anything
        mock_memory.store.assert_not_called()
        assert result.learned is False


# =============================================================================
# Proactive Question Timing Integration Tests
# =============================================================================


class TestProactiveQuestionTimingIntegration:
    """Integration tests for ProactiveQuestionTimingService."""

    @pytest.fixture
    def mock_curiosity_engine(self, mock_llm, mock_memory, mock_trait_provider):
        """Create a mock curiosity engine."""
        engine = MagicMock()
        engine.get_pending_questions = MagicMock(return_value=[])
        return engine

    @pytest.mark.asyncio
    async def test_timing_service_initialization(
        self,
        mock_llm,
        mock_trait_provider,
        mock_user_preferences_provider,
        mock_curiosity_engine,
    ):
        """Test that timing service can be initialized."""
        service = ProactiveQuestionTimingService(
            llm=mock_llm,
            curiosity_engine=mock_curiosity_engine,
            user_prefs_provider=mock_user_preferences_provider,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
        )

        assert service.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_rate_limiting_logic(
        self,
        mock_llm,
        mock_trait_provider,
        mock_user_preferences_provider,
        mock_curiosity_engine,
    ):
        """Test that rate limiting prevents too many questions."""
        service = ProactiveQuestionTimingService(
            llm=mock_llm,
            curiosity_engine=mock_curiosity_engine,
            user_prefs_provider=mock_user_preferences_provider,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
            max_questions_per_day=3,
            min_time_between_questions_minutes=30,
        )

        # Simulate 3 questions already asked today using correct format
        today = datetime.now().strftime("%Y-%m-%d")
        service._questions_asked_today = {
            f"doug:{today}": 3,  # Already at limit
        }

        # Use private method to check rate limiting
        can_ask = service._should_even_try("doug")
        assert can_ask is False

    @pytest.mark.asyncio
    async def test_can_ask_when_under_limit(
        self,
        mock_llm,
        mock_trait_provider,
        mock_user_preferences_provider,
        mock_curiosity_engine,
    ):
        """Test that questions can be asked when under limit."""
        service = ProactiveQuestionTimingService(
            llm=mock_llm,
            curiosity_engine=mock_curiosity_engine,
            user_prefs_provider=mock_user_preferences_provider,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
            max_questions_per_day=3,
            min_time_between_questions_minutes=30,
        )

        # No questions asked today
        service._questions_asked_today = {}
        service._last_question_asked = {}

        can_ask = service._should_even_try("doug")
        assert can_ask is True

    @pytest.mark.asyncio
    async def test_min_gap_enforcement(
        self,
        mock_llm,
        mock_trait_provider,
        mock_user_preferences_provider,
        mock_curiosity_engine,
    ):
        """Test that minimum gap between questions is enforced."""
        service = ProactiveQuestionTimingService(
            llm=mock_llm,
            curiosity_engine=mock_curiosity_engine,
            user_prefs_provider=mock_user_preferences_provider,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
            max_questions_per_day=10,
            min_time_between_questions_minutes=30,
        )

        # Asked a question 10 minutes ago - too soon
        service._last_question_asked = {
            "doug": datetime.now() - timedelta(minutes=10),
        }
        assert service._should_even_try("doug") is False

        # Asked a question 45 minutes ago - enough time passed
        service._last_question_asked = {
            "doug": datetime.now() - timedelta(minutes=45),
        }
        assert service._should_even_try("doug") is True


# =============================================================================
# Cross-Service Integration Tests
# =============================================================================


class TestCrossServiceIntegration:
    """Tests for interactions between cognitive services."""

    @pytest.mark.asyncio
    async def test_learning_stores_facts(self, mock_llm, mock_memory):
        """Test that learning service properly stores facts."""
        mock_llm.chat_json.return_value = {
            "parsed": {
                "learned_something": True,
                "learning_type": "fact",
                "content": "The household has 5 cats now",
                "confidence": 0.95,
                "entities": ["cats", "household"],
                "reasoning": "User updated the cat count",
            }
        }

        learning_service = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        result = await learning_service.process_interaction(
            user_query="Actually we got another cat, so we have 5 now",
            response="Wonderful! I'll update that you have 5 cats.",
            tool_calls=[],
            user_id="doug",
            conversation_id="conv_123",
        )

        # The learning should be stored
        assert result.learned is True
        mock_memory.store.assert_called()

    @pytest.mark.asyncio
    async def test_curiosity_and_learning_flow(
        self,
        mock_llm,
        mock_memory,
        mock_trait_provider,
    ):
        """Test the flow from curiosity question to learning."""
        # First, curiosity detects a knowledge gap
        mock_llm.chat.return_value = ChatResponse(
            content="""{
                "analysis": "User's birthday is unknown",
                "knowledge_gaps": [{"topic": "birthday", "importance": 0.8, "why_important": "For reminders"}],
                "potential_questions": [{
                    "question": "When is your birthday?",
                    "purpose": "learn_about_user",
                    "type": "knowledge_gap",
                    "priority": "medium",
                    "target_user": "doug",
                    "why_this_question": "Can set reminders",
                    "what_agent_will_do_with_answer": "Store for calendar",
                    "context_to_remember": "Getting to know user",
                    "interesting_to_user": true
                }],
                "should_explore": false,
                "exploration_topic": null
            }""",
            role="assistant",
        )

        curiosity = CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
        )

        # Analyze conversation for curiosity
        curiosity_result = await curiosity.analyze_for_curiosity(
            conversation="User: Hi there!\nAssistant: Hello!",
            user_id="doug",
        )

        assert len(curiosity_result) > 0

        # Later, user answers the question - learning service picks it up
        mock_llm.chat_json.return_value = {
            "parsed": {
                "learned_something": True,
                "learning_type": "fact",
                "content": "Doug's birthday is March 15",
                "confidence": 0.95,
                "entities": ["doug", "birthday", "march"],
                "reasoning": "User answered curiosity question",
            }
        }

        learning = LearningService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
        )

        learning_result = await learning.process_interaction(
            user_query="My birthday is March 15th",
            response="Great, I'll remember March 15th!",
            tool_calls=[],
            user_id="doug",
            conversation_id="conv_123",
        )

        assert learning_result.learned is True

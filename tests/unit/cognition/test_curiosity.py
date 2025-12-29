"""Unit tests for Curiosity Engine.

These tests verify the curiosity engine logic works correctly
with the draagon-ai LLM and memory protocols.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from draagon_ai.llm import MockLLM, ChatResponse
from draagon_ai.memory import Memory, MemoryType, MemoryScope, SearchResult
from draagon_ai.cognition.curiosity import (
    CuriosityEngine,
    CuriousQuestion,
    KnowledgeGap,
    QuestionType,
    QuestionPriority,
    QuestionPurpose,
    TraitProvider,
)


class MockTraitProvider:
    """Mock trait provider for testing."""

    def __init__(self, curiosity: float = 0.7):
        self._curiosity = curiosity

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        if trait_name == "curiosity_intensity":
            return self._curiosity
        return default


class TestCuriosityEngine:
    """Tests for CuriosityEngine."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM service that returns JSON responses."""
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory service."""
        memory = MagicMock()
        memory.store = AsyncMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def mock_trait_provider(self):
        """Create a mock trait provider."""
        return MockTraitProvider(curiosity=0.7)

    @pytest.fixture
    def engine(self, mock_llm, mock_memory, mock_trait_provider):
        """Create engine with mocked dependencies."""
        return CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=mock_trait_provider,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_analyze_for_curiosity_finds_questions(self, engine, mock_llm):
        """Test that curiosity analysis finds questions in conversation."""
        json_response = json.dumps({
            "analysis": "User mentioned alarms - could learn about their morning preferences",
            "knowledge_gaps": [
                {
                    "topic": "morning_routine",
                    "importance": 0.6,
                    "why_important": "Could personalize alarm suggestions",
                }
            ],
            "potential_questions": [
                {
                    "question": "Do you prefer a gentle wake-up or jumping right into action?",
                    "purpose": "learn_about_user",
                    "type": "preference",
                    "priority": "medium",
                    "target_user": "doug",
                    "why_this_question": "Would help personalize alarm suggestions",
                    "what_agent_will_do_with_answer": "Remember preference for future alarm settings",
                    "context_to_remember": "User sets alarms regularly",
                    "interesting_to_user": True,
                }
            ],
            "should_explore": False,
            "exploration_topic": None,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        questions = await engine.analyze_for_curiosity(
            conversation="Set my alarm for 6am",
            user_id="doug",
        )

        assert len(questions) == 1
        assert questions[0].question_type == QuestionType.PREFERENCE
        assert questions[0].priority == QuestionPriority.MEDIUM
        assert questions[0].purpose == QuestionPurpose.LEARN_ABOUT_USER
        assert len(mock_llm.calls) > 0

    @pytest.mark.asyncio
    async def test_low_curiosity_skips_analysis(self, mock_llm, mock_memory):
        """Test that low curiosity trait skips detailed analysis."""
        low_curiosity_provider = MockTraitProvider(curiosity=0.2)
        engine = CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=low_curiosity_provider,
            agent_name="TestBot",
            agent_id="testbot",
        )

        questions = await engine.analyze_for_curiosity(
            conversation="Some conversation",
            user_id="doug",
        )

        assert len(questions) == 0
        # LLM should not have been called due to low curiosity
        assert len(mock_llm.calls) == 0

    @pytest.mark.asyncio
    async def test_analyze_stores_knowledge_gaps(self, engine, mock_llm, mock_memory):
        """Test that knowledge gaps are stored to memory."""
        json_response = json.dumps({
            "analysis": "Found a knowledge gap",
            "knowledge_gaps": [
                {
                    "topic": "user_preferences",
                    "importance": 0.8,
                    "why_important": "To serve user better",
                }
            ],
            "potential_questions": [],
            "should_explore": False,
            "exploration_topic": None,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        await engine.analyze_for_curiosity(
            conversation="I like my coffee black",
            user_id="doug",
        )

        # Should have stored the knowledge gap
        assert mock_memory.store.called

    @pytest.mark.asyncio
    async def test_get_question_for_moment_prioritizes_high(self, engine):
        """Test that high priority questions are selected first."""
        # Add questions to queue
        low_q = CuriousQuestion(
            question_id="low1",
            question="Low priority question?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.LOW,
            target_user="doug",
            context="test",
        )
        high_q = CuriousQuestion(
            question_id="high1",
            question="High priority question?",
            question_type=QuestionType.CLARIFICATION,
            priority=QuestionPriority.HIGH,
            target_user="doug",
            context="test",
        )

        engine._question_queue = [low_q, high_q]

        selected = await engine.get_question_for_moment(
            user_id="doug",
            conversation_context="test",
        )

        assert selected is not None
        assert selected.question_id == "high1"

    @pytest.mark.asyncio
    async def test_get_question_excludes_expired(self, engine):
        """Test that expired questions are excluded."""
        from datetime import timedelta

        expired_q = CuriousQuestion(
            question_id="expired1",
            question="Expired question?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.HIGH,
            target_user="doug",
            context="test",
            expires_at=datetime.now() - timedelta(days=1),  # Expired
        )

        engine._question_queue = [expired_q]

        selected = await engine.get_question_for_moment(
            user_id="doug",
            conversation_context="test",
        )

        assert selected is None

    @pytest.mark.asyncio
    async def test_get_question_excludes_already_asked(self, engine):
        """Test that already asked questions are excluded."""
        asked_q = CuriousQuestion(
            question_id="asked1",
            question="Already asked?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.HIGH,
            target_user="doug",
            context="test",
            asked=True,
        )

        engine._question_queue = [asked_q]

        selected = await engine.get_question_for_moment(
            user_id="doug",
            conversation_context="test",
        )

        assert selected is None

    @pytest.mark.asyncio
    async def test_mark_question_asked(self, engine):
        """Test marking a question as asked."""
        question = CuriousQuestion(
            question_id="q1",
            question="Test question?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user="doug",
            context="test",
        )
        engine._question_queue = [question]

        await engine.mark_question_asked("q1")

        assert question.asked is True
        assert question.asked_at is not None

    @pytest.mark.asyncio
    async def test_process_answer_stores_learned_fact(self, engine, mock_llm, mock_memory):
        """Test that processing an answer stores learned facts."""
        # Set up a question
        question = CuriousQuestion(
            question_id="q1",
            question="What time do you usually wake up?",
            question_type=QuestionType.PREFERENCE,
            priority=QuestionPriority.MEDIUM,
            target_user="doug",
            context="Alarm discussion",
            why_asking="To set better alarms",
            follow_up_plan="Remember for future alarm suggestions",
        )
        engine._question_queue = [question]

        # Set up LLM response
        json_response = json.dumps({
            "answer_extracted": "Doug usually wakes up at 6:30 AM",
            "entities": ["wake_time", "6:30 AM"],
            "can_execute_follow_up": True,
            "follow_up_action": "Set alarms for 6:15 AM by default",
            "something_to_share": None,
            "implies_follow_up": False,
            "follow_up_topic": None,
            "user_receptivity": "positive",
            "should_remember": True,
            "what_to_remember": "Doug prefers waking at 6:30 AM",
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        result = await engine.process_answer(
            question_id="q1",
            response="I usually wake up around 6:30",
            user_id="doug",
        )

        assert result.get("should_remember") is True
        assert mock_memory.store.called


class TestQuestionFiltering:
    """Tests for question filtering based on user targeting."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.store = AsyncMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def engine(self, mock_llm, mock_memory):
        return CuriosityEngine(
            llm=mock_llm,
            memory=mock_memory,
            trait_provider=MockTraitProvider(),
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_filters_by_target_user(self, engine):
        """Test that questions are filtered by target user."""
        doug_q = CuriousQuestion(
            question_id="doug1",
            question="Question for Doug?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user="doug",
            context="test",
        )
        lisa_q = CuriousQuestion(
            question_id="lisa1",
            question="Question for Lisa?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user="lisa",
            context="test",
        )

        engine._question_queue = [doug_q, lisa_q]

        # When asking for Doug, should only get Doug's question
        selected = await engine.get_question_for_moment(
            user_id="doug",
            conversation_context="test",
        )

        assert selected is not None
        assert selected.question_id == "doug1"

    @pytest.mark.asyncio
    async def test_null_target_user_matches_anyone(self, engine):
        """Test that questions with null target_user match anyone."""
        general_q = CuriousQuestion(
            question_id="general1",
            question="Question for anyone?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user=None,
            context="test",
        )

        engine._question_queue = [general_q]

        selected = await engine.get_question_for_moment(
            user_id="doug",
            conversation_context="test",
        )

        assert selected is not None
        assert selected.question_id == "general1"


class TestCuriousQuestion:
    """Tests for CuriousQuestion dataclass."""

    def test_is_expired_returns_true_for_past_date(self):
        """Test is_expired returns True for expired questions."""
        from datetime import timedelta

        question = CuriousQuestion(
            question_id="q1",
            question="Test?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user="doug",
            context="test",
            expires_at=datetime.now() - timedelta(days=1),
        )

        assert question.is_expired() is True

    def test_is_expired_returns_false_for_future_date(self):
        """Test is_expired returns False for non-expired questions."""
        from datetime import timedelta

        question = CuriousQuestion(
            question_id="q1",
            question="Test?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user="doug",
            context="test",
            expires_at=datetime.now() + timedelta(days=1),
        )

        assert question.is_expired() is False

    def test_get_full_context_combines_fields(self):
        """Test get_full_context combines context fields."""
        question = CuriousQuestion(
            question_id="q1",
            question="Test?",
            question_type=QuestionType.FOLLOW_UP,
            priority=QuestionPriority.MEDIUM,
            target_user="doug",
            context="Original context",
            why_asking="Because I want to know",
            follow_up_plan="I will remember this",
        )

        full_context = question.get_full_context()

        assert "Original context" in full_context
        assert "Because I want to know" in full_context
        assert "I will remember this" in full_context

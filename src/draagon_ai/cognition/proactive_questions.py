"""Proactive Question Timing Service for Draagon AI.

Determines when it's appropriate to ask queued curiosity questions.
Balances being curious with not being annoying.

Key responsibilities:
- Detect natural conversation pauses
- Assess user availability/mood
- Select appropriate questions for the moment
- Respect user preferences for questions
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from draagon_ai.cognition.curiosity import CuriosityEngine, CuriousQuestion
from draagon_ai.core import UserInteractionPreferences
from draagon_ai.llm import LLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class ConversationMoment(str, Enum):
    """Types of conversation moments."""
    TASK_COMPLETE = "task_complete"  # Just finished helping
    NATURAL_PAUSE = "natural_pause"  # Conversation lull
    GREETING = "greeting"  # User just said hello
    FAREWELL = "farewell"  # User is leaving
    FOLLOWUP = "followup"  # Natural follow-up opportunity
    TOPIC_CHANGE = "topic_change"  # Conversation shifting
    USER_CURIOUS = "user_curious"  # User is asking questions
    BUSY = "busy"  # User is clearly busy
    FRUSTRATED = "frustrated"  # User is frustrated


class UserSentiment(str, Enum):
    """User sentiment states."""
    HAPPY = "happy"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    RUSHED = "rushed"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: UserSentiment
    confidence: float
    indicators: list[str]
    response_guidance: str


@dataclass
class QuestionOpportunity:
    """An opportunity to ask a question."""
    moment: ConversationMoment
    question: CuriousQuestion
    timing_score: float  # 0-1, how good is this moment
    transition_phrase: str  # How to lead into the question
    ask_now: bool  # Whether to ask now


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class UserPreferencesProvider(Protocol):
    """Protocol for getting user interaction preferences.

    Host applications implement this to provide per-user preferences.
    """

    async def get_user_prefs(self, user_id: str) -> UserInteractionPreferences:
        """Get interaction preferences for a user."""
        ...


@runtime_checkable
class TraitProvider(Protocol):
    """Protocol for getting agent trait values.

    Host applications implement this to provide personality traits.
    """

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get a trait value by name."""
        ...


# =============================================================================
# Prompts
# =============================================================================

TIMING_ASSESSMENT_PROMPT = """Assess whether this is a good moment to ask a curious question.

CONVERSATION CONTEXT:
{context}

USER'S LAST MESSAGE: {last_message}
{agent_name}'S LAST RESPONSE: {last_response}
TASK JUST COMPLETED: {task_complete}

QUESTION {agent_name} WANTS TO ASK: {question}
QUESTION PRIORITY: {priority}

USER PREFERENCES:
- Question tolerance: {question_tolerance} (0=hates questions, 1=loves them)
- Current mood: {mood}

{agent_name}'S CURIOSITY LEVEL: {curiosity}

Consider:
1. Did we just complete a task? (good moment)
2. Is the user in a hurry? (bad moment)
3. Is the user frustrated? (bad moment)
4. Is the question relevant to recent discussion? (better)
5. Is this a greeting/casual moment? (can be good)

Output JSON:
{{
    "ask_now": true/false,
    "timing_score": 0.0-1.0,
    "moment_type": "task_complete" | "natural_pause" | "greeting" | "farewell" | "followup" | "topic_change" | "user_curious" | "busy" | "frustrated",
    "transition_phrase": "SHORT 3-6 word lead-in only (e.g. 'By the way,' or 'I was curious,' - DO NOT include the question itself)",
    "reason": "why this is/isn't a good moment",
    "defer_reason": "if not asking now, why" or null
}}
"""

QUESTION_SELECTION_PROMPT = """Select the best question to ask at this moment.

AVAILABLE QUESTIONS:
{questions}

CONVERSATION CONTEXT:
{context}

USER: {user_id}

Select the question that:
1. Is most relevant to recent discussion
2. Would feel natural to ask now
3. Respects the user's time and mood

Output JSON:
{{
    "selected_question_id": "question_id",
    "reason": "why this one",
    "skip_all": true/false,
    "skip_reason": "if skipping all, why"
}}
"""


# =============================================================================
# Service
# =============================================================================


class ProactiveQuestionTimingService:
    """Determines when to ask curiosity questions.

    Works with CuriosityEngine to time questions appropriately.
    """

    def __init__(
        self,
        llm: LLMProvider,
        curiosity_engine: CuriosityEngine,
        user_prefs_provider: UserPreferencesProvider,
        trait_provider: TraitProvider,
        agent_name: str = "the agent",
        min_time_between_questions_minutes: int = 30,
        max_questions_per_day: int = 3,
    ):
        """Initialize the proactive question timing service.

        Args:
            llm: LLM provider for timing assessment
            curiosity_engine: Engine that manages curious questions
            user_prefs_provider: Provider for user interaction preferences
            trait_provider: Provider for agent personality traits
            agent_name: Display name for this agent
            min_time_between_questions_minutes: Minimum gap between questions
            max_questions_per_day: Maximum questions to ask per user per day
        """
        self.llm = llm
        self.curiosity_engine = curiosity_engine
        self.user_prefs_provider = user_prefs_provider
        self.trait_provider = trait_provider
        self.agent_name = agent_name

        # Rate limiting settings
        self._min_time_between_questions_minutes = min_time_between_questions_minutes
        self._max_questions_per_day = max_questions_per_day

        # Track recent asks to avoid being annoying
        self._last_question_asked: dict[str, datetime] = {}  # user_id -> time
        self._questions_asked_today: dict[str, int] = {}  # user_id:date -> count

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def check_for_question_opportunity(
        self,
        user_id: str,
        context: str,
        last_message: str,
        last_response: str,
        task_complete: bool = False,
        sentiment: SentimentResult | None = None,
    ) -> QuestionOpportunity | None:
        """Check if now is a good time to ask a question.

        Called after completing a task or at natural pauses.

        Args:
            user_id: Current user
            context: Conversation context
            last_message: User's last message
            last_response: Agent's last response
            task_complete: Whether we just finished a task
            sentiment: Current user sentiment

        Returns:
            QuestionOpportunity if should ask, None otherwise
        """
        # Quick checks first
        if not self._should_even_try(user_id):
            return None

        # Get pending questions
        questions = self.curiosity_engine.get_pending_questions()

        # Filter to relevant for this user
        user_questions = [
            q for q in questions
            if q.target_user is None or q.target_user == user_id
        ]

        if not user_questions:
            return None

        # Get user preferences
        user_prefs = await self.user_prefs_provider.get_user_prefs(user_id)
        question_tolerance = user_prefs.question_tolerance

        # If user hates questions, only ask high priority
        if question_tolerance < 0.3:
            user_questions = [q for q in user_questions if q.priority.value == "high"]
            if not user_questions:
                return None

        # Get curiosity level
        curiosity = self.trait_provider.get_trait_value(
            "curiosity_intensity",
            default=0.7,
        )

        # Determine mood from sentiment enum
        mood = "neutral"
        if sentiment:
            if sentiment.sentiment == UserSentiment.FRUSTRATED:
                mood = "frustrated"
                return None  # Never ask when frustrated
            elif sentiment.sentiment == UserSentiment.HAPPY:
                mood = "satisfied"
            elif sentiment.sentiment in (UserSentiment.CONFUSED, UserSentiment.RUSHED):
                mood = "slightly_negative"

        # Pick best question and assess timing
        opportunity = await self._assess_timing(
            questions=user_questions,
            user_id=user_id,
            context=context,
            last_message=last_message,
            last_response=last_response,
            task_complete=task_complete,
            question_tolerance=question_tolerance,
            mood=mood,
            curiosity=curiosity,
        )

        return opportunity

    # =========================================================================
    # Timing Assessment
    # =========================================================================

    async def _assess_timing(
        self,
        questions: list[CuriousQuestion],
        user_id: str,
        context: str,
        last_message: str,
        last_response: str,
        task_complete: bool,
        question_tolerance: float,
        mood: str,
        curiosity: float,
    ) -> QuestionOpportunity | None:
        """Assess timing for asking a question."""
        # Select best question first
        selected = await self._select_best_question(
            questions=questions,
            context=context,
            user_id=user_id,
        )

        if not selected:
            return None

        # Now assess timing for this question
        prompt = TIMING_ASSESSMENT_PROMPT.format(
            context=context[:500],
            last_message=last_message,
            last_response=last_response[:200],
            task_complete=task_complete,
            question=selected.question,
            priority=selected.priority.value,
            question_tolerance=question_tolerance,
            mood=mood,
            curiosity=curiosity,
            agent_name=self.agent_name,
        )

        result = await self.llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Should I ask now?"},
            ],
            max_tokens=300,
        )

        if not result or not result.get("parsed"):
            return None

        parsed = result["parsed"]

        if not parsed.get("ask_now"):
            logger.debug(f"Question deferred: {parsed.get('defer_reason')}")
            return None

        # Map moment type
        moment_map = {
            "task_complete": ConversationMoment.TASK_COMPLETE,
            "natural_pause": ConversationMoment.NATURAL_PAUSE,
            "greeting": ConversationMoment.GREETING,
            "farewell": ConversationMoment.FAREWELL,
            "followup": ConversationMoment.FOLLOWUP,
            "topic_change": ConversationMoment.TOPIC_CHANGE,
            "user_curious": ConversationMoment.USER_CURIOUS,
            "busy": ConversationMoment.BUSY,
            "frustrated": ConversationMoment.FRUSTRATED,
        }

        opportunity = QuestionOpportunity(
            moment=moment_map.get(parsed.get("moment_type", "natural_pause"), ConversationMoment.NATURAL_PAUSE),
            question=selected,
            timing_score=parsed.get("timing_score", 0.5),
            transition_phrase=parsed.get("transition_phrase", "By the way,"),
            ask_now=True,
        )

        # Update tracking
        self._record_question_asked(user_id)

        return opportunity

    async def _select_best_question(
        self,
        questions: list[CuriousQuestion],
        context: str,
        user_id: str,
    ) -> CuriousQuestion | None:
        """Select the best question to ask now."""
        if len(questions) == 1:
            return questions[0]

        # Format questions for selection
        questions_str = "\n".join([
            f"- ID: {q.question_id[:8]}, Priority: {q.priority.value}, "
            f"Question: {q.question}, Context: {q.context[:50]}"
            for q in questions[:5]
        ])

        prompt = QUESTION_SELECTION_PROMPT.format(
            questions=questions_str,
            context=context[:300],
            user_id=user_id,
        )

        result = await self.llm.chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Select the best question."},
            ],
            max_tokens=200,
        )

        if not result or not result.get("parsed"):
            # Default to first high priority
            for q in questions:
                if q.priority.value == "high":
                    return q
            return questions[0]

        parsed = result["parsed"]

        if parsed.get("skip_all"):
            return None

        # Find selected question
        selected_id = parsed.get("selected_question_id", "")
        for q in questions:
            if q.question_id.startswith(selected_id) or selected_id.startswith(q.question_id[:8]):
                return q

        # Fallback
        return questions[0]

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def _should_even_try(self, user_id: str) -> bool:
        """Quick check if we should even consider asking."""
        # Check time since last question
        if user_id in self._last_question_asked:
            last = self._last_question_asked[user_id]
            min_gap = timedelta(minutes=self._min_time_between_questions_minutes)
            if datetime.now() - last < min_gap:
                return False

        # Check daily limit
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{user_id}:{today}"
        if self._questions_asked_today.get(key, 0) >= self._max_questions_per_day:
            return False

        return True

    def _record_question_asked(self, user_id: str) -> None:
        """Record that we asked a question."""
        self._last_question_asked[user_id] = datetime.now()

        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{user_id}:{today}"
        self._questions_asked_today[key] = self._questions_asked_today.get(key, 0) + 1

        # Clean up old entries
        old_keys = [
            k for k in self._questions_asked_today.keys()
            if not k.endswith(today)
        ]
        for k in old_keys:
            del self._questions_asked_today[k]

    # =========================================================================
    # Integration Points
    # =========================================================================

    async def on_task_complete(
        self,
        user_id: str,
        context: str,
        response: str,
        sentiment: SentimentResult | None = None,
    ) -> QuestionOpportunity | None:
        """Called when a task is complete - prime moment for questions.

        Args:
            user_id: User
            context: What was the task
            response: Agent's completion response
            sentiment: User sentiment

        Returns:
            Question opportunity if appropriate
        """
        return await self.check_for_question_opportunity(
            user_id=user_id,
            context=context,
            last_message="",  # Will be extracted from context
            last_response=response,
            task_complete=True,
            sentiment=sentiment,
        )

    async def on_conversation_pause(
        self,
        user_id: str,
        context: str,
        last_exchange: tuple[str, str],  # (user_msg, agent_msg)
        sentiment: SentimentResult | None = None,
    ) -> QuestionOpportunity | None:
        """Called when there's a natural pause.

        Args:
            user_id: User
            context: Conversation context
            last_exchange: Last (user, agent) message pair
            sentiment: User sentiment

        Returns:
            Question opportunity if appropriate
        """
        return await self.check_for_question_opportunity(
            user_id=user_id,
            context=context,
            last_message=last_exchange[0],
            last_response=last_exchange[1],
            task_complete=False,
            sentiment=sentiment,
        )

    def format_question_with_transition(
        self,
        opportunity: QuestionOpportunity,
    ) -> str:
        """Format the question with its transition phrase.

        Args:
            opportunity: The question opportunity

        Returns:
            Formatted question string
        """
        transition = opportunity.transition_phrase.strip()
        question = opportunity.question.question.strip()

        # Ensure transition ends with appropriate punctuation
        if not transition.endswith((",", ":", "?")):
            transition += ","

        return f"{transition} {question}"

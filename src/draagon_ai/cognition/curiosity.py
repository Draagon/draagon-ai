"""Curiosity Engine for Draagon AI.

Identifies knowledge gaps and queues questions for natural conversation moments.
Driven by the agent's curiosity_intensity trait.

Key responsibilities:
- Detect knowledge gaps during conversations
- Generate thoughtful questions
- Queue questions for appropriate moments
- Track what the agent is curious about
- Prioritize questions based on relevance and interest
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from draagon_ai.core import AgentIdentity
from draagon_ai.llm import LLMProvider, ModelTier
from draagon_ai.memory import MemoryProvider, MemoryType, MemoryScope

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class QuestionType(str, Enum):
    """Types of curious questions."""
    CLARIFICATION = "clarification"  # Need more info about something mentioned
    FOLLOW_UP = "follow_up"  # Curious about related topic
    KNOWLEDGE_GAP = "knowledge_gap"  # Agent realizes they don't know something
    PREFERENCE = "preference"  # Curious about user's preferences
    CONTEXT = "context"  # Need context to understand something better
    VERIFICATION = "verification"  # Want to confirm understanding


class QuestionPriority(str, Enum):
    """Priority levels for questions."""
    LOW = "low"  # Nice to know, ask if there's a natural pause
    MEDIUM = "medium"  # Would help improve interactions
    HIGH = "high"  # Need to know to provide good service


class QuestionPurpose(str, Enum):
    """Purpose of asking a question."""
    LEARN_ABOUT_USER = "learn_about_user"  # To serve them better
    SHARE_KNOWLEDGE = "share_knowledge"  # Lead to teaching something
    DEEPEN_UNDERSTANDING = "deepen_understanding"  # Clarify ambiguity
    GENUINE_CURIOSITY = "genuine_curiosity"  # Agent authentically wants to know


@dataclass
class CuriousQuestion:
    """A question the agent is curious about."""
    question_id: str
    question: str
    question_type: QuestionType
    priority: QuestionPriority
    target_user: str | None  # Who to ask (None = anyone)
    context: str  # What prompted the curiosity

    # Rich context for meaningful follow-up
    purpose: QuestionPurpose = QuestionPurpose.GENUINE_CURIOSITY
    why_asking: str = ""  # Why agent wants to know (from their perspective)
    follow_up_plan: str = ""  # What agent will do with the answer
    interesting_to_user: bool = True  # Whether user would find this valuable

    created_at: datetime | None = None
    expires_at: datetime | None = None  # Some questions become stale
    asked: bool = False
    asked_at: datetime | None = None
    answered: bool = False
    answer: str | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            # Default expiration: 7 days
            self.expires_at = datetime.now() + timedelta(days=7)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def get_full_context(self) -> str:
        """Get full context for when agent asks the question."""
        parts = [f"Context: {self.context}"]
        if self.why_asking:
            parts.append(f"Why I'm asking: {self.why_asking}")
        if self.follow_up_plan:
            parts.append(f"What I'll do with the answer: {self.follow_up_plan}")
        return " | ".join(parts)


@dataclass
class KnowledgeGap:
    """A gap in the agent's knowledge."""
    topic: str
    description: str
    importance: float  # 0-1, how important is this to know
    discovered_at: datetime | None = None
    related_entities: list[str] = field(default_factory=list)
    filled: bool = False

    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.now()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class TraitProvider(Protocol):
    """Protocol for accessing agent traits.

    Host applications implement this to provide trait values.
    """

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get a trait value with default fallback."""
        ...


# =============================================================================
# Prompts
# =============================================================================

CURIOSITY_DETECTION_PROMPT = """You are {agent_name}, analyzing a conversation to identify genuinely interesting questions to ask.

CONVERSATION:
{conversation}

WHAT {agent_name} ALREADY KNOWS about this topic:
{current_knowledge}

{agent_name}'S WORLDVIEW AND INTERESTS:
{worldview}

{agent_name}'S CURIOSITY INTENSITY: {curiosity_level} (0=not curious, 1=very curious)

QUESTION PURPOSES (choose wisely - only ask questions with clear purpose):

1. **LEARN ABOUT USER** - To serve them better in the future
   - What they care about, how they think, their challenges
   - Example: After "I'm planning a trip to Japan" → "What draws you to Japan?" (learns their interests)
   - Example: After frustrated debugging → "What's your usual approach when you hit a wall like this?" (learns their process)

2. **SHARE KNOWLEDGE** - Lead to something valuable you can teach
   - Ask something where the answer lets you share interesting info
   - Example: "Have you tried X?" when you know X is helpful
   - Example: "What do you know about Y?" when you can expand on it

3. **DEEPEN UNDERSTANDING** - Clarify something ambiguous
   - Only when the ambiguity actually matters for helping them
   - NOT generic "tell me more" questions

4. **GENUINE CURIOSITY** - Something {agent_name} would authentically want to know
   - Based on their values (truth-seeking, helping, learning)
   - Example: If user mentions an ethical dilemma → {agent_name} genuinely wants to understand their reasoning

**ANTI-PATTERNS TO AVOID:**
- "What's your favorite X?" (boring, doesn't lead anywhere)
- "Can you tell me more about X?" (lazy, user already explained)
- Data collection questions ("What time do you usually wake up?")
- Questions {agent_name} should already know from context or memory
- Questions with no follow-up value

**QUALITY OVER QUANTITY:**
- Generate 0-2 questions MAX (often zero is correct)
- Each question must have clear PURPOSE and FOLLOW_UP_PLAN
- If nothing genuinely interesting emerged, output empty list

Output JSON:
{{
    "analysis": "Brief analysis of what's worth exploring in this conversation (1-2 sentences)",
    "knowledge_gaps": [
        {{
            "topic": "specific gap in {agent_name}'s knowledge",
            "importance": 0.0-1.0,
            "why_important": "how filling this gap helps {agent_name} serve better"
        }}
    ],
    "potential_questions": [
        {{
            "question": "The specific question to ask",
            "purpose": "learn_about_user" | "share_knowledge" | "deepen_understanding" | "genuine_curiosity",
            "type": "clarification" | "follow_up" | "knowledge_gap" | "preference" | "context" | "verification",
            "priority": "low" | "medium" | "high",
            "target_user": "user_id" or null,
            "why_this_question": "Why {agent_name} genuinely wants to know this (from their perspective)",
            "what_agent_will_do_with_answer": "Specific follow-up: teach something, remember for later, adjust behavior, etc.",
            "context_to_remember": "Full context so {agent_name} remembers why they asked when user answers",
            "interesting_to_user": true/false
        }}
    ],
    "should_explore": true/false,
    "exploration_topic": "topic for {agent_name} to research on their own" or null
}}
"""

QUESTION_FORMULATION_PROMPT = """Formulate a natural, conversational question for {agent_name} to ask.

TOPIC: {topic}
CONTEXT: {context}
QUESTION TYPE: {question_type}
USER: {user}

Create a question that:
1. Sounds natural and conversational, not interrogative
2. Shows genuine curiosity, not just data collection
3. Is appropriate for the relationship/context
4. Is voice-friendly (can be spoken naturally)

Output JSON:
{{
    "question": "The natural question",
    "alternative": "An alternative phrasing",
    "too_intrusive": true/false,
    "suggested_timing": "when to ask (now, later, never)"
}}
"""

ANSWER_PROCESSING_PROMPT = """Process the user's answer to the agent's curious question and determine next steps.

QUESTION ASKED: {question}
WHY AGENT ASKED: {why_asking}
AGENT'S PLANNED FOLLOW-UP: {follow_up_plan}
USER'S RESPONSE: {response}

Based on the user's answer, determine:
1. What specific information did they provide?
2. Can the agent now execute their planned follow-up? (teach something, remember this, adjust behavior)
3. Did the user seem interested in the topic?
4. Is there a natural follow-up question?

Output JSON:
{{
    "answer_extracted": "The key information from their response",
    "entities": ["key", "entities", "to", "remember"],
    "can_execute_follow_up": true/false,
    "follow_up_action": "What the agent should now do based on the answer (be specific)",
    "something_to_share": "If agent can teach/share something interesting based on this answer, what?" or null,
    "implies_follow_up": true/false,
    "follow_up_topic": "Natural follow-up topic" or null,
    "user_receptivity": "positive" | "neutral" | "negative",
    "should_remember": true/false,
    "what_to_remember": "Specific fact to store for future" or null
}}
"""


# =============================================================================
# Service
# =============================================================================


class CuriosityEngine:
    """Manages agent curiosity and question-asking behavior.

    Driven by the curiosity_intensity trait. This service is backend-agnostic,
    using LLMProvider and MemoryProvider interfaces.
    """

    # Record types for storage
    QUESTION_RECORD_TYPE = "curious_question"
    KNOWLEDGE_GAP_RECORD_TYPE = "knowledge_gap"

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        trait_provider: TraitProvider,
        agent_name: str = "the agent",
        agent_id: str = "agent",
        ask_cooldown_hours: int = 24,
    ):
        """Initialize the curiosity engine.

        Args:
            llm: LLM provider for reasoning
            memory: Memory provider for storage
            trait_provider: Provider for agent trait values
            agent_name: Name of the agent (for prompts)
            agent_id: Agent ID for namespacing
            ask_cooldown_hours: Hours before asking similar questions
        """
        self.llm = llm
        self.memory = memory
        self.trait_provider = trait_provider
        self.agent_name = agent_name
        self.agent_id = agent_id

        # In-memory question queue (also persisted to storage)
        self._question_queue: list[CuriousQuestion] = []

        # Track what we've asked recently to avoid repetition
        self._recently_asked: dict[str, datetime] = {}
        self._ask_cooldown_hours = ask_cooldown_hours

    # =========================================================================
    # Curiosity Detection
    # =========================================================================

    async def analyze_for_curiosity(
        self,
        conversation: str,
        user_id: str,
        topic_hint: str | None = None,
        current_knowledge: str | None = None,
        worldview_str: str | None = None,
    ) -> list[CuriousQuestion]:
        """Analyze a conversation for things to be curious about.

        Called after interactions to identify questions.

        Args:
            conversation: The conversation text
            user_id: The user involved
            topic_hint: Optional topic hint
            current_knowledge: What agent already knows (if pre-fetched)
            worldview_str: Agent's worldview (if pre-fetched)

        Returns:
            List of questions generated
        """
        # Get curiosity level
        curiosity = self.trait_provider.get_trait_value(
            "curiosity_intensity",
            default=0.7,
        )

        # If low curiosity, only look for high-priority gaps
        if curiosity < 0.3:
            logger.debug("Low curiosity - skipping detailed analysis")
            return []

        # Use provided knowledge or default
        if current_knowledge is None:
            current_knowledge = "No prior knowledge about this topic or user"

        if worldview_str is None:
            worldview_str = "Values: truth-seeking, helpfulness, genuine curiosity"

        # Analyze with LLM
        prompt = CURIOSITY_DETECTION_PROMPT.format(
            agent_name=self.agent_name,
            conversation=conversation[:2000],
            current_knowledge=current_knowledge,
            worldview=worldview_str,
            curiosity_level=curiosity,
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze for curiosity."},
            ],
            max_tokens=500,
            tier=ModelTier.LOCAL,
        )

        parsed = self._parse_json_response(response.content)

        if not parsed or not isinstance(parsed, dict):
            return []

        # Process knowledge gaps
        for gap in parsed.get("knowledge_gaps", []):
            await self._store_knowledge_gap(KnowledgeGap(
                topic=gap.get("topic", ""),
                description=gap.get("why_important", ""),
                importance=gap.get("importance", 0.5),
                related_entities=[],
            ))

        # Process questions
        new_questions = []
        for q in parsed.get("potential_questions", []):
            # Skip if we asked something similar recently
            if self._asked_recently(q.get("question", "")):
                continue

            # Skip questions marked as not interesting to user
            if not q.get("interesting_to_user", True):
                logger.debug(f"Skipping question not interesting to user: {q.get('question', '')[:50]}")
                continue

            # Map types
            type_map = {
                "clarification": QuestionType.CLARIFICATION,
                "follow_up": QuestionType.FOLLOW_UP,
                "knowledge_gap": QuestionType.KNOWLEDGE_GAP,
                "preference": QuestionType.PREFERENCE,
                "context": QuestionType.CONTEXT,
                "verification": QuestionType.VERIFICATION,
            }
            priority_map = {
                "low": QuestionPriority.LOW,
                "medium": QuestionPriority.MEDIUM,
                "high": QuestionPriority.HIGH,
            }
            purpose_map = {
                "learn_about_user": QuestionPurpose.LEARN_ABOUT_USER,
                "share_knowledge": QuestionPurpose.SHARE_KNOWLEDGE,
                "deepen_understanding": QuestionPurpose.DEEPEN_UNDERSTANDING,
                "genuine_curiosity": QuestionPurpose.GENUINE_CURIOSITY,
            }

            question = CuriousQuestion(
                question_id=str(uuid.uuid4()),
                question=q.get("question", ""),
                question_type=type_map.get(q.get("type", "follow_up"), QuestionType.FOLLOW_UP),
                priority=priority_map.get(q.get("priority", "medium"), QuestionPriority.MEDIUM),
                target_user=q.get("target_user") or user_id,
                context=q.get("context_to_remember", conversation[:200]),
                # Rich context fields
                purpose=purpose_map.get(q.get("purpose", "genuine_curiosity"), QuestionPurpose.GENUINE_CURIOSITY),
                why_asking=q.get("why_this_question", ""),
                follow_up_plan=q.get("what_agent_will_do_with_answer", ""),
                interesting_to_user=q.get("interesting_to_user", True),
            )

            new_questions.append(question)
            self._question_queue.append(question)

            # Persist to memory with rich context
            await self._store_question(question)

            logger.info(
                f"Generated question: '{question.question[:50]}...' "
                f"purpose={question.purpose.value}, "
                f"follow_up_plan={question.follow_up_plan[:50]}..."
            )

        # If should explore, trigger research
        if parsed.get("should_explore") and parsed.get("exploration_topic"):
            await self._queue_for_research(parsed["exploration_topic"])

        logger.info(f"Curiosity analysis found {len(new_questions)} questions")
        return new_questions

    # =========================================================================
    # Question Selection
    # =========================================================================

    async def get_question_for_moment(
        self,
        user_id: str,
        conversation_context: str,
    ) -> CuriousQuestion | None:
        """Get a question appropriate for the current moment.

        Called when there's a natural pause or conversation end.

        Args:
            user_id: Current user
            conversation_context: Recent context

        Returns:
            A question to ask, or None if not appropriate
        """
        # Get curiosity level
        curiosity = self.trait_provider.get_trait_value(
            "curiosity_intensity",
            default=0.7,
        )

        # Low curiosity = rarely ask questions unprompted
        if curiosity < 0.4:
            # Only ask high priority questions
            candidates = [
                q for q in self._question_queue
                if q.priority == QuestionPriority.HIGH
                and not q.asked
                and not q.is_expired()
                and (q.target_user is None or q.target_user == user_id)
            ]
        else:
            # Consider all questions
            candidates = [
                q for q in self._question_queue
                if not q.asked
                and not q.is_expired()
                and (q.target_user is None or q.target_user == user_id)
            ]

        if not candidates:
            return None

        # Sort by priority and recency
        candidates.sort(
            key=lambda q: (
                {"high": 0, "medium": 1, "low": 2}[q.priority.value],
                q.created_at,
            )
        )

        # Return best candidate
        return candidates[0]

    async def mark_question_asked(
        self,
        question_id: str,
    ) -> None:
        """Mark a question as asked."""
        for q in self._question_queue:
            if q.question_id == question_id:
                q.asked = True
                q.asked_at = datetime.now()
                self._recently_asked[q.question[:50]] = datetime.now()
                break

    async def process_answer(
        self,
        question_id: str,
        response: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process user's answer to a question and execute follow-up plan.

        Args:
            question_id: The question that was answered
            response: User's response
            user_id: User ID for storage

        Returns:
            Extracted information including what agent should do next
        """
        # Find the question
        question = None
        for q in self._question_queue:
            if q.question_id == question_id:
                question = q
                break

        if not question:
            return {"error": "Question not found"}

        # Analyze response WITH the original context and follow-up plan
        prompt = ANSWER_PROCESSING_PROMPT.format(
            question=question.question,
            why_asking=question.why_asking or "General curiosity",
            follow_up_plan=question.follow_up_plan or "Remember for future reference",
            response=response,
        )

        llm_response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Process the answer and determine what I should do next."},
            ],
            max_tokens=400,
            tier=ModelTier.LOCAL,
        )

        parsed = self._parse_json_response(llm_response.content)

        if not parsed:
            return {"error": "Processing failed"}

        # Update question
        question.answered = True
        question.answer = parsed.get("answer_extracted", response)

        # Store what we learned (use more specific content if available)
        if parsed.get("should_remember"):
            what_to_remember = parsed.get("what_to_remember") or question.answer
            target_user = user_id or question.target_user

            await self.memory.store(
                content=what_to_remember,
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER if target_user else MemoryScope.AGENT,
                agent_id=self.agent_id,
                user_id=target_user,
                importance=0.75,  # Higher - we specifically asked for this
                entities=parsed.get("entities", []),
                metadata={
                    "learned_from_question": question.question[:100],
                    "question_purpose": question.purpose.value,
                },
            )
            logger.info(f"Stored learned fact from question: {what_to_remember[:50]}...")

        # If there's something agent can share/teach, note it for the next response
        if parsed.get("something_to_share"):
            parsed["agent_should_share"] = parsed["something_to_share"]
            logger.info(f"Agent can share: {parsed['something_to_share'][:50]}...")

        # If follow-up implied, queue it
        if parsed.get("implies_follow_up") and parsed.get("follow_up_topic"):
            await self.analyze_for_curiosity(
                conversation=f"Q: {question.question}\nA: {response}",
                user_id=question.target_user or "system",
                topic_hint=parsed["follow_up_topic"],
            )

        # Log the follow-up action for transparency
        if parsed.get("follow_up_action"):
            logger.info(f"Question follow-up action: {parsed['follow_up_action']}")

        return parsed

    # =========================================================================
    # Storage
    # =========================================================================

    async def _store_question(self, question: CuriousQuestion) -> None:
        """Store a question with full context."""
        # Include rich context in content for better retrieval
        content = f"Question: {question.question}"
        if question.why_asking:
            content += f" (Why: {question.why_asking})"
        if question.follow_up_plan:
            content += f" (Plan: {question.follow_up_plan})"

        await self.memory.store(
            content=content,
            memory_type=MemoryType.INSIGHT,
            scope=MemoryScope.AGENT,
            agent_id=self.agent_id,
            importance=0.6,  # Slightly higher - these are purposeful questions
            entities=["curious_question", question.purpose.value],
            metadata={
                "record_type": self.QUESTION_RECORD_TYPE,
                "question_id": question.question_id,
                "question_type": question.question_type.value,
                "priority": question.priority.value,
                "target_user": question.target_user,
                "context": question.context,
                # Rich context fields
                "purpose": question.purpose.value,
                "why_asking": question.why_asking,
                "follow_up_plan": question.follow_up_plan,
                "interesting_to_user": question.interesting_to_user,
                # Timestamps
                "created_at": question.created_at.isoformat() if question.created_at else None,
                "expires_at": question.expires_at.isoformat() if question.expires_at else None,
            },
        )

    async def _store_knowledge_gap(self, gap: KnowledgeGap) -> None:
        """Store a knowledge gap."""
        await self.memory.store(
            content=f"Knowledge gap: {gap.topic} - {gap.description}",
            memory_type=MemoryType.INSIGHT,
            scope=MemoryScope.AGENT,
            agent_id=self.agent_id,
            importance=gap.importance,
            entities=["knowledge_gap"] + gap.related_entities,
            metadata={
                "record_type": self.KNOWLEDGE_GAP_RECORD_TYPE,
                "topic": gap.topic,
                "importance": gap.importance,
                "discovered_at": gap.discovered_at.isoformat() if gap.discovered_at else None,
            },
        )

    async def _queue_for_research(self, topic: str) -> None:
        """Queue a topic for autonomous research."""
        await self.memory.store(
            content=f"Research topic: {topic}",
            memory_type=MemoryType.INSIGHT,
            scope=MemoryScope.AGENT,
            agent_id=self.agent_id,
            importance=0.6,
            entities=["research_queue", "autonomous_task"],
            metadata={
                "record_type": "research_queue",
                "topic": topic,
                "queued_at": datetime.now().isoformat(),
            },
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _asked_recently(self, question: str) -> bool:
        """Check if we asked a similar question recently."""
        key = question[:50].lower()

        # Check exact match
        if key in self._recently_asked:
            asked_at = self._recently_asked[key]
            if datetime.now() - asked_at < timedelta(hours=self._ask_cooldown_hours):
                return True

        return False

    async def load_questions_from_storage(self) -> None:
        """Load queued questions from storage."""
        try:
            results = await self.memory.search(
                query="curious question",
                agent_id=self.agent_id,
                scopes=[MemoryScope.AGENT],
                limit=50,
            )

            for r in results:
                # Handle SearchResult objects
                if hasattr(r, 'memory'):
                    content = r.memory.content
                    # Try to get metadata from the search result
                    # For now, we'll need host to provide metadata access
                    continue  # Skip for now - host should implement proper loading

                # Handle dict format (legacy)
                metadata = r.get("metadata", {})
                if metadata.get("record_type") != self.QUESTION_RECORD_TYPE:
                    continue

                # Skip expired
                expires_at_str = metadata.get("expires_at")
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if datetime.now() > expires_at:
                        continue

                # Skip already in queue
                question_id = metadata.get("question_id")
                if any(q.question_id == question_id for q in self._question_queue):
                    continue

                # Extract just the question from content
                content = r.get("content", "")
                question_text = content.replace("Question: ", "").split(" (Why:")[0].split(" (Plan:")[0]

                # Map purpose
                purpose_str = metadata.get("purpose", "genuine_curiosity")
                try:
                    purpose = QuestionPurpose(purpose_str)
                except ValueError:
                    purpose = QuestionPurpose.GENUINE_CURIOSITY

                # Reconstruct question with new fields
                question = CuriousQuestion(
                    question_id=question_id,
                    question=question_text,
                    question_type=QuestionType(metadata.get("question_type", "follow_up")),
                    priority=QuestionPriority(metadata.get("priority", "medium")),
                    target_user=metadata.get("target_user"),
                    context=metadata.get("context", ""),
                    # Rich context fields
                    purpose=purpose,
                    why_asking=metadata.get("why_asking", ""),
                    follow_up_plan=metadata.get("follow_up_plan", ""),
                    interesting_to_user=metadata.get("interesting_to_user", True),
                    # Timestamps
                    created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else datetime.now(),
                    expires_at=datetime.fromisoformat(metadata["expires_at"]) if metadata.get("expires_at") else None,
                )

                self._question_queue.append(question)

            logger.info(f"Loaded {len(self._question_queue)} questions from storage")

        except Exception as e:
            logger.error(f"Error loading questions: {e}")

    # =========================================================================
    # Transparency
    # =========================================================================

    def get_pending_questions(self) -> list[CuriousQuestion]:
        """Get all pending questions (for dashboard)."""
        return [q for q in self._question_queue if not q.asked and not q.is_expired()]

    def get_knowledge_gaps_count(self) -> int:
        """Get count of known knowledge gaps."""
        # Would query storage in a real implementation
        return 0

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response content."""
        import json
        import re

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return None

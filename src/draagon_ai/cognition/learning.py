"""Autonomous learning service for Draagon AI agents.

Implements fully LLM-driven learning with:
- Semantic detection (no keyword patterns)
- Failure-triggered relearning (when skills don't work)
- Skill verification and confidence decay
- Knowledge correction, refinement, and deletion
- User credibility tracking and trust calibration
- Multi-user support with knowledge scoping

Based on:
- A-Mem: Agentic Memory for LLM Agents (2025)
- Memp: Procedural Memory Framework (2025)
- Cognitive Memory in LLMs (2025)
- CRITIC: Tool-Interactive Self-Correction (2023)

Learning is triggered:
1. After each successful interaction (extract new knowledge)
2. After tool failures (detect skill problems, research, fix)
3. When user provides corrections (verify, then update/delete memories)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from draagon_ai.llm import LLMProvider
from draagon_ai.memory import MemoryProvider, MemoryType

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class LearningType(str, Enum):
    """Types of learnable content."""
    SKILL = "skill"  # How to do something (procedural)
    FACT = "fact"  # New information learned
    INSIGHT = "insight"  # Meta-learning about tasks/patterns
    PREFERENCE = "preference"  # User preference discovered
    CORRECTION = "correction"  # Error correction or update
    REFINEMENT = "refinement"  # Adding detail to existing knowledge
    DELETION = "deletion"  # Something is no longer true


class MemoryAction(str, Enum):
    """Actions to take on existing memories."""
    CREATE = "create"  # Store new memory
    UPDATE = "update"  # Replace existing with new content
    REFINE = "refine"  # Add detail to existing memory
    SUPERSEDE = "supersede"  # Mark old as superseded, store new
    DELETE = "delete"  # Remove existing memory
    SKIP = "skip"  # No action needed


class FailureType(str, Enum):
    """Types of skill/tool failures."""
    EXECUTION_ERROR = "execution_error"  # Command/tool threw an error
    UNEXPECTED_RESULT = "unexpected_result"  # Result doesn't match expectations
    OUTDATED_INFO = "outdated_info"  # Information in memory is stale
    WRONG_APPROACH = "wrong_approach"  # The entire approach was incorrect


class VerificationResult(str, Enum):
    """Results of verifying a user correction."""
    VERIFIED = "verified"  # User is correct, accept the correction
    CONTRADICTED = "contradicted"  # Evidence shows user is wrong
    UNCERTAIN = "uncertain"  # Can't verify either way
    NOT_VERIFIABLE = "not_verifiable"  # This type of claim can't be verified
    PARTIALLY_CORRECT = "partially_correct"  # User is right about some parts


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SkillConfidence:
    """Track skill confidence with decay on failures."""
    skill_id: str
    content: str
    confidence: float = 1.0
    successes: int = 0
    failures: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    last_verified: datetime | None = None

    def record_success(self) -> None:
        """Record successful use of skill."""
        self.successes += 1
        self.last_used = datetime.now()
        self.confidence = min(1.0, self.confidence + 0.1)

    def record_failure(self) -> None:
        """Record failed use of skill."""
        self.failures += 1
        self.last_used = datetime.now()
        self.confidence = max(0.0, self.confidence - 0.25)

    def needs_relearning(self) -> bool:
        """Check if skill has failed enough to need relearning."""
        return self.confidence < 0.3 or self.failures >= 3


@dataclass
class LearningCandidate:
    """A potential learning to be stored."""
    content: str
    learning_type: LearningType
    confidence: float
    source: str
    entities: list[str]
    procedure: str | None = None
    success_indicators: list[str] | None = None


@dataclass
class LearningResult:
    """Result of a learning operation."""
    learned: bool
    learning_type: str | None = None
    action: str | None = None
    title: str | None = None
    memory_id: str | None = None
    confidence: float | None = None
    reason: str | None = None
    verification_result: VerificationResult | None = None
    correct_information: str | None = None


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for web search functionality.

    Used for relearning when skills fail and need to be corrected
    via web search.
    """

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search the web and return results.

        Args:
            query: Search query
            limit: Max results to return

        Returns:
            List of search results with title, snippet/content, url
        """
        ...


@runtime_checkable
class CredibilityProvider(Protocol):
    """Protocol for user credibility tracking.

    Host applications implement this to track user correction accuracy.
    """

    def should_verify_correction(
        self,
        user_id: str,
        domain: str | None = None,
    ) -> tuple[bool, float]:
        """Determine if a correction should be verified.

        Args:
            user_id: User making the correction
            domain: Domain of the correction (optional)

        Returns:
            Tuple of (should_verify, threshold)
        """
        ...

    def record_correction_result(
        self,
        user_id: str,
        result: str,
        domain: str | None = None,
        user_was_confident: bool = False,
    ) -> dict[str, Any]:
        """Record the result of a correction verification.

        Args:
            user_id: User who made the correction
            result: Verification result string
            domain: Domain of the correction
            user_was_confident: Whether user seemed confident

        Returns:
            Updated credibility info
        """
        ...

    def get_user_credibility(self, user_id: str) -> Any | None:
        """Get credibility info for a user."""
        ...


@runtime_checkable
class UserProvider(Protocol):
    """Protocol for user information.

    Host applications implement this to provide user details.
    """

    async def get_user(self, user_id: str) -> Any | None:
        """Get user by ID."""
        ...

    async def get_display_name(self, user_id: str) -> str:
        """Get user's display name."""
        ...


# =============================================================================
# Prompts
# =============================================================================

LEARNING_DETECTION_PROMPT = """You are analyzing an interaction to determine if something was learned.

INTERACTION:
Previous Assistant Response: {previous_response}
User: {user_query}
Assistant's Response: {response}
Tools Used: {tools_used}
Tool Results: {tool_results}
{mode_guidance}
Analyze this interaction semantically. Do NOT look for specific keywords or phrases.
Instead, understand the MEANING and determine:

1. Did the assistant learn HOW to do something new? (A procedure, command, or technique that worked)
2. Did the assistant discover NEW INFORMATION through search or tools?
3. Did the user SHARE A PERSONAL FACT about themselves? (Name, birthday, possessions, relationships, job, preferences, etc.)
4. Did the user EXPRESS A PREFERENCE or give an instruction about how they want things done?
5. Did the assistant gain an INSIGHT about patterns or better approaches?
6. Is the user CORRECTING something the assistant believed or said before?
7. Is the user ADDING DETAIL to something the assistant already knows?
8. Is the user indicating something is NO LONGER TRUE and should be forgotten?
9. **CRITICAL**: Is the user ANSWERING A QUESTION the assistant asked?
   - Look at "Previous Assistant Response" - did the assistant ask a question?
   - If so, is the user's message an answer to that question?
   - Example: Assistant asked "What's your daughter's name?" → User says "Maya" → LEARN: daughter's name is Maya

Think about the semantic meaning, not surface patterns. Examples:
- Personal facts: "I drive a Honda Civic" / "My birthday is March 15" / "I work at Google" / "I have a daughter named Maya"
- Corrections: "No, it's actually X" / "You're wrong, it's X" / "It changed to X" / "Update that to X"
- Answers to questions: Assistant asked "What team?" → User says "Eagles" → User's team is the Eagles
- All mean the same thing based on INTENT, not exact words.

Output JSON:
{{
    "learned_something": true/false,
    "learning_type": "skill" | "fact" | "insight" | "preference" | "correction" | "refinement" | "deletion" | null,
    "is_modification": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "semantic explanation of what was detected",
    "answered_question": true/false
}}

Rules:
- learned_something=false if this is just retrieval, clarification, or small talk
- learned_something=true if user shares ANY personal fact, even casually ("I drive X", "I like Y", "My birthday is Z")
- learned_something=true if user ANSWERS A QUESTION the assistant asked (even short answers like "Maya" or "Eagles")
- is_modification=true if this changes/updates/deletes existing knowledge
- answered_question=true if the user is responding to a question in the previous assistant response
- For personal facts: extract them even if mentioned casually - they're valuable for personalization
"""

LEARNING_EXTRACTION_PROMPT = """Extract the specific learning from this interaction as a STRUCTURED FACT.

INTERACTION:
Previous Assistant Response: {previous_response}
User: {user_query}
Assistant's Response: {response}
Tools Used: {tools_used}
Tool Results: {tool_results}
Learning Type: {learning_type}
Answered Question: {answered_question}

**CRITICAL - STRUCTURED FACT FORMAT:**
Extract learnings as semantic triples (Subject → Predicate → Object).

GOOD examples (structured facts):
- "{user_display_name}'s birthday is March 15" ✓
- "Maya is {user_display_name}'s daughter" ✓
- "WiFi password is hunter123" ✓
- "Restart service with: docker restart service" ✓

BAD examples (self-referential, meta-commentary):
- "I should remember that {user_display_name}'s birthday is March 15" ✗
- "I learned that Maya is {user_display_name}'s daughter" ✗
- "The user told me the WiFi password" ✗
- "I need to remember to restart with docker" ✗

**NEVER include:**
- "I should...", "I need to...", "I learned..."
- "Remember that...", "Note that..."
- "The user said...", "The user told me..."
- "According to...", "Based on..."
- Any meta-commentary about the act of learning

**ALWAYS write as:**
- Direct factual statements
- Third-person about the user (use their name or "User")
- Commands/procedures as imperative statements

**IF ANSWERED_QUESTION IS TRUE:**
The user was answering a question from the previous assistant response. Extract the FULL meaning:
- If assistant asked "What's your daughter's name?" and user said "Maya"
  → Store: "User's daughter is named Maya"
- If assistant asked "What team do you follow?" and user said "Eagles"
  → Store: "User follows/supports the Eagles"
Combine the question context with the answer to create a complete memory.

For modifications (corrections/refinements/deletions):
- Identify what existing knowledge needs to change
- Provide search terms to find that knowledge
- Specify the action needed

SCOPE DETECTION:
Determine memory sharing level - choose ONE of:

"private" - Personal sensitive info for THIS USER ONLY:
  - PINs, passwords, passcodes, authentication codes
  - Bank account numbers, SSN, credit cards
  - Personal secrets, medical info

"shared" - Info that can be shared with family/household (DEFAULT for most info):
  - WiFi passwords, garage codes, alarm codes
  - Shared calendar access, streaming accounts
  - Skills and how-to instructions
  - User preferences and interests
  - General personal facts like jobs, vehicles

"public" - Publicly available information:
  - Facts from web searches
  - Technical documentation
  - General knowledge

Output JSON:
{{
    "title": "short descriptive title",
    "content": "the learning as a complete, retrievable memory",
    "entities": ["key", "entities", "for", "linking"],
    "perspective": "user" | "system" | "third_party",
    "subject": "who this memory is about",
    "scope": "private" | "shared" | "public",
    "procedure": "for skills: the exact steps" or null,
    "success_indicators": ["how to verify it worked"] or null,
    "source": "web_search" | "tool_result" | "user_statement" | "error_recovery",
    "search_for_existing": "query to find related/conflicting memories" or null,
    "action": "create" | "update" | "refine" | "delete"
}}
"""

FAILURE_DETECTION_PROMPT = """Analyze this tool execution to determine if it failed.

TOOL: {tool_name}
ARGUMENTS: {tool_args}
RESULT: {tool_result}
EXPECTED (if skill-based): {expected_outcome}

Determine:
1. Did the tool execution FAIL or produce an ERROR?
2. Does the result match what was EXPECTED?
3. Is there evidence the INFORMATION used was OUTDATED or WRONG?
4. Was the APPROACH fundamentally incorrect?

Output JSON:
{{
    "is_failure": true/false,
    "failure_type": "execution_error" | "unexpected_result" | "outdated_info" | "wrong_approach" | null,
    "confidence": 0.0-1.0,
    "failure_description": "what went wrong",
    "needs_relearning": true/false,
    "search_topic": "what to search for to fix this" or null
}}

Notes:
- Not all unexpected results are failures (tool may have succeeded but returned empty)
- A command returning error codes or exceptions is an execution_error
- If the tool worked but gave wrong info, that's outdated_info
"""

SKILL_RELEARNING_PROMPT = """A skill/approach failed and needs to be relearned.

FAILED SKILL:
{failed_skill}

FAILURE REASON:
{failure_reason}

WEB SEARCH RESULTS:
{search_results}

Based on the search results, extract the CORRECT way to do this:
1. What is the updated procedure/command?
2. What changed from the old approach?
3. What should be remembered for next time?

Output JSON:
{{
    "corrected_skill": "the new correct procedure or approach",
    "what_changed": "explanation of what was wrong and what's different",
    "confidence": 0.0-1.0,
    "should_update_memory": true/false,
    "search_query_for_old": "query to find the old skill to replace"
}}
"""

MEMORY_COMPARISON_PROMPT = """Compare new information against existing memory.

NEW INFORMATION:
{new_info}

EXISTING MEMORY:
{existing_memory}

Determine the semantic relationship:
- Are these about the SAME TOPIC?
- Does new info REPLACE old (correction)?
- Does new info ADD TO old (refinement)?
- Does new info make old OBSOLETE (supersede)?
- Should old be DELETED (no longer valid)?
- Are both VALID but different aspects?

Output JSON:
{{
    "action": "update" | "refine" | "supersede" | "delete" | "keep_both" | "no_action",
    "confidence": 0.0-1.0,
    "merged_content": "combined content if refining" or null,
    "reasoning": "semantic explanation"
}}
"""

MEMORY_LINKING_PROMPT = """Find semantic connections between memories.

NEW MEMORY:
{new_memory}

POTENTIALLY RELATED MEMORIES:
{existing_memories}

For each existing memory, determine:
1. Is there a MEANINGFUL semantic connection?
2. What TYPE of connection? (same_topic, prerequisite, builds_on, contradicts, complements)
3. How STRONG is the connection?

Output JSON:
{{
    "links": [
        {{
            "memory_id": "id",
            "connection_type": "same_topic" | "prerequisite" | "builds_on" | "contradicts" | "complements",
            "strength": 0.0-1.0,
            "reason": "explanation"
        }}
    ]
}}

Only include links with strength >= 0.5.
"""

CORRECTION_VERIFIABILITY_PROMPT = """The user is correcting something. Determine if this correction can/should be verified.

USER'S CORRECTION: {user_correction}
WHAT THE AGENT BELIEVED: {original_belief}
TOPIC: {topic}

Determine:
1. Is this a VERIFIABLE FACT that can be checked via web search?
   - Dates, names, technical specifications, public information = VERIFIABLE
2. Is this a PERSONAL PREFERENCE or OPINION that only the user would know?
   - "I prefer X", "My birthday is Y", "I like Z" = NOT_VERIFIABLE (trust user)
3. Is this about PRIVATE/PERSONAL information?
   - Passwords, personal details, family info = NOT_VERIFIABLE (trust user)
4. Is this a SKILL or PROCEDURE that can be verified?
   - Commands, technical steps, how-to = VERIFIABLE

Output JSON:
{{
    "is_verifiable": true/false,
    "reason": "why it can or cannot be verified",
    "verification_query": "search query to verify this" or null,
    "category": "fact" | "preference" | "personal" | "skill" | "opinion"
}}
"""

CORRECTION_VERIFICATION_PROMPT = """The user corrected the assistant. Verify if the user is correct using search results.

USER'S CLAIM: {user_claim}
WHAT THE ASSISTANT BELIEVED: {original_belief}

SEARCH RESULTS:
{search_results}

Analyze the evidence and determine:
1. Do the search results SUPPORT the user's claim?
2. Do the search results CONTRADICT the user's claim?
3. Is there not enough evidence either way?
4. Is the user PARTIALLY correct (some parts right, some wrong)?

Be fair and objective. Users can be wrong, but so can the assistant's original belief.

Output JSON:
{{
    "verification_result": "verified" | "contradicted" | "uncertain" | "partially_correct",
    "confidence": 0.0-1.0,
    "evidence_summary": "what the search results say",
    "correct_information": "the actual correct information based on evidence" or null,
    "user_error": "if user is wrong, what they got wrong" or null,
    "polite_response": "suggested response if user needs to be corrected (be kind and helpful)"
}}

Rules:
- If evidence clearly supports user → verified
- If evidence clearly contradicts user → contradicted (provide correct info)
- If evidence is mixed or unclear → uncertain (accept with lower confidence)
- If user is right about some but not all → partially_correct
- Always be respectful - users make honest mistakes
"""


# =============================================================================
# Mode-Specific Learning Guidance
# =============================================================================

MODE_LEARNING_GUIDANCE = {
    "learning": """
LEARNING MODE ACTIVE - Also look for:
- Topics the user wants to learn about (store as interest/preference)
- Subject areas they're curious about
- Their current level of understanding (beginner, intermediate, advanced)
""",
    "support": """
SUPPORT MODE ACTIVE - Also look for:
- Sports teams they follow (if discussing sports losses/wins)
- Their job/workplace (if discussing work stress)
- Recurring challenges or frustrations (work patterns, relationships)
""",
    "casual": """
CASUAL MODE ACTIVE - Also look for:
- Hobbies and interests
- Food preferences, favorite activities
- Entertainment preferences (shows, music, games)
""",
    "brainstorm": """
BRAINSTORM MODE ACTIVE - Also look for:
- Projects they're working on
- Goals they're trying to achieve
- Skills they want to develop
""",
}


# =============================================================================
# Service
# =============================================================================


class LearningService:
    """Service for autonomous learning and memory extraction.

    Features:
    - Fully LLM-driven detection (no keyword patterns)
    - Failure-triggered relearning
    - Skill confidence tracking and decay
    - Knowledge correction, refinement, deletion
    - User credibility tracking and trust calibration
    - Multi-user support with knowledge scoping
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        agent_name: str = "the agent",
        agent_id: str = "agent",
        search_provider: SearchProvider | None = None,
        credibility_provider: CredibilityProvider | None = None,
        user_provider: UserProvider | None = None,
        learning_cooldown_minutes: int = 30,
    ):
        """Initialize the learning service.

        Args:
            llm: LLM provider for semantic analysis
            memory: Memory provider for storage
            agent_name: Display name of the agent
            agent_id: ID of the agent
            search_provider: Optional search provider for relearning
            credibility_provider: Optional provider for user credibility
            user_provider: Optional provider for user information
            learning_cooldown_minutes: Cooldown for duplicate detection
        """
        self.llm = llm
        self.memory = memory
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.search_provider = search_provider
        self.credibility_provider = credibility_provider
        self.user_provider = user_provider

        # Track recent learnings to avoid duplicates
        self._recent_learnings: dict[str, datetime] = {}
        self._learning_cooldown_minutes = learning_cooldown_minutes

        # Track skill confidence for decay on failures
        self._skill_confidence: dict[str, SkillConfidence] = {}

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    async def process_interaction(
        self,
        user_query: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        user_id: str,
        conversation_id: str,
        conversation_mode: str | None = None,
        previous_response: str | None = None,
    ) -> LearningResult:
        """Process an interaction to extract and store learnings.

        This should be called asynchronously after the response is sent.

        Args:
            user_query: What the user asked
            response: Agent's response
            tool_calls: List of tool calls made (with results)
            user_id: User ID for storing memories
            conversation_id: Conversation ID for context
            conversation_mode: Current conversation mode
            previous_response: The agent's previous response (for answered questions)

        Returns:
            LearningResult with learning outcome
        """
        try:
            logger.info(f"[LEARNING] Starting detection for user={user_id}, query='{user_query[:50]}...'")

            # Get user display name
            user_display_name = user_id
            if self.user_provider:
                user_display_name = await self.user_provider.get_display_name(user_id)

            # Phase 1: Detect if we learned something (fully LLM-driven)
            detection = await self._detect_learning(
                user_query, response, tool_calls, conversation_mode, previous_response
            )

            logger.info(
                f"[LEARNING] Detection result: learned={detection.get('learned_something')}, "
                f"type={detection.get('learning_type')}, conf={detection.get('confidence')}"
            )

            if not detection.get("learned_something"):
                return LearningResult(learned=False, reason=detection.get("reasoning"))

            learning_type = detection.get("learning_type")
            confidence = detection.get("confidence", 0.5)
            is_modification = detection.get("is_modification", False)
            answered_question = detection.get("answered_question", False)

            # Skip low confidence learnings
            if confidence < 0.6:
                return LearningResult(learned=False, reason=f"low confidence ({confidence})")

            logger.info(
                f"Learning detected ({learning_type}, confidence={confidence}, "
                f"modification={is_modification}): {detection.get('reasoning')}"
            )

            # Phase 2: Extract the specific learning
            extraction = await self._extract_learning(
                user_query, response, tool_calls, learning_type, user_display_name,
                previous_response, answered_question
            )

            if not extraction or not extraction.get("content"):
                return LearningResult(learned=False, reason="extraction failed")

            # Check for duplicate/recent learning
            content_key = f"{user_id}:{extraction.get('title', '')[:50]}"
            if not answered_question and self._is_recent_learning(content_key):
                return LearningResult(learned=False, reason="duplicate")

            # Phase 2.5: Verify corrections before accepting
            verification_result_str = None

            if learning_type == "correction" and self.search_provider and self.credibility_provider:
                should_verify, _ = self.credibility_provider.should_verify_correction(
                    user_id=user_id,
                    domain=extraction.get("title", "").split()[0].lower() if extraction.get("title") else None,
                )

                if should_verify:
                    verification = await self._verify_user_correction(
                        user_claim=extraction.get("content"),
                        original_belief=response,
                        topic=extraction.get("title", ""),
                    )

                    if verification:
                        result = verification.get("result")
                        verification_result_str = result.value if hasattr(result, 'value') else str(result)

                        if result == VerificationResult.CONTRADICTED:
                            self.credibility_provider.record_correction_result(
                                user_id=user_id,
                                result="contradicted",
                                domain=extraction.get("title", "").split()[0].lower() if extraction.get("title") else None,
                                user_was_confident=confidence > 0.8,
                            )
                            return LearningResult(
                                learned=False,
                                reason="correction_contradicted",
                                verification_result=result,
                                correct_information=verification.get("correct_information"),
                            )

                        elif result == VerificationResult.VERIFIED:
                            confidence = max(confidence, verification.get("confidence", 0.9))

                        elif result == VerificationResult.UNCERTAIN:
                            confidence = min(confidence, 0.65)

                # Update credibility if verified
                if verification_result_str and verification_result_str != "contradicted":
                    self.credibility_provider.record_correction_result(
                        user_id=user_id,
                        result=verification_result_str,
                        domain=extraction.get("title", "").split()[0].lower() if extraction.get("title") else None,
                        user_was_confident=confidence > 0.8,
                    )

            # Phase 3: Handle modifications if needed
            action = extraction.get("action", "create")
            search_query = extraction.get("search_for_existing")

            if action in ("update", "refine", "delete") and search_query:
                result = await self._handle_memory_modification(
                    action=action,
                    search_query=search_query,
                    new_content=extraction.get("content"),
                    user_id=user_id,
                    confidence=confidence,
                    entities=extraction.get("entities", []),
                    conversation_id=conversation_id,
                    learning_type=learning_type,
                )
                if result.get("handled"):
                    self._mark_learned(content_key)
                    return LearningResult(
                        learned=True,
                        learning_type=result.get("learning_type"),
                        action=result.get("action"),
                        memory_id=result.get("memory_id"),
                        confidence=result.get("confidence"),
                    )

            # Phase 4: Check for contradictions with existing memories
            if is_modification:
                contradiction_result = await self._check_and_resolve_contradictions(
                    new_content=extraction["content"],
                    user_id=user_id,
                    confidence=confidence,
                    entities=extraction.get("entities", []),
                    conversation_id=conversation_id,
                )
                if contradiction_result.get("resolved"):
                    self._mark_learned(content_key)
                    return LearningResult(
                        learned=True,
                        learning_type=contradiction_result.get("learning_type", learning_type),
                        action=contradiction_result.get("action"),
                        memory_id=contradiction_result.get("memory_id"),
                        confidence=confidence,
                    )

            # Phase 5: Store as new learning
            memory_type = self._learning_type_to_memory_type(learning_type)
            detected_scope = extraction.get("scope", "private")

            logger.info(
                f"[LEARNING] Storing: type={memory_type}, scope={detected_scope}, "
                f"content='{extraction['content'][:100]}...'"
            )

            store_result = await self.memory.store(
                content=extraction["content"],
                user_id=user_id,
                scope=detected_scope,
                memory_type=memory_type,
                importance=confidence,
                entities=extraction.get("entities", []),
                conversation_id=conversation_id,
            )

            if not store_result.get("success"):
                return LearningResult(learned=False, reason=store_result.get("error"))

            logger.info(f"[LEARNING] SUCCESS! Stored memory_id={store_result.get('memory_id')}")

            # Track skill confidence
            if learning_type == "skill":
                self._track_skill(
                    store_result["memory_id"],
                    extraction["content"],
                    confidence,
                )

            self._mark_learned(content_key)

            # Phase 6: Link to related memories (async, don't block)
            asyncio.create_task(
                self._link_memory(
                    store_result["memory_id"],
                    extraction["content"],
                    extraction.get("entities", []),
                    user_id,
                )
            )

            return LearningResult(
                learned=True,
                learning_type=learning_type,
                action="create",
                title=extraction.get("title"),
                memory_id=store_result["memory_id"],
                confidence=confidence,
            )

        except Exception as e:
            logger.exception(f"Error in learning process: {e}")
            return LearningResult(learned=False, reason=str(e))

    async def process_tool_failure(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        skill_used: dict[str, Any] | None,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Process a tool failure to trigger relearning.

        When a tool fails, especially if it was based on a stored skill,
        we should:
        1. Detect what failed and why
        2. Search for correct information
        3. Update/fix the stored skill

        Args:
            tool_name: Name of the tool that failed
            tool_args: Arguments passed to the tool
            tool_result: The result (error or unexpected output)
            skill_used: The skill memory that was used (if any)
            user_id: User ID
            conversation_id: Conversation ID

        Returns:
            Dict with relearning results
        """
        try:
            # Phase 1: Detect if this is actually a failure
            expected_outcome = None
            if skill_used:
                expected_outcome = skill_used.get("success_indicators", [])

            detection = await self._detect_failure(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                expected_outcome=expected_outcome,
            )

            if not detection.get("is_failure"):
                return {"relearned": False, "reason": "not a failure"}

            failure_type = detection.get("failure_type")
            needs_relearning = detection.get("needs_relearning", False)
            search_topic = detection.get("search_topic")

            logger.warning(
                f"Tool failure detected ({failure_type}): "
                f"{detection.get('failure_description')}"
            )

            # Record failure for skill confidence decay
            if skill_used:
                skill_id = skill_used.get("memory_id")
                if skill_id and skill_id in self._skill_confidence:
                    self._skill_confidence[skill_id].record_failure()
                    if self._skill_confidence[skill_id].needs_relearning():
                        needs_relearning = True

            if not needs_relearning or not search_topic:
                return {
                    "relearned": False,
                    "failure_detected": True,
                    "failure_type": failure_type,
                    "reason": "failure detected but no relearning needed",
                }

            # Phase 2: Search for correct information
            if not self.search_provider:
                return {
                    "relearned": False,
                    "failure_detected": True,
                    "reason": "no search service for relearning",
                }

            search_results = await self.search_provider.search(search_topic, limit=5)

            if not search_results:
                return {
                    "relearned": False,
                    "failure_detected": True,
                    "reason": "search returned no results",
                }

            # Phase 3: Extract corrected skill
            failed_skill_content = skill_used.get("content", "") if skill_used else ""
            corrected = await self._extract_corrected_skill(
                failed_skill=failed_skill_content,
                failure_reason=detection.get("failure_description", ""),
                search_results=search_results,
            )

            if not corrected or not corrected.get("should_update_memory"):
                return {
                    "relearned": False,
                    "failure_detected": True,
                    "searched": True,
                    "reason": "could not extract corrected skill",
                }

            # Phase 4: Update the stored skill
            corrected_content = corrected.get("corrected_skill")
            search_for_old = corrected.get("search_query_for_old")

            if search_for_old:
                update_result = await self._handle_memory_modification(
                    action="update",
                    search_query=search_for_old,
                    new_content=corrected_content,
                    user_id=user_id,
                    confidence=corrected.get("confidence", 0.8),
                    entities=[],
                    conversation_id=conversation_id,
                    learning_type="skill",
                )

                if update_result.get("handled"):
                    logger.info(f"Successfully relearned skill: {corrected.get('what_changed')}")
                    return {
                        "relearned": True,
                        "failure_type": failure_type,
                        "what_changed": corrected.get("what_changed"),
                        "new_skill": corrected_content[:100],
                    }

            # Store as new if no old skill to update
            store_result = await self.memory.store(
                content=corrected_content,
                user_id=user_id,
                scope="private",
                memory_type=MemoryType.SKILL,
                importance=corrected.get("confidence", 0.8),
                entities=[],
                conversation_id=conversation_id,
            )

            if store_result.get("success"):
                return {
                    "relearned": True,
                    "failure_type": failure_type,
                    "what_changed": corrected.get("what_changed"),
                    "memory_id": store_result["memory_id"],
                }

            return {"relearned": False, "reason": "failed to store corrected skill"}

        except Exception as e:
            logger.exception(f"Error in failure relearning: {e}")
            return {"relearned": False, "reason": str(e)}

    async def record_skill_success(self, skill_id: str, skill_content: str) -> None:
        """Record successful use of a skill."""
        if skill_id in self._skill_confidence:
            self._skill_confidence[skill_id].record_success()
        else:
            self._track_skill(skill_id, skill_content, 1.0)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _detect_learning(
        self,
        user_query: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        conversation_mode: str | None = None,
        previous_response: str | None = None,
    ) -> dict[str, Any]:
        """Detect if the interaction contains something worth learning."""
        tools_used = [tc.get("tool", "unknown") for tc in tool_calls]
        tool_results = []
        for tc in tool_calls:
            result = tc.get("result")
            if result:
                result_str = str(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                tool_results.append(f"{tc.get('tool')}: {result_str}")

        mode_guidance = ""
        if conversation_mode:
            mode_guidance = MODE_LEARNING_GUIDANCE.get(conversation_mode, "")

        prompt = LEARNING_DETECTION_PROMPT.format(
            user_query=user_query,
            response=response,
            tools_used=", ".join(tools_used) if tools_used else "(none)",
            tool_results="\n".join(tool_results) if tool_results else "(none)",
            mode_guidance=mode_guidance,
            previous_response=previous_response or "(no previous response)",
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze this interaction semantically."},
        ]

        result = await self.llm.chat_json(messages, max_tokens=200)

        if not result or not result.get("parsed"):
            return {"learned_something": False, "reasoning": "detection failed"}

        return result["parsed"]

    async def _extract_learning(
        self,
        user_query: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        learning_type: str,
        user_display_name: str,
        previous_response: str | None = None,
        answered_question: bool = False,
    ) -> dict[str, Any] | None:
        """Extract specific learning content from the interaction."""
        tools_used = [tc.get("tool", "unknown") for tc in tool_calls]
        tool_results = []
        for tc in tool_calls:
            result = tc.get("result")
            if result:
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "..."
                tool_results.append(f"{tc.get('tool')}: {result_str}")

        prompt = LEARNING_EXTRACTION_PROMPT.format(
            user_query=user_query,
            response=response,
            tools_used=", ".join(tools_used) if tools_used else "(none)",
            tool_results="\n".join(tool_results) if tool_results else "(none)",
            learning_type=learning_type,
            user_display_name=user_display_name,
            previous_response=previous_response or "(no previous response)",
            answered_question="true" if answered_question else "false",
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Extract the learning for storage."},
        ]

        result = await self.llm.chat_json(messages, max_tokens=500)

        if not result or not result.get("parsed"):
            return None

        return result["parsed"]

    async def _verify_user_correction(
        self,
        user_claim: str,
        original_belief: str,
        topic: str,
    ) -> dict[str, Any] | None:
        """Verify a user's correction before accepting it."""
        try:
            # Step 1: Determine if verifiable
            verifiability_prompt = CORRECTION_VERIFIABILITY_PROMPT.format(
                user_correction=user_claim,
                original_belief=original_belief,
                topic=topic,
            )

            messages = [
                {"role": "system", "content": verifiability_prompt},
                {"role": "user", "content": "Should this correction be verified?"},
            ]

            verifiability = await self.llm.chat_json(messages, max_tokens=200)

            if not verifiability or not verifiability.get("parsed"):
                return None

            parsed = verifiability["parsed"]

            if not parsed.get("is_verifiable"):
                return {
                    "result": VerificationResult.NOT_VERIFIABLE,
                    "reason": parsed.get("reason"),
                    "category": parsed.get("category"),
                }

            # Step 2: Search for verification evidence
            search_query = parsed.get("verification_query")
            if not search_query or not self.search_provider:
                return {
                    "result": VerificationResult.UNCERTAIN,
                    "reason": "no search query or service available",
                }

            search_results = await self.search_provider.search(search_query, limit=5)

            if not search_results:
                return {
                    "result": VerificationResult.UNCERTAIN,
                    "reason": "no search results found",
                }

            # Step 3: Analyze search results
            results_str = "\n".join([
                f"- {r.get('title', 'Untitled')}: {r.get('snippet', r.get('content', ''))[:300]}"
                for r in search_results[:5]
            ])

            verification_prompt = CORRECTION_VERIFICATION_PROMPT.format(
                user_claim=user_claim,
                original_belief=original_belief,
                search_results=results_str,
            )

            messages = [
                {"role": "system", "content": verification_prompt},
                {"role": "user", "content": "Verify the user's correction against the evidence."},
            ]

            verification = await self.llm.chat_json(messages, max_tokens=400)

            if not verification or not verification.get("parsed"):
                return {
                    "result": VerificationResult.UNCERTAIN,
                    "reason": "verification analysis failed",
                }

            result_parsed = verification["parsed"]
            result_str = result_parsed.get("verification_result", "uncertain")

            result_map = {
                "verified": VerificationResult.VERIFIED,
                "contradicted": VerificationResult.CONTRADICTED,
                "uncertain": VerificationResult.UNCERTAIN,
                "partially_correct": VerificationResult.PARTIALLY_CORRECT,
            }

            return {
                "result": result_map.get(result_str, VerificationResult.UNCERTAIN),
                "confidence": result_parsed.get("confidence", 0.5),
                "evidence_summary": result_parsed.get("evidence_summary"),
                "correct_information": result_parsed.get("correct_information"),
                "user_error": result_parsed.get("user_error"),
                "polite_response": result_parsed.get("polite_response"),
            }

        except Exception as e:
            logger.warning(f"Error verifying user correction: {e}")
            return {
                "result": VerificationResult.UNCERTAIN,
                "reason": str(e),
            }

    async def _detect_failure(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        expected_outcome: list[str] | None,
    ) -> dict[str, Any]:
        """Detect if a tool execution failed."""
        prompt = FAILURE_DETECTION_PROMPT.format(
            tool_name=tool_name,
            tool_args=str(tool_args),
            tool_result=str(tool_result)[:1000],
            expected_outcome=", ".join(expected_outcome) if expected_outcome else "(not specified)",
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze this tool execution."},
        ]

        result = await self.llm.chat_json(messages, max_tokens=200)

        if not result or not result.get("parsed"):
            return {"is_failure": False}

        return result["parsed"]

    async def _extract_corrected_skill(
        self,
        failed_skill: str,
        failure_reason: str,
        search_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Extract corrected skill from search results."""
        results_str = "\n".join([
            f"- {r.get('title', 'Untitled')}: {r.get('snippet', r.get('content', ''))[:300]}"
            for r in search_results[:5]
        ])

        prompt = SKILL_RELEARNING_PROMPT.format(
            failed_skill=failed_skill or "(no previous skill)",
            failure_reason=failure_reason,
            search_results=results_str,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Extract the corrected approach."},
        ]

        result = await self.llm.chat_json(messages, max_tokens=400)

        if not result or not result.get("parsed"):
            return None

        return result["parsed"]

    async def _link_memory(
        self,
        memory_id: str,
        content: str,
        entities: list[str],
        user_id: str,
    ) -> None:
        """Find and create links to related memories."""
        try:
            search_query = " ".join(entities[:3]) if entities else content[:100]
            existing = await self.memory.search(
                query=search_query,
                user_id=user_id,
                limit=5,
                score_threshold=0.5,
            )

            if not existing:
                return

            existing_str = "\n".join([
                f"- [ID: {m.get('memory_id', 'unknown')}] {m.get('content', '')[:200]}"
                for m in existing
            ])

            prompt = MEMORY_LINKING_PROMPT.format(
                new_memory=content,
                existing_memories=existing_str,
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Find semantic connections."},
            ]

            result = await self.llm.chat_json(messages, max_tokens=300)

            if not result or not result.get("parsed"):
                return

            links = result["parsed"].get("links", [])
            for link in links:
                if link.get("strength", 0) >= 0.5:
                    logger.info(
                        f"Memory link: {memory_id} --[{link.get('connection_type')}]--> "
                        f"{link.get('memory_id')} (strength={link.get('strength')})"
                    )

        except Exception as e:
            logger.warning(f"Error linking memory: {e}")

    async def _handle_memory_modification(
        self,
        action: str,
        search_query: str,
        new_content: str | None,
        user_id: str,
        confidence: float,
        entities: list[str],
        conversation_id: str,
        learning_type: str,
    ) -> dict[str, Any]:
        """Handle update, refine, or delete actions on existing memories."""
        try:
            existing = await self.memory.search(
                query=search_query,
                user_id=user_id,
                limit=3,
                score_threshold=0.6,
            )

            if not existing:
                return {"handled": False, "reason": "no matching memory found"}

            best_match = existing[0]
            match_score = best_match.get("weighted_score", best_match.get("score", 0))

            if match_score < 0.65:
                return {"handled": False, "reason": "match confidence too low"}

            if action == "delete":
                invalidate_result = await self.memory.invalidate(
                    content=best_match["content"],
                    user_id=user_id,
                    reason="User requested deletion",
                )
                if invalidate_result.get("success"):
                    return {
                        "handled": True,
                        "learned": True,
                        "learning_type": "deletion",
                        "action": "delete",
                        "deleted_content": best_match["content"][:100],
                        "confidence": confidence,
                    }
                return {"handled": False, "reason": invalidate_result.get("error")}

            elif action == "update":
                memory_type = self._learning_type_to_memory_type(learning_type)
                store_result = await self.memory.store(
                    content=new_content,
                    user_id=user_id,
                    scope="private",
                    memory_type=memory_type,
                    importance=confidence,
                    entities=entities,
                    conversation_id=conversation_id,
                )

                if store_result.get("success"):
                    await self.memory.invalidate(
                        content=best_match["content"],
                        user_id=user_id,
                        superseded_by=store_result["memory_id"],
                        reason="Superseded by correction",
                    )
                    return {
                        "handled": True,
                        "learned": True,
                        "learning_type": "correction",
                        "action": "update",
                        "old_content": best_match["content"][:100],
                        "new_content": new_content[:100],
                        "memory_id": store_result["memory_id"],
                        "confidence": confidence,
                    }
                return {"handled": False, "reason": store_result.get("error")}

            elif action == "refine":
                merged = await self._merge_memories(
                    existing=best_match["content"],
                    new_info=new_content,
                )

                if not merged:
                    return {"handled": False, "reason": "merge failed"}

                memory_type_str = best_match.get("memory_type", "fact")
                try:
                    memory_type = MemoryType(memory_type_str) if isinstance(memory_type_str, str) else memory_type_str
                except ValueError:
                    memory_type = MemoryType.FACT

                store_result = await self.memory.store(
                    content=merged,
                    user_id=user_id,
                    scope="private",
                    memory_type=memory_type,
                    importance=max(confidence, best_match.get("importance", 0.5)),
                    entities=list(set(entities + best_match.get("entities", []))),
                    conversation_id=conversation_id,
                )

                if store_result.get("success"):
                    await self.memory.invalidate(
                        content=best_match["content"],
                        user_id=user_id,
                        superseded_by=store_result["memory_id"],
                        reason="Refined with additional detail",
                    )
                    return {
                        "handled": True,
                        "learned": True,
                        "learning_type": "refinement",
                        "action": "refine",
                        "merged": merged[:100],
                        "memory_id": store_result["memory_id"],
                        "confidence": confidence,
                    }
                return {"handled": False, "reason": store_result.get("error")}

            return {"handled": False, "reason": f"unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Error handling memory modification: {e}")
            return {"handled": False, "reason": str(e)}

    async def _check_and_resolve_contradictions(
        self,
        new_content: str,
        user_id: str,
        confidence: float,
        entities: list[str],
        conversation_id: str,
    ) -> dict[str, Any]:
        """Check if new content contradicts existing memories and resolve."""
        try:
            existing = await self.memory.search(
                query=new_content,
                user_id=user_id,
                limit=5,
                score_threshold=0.5,
            )

            if not existing:
                return {"resolved": False, "reason": "no related memories"}

            for memory in existing:
                comparison = await self._compare_memories(
                    new_info=new_content,
                    existing=memory["content"],
                )

                if not comparison:
                    continue

                action = comparison.get("action")
                action_confidence = comparison.get("confidence", 0)

                if action_confidence < 0.7:
                    continue

                if action == "update":
                    store_result = await self.memory.store(
                        content=new_content,
                        user_id=user_id,
                        scope="private",
                        memory_type=memory.get("memory_type", MemoryType.FACT),
                        importance=confidence,
                        entities=entities,
                        conversation_id=conversation_id,
                    )

                    if store_result.get("success"):
                        await self.memory.invalidate(
                            content=memory["content"],
                            user_id=user_id,
                            superseded_by=store_result["memory_id"],
                            reason="Contradiction resolved - updated",
                        )

                    return {
                        "resolved": True,
                        "learned": True,
                        "action": "update",
                        "memory_id": store_result.get("memory_id"),
                        "confidence": confidence,
                    }

                elif action == "refine":
                    merged = comparison.get("merged_content")
                    if not merged:
                        merged = await self._merge_memories(
                            existing=memory["content"],
                            new_info=new_content,
                        )

                    if merged:
                        store_result = await self.memory.store(
                            content=merged,
                            user_id=user_id,
                            scope="private",
                            memory_type=memory.get("memory_type", MemoryType.FACT),
                            importance=max(confidence, memory.get("importance", 0.5)),
                            entities=list(set(entities + memory.get("entities", []))),
                            conversation_id=conversation_id,
                        )

                        if store_result.get("success"):
                            await self.memory.invalidate(
                                content=memory["content"],
                                user_id=user_id,
                                superseded_by=store_result["memory_id"],
                                reason="Refined with additional detail",
                            )

                        return {
                            "resolved": True,
                            "learned": True,
                            "action": "refine",
                            "memory_id": store_result.get("memory_id"),
                            "confidence": confidence,
                        }

                elif action == "supersede":
                    store_result = await self.memory.store(
                        content=new_content,
                        user_id=user_id,
                        scope="private",
                        memory_type=memory.get("memory_type", MemoryType.FACT),
                        importance=confidence,
                        entities=entities,
                        conversation_id=conversation_id,
                    )

                    if store_result.get("success"):
                        await self.memory.invalidate(
                            content=memory["content"],
                            user_id=user_id,
                            superseded_by=store_result["memory_id"],
                            reason="Superseded by newer information",
                        )

                    return {
                        "resolved": True,
                        "learned": True,
                        "action": "supersede",
                        "memory_id": store_result.get("memory_id"),
                        "confidence": confidence,
                    }

            return {"resolved": False, "reason": "no contradictions"}

        except Exception as e:
            logger.exception(f"Error checking contradictions: {e}")
            return {"resolved": False, "reason": str(e)}

    async def _compare_memories(
        self,
        new_info: str,
        existing: str,
    ) -> dict[str, Any] | None:
        """Compare new information against existing memory."""
        prompt = MEMORY_COMPARISON_PROMPT.format(
            new_info=new_info,
            existing_memory=existing,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Compare these semantically."},
        ]

        result = await self.llm.chat_json(messages, max_tokens=300)

        if not result or not result.get("parsed"):
            return None

        return result["parsed"]

    async def _merge_memories(
        self,
        existing: str,
        new_info: str,
    ) -> str | None:
        """Merge existing memory with new information."""
        prompt = f"""Merge these two pieces of information into a single, coherent memory.
Keep all relevant details from both. Remove contradictions by preferring newer information.

EXISTING MEMORY:
{existing}

NEW INFORMATION:
{new_info}

Output ONLY the merged content as a single paragraph."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Merge these memories."},
        ]

        result = await self.llm.chat(messages, max_tokens=300)

        if result and result.get("content"):
            return result["content"].strip()

        return None

    def _track_skill(self, skill_id: str, content: str, confidence: float) -> None:
        """Track a skill for confidence monitoring."""
        self._skill_confidence[skill_id] = SkillConfidence(
            skill_id=skill_id,
            content=content,
            confidence=confidence,
        )

    def _learning_type_to_memory_type(self, learning_type: str) -> MemoryType:
        """Convert learning type to memory type for storage."""
        mapping = {
            "skill": MemoryType.SKILL,
            "fact": MemoryType.FACT,
            "insight": MemoryType.INSIGHT,
            "preference": MemoryType.PREFERENCE,
            "correction": MemoryType.FACT,
            "refinement": MemoryType.FACT,
            "deletion": MemoryType.FACT,
        }
        return mapping.get(learning_type, MemoryType.FACT)

    def _is_recent_learning(self, content_key: str) -> bool:
        """Check if we recently learned something similar."""
        if content_key not in self._recent_learnings:
            return False

        last_learned = self._recent_learnings[content_key]
        cooldown = timedelta(minutes=self._learning_cooldown_minutes)
        return datetime.now() - last_learned < cooldown

    def _mark_learned(self, content_key: str) -> None:
        """Mark something as recently learned."""
        self._recent_learnings[content_key] = datetime.now()

        # Clean up old entries
        cutoff = datetime.now() - timedelta(hours=2)
        self._recent_learnings = {
            k: v for k, v in self._recent_learnings.items() if v > cutoff
        }

    # =========================================================================
    # Public Getters
    # =========================================================================

    def get_skill_confidence(self, skill_id: str) -> float | None:
        """Get current confidence for a skill."""
        if skill_id in self._skill_confidence:
            return self._skill_confidence[skill_id].confidence
        return None

    def get_degraded_skills(self) -> list[SkillConfidence]:
        """Get skills that have degraded and may need relearning."""
        return [
            skill for skill in self._skill_confidence.values()
            if skill.needs_relearning()
        ]

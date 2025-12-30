"""Query contextualization service for multi-turn conversations.

This service transforms ambiguous follow-up queries into standalone queries
that can be effectively used for RAG retrieval.

Example:
    History: "What events do I have this week?" → "You have a dentist appointment..."
    Query: "What about next week?"
    Contextualized: "What events do I have next week?"

The contextualization happens BEFORE RAG retrieval to ensure we search
for the right thing, not after when it's too late.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .protocols import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ContextualizerConfig:
    """Configuration for query contextualization.

    Attributes:
        max_history_turns: Maximum conversation turns to include in context
        use_large_model: Whether to use larger model for better quality
        max_result_length: Maximum length for summarized tool results
        identity_patterns: Patterns that indicate identity questions (skip contextualizing)
        context_indicators: Words that indicate context is needed
        affirmations: Short words that need context
    """

    max_history_turns: int = 5
    use_large_model: bool = True
    max_result_length: int = 500
    identity_patterns: list[str] = field(default_factory=lambda: [
        "who are you",
        "what are you",
        "what's your name",
        "what is your name",
        "who made you",
        "who created you",
        "who built you",
        "what can you do",
        "tell me about yourself",
        "introduce yourself",
        "where are you",
        "what room are you in",
        "your location",
    ])
    context_indicators: list[str] = field(default_factory=lambda: [
        "it", "that", "this", "those", "these",
        "them", "they", "he", "she", "him", "her",
        "the first", "the second", "the last",
        "next", "previous", "again",
        "what about", "how about", "and the",
        "also", "too", "as well",
    ])
    affirmations: list[str] = field(default_factory=lambda: [
        "yes", "no", "okay", "ok", "sure", "yep", "nope", "thanks", "thank you"
    ])


@dataclass
class ContextualizedQuery:
    """Result of query contextualization.

    Attributes:
        original_query: The original query before contextualization
        standalone_query: The rewritten standalone query
        needs_retrieval: Whether RAG retrieval is needed
        intent: Detected intent (calendar, memory, home_control, etc.)
        direct_answer: If answerable from history alone
        confidence: Confidence in the contextualization
        processing_time_ms: Time taken to contextualize
    """

    original_query: str
    standalone_query: str
    needs_retrieval: bool
    intent: str  # calendar, memory, home_control, general_knowledge, etc.
    direct_answer: str | None
    confidence: float
    processing_time_ms: int

    def should_skip_rag(self) -> bool:
        """Check if we can skip RAG retrieval."""
        return not self.needs_retrieval or self.direct_answer is not None


# Default prompt for query contextualization
DEFAULT_CONTEXTUALIZE_PROMPT = """You are a query preprocessor. Your job is to:
1. REWRITE the query to be completely self-contained (NO pronouns, NO references to history)
2. Determine if retrieval is needed
3. Identify the intent

CONVERSATION HISTORY:
{history}

CURRENT QUERY: "{query}"

Output XML:
<contextualization>
    <standalone_query>FULLY REWRITTEN query with all pronouns/references replaced</standalone_query>
    <needs_retrieval>true or false</needs_retrieval>
    <intent>calendar | memory | home_control | web_search | system_command | general_knowledge | clarification | acknowledgment | greeting</intent>
    <direct_answer>Answer if needs_retrieval=false AND answerable from history, else empty</direct_answer>
    <confidence>0.0-1.0</confidence>
</contextualization>

CRITICAL - standalone_query RULES:
1. ALWAYS rewrite to be completely standalone - someone with NO context should understand it
2. Replace ALL pronouns: "it" → "the dog", "its" → "the dog's", "that" → "the concert"
3. Replace ALL references: "next week" → "What events do I have next week"
4. The standalone_query should be a COMPLETE question that makes sense alone
5. If the original has pronouns/references, the standalone_query MUST be different
6. SPELLING DICTATION: When users spell words letter-by-letter with hyphens/spaces (like "C-A-R-E-M-E-T-X" or "A B C"), convert to the actual word ("CareMetx" or "ABC") in the standalone_query.

Examples of CORRECT rewrites:
- "What about Germany?" after France discussion → "What is the capital of Germany?"
- "What is its breed?" after dog discussion → "What breed is my dog Max?"
- "And the second one?" after listing events → "Tell me about the second event on my calendar"
- "Add it" after discussing concert → "Add the Taylor Swift concert to my calendar"
- "It's spelled C-A-R-E-M-E-T-X" → "The company is named CareMetx"

needs_retrieval RULES:
- FALSE for: acknowledgments ("yes", "no", "thanks"), greetings, direct follow-ups answerable from history
- TRUE for: questions needing knowledge/memory search, calendar queries, web searches

intent VALUES: calendar, memory, home_control, web_search, system_command, general_knowledge, clarification, acknowledgment, greeting

direct_answer: Only if you can FULLY answer from history alone. Otherwise empty.

Output ONLY valid XML, no other text."""


class QueryContextualizer:
    """Service for contextualizing queries using conversation history.

    This runs BEFORE RAG retrieval to ensure ambiguous follow-up queries
    are transformed into standalone queries that can be effectively searched.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: ContextualizerConfig | None = None,
        prompt_template: str | None = None,
    ):
        """Initialize contextualizer.

        Args:
            llm_provider: LLM provider for contextualization
            config: Configuration options
            prompt_template: Custom prompt template (must have {history} and {query} placeholders)
        """
        self.llm = llm_provider
        self.config = config or ContextualizerConfig()
        self.prompt_template = prompt_template or DEFAULT_CONTEXTUALIZE_PROMPT

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "(No previous conversation)"

        parts = []
        for turn in history[-self.config.max_history_turns:]:
            parts.append(f"User: {turn.get('user', '')}")
            parts.append(f"Assistant: {turn.get('assistant', '')}")

            # Include tool results for context
            tool_results = turn.get("tool_results", [])
            for tr in tool_results:
                tool_name = tr.get("tool", "unknown")
                result = tr.get("result")
                if result:
                    result_str = str(result)
                    if len(result_str) > self.config.max_result_length:
                        result_str = result_str[:self.config.max_result_length] + "..."
                    parts.append(f"  [{tool_name} result: {result_str}]")

        return "\n".join(parts)

    async def contextualize(
        self,
        query: str,
        history: list[dict[str, Any]],
    ) -> ContextualizedQuery:
        """Contextualize a query using conversation history.

        Args:
            query: The current user query
            history: List of conversation turns with user/assistant/tool_results

        Returns:
            ContextualizedQuery with standalone query and metadata
        """
        start_time = time.time()

        # If no history, the query is already standalone
        if not history:
            return ContextualizedQuery(
                original_query=query,
                standalone_query=query,
                needs_retrieval=True,
                intent="general_knowledge",
                direct_answer=None,
                confidence=1.0,
                processing_time_ms=0,
            )

        # Format history for prompt
        history_str = self._format_history(history)

        # Build prompt
        prompt = self.prompt_template.format(
            history=history_str,
            query=query,
        )

        # Call LLM
        try:
            messages = [
                {"role": "system", "content": "You are a query preprocessing assistant. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ]

            result = await self.llm.chat(
                messages=messages,
                max_tokens=300,
            )

            processing_time = int((time.time() - start_time) * 1000)

            if not result:
                logger.warning("Contextualization failed: no result from LLM")
                return self._fallback_result(query, processing_time)

            # Extract content
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            try:
                parsed = json.loads(content)
                logger.info(
                    f"Contextualization: '{query}' -> '{parsed.get('standalone_query', query)}'"
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse contextualization JSON: {e}")
                return self._fallback_result(query, processing_time)

            return ContextualizedQuery(
                original_query=query,
                standalone_query=parsed.get("standalone_query", query),
                needs_retrieval=parsed.get("needs_retrieval", True),
                intent=parsed.get("intent", "general_knowledge"),
                direct_answer=parsed.get("direct_answer"),
                confidence=parsed.get("confidence", 0.8),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Contextualization error: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            return self._fallback_result(query, processing_time)

    def _fallback_result(self, query: str, processing_time_ms: int) -> ContextualizedQuery:
        """Return a safe fallback when contextualization fails."""
        return ContextualizedQuery(
            original_query=query,
            standalone_query=query,
            needs_retrieval=True,
            intent="general_knowledge",
            direct_answer=None,
            confidence=0.5,
            processing_time_ms=processing_time_ms,
        )

    def should_contextualize(self, query: str, history: list[dict[str, Any]]) -> bool:
        """Quick check if contextualization is likely needed.

        This is a fast pre-filter to avoid LLM calls for obviously standalone queries.
        Returns True if contextualization might help.
        """
        # No history = no context to add
        if not history:
            return False

        query_lower = query.lower().strip()

        # Identity questions should NOT be contextualized
        if any(pattern in query_lower for pattern in self.config.identity_patterns):
            return False

        # Short queries often need context
        if len(query.split()) <= 3:
            return True

        # Pronouns and references need context
        if any(indicator in query_lower for indicator in self.config.context_indicators):
            return True

        # Short affirmations/negations
        if query_lower in self.config.affirmations:
            return True

        # Spelled-out words need contextualization
        spelled_pattern = re.compile(r'\b[A-Za-z](?:[-\s][A-Za-z]){2,}\b')
        if spelled_pattern.search(query):
            return True

        return False

"""Reflection service for post-interaction quality evaluation.

After every interaction, this service uses an LLM to evaluate:
- Quality of the response (1-5)
- Any issues that occurred
- Root cause analysis
- Suggested fixes
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from .models import (
    DiscoveredIssue,
    IssueSeverity,
    IssueType,
    ReflectionResult,
)
from .protocols import LLMProvider

logger = logging.getLogger(__name__)


# Default reflection prompt
DEFAULT_REFLECTION_PROMPT = """You are reviewing an AI assistant conversation to identify quality issues.

## Full Conversation History

{conversation_history}

## Most Recent Interaction

**User Query:** {query}
**Action Taken:** {action}
**Tools Called:** {tool_calls}
**Response:** {response}

## Evaluation Task

Review the ENTIRE conversation, paying special attention to:
1. **User Feedback in History**: If the user's current or previous messages indicate dissatisfaction, correction, or had to repeat/rephrase - this is CRITICAL feedback (ground truth!)
2. **Response Quality**: Was the response accurate, helpful, and appropriate?
3. **Conversation Flow**: Did the assistant maintain context and understand references?

### User Feedback Signals (in conversation history)
Look for these patterns that indicate the PREVIOUS response had issues:
- Direct corrections: "No, I meant...", "That's wrong", "Actually..."
- Rephrasing: User asks the same thing differently
- Frustration: "I already told you", "That's not what I asked"
- Clarification needed: "What?", "I don't understand"
- Positive signals: "Thanks", "Perfect", "Great" (indicates good quality)

**IMPORTANT**: User feedback is ground truth. If the user says something was wrong, it was wrong.

### Quality Score (1-5)
- 5 = Perfect - exactly what user needed, natural response, no corrections needed
- 4 = Good - correct and helpful, user satisfied
- 3 = Acceptable - got the job done but awkward or incomplete
- 2 = Poor - partially wrong, confusing, or user had to correct/repeat
- 1 = Failed - wrong answer, wrong action, or user frustrated

### Issue Categories
- **Misunderstood intent**: Interpreted the query wrong
- **Wrong tool choice**: Used wrong tool or missed a better option
- **Incomplete response**: Left out important information
- **Incorrect information**: Factually wrong
- **Poor phrasing**: Too long, hard to understand
- **Missing context**: Didn't use conversation history properly
- **Unnecessary action**: Did more than asked
- **Required correction**: User had to correct or repeat

### Severity Levels
- CRITICAL: Wrong or harmful action taken
- HIGH: User had to correct, repeat, or showed frustration
- MEDIUM: Suboptimal but achieved the goal
- LOW: Minor polish issue

## Output

Return XML:
<reflection>
  <quality_score>1-5</quality_score>
  <no_issues>true if score >= 4 and conversation shows user satisfaction</no_issues>
  <user_sentiment>positive|neutral|negative|frustrated</user_sentiment>
  <issues>
    <issue>
      <description>clear description of the problem</description>
      <prompt_blamed>which prompt caused this, or empty if unsure</prompt_blamed>
      <root_cause>why this happened</root_cause>
      <severity>CRITICAL|HIGH|MEDIUM|LOW</severity>
      <suggested_fix>specific fix</suggested_fix>
    </issue>
  </issues>
</reflection>

If the conversation shows user satisfaction (positive feedback, no corrections), return empty issues element.
Be specific - these will be used to improve the assistant."""


# Classification prompt
DEFAULT_CLASSIFICATION_PROMPT = """Classify what type of fix is needed for this issue:

## Issue
**Description:** {description}
**Root Cause:** {root_cause}
**Suggested Fix:** {suggested_fix}
**Prompt Blamed:** {prompt_blamed}

## Categories

- **PROMPT**: Can be fixed by changing LLM instructions/prompts
- **KNOWLEDGE**: Missing factual information
- **TOOL**: Need to modify existing tool or add new tool capability
- **BUG**: Code is broken and needs fixing
- **FEATURE**: Need entirely new capability
- **EXTERNAL**: Issue outside the assistant's control

## Output

Return XML:
<classification>
  <issue_type>PROMPT|KNOWLEDGE|TOOL|BUG|FEATURE|EXTERNAL</issue_type>
  <fixable_by_agent>true if PROMPT or KNOWLEDGE, false otherwise</fixable_by_agent>
  <reasoning>why this classification</reasoning>
  <suggested_approach>for non-prompt issues, how to fix this</suggested_approach>
  <files_involved>
    <file>likely file if code change needed</file>
  </files_involved>
</classification>"""


@dataclass
class ReflectionConfig:
    """Configuration for reflection service.

    Attributes:
        max_history_turns: Maximum conversation turns to include
        max_response_length: Maximum response length in prompt
        max_context_length: Maximum context length for issue
        use_fast_model: Whether to use a faster model for reflection
        reflection_prompt: Custom reflection prompt template
        classification_prompt: Custom classification prompt template
        prompts_that_can_be_blamed: List of prompt names that can be blamed
    """

    max_history_turns: int = 5
    max_response_length: int = 1000
    max_context_length: int = 500
    use_fast_model: bool = True
    reflection_prompt: str | None = None
    classification_prompt: str | None = None
    prompts_that_can_be_blamed: list[str] = field(default_factory=lambda: [
        "DECISION_PROMPT",
        "SYNTHESIS_PROMPT",
        "FAST_ROUTE_PROMPT",
    ])


class ReflectionService:
    """Evaluates interaction quality and identifies issues."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        config: ReflectionConfig | None = None,
    ):
        """Initialize reflection service.

        Args:
            llm_provider: LLM provider for evaluation
            config: Configuration options
        """
        self.llm = llm_provider
        self.config = config or ReflectionConfig()
        self.reflection_prompt = self.config.reflection_prompt or DEFAULT_REFLECTION_PROMPT
        self.classification_prompt = self.config.classification_prompt or DEFAULT_CLASSIFICATION_PROMPT

    async def reflect(
        self,
        interaction_id: str,
        query: str,
        response: str,
        action: str,
        tool_calls: list[str] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        context: str = "",
    ) -> ReflectionResult:
        """Reflect on an interaction to evaluate quality and find issues.

        Args:
            interaction_id: ID of the interaction
            query: User's original query
            response: Assistant's response
            action: Action taken (answer, tool_call, etc.)
            tool_calls: List of tools called
            conversation_history: Full conversation history
            context: Optional additional context

        Returns:
            ReflectionResult with quality score and any issues found
        """
        if not self.llm:
            logger.warning("No LLM available for reflection")
            return ReflectionResult(
                quality_score=3,
                issues=[],
                no_issues=True,
                interaction_id=interaction_id,
            )

        tool_calls = tool_calls or []
        conversation_history = conversation_history or []

        # Format conversation history
        history_text = self._format_conversation_history(conversation_history)

        # Build the reflection prompt
        prompt = self.reflection_prompt.format(
            conversation_history=history_text,
            query=query,
            action=action,
            tool_calls=", ".join(tool_calls) if tool_calls else "None",
            response=response[:self.config.max_response_length],
        )

        try:
            # Call LLM for reflection
            if hasattr(self.llm, "chat_json"):
                result = await self.llm.chat_json(
                    messages=[
                        {"role": "system", "content": "You are a quality assurance reviewer."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                )
                parsed = result.get("parsed") if result else None
            else:
                result = await self.llm.chat(
                    messages=[
                        {"role": "system", "content": "You are a quality assurance reviewer. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                )
                content = result.get("content", "") if result else ""
                try:
                    # Handle markdown code blocks
                    if content.strip().startswith("```"):
                        lines = content.strip().split("\n")
                        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None

            if not parsed:
                logger.warning("Failed to parse reflection result")
                return ReflectionResult(
                    quality_score=3,
                    issues=[],
                    no_issues=True,
                    interaction_id=interaction_id,
                )

            quality_score = parsed.get("quality_score", 3)
            no_issues = parsed.get("no_issues", True)
            raw_issues = parsed.get("issues", [])
            user_sentiment = parsed.get("user_sentiment", "neutral")

            # Convert raw issues to DiscoveredIssue objects
            issues = []
            for raw in raw_issues:
                if not raw.get("description"):
                    continue

                issue = DiscoveredIssue(
                    id=str(uuid4()),
                    timestamp=datetime.now(),
                    interaction_id=interaction_id,
                    query=query,
                    response=response,
                    action_taken=action,
                    tool_calls=tool_calls,
                    description=raw.get("description", "Unknown issue"),
                    prompt_blamed=raw.get("prompt_blamed"),
                    root_cause=raw.get("root_cause", "Unknown"),
                    severity=self._parse_severity(raw.get("severity", "MEDIUM")),
                    suggested_fix=raw.get("suggested_fix", ""),
                    conversation_context=history_text[:self.config.max_context_length] if history_text else None,
                    user_sentiment=user_sentiment,
                )

                # Classify the issue type
                issue = await self._classify_issue(issue)
                issues.append(issue)

            logger.info(
                f"Reflection complete: quality={quality_score}, issues={len(issues)}"
            )

            return ReflectionResult(
                quality_score=quality_score,
                issues=issues,
                no_issues=no_issues and len(issues) == 0,
                interaction_id=interaction_id,
                user_sentiment=user_sentiment,
            )

        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            return ReflectionResult(
                quality_score=3,
                issues=[],
                no_issues=True,
                interaction_id=interaction_id,
            )

    async def _classify_issue(self, issue: DiscoveredIssue) -> DiscoveredIssue:
        """Classify what type of fix is needed for an issue."""
        if not self.llm:
            return issue

        prompt = self.classification_prompt.format(
            description=issue.description,
            root_cause=issue.root_cause,
            suggested_fix=issue.suggested_fix,
            prompt_blamed=issue.prompt_blamed or "Unknown",
        )

        try:
            if hasattr(self.llm, "chat_json"):
                result = await self.llm.chat_json(
                    messages=[
                        {"role": "system", "content": "You are classifying issues for an AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                )
                parsed = result.get("parsed") if result else None
            else:
                result = await self.llm.chat(
                    messages=[
                        {"role": "system", "content": "You are classifying issues. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                )
                content = result.get("content", "") if result else ""
                try:
                    if content.strip().startswith("```"):
                        lines = content.strip().split("\n")
                        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None

            if parsed:
                issue.issue_type = self._parse_issue_type(parsed.get("issue_type", "PROMPT"))
                issue.fixable_by_agent = parsed.get("fixable_by_agent", True)
                issue.suggested_approach = parsed.get("suggested_approach")
                issue.files_involved = parsed.get("files_involved", [])

        except Exception as e:
            logger.warning(f"Error classifying issue: {e}")

        return issue

    def _format_conversation_history(
        self,
        history: list[dict[str, Any]],
    ) -> str:
        """Format conversation history for the reflection prompt."""
        if not history:
            return "(No previous conversation history)"

        max_turns = self.config.max_history_turns
        recent = history[-max_turns:] if len(history) > max_turns else history

        lines = []
        for i, turn in enumerate(recent, 1):
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")

            # Truncate long messages
            if len(user_msg) > 300:
                user_msg = user_msg[:300] + "..."
            if len(assistant_msg) > 300:
                assistant_msg = assistant_msg[:300] + "..."

            lines.append(f"**Turn {i}:**")
            lines.append(f"  User: {user_msg}")
            lines.append(f"  Assistant: {assistant_msg}")
            lines.append("")

        return "\n".join(lines)

    def _parse_severity(self, severity_input: str | dict | None) -> IssueSeverity:
        """Parse severity string/dict to enum."""
        try:
            if isinstance(severity_input, dict):
                severity_input = severity_input.get("value", "MEDIUM")
            if severity_input is None:
                return IssueSeverity.MEDIUM
            return IssueSeverity(str(severity_input).lower())
        except (ValueError, AttributeError):
            return IssueSeverity.MEDIUM

    def _parse_issue_type(self, type_input: str | dict | None) -> IssueType:
        """Parse issue type string/dict to enum."""
        try:
            if isinstance(type_input, dict):
                type_input = type_input.get("value", "PROMPT")
            if type_input is None:
                return IssueType.PROMPT
            return IssueType(str(type_input).lower())
        except (ValueError, AttributeError):
            return IssueType.PROMPT

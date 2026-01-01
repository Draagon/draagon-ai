"""Decision engine for determining what action to take.

The decision engine is responsible for:
1. Building the decision prompt from behavior and context
2. Calling the LLM to make a decision
3. Parsing the decision output
4. Selecting the appropriate action
5. Validating action is in behavior's action list

REQ-002-02: DecisionEngine Integration
- Selects appropriate tool for query
- Extracts tool arguments correctly
- Returns confidence score with decision
- Falls back to "no action" when appropriate
- Supports all behavior-defined tools
"""

from dataclasses import dataclass, field
from typing import Any
import logging
import re
import xml.etree.ElementTree as ET

from ..behaviors import Action, Behavior
from .protocols import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Result of a decision made by the engine.

    REQ-002-02: Enhanced with validation and confidence tracking.
    """

    # Primary decision
    action: str  # Action name to execute
    reasoning: str = ""

    # Action-specific data
    args: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None  # Direct answer (when action=answer)

    # Metadata
    model_tier: str = "local"  # local, complex, deep
    confidence: float = 1.0
    additional_actions: list[str] = field(default_factory=list)

    # Memory operations
    memory_update: dict[str, Any] | None = None

    # Validation status (REQ-002-02)
    is_valid_action: bool = True  # Whether action is in behavior's action list
    original_action: str | None = None  # Pre-validation action (if remapped)
    validation_notes: str = ""  # Why action was remapped or rejected

    # Raw response for debugging
    raw_response: str = ""

    def is_final_answer(self) -> bool:
        """Check if this decision is a final answer.

        Returns:
            True if action is "answer" and has an answer value
        """
        return self.action == "answer" and self.answer is not None

    def is_no_action(self) -> bool:
        """Check if this decision is a no-action fallback.

        Returns:
            True if action is "no_action" or "none"
        """
        return self.action in ("no_action", "none", "")


# Common action aliases that should map to canonical names
ACTION_ALIASES: dict[str, str] = {
    "respond": "answer",
    "reply": "answer",
    "say": "answer",
    "search": "search_web",
    "web_search": "search_web",
    "lookup": "search_web",
    "find": "search_web",
    "no_action": "answer",
    "none": "answer",
}


class DecisionEngine:
    """Engine for making action decisions based on behavior and context.

    The decision engine takes a behavior's decision prompt template,
    fills it with context, and uses an LLM to decide what action to take.

    REQ-002-02: Enhanced with action validation, alias resolution, and
    confidence tracking.
    """

    def __init__(
        self,
        llm: LLMProvider,
        default_model_tier: str = "local",
        validate_actions: bool = True,
        fallback_to_answer: bool = True,
        action_aliases: dict[str, str] | None = None,
    ):
        """Initialize the decision engine.

        Args:
            llm: LLM provider for making decisions
            default_model_tier: Default model tier to use
            validate_actions: Whether to validate actions against behavior
            fallback_to_answer: Whether to fallback to "answer" for invalid actions
            action_aliases: Custom action aliases (merged with defaults)
        """
        self.llm = llm
        self.default_model_tier = default_model_tier
        self.validate_actions = validate_actions
        self.fallback_to_answer = fallback_to_answer

        # Merge custom aliases with defaults
        self.action_aliases = ACTION_ALIASES.copy()
        if action_aliases:
            self.action_aliases.update(action_aliases)

    async def decide(
        self,
        behavior: Behavior,
        query: str,
        context: "DecisionContext",
    ) -> DecisionResult:
        """Make a decision about what action to take.

        Args:
            behavior: The behavior providing available actions
            query: User's query
            context: Context for the decision

        Returns:
            DecisionResult with the selected action and arguments
        """
        # Build the decision prompt
        prompt = self._build_decision_prompt(behavior, query, context)

        # Get LLM decision
        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=query),
        ]

        response = await self.llm.chat(
            messages=messages,
            model=self._get_model_for_tier(self.default_model_tier),
        )

        # Parse the response
        result = self._parse_decision(response, behavior, query)
        result.raw_response = response.content

        # Validate and normalize the action (REQ-002-02)
        if self.validate_actions:
            result = self._validate_action(result, behavior)

        return result

    def _validate_action(
        self,
        result: DecisionResult,
        behavior: Behavior,
    ) -> DecisionResult:
        """Validate that the action exists in the behavior and normalize aliases.

        REQ-002-02: Ensures selected action is valid for the behavior.

        Args:
            result: The parsed decision result
            behavior: The behavior with valid actions

        Returns:
            DecisionResult with validated/normalized action
        """
        original_action = result.action
        action = original_action.lower().strip()

        # Get valid action names from behavior
        valid_actions = {a.name.lower() for a in behavior.actions}

        # Check for alias resolution
        if action in self.action_aliases:
            canonical = self.action_aliases[action]
            if canonical.lower() in valid_actions:
                result.original_action = original_action
                result.action = canonical
                result.validation_notes = f"Alias '{original_action}' → '{canonical}'"
                logger.debug(f"Resolved action alias: {original_action} → {canonical}")
                return result

        # Check if action is valid as-is
        if action in valid_actions:
            result.is_valid_action = True
            return result

        # Action is not valid - handle fallback
        result.is_valid_action = False
        result.original_action = original_action

        if self.fallback_to_answer:
            # Fall back to answer action if available
            if "answer" in valid_actions:
                result.action = "answer"
                result.validation_notes = (
                    f"Unknown action '{original_action}' → fallback to 'answer'"
                )
                logger.warning(
                    f"Unknown action '{original_action}' not in behavior actions, "
                    f"falling back to 'answer'"
                )
                # If we have reasoning but no answer, use reasoning as answer
                if not result.answer and result.reasoning:
                    result.answer = result.reasoning
            else:
                result.validation_notes = (
                    f"Unknown action '{original_action}' and no 'answer' fallback"
                )
                logger.warning(
                    f"Unknown action '{original_action}' and no 'answer' action "
                    f"available in behavior"
                )
        else:
            result.validation_notes = f"Unknown action '{original_action}'"
            logger.warning(f"Unknown action '{original_action}' (no fallback enabled)")

        return result

    def get_valid_actions(self, behavior: Behavior) -> list[str]:
        """Get list of valid action names for a behavior.

        Args:
            behavior: The behavior to get actions from

        Returns:
            List of valid action names
        """
        return [a.name for a in behavior.actions]

    def _build_decision_prompt(
        self,
        behavior: Behavior,
        query: str,
        context: "DecisionContext",
    ) -> str:
        """Build the decision prompt from behavior template and context.

        Args:
            behavior: The behavior with the decision prompt template
            query: User's query
            context: Context for substitution

        Returns:
            Completed decision prompt
        """
        if not behavior.prompts:
            raise ValueError(f"Behavior {behavior.behavior_id} has no prompts defined")

        template = behavior.prompts.decision_prompt

        # Build available actions section
        actions_text = self._format_actions(behavior.actions)

        # Substitute placeholders
        prompt = template.format(
            assistant_intro=context.assistant_intro,
            question=query,
            user_id=context.user_id,
            conversation_history=context.conversation_history,
            context=context.gathered_context,
            pending_details=context.pending_details or "None",
        )

        return prompt

    def _format_actions(self, actions: list[Action]) -> str:
        """Format actions for the prompt.

        Args:
            actions: List of available actions

        Returns:
            Formatted string describing available actions
        """
        lines = []
        for action in actions:
            desc = f"- {action.name}: {action.description}"
            if action.parameters:
                params = ", ".join(
                    f"{name}={p.type}" for name, p in action.parameters.items()
                )
                desc += f" ({params})"
            lines.append(desc)
        return "\n".join(lines)

    def _parse_decision(
        self,
        response: LLMResponse,
        behavior: Behavior,
        query: str = "",
    ) -> DecisionResult:
        """Parse the LLM response into a DecisionResult.

        Args:
            response: LLM response
            behavior: Behavior for validation
            query: Original user query (for fallback action inference)

        Returns:
            Parsed DecisionResult
        """
        # Handle None content defensively (can happen with tool-only responses)
        raw_content = response.content if response.content is not None else ""
        content = raw_content.strip()
        logger.debug(f"Raw LLM decision content: {content[:500]}...")

        # Try XML parsing first (our default format)
        if "<response>" in content:
            result = self._parse_xml_decision(content, behavior)
            logger.debug(f"Parsed XML decision: action={result.action}, args={result.args}")
            if result.memory_update:
                logger.info(f"Memory update found: {result.memory_update}")
            return result

        # Fallback to JSON parsing
        if content.startswith("{"):
            result = self._parse_json_decision(content, behavior)
            return result

        # If neither, try to extract action from plain text
        return self._parse_text_decision(content, behavior, query)

    def _parse_xml_decision(self, content: str, behavior: Behavior) -> DecisionResult:
        """Parse XML-formatted decision.

        Args:
            content: XML content
            behavior: Behavior for validation

        Returns:
            Parsed DecisionResult
        """
        result = DecisionResult(action="answer")

        try:
            # Extract the response element
            match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
            if not match:
                result.answer = content
                return result

            xml_content = f"<response>{match.group(1)}</response>"
            root = ET.fromstring(xml_content)

            # Extract basic fields
            action_elem = root.find("action")
            result.action = action_elem.text.strip() if action_elem is not None else "answer"

            reasoning_elem = root.find("reasoning")
            result.reasoning = reasoning_elem.text.strip() if reasoning_elem is not None else ""

            answer_elem = root.find("answer")
            result.answer = answer_elem.text.strip() if answer_elem is not None else None

            model_tier_elem = root.find("model_tier")
            result.model_tier = (
                model_tier_elem.text.strip() if model_tier_elem is not None else "local"
            )

            # Extract confidence if present (REQ-002-02)
            confidence_elem = root.find("confidence")
            if confidence_elem is not None and confidence_elem.text:
                try:
                    result.confidence = float(confidence_elem.text.strip())
                    # Clamp to 0-1 range
                    result.confidence = max(0.0, min(1.0, result.confidence))
                except ValueError:
                    result.confidence = 1.0  # Default if parsing fails

            # Extract action-specific args
            result.args = self._extract_action_args(root, result.action)

            # Extract additional actions
            additional_elem = root.find("additional_actions")
            if additional_elem is not None and additional_elem.text:
                result.additional_actions = [
                    a.strip() for a in additional_elem.text.split(",") if a.strip()
                ]

            # Extract memory update
            memory_elem = root.find("memory_update")
            if memory_elem is not None:
                result.memory_update = self._parse_memory_update(memory_elem)

        except ET.ParseError:
            # Fall back to answer with content
            result.answer = content
            result.action = "answer"

        return result

    def _extract_action_args(
        self,
        root: ET.Element,
        action: str,
    ) -> dict[str, Any]:
        """Extract action-specific arguments from XML.

        Args:
            root: XML root element
            action: Action name

        Returns:
            Dictionary of arguments
        """
        args: dict[str, Any] = {}

        # Search query
        query_elem = root.find("query")
        if query_elem is not None and query_elem.text:
            args["query"] = query_elem.text.strip()

        # Calendar event fields (structured)
        summary_elem = root.find("summary")
        if summary_elem is not None and summary_elem.text:
            args["summary"] = summary_elem.text.strip()

        start_elem = root.find("start")
        if start_elem is not None and start_elem.text:
            args["start"] = start_elem.text.strip()

        end_elem = root.find("end")
        if end_elem is not None and end_elem.text:
            args["end"] = end_elem.text.strip()

        location_elem = root.find("location")
        if location_elem is not None and location_elem.text:
            args["location"] = location_elem.text.strip()

        event_id_elem = root.find("event_id")
        if event_id_elem is not None and event_id_elem.text:
            args["event_id"] = event_id_elem.text.strip()

        # Legacy event details (backwards compatibility)
        event_elem = root.find("event")
        if event_elem is not None and event_elem.text:
            args["event"] = event_elem.text.strip()

        # Home Assistant specific
        ha_domain = root.find("ha_domain")
        if ha_domain is not None and ha_domain.text:
            args["domain"] = ha_domain.text.strip()

        ha_service = root.find("ha_service")
        if ha_service is not None and ha_service.text:
            args["service"] = ha_service.text.strip()

        ha_entity = root.find("ha_entity")
        if ha_entity is not None and ha_entity.text:
            args["entity"] = ha_entity.text.strip()

        ha_brightness = root.find("ha_brightness")
        if ha_brightness is not None and ha_brightness.text:
            try:
                args["brightness"] = int(ha_brightness.text.strip())
            except ValueError:
                pass

        ha_color = root.find("ha_color")
        if ha_color is not None and ha_color.text:
            args["color"] = ha_color.text.strip()

        # Sensor query filter (for querying HA sensors)
        ha_query = root.find("ha_query")
        if ha_query is not None and ha_query.text:
            args["query"] = ha_query.text.strip()

        # Entity query (for get_entity action)
        entity_id_elem = root.find("entity_id")
        if entity_id_elem is not None and entity_id_elem.text:
            args["entity_id"] = entity_id_elem.text.strip()

        # Entity search filter (for search_entities action)
        filter_elem = root.find("filter")
        if filter_elem is not None and filter_elem.text:
            args["filter"] = filter_elem.text.strip()

        # Code-related
        code_query = root.find("code_query")
        if code_query is not None and code_query.text:
            args["code_query"] = code_query.text.strip()

        code_file = root.find("code_file")
        if code_file is not None and code_file.text:
            args["code_file"] = code_file.text.strip()

        # Timer fields
        minutes_elem = root.find("minutes")
        if minutes_elem is not None and minutes_elem.text:
            try:
                args["minutes"] = float(minutes_elem.text.strip())
            except ValueError:
                pass

        # Fallback: extract minutes from query text if action is set_timer and no minutes specified
        if action == "set_timer" and "minutes" not in args:
            query_text = args.get("query", "")
            answer_text = root.find("answer")
            if answer_text is not None and answer_text.text:
                query_text = answer_text.text + " " + query_text

            # Try to extract duration like "5 minutes", "30 seconds", "1 hour"
            import re
            # Match patterns like "5 minutes", "30 seconds", "1 hour", "2 hours"
            duration_match = re.search(
                r'(\d+(?:\.\d+)?)\s*(minute|min|second|sec|hour|hr)s?',
                query_text,
                re.IGNORECASE
            )
            if duration_match:
                value = float(duration_match.group(1))
                unit = duration_match.group(2).lower()
                if unit in ("second", "sec"):
                    args["minutes"] = value / 60
                elif unit in ("hour", "hr"):
                    args["minutes"] = value * 60
                else:  # minutes
                    args["minutes"] = value

        label_elem = root.find("label")
        if label_elem is not None and label_elem.text:
            args["label"] = label_elem.text.strip()

        timer_id_elem = root.find("timer_id")
        if timer_id_elem is not None and timer_id_elem.text:
            args["timer_id"] = timer_id_elem.text.strip()

        # Command execution fields
        command_elem = root.find("command")
        if command_elem is not None and command_elem.text:
            args["command"] = command_elem.text.strip()

        host_elem = root.find("host")
        if host_elem is not None and host_elem.text:
            args["host"] = host_elem.text.strip()

        return args

    def _parse_memory_update(self, elem: ET.Element) -> dict[str, Any]:
        """Parse a memory update element.

        Args:
            elem: memory_update XML element

        Returns:
            Dictionary with memory update details
        """
        update: dict[str, Any] = {
            "action": elem.get("action", "store"),
        }

        content_elem = elem.find("content")
        if content_elem is not None and content_elem.text:
            update["content"] = content_elem.text.strip()

        type_elem = elem.find("type")
        if type_elem is not None and type_elem.text:
            update["type"] = type_elem.text.strip()

        confidence_elem = elem.find("confidence")
        if confidence_elem is not None and confidence_elem.text:
            try:
                update["confidence"] = float(confidence_elem.text.strip())
            except ValueError:
                update["confidence"] = 0.9

        entities_elem = elem.find("entities")
        if entities_elem is not None and entities_elem.text:
            update["entities"] = [e.strip() for e in entities_elem.text.split(",")]

        return update

    def _parse_json_decision(self, content: str, behavior: Behavior) -> DecisionResult:
        """Parse JSON-formatted decision.

        Args:
            content: JSON content
            behavior: Behavior for validation

        Returns:
            Parsed DecisionResult
        """
        import json

        result = DecisionResult(action="answer")

        try:
            data = json.loads(content)
            result.action = data.get("action", "answer")
            result.reasoning = data.get("reasoning", "")
            result.answer = data.get("answer")
            result.args = data.get("args", {})
            result.model_tier = data.get("model_tier", "local")

            # Extract confidence if present (REQ-002-02)
            if "confidence" in data:
                try:
                    result.confidence = float(data["confidence"])
                    result.confidence = max(0.0, min(1.0, result.confidence))
                except (ValueError, TypeError):
                    result.confidence = 1.0

            # Extract additional actions if present
            if "additional_actions" in data:
                result.additional_actions = data["additional_actions"]

            # Extract memory update if present
            if "memory_update" in data:
                result.memory_update = data["memory_update"]

        except json.JSONDecodeError:
            result.answer = content

        return result

    def _parse_text_decision(
        self,
        content: str,
        behavior: Behavior,
        query: str = "",
    ) -> DecisionResult:
        """Parse plain text decision (fallback).

        REQ-002-02: Returns lower confidence since this is a fallback parser.

        The parsing strategy:
        1. First, try to find tool-specific action keywords (get_time, get_weather, etc.)
           These are more specific than "answer" and indicate the LLM wants to use a tool.
        2. Try to extract JSON from the content (may be embedded in markdown)
        3. Look for action patterns like "action: X" or "Action: X"
        4. Check the original query to infer the required action
        5. Default to "answer" only if no tool action is found

        Args:
            content: Plain text content
            behavior: Behavior for validation
            query: Original user query (for action inference)

        Returns:
            Parsed DecisionResult with lower confidence
        """
        content_lower = content.lower()
        query_lower = query.lower() if query else ""

        # First, try to extract embedded JSON (LLM often outputs markdown with JSON)
        json_match = re.search(r'\{[^{}]*"(?:action|answer)"[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                import json
                data = json.loads(json_match.group(0))
                return DecisionResult(
                    action=data.get("action", "answer"),
                    answer=data.get("answer"),
                    args=data.get("args", {}),
                    confidence=0.6,
                    reasoning="Parsed from embedded JSON",
                    validation_notes="Extracted JSON from text",
                )
            except json.JSONDecodeError:
                pass

        # Tool actions to prioritize (NOT "answer" - that's too common)
        # Order matters: check more specific actions first
        # NOTE: These names must match the actual tool names registered with the agent
        tool_actions = [
            "get_time", "get_weather", "get_location",
            "search_web", "home_assistant", "get_calendar_events",
            "create_calendar_event", "delete_calendar_event", "execute_command",
            "use_tools", "form_opinion", "more_details", "clarify",
        ]

        # Look for explicit action patterns first
        action_patterns = [
            r'<action>\s*(\w+)\s*</action>',  # XML-like
            r'"action"\s*:\s*"(\w+)"',  # JSON-like
            r'action\s*[:=]\s*["\']?(\w+)',  # Key-value like
            r'I (?:will|should|need to) use (\w+)',  # Natural language
            r'using (?:the )?(\w+) (?:action|tool)',  # Natural language
        ]

        for pattern in action_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                action_name = match.group(1).lower()
                # Check if it's a valid action
                valid_action_names = {a.name.lower() for a in behavior.actions}
                if action_name in valid_action_names:
                    return DecisionResult(
                        action=action_name,
                        answer=content,
                        confidence=0.7,
                        reasoning=f"Found explicit action pattern: {action_name}",
                        validation_notes="Matched action pattern in text",
                    )

        # Check for tool action keywords in content (exact keyword matching, not semantic)
        # This is acceptable because we're matching exact tool names the LLM mentioned,
        # not trying to infer intent from user language
        valid_action_names = {a.name.lower() for a in behavior.actions}
        for tool_action in tool_actions:
            # Use word boundaries to avoid partial matches
            if re.search(rf'\b{tool_action}\b', content_lower):
                # Verify this action exists in behavior
                if tool_action in valid_action_names:
                    return DecisionResult(
                        action=tool_action,
                        answer=content,
                        confidence=0.5,
                        reasoning=f"Detected tool action keyword: {tool_action}",
                        validation_notes="Matched tool action keyword in LLM response",
                    )

        # NOTE: Semantic intent detection (time/weather/calendar patterns) has been
        # intentionally removed. Using regex to infer user intent violates the
        # LLM-first architecture principle. The LLM decision prompt should determine
        # the action; if it produces unstructured output, we fall back to "answer".
        # See CLAUDE.md: "NEVER use regex or keyword patterns for semantic understanding"

        # Default fallback: answer action
        # Lower confidence because we couldn't determine the action
        return DecisionResult(
            action="answer",
            answer=content,
            confidence=0.3,
            reasoning="Parsed from plain text (fallback)",
            validation_notes="Used text fallback parser - no action pattern matched",
        )

    def _get_model_for_tier(self, tier: str) -> str | None:
        """Get the model name for a tier.

        This is a placeholder - actual mapping is provider-specific.

        Args:
            tier: Model tier (local, complex, deep)

        Returns:
            Model name or None for provider default
        """
        # Return None to use provider's default
        # Applications should override this in their LLM provider
        return None


@dataclass
class DecisionContext:
    """Context provided to the decision engine."""

    # User info
    user_id: str

    # Personality/identity
    assistant_intro: str = ""

    # Conversation state
    conversation_history: str = ""
    pending_details: str | None = None

    # Gathered context (memories, knowledge, etc.)
    gathered_context: str = ""

    # Room/area context for smart home
    area_id: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

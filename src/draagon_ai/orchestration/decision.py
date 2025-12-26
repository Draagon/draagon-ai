"""Decision engine for determining what action to take.

The decision engine is responsible for:
1. Building the decision prompt from behavior and context
2. Calling the LLM to make a decision
3. Parsing the decision output
4. Selecting the appropriate action
"""

from dataclasses import dataclass, field
from typing import Any
import re
import xml.etree.ElementTree as ET

from ..behaviors import Action, Behavior
from .protocols import LLMMessage, LLMProvider, LLMResponse


@dataclass
class DecisionResult:
    """Result of a decision made by the engine."""

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

    # Raw response for debugging
    raw_response: str = ""


class DecisionEngine:
    """Engine for making action decisions based on behavior and context.

    The decision engine takes a behavior's decision prompt template,
    fills it with context, and uses an LLM to decide what action to take.
    """

    def __init__(
        self,
        llm: LLMProvider,
        default_model_tier: str = "local",
    ):
        """Initialize the decision engine.

        Args:
            llm: LLM provider for making decisions
            default_model_tier: Default model tier to use
        """
        self.llm = llm
        self.default_model_tier = default_model_tier

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
        result = self._parse_decision(response, behavior)
        result.raw_response = response.content

        return result

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
    ) -> DecisionResult:
        """Parse the LLM response into a DecisionResult.

        Args:
            response: LLM response
            behavior: Behavior for validation

        Returns:
            Parsed DecisionResult
        """
        content = response.content.strip()

        # Try XML parsing first (our default format)
        if "<response>" in content:
            return self._parse_xml_decision(content, behavior)

        # Fallback to JSON parsing
        if content.startswith("{"):
            return self._parse_json_decision(content, behavior)

        # If neither, try to extract action from plain text
        return self._parse_text_decision(content, behavior)

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

        # Event details
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

        # Code-related
        code_query = root.find("code_query")
        if code_query is not None and code_query.text:
            args["code_query"] = code_query.text.strip()

        code_file = root.find("code_file")
        if code_file is not None and code_file.text:
            args["code_file"] = code_file.text.strip()

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
        except json.JSONDecodeError:
            result.answer = content

        return result

    def _parse_text_decision(self, content: str, behavior: Behavior) -> DecisionResult:
        """Parse plain text decision (fallback).

        Args:
            content: Plain text content
            behavior: Behavior for validation

        Returns:
            Parsed DecisionResult
        """
        # Try to detect action keywords
        action = "answer"
        for act in behavior.actions:
            if act.name.lower() in content.lower():
                action = act.name
                break

        return DecisionResult(
            action=action,
            answer=content,
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

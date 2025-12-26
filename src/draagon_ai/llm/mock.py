"""Mock LLM providers for testing.

Provides realistic mock responses for testing behavior creation
without requiring actual API calls.
"""

import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    ModelTier,
    ToolDefinition,
)


class MockLLM(LLMProvider):
    """Simple mock LLM that returns configurable responses.

    Usage:
        llm = MockLLM(default_response="Hello!")

        # Or with a response sequence
        llm = MockLLM(responses=["First", "Second", "Third"])
    """

    def __init__(
        self,
        default_response: str = "Mock response",
        responses: list[str] | None = None,
    ):
        """Initialize mock LLM.

        Args:
            default_response: Default response when no others available
            responses: Optional list of responses to cycle through
        """
        self.default_response = default_response
        self.responses = list(responses) if responses else []
        self.response_index = 0
        self.calls: list[dict] = []  # Track all calls for assertions

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[ToolDefinition] | None = None,
        tier: ModelTier = ModelTier.LOCAL,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Return mock response."""
        # Track the call
        self.calls.append({
            "messages": messages,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "tier": tier,
        })

        # Get response
        if self.responses:
            response = self.responses[self.response_index % len(self.responses)]
            self.response_index += 1
        else:
            response = self.default_response

        return ChatResponse(
            content=response,
            role="assistant",
            model="mock",
            latency_ms=10.0,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> AsyncIterator[str]:
        """Stream mock response word by word."""
        response = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )

        for word in response.content.split():
            yield word + " "

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> str:
        """Generate text (compatibility method)."""
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )
        return response.content


@dataclass
class RealisticResponseTemplate:
    """Template for generating realistic LLM responses."""

    # Phase patterns
    pattern: str  # Regex to match in prompt
    template: Callable[[str, dict], str]  # Function to generate response


class RealisticMockLLM(LLMProvider):
    """Mock LLM that generates realistic, contextual responses.

    This mock understands the behavior architect prompts and generates
    responses that would actually work, enabling meaningful testing
    without API calls.

    Usage:
        llm = RealisticMockLLM()
        architect = BehaviorArchitectService(llm=llm)
        behavior = await architect.create_behavior("kitchen timer")
        # behavior will have realistic prompts and test cases
    """

    def __init__(self, variability: float = 0.1):
        """Initialize realistic mock.

        Args:
            variability: How much randomness to add (0-1)
        """
        self.variability = variability
        self.calls: list[dict] = []
        self._context: dict[str, Any] = {}  # Track context across calls

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[ToolDefinition] | None = None,
        tier: ModelTier = ModelTier.LOCAL,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Generate realistic contextual response."""
        # Track call
        self.calls.append({
            "messages": messages,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "tier": tier,
        })

        # Get prompt content
        prompt = self._extract_prompt(messages)

        # Determine phase and generate appropriate response
        response = self._generate_response(prompt, system_prompt or "")

        return ChatResponse(
            content=response,
            role="assistant",
            model="realistic-mock",
            latency_ms=50.0 + random.random() * 100,  # Simulate realistic latency
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> AsyncIterator[str]:
        """Stream response word by word."""
        response = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )

        for word in response.content.split():
            yield word + " "

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> str:
        """Generate text (compatibility method)."""
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )
        return response.content

    def _extract_prompt(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
    ) -> str:
        """Extract prompt text from messages."""
        if not messages:
            return ""

        last = messages[-1]
        if isinstance(last, ChatMessage):
            return last.content
        return last.get("content", "")

    def _generate_response(self, prompt: str, system_prompt: str) -> str:
        """Generate appropriate response based on prompt content."""
        prompt_lower = prompt.lower()

        # Research phase
        if "researching a domain" in prompt_lower or "research" in system_prompt.lower():
            return self._generate_research_response(prompt)

        # Design phase
        if "designing an ai behavior" in prompt_lower or "design" in system_prompt.lower():
            return self._generate_design_response(prompt)

        # Prompt generation phase
        if "writing prompts" in prompt_lower or "prompt engineer" in system_prompt.lower():
            return self._generate_prompts_response(prompt)

        # Test generation phase
        if "generating test cases" in prompt_lower or "qa engineer" in system_prompt.lower():
            return self._generate_test_cases_response(prompt)

        # Failure analysis phase
        if "analyzing test failures" in prompt_lower or "qa analyst" in system_prompt.lower():
            return self._generate_failure_analysis_response(prompt)

        # Mutation phase (evolution)
        if "mutation" in prompt_lower or "mutate" in prompt_lower:
            return self._generate_mutation_response(prompt)

        # Decision/action phase (test execution)
        if "choose" in prompt_lower or "action" in prompt_lower:
            return self._generate_action_response(prompt)

        # Fix application
        if "improve this prompt" in prompt_lower:
            return self._generate_fix_response(prompt)

        # Default
        return "I understand and will help with this task."

    def _generate_research_response(self, prompt: str) -> str:
        """Generate realistic research response."""
        # Extract domain from prompt
        domain = self._extract_domain(prompt)
        self._context["domain"] = domain

        # Generate domain-appropriate actions
        actions = self._generate_domain_actions(domain)
        self._context["actions"] = actions

        actions_xml = "\n".join([
            f'    <action name="{a["name"]}" description="{a["description"]}">\n'
            f'      <parameter name="{a["param"]}" type="{a["type"]}" required="true">{a["param_desc"]}</parameter>\n'
            f'    </action>'
            for a in actions
        ])

        triggers = self._generate_domain_triggers(domain)
        triggers_xml = "\n".join([f"    <trigger>{t}</trigger>" for t in triggers])

        constraints = self._generate_domain_constraints(domain)
        constraints_xml = "\n".join([f"    <constraint>{c}</constraint>" for c in constraints])

        knowledge = self._generate_domain_knowledge(domain)

        return f'''<research>
  <core_tasks>
    <task>Handle primary {domain} operations</task>
    <task>Manage {domain} state and queries</task>
    <task>Provide helpful {domain} information</task>
  </core_tasks>
  <suggested_actions>
{actions_xml}
  </suggested_actions>
  <triggers>
{triggers_xml}
  </triggers>
  <constraints>
{constraints_xml}
  </constraints>
  <domain_knowledge>
{knowledge}
  </domain_knowledge>
  <sources>
    <source>Domain analysis</source>
    <source>Best practices research</source>
  </sources>
</research>'''

    def _generate_design_response(self, prompt: str) -> str:
        """Generate realistic design response."""
        domain = self._context.get("domain", self._extract_domain(prompt))
        actions = self._context.get("actions", self._generate_domain_actions(domain))

        behavior_id = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
        name = domain.title().replace("_", " ")

        actions_xml = "\n".join([
            f'    <action name="{a["name"]}" requires_confirmation="false">\n'
            f'      <description>{a["description"]}</description>\n'
            f'      <parameter name="{a["param"]}" type="{a["type"]}" required="true">{a["param_desc"]}</parameter>\n'
            f'      <trigger>{a["trigger"]}</trigger>\n'
            f'      <example query="{a["example"]}" outcome="{a["outcome"]}"/>\n'
            f'    </action>'
            for a in actions
        ])

        return f'''<design>
  <behavior_id>{behavior_id}</behavior_id>
  <name>{name}</name>
  <description>Manages {domain} operations with voice-friendly responses</description>

  <actions>
{actions_xml}
  </actions>

  <triggers>
    <trigger name="main" priority="70">
      <semantic>{domain} related requests</semantic>
      <keyword>{behavior_id}</keyword>
    </trigger>
  </triggers>

  <constraints>
    <style>Be concise and helpful. Confirm actions when appropriate.</style>
  </constraints>

  <domain_context>
    This behavior handles {domain} operations. Provide clear, actionable responses.
  </domain_context>
</design>'''

    def _generate_prompts_response(self, prompt: str) -> str:
        """Generate realistic prompts."""
        domain = self._context.get("domain", "general")
        actions = self._context.get("actions", [{"name": "help", "description": "Provide help"}])

        action_list = "\n".join([
            f"- {a['name']}: {a['description']}"
            for a in actions
        ])

        return f'''<prompts>
  <decision_prompt>
You are an AI assistant specialized in {domain}. Your role is to understand user requests and select the most appropriate action.

AVAILABLE ACTIONS:
{action_list}

DECISION PROCESS:
1. Analyze the user's query for intent
2. Match intent to available actions
3. Extract required parameters from the query
4. Return your decision in XML format

USER QUERY: {{{{query}}}}
CONTEXT: {{{{context}}}}

Respond with your decision:
<decision>
  <action name="action_name">
    <parameter name="param_name">value</parameter>
  </action>
  <reasoning>Brief explanation of why this action was chosen</reasoning>
</decision>
  </decision_prompt>

  <synthesis_prompt>
You are formatting a response for a voice assistant. The response should be:
- Concise (1-2 sentences for simple confirmations)
- Natural and conversational
- Action-focused (confirm what was done)

ACTION RESULT: {{{{action_result}}}}
STYLE: {{{{style}}}}

Format a friendly, voice-appropriate response.
  </synthesis_prompt>
</prompts>'''

    def _generate_test_cases_response(self, prompt: str) -> str:
        """Generate realistic test cases."""
        actions = self._context.get("actions", [{"name": "help", "description": "Help"}])

        test_cases = []
        test_id = 1

        # Positive tests - one per action
        for action in actions:
            test_cases.append(f'''  <test id="test_{test_id:03d}" name="Test {action['name']} basic" priority="high">
    <description>Test basic {action['name']} functionality</description>
    <user_query>{action.get('example', f"Use {action['name']}")}</user_query>
    <expected_actions>
      <action>{action['name']}</action>
    </expected_actions>
    <expected_response_contains>
      <phrase>{action['name'].replace('_', ' ')}</phrase>
    </expected_response_contains>
  </test>''')
            test_id += 1

        # Negative tests
        test_cases.append(f'''  <test id="test_{test_id:03d}" name="Reject invalid request" priority="high">
    <description>Should not perform action for unrelated request</description>
    <user_query>What's the weather like?</user_query>
    <forbidden_actions>
      <action>{actions[0]['name']}</action>
    </forbidden_actions>
  </test>''')
        test_id += 1

        test_cases.append(f'''  <test id="test_{test_id:03d}" name="Handle ambiguous input" priority="medium">
    <description>Should handle unclear requests gracefully</description>
    <user_query>Do something</user_query>
    <expected_response_contains>
      <phrase>help</phrase>
    </expected_response_contains>
  </test>''')
        test_id += 1

        # Edge case tests
        test_cases.append(f'''  <test id="test_{test_id:03d}" name="Handle empty context" priority="medium">
    <description>Should work without additional context</description>
    <user_query>{actions[0].get('example', 'Help me')}</user_query>
    <expected_actions>
      <action>{actions[0]['name']}</action>
    </expected_actions>
  </test>''')

        return f'''<test_cases>
{"".join(test_cases)}
</test_cases>'''

    def _generate_failure_analysis_response(self, prompt: str) -> str:
        """Generate failure analysis response."""
        return '''<analysis>
  <patterns>
    <pattern type="wrong_action" count="1">Action selection unclear for edge cases</pattern>
  </patterns>
  <root_causes>
    <cause id="cause_1">Decision prompt lacks explicit guidance for edge cases</cause>
  </root_causes>
  <fixes>
    <fix cause_id="cause_1" target="decision_prompt">
      Add explicit instructions for handling ambiguous requests. Include examples of edge cases and how to respond.
    </fix>
  </fixes>
</analysis>'''

    def _generate_mutation_response(self, prompt: str) -> str:
        """Generate mutated prompt."""
        # Extract the current prompt from the mutation request
        match = re.search(r"CURRENT PROMPT:\s*(.+?)(?:$|\n\n)", prompt, re.DOTALL)
        if match:
            original = match.group(1).strip()
            # Apply some realistic mutation
            mutations = [
                lambda p: p + "\n\nIMPORTANT: When uncertain, ask for clarification.",
                lambda p: p.replace("appropriate", "most suitable"),
                lambda p: p + "\n\nConsider the user's context when making decisions.",
                lambda p: re.sub(r"Be (concise|brief)", "Be clear and concise", p),
            ]
            return random.choice(mutations)(original)

        return prompt

    def _generate_action_response(self, prompt: str) -> str:
        """Generate action decision response."""
        actions = self._context.get("actions", [])

        # Try to match query to an action
        prompt_lower = prompt.lower()
        for action in actions:
            if action["name"] in prompt_lower or any(
                word in prompt_lower
                for word in action.get("trigger", "").lower().split()
            ):
                return f'''<decision>
  <action name="{action['name']}">
    <parameter name="{action['param']}">extracted_value</parameter>
  </action>
  <reasoning>User request matches {action['name']} action</reasoning>
</decision>'''

        # Default to first action or help
        if actions:
            return f'''<decision>
  <action name="{actions[0]['name']}">
  </action>
  <reasoning>Best match for user request</reasoning>
</decision>'''

        return '''<decision>
  <action name="help">
  </action>
  <reasoning>Request requires clarification</reasoning>
</decision>'''

    def _generate_fix_response(self, prompt: str) -> str:
        """Generate fixed prompt."""
        match = re.search(r"CURRENT PROMPT:\s*(.+?)SUGGESTED FIX:", prompt, re.DOTALL)
        if match:
            original = match.group(1).strip()
            # Add the suggested improvements
            return original + "\n\nAdditional guidance: When the intent is unclear, prioritize asking clarifying questions over making assumptions."

        return "Improved prompt with better clarity."

    def _extract_domain(self, prompt: str) -> str:
        """Extract domain from prompt."""
        prompt_lower = prompt.lower()

        # Check for specific domain keywords first (most reliable)
        domain_keywords = {
            "timer": ["timer", "countdown", "alarm", "stopwatch"],
            "kitchen timer": ["kitchen timer", "cooking timer", "kitchen"],
            "calendar": ["calendar", "schedule", "event", "meeting", "appointment"],
            "smart home": ["smart home", "home automation", "lights", "thermostat", "devices", "iot"],
            "weather": ["weather", "forecast", "temperature", "climate"],
            "music": ["music", "playlist", "song", "audio", "spotify"],
            "reminder": ["reminder", "remind me", "todo", "task"],
        }

        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return domain

        # Look for common patterns
        patterns = [
            r"behavior for (?:managing |handling )?(.+?)(?:\.|,|$)",
            r"create (?:a |an )?(.+?) behavior",
            r"domain: (.+?)(?:\.|,|\n|$)",
            r"about (.+?)(?:\.|,|$)",
            r"for (.+?)(?:\.|,|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                domain = match.group(1).strip()
                # Clean up
                domain = re.sub(r"^(a|an|the)\s+", "", domain)
                domain = re.sub(r"\s+(?:behavior|management|operations|system)$", "", domain)
                if domain and len(domain) > 2:
                    return domain[:50]  # Limit length

        return "general assistant"

    def _generate_domain_actions(self, domain: str) -> list[dict]:
        """Generate domain-specific actions."""
        domain_lower = domain.lower()

        # Domain-specific action sets
        if "timer" in domain_lower or "kitchen" in domain_lower:
            return [
                {
                    "name": "set_timer",
                    "description": "Set a countdown timer for a specified duration",
                    "param": "duration",
                    "type": "string",
                    "param_desc": "Duration like '5 minutes' or '30 seconds'",
                    "trigger": "set a timer",
                    "example": "Set a 5 minute timer",
                    "outcome": "Timer started for 5 minutes",
                },
                {
                    "name": "cancel_timer",
                    "description": "Cancel an active timer",
                    "param": "timer_name",
                    "type": "string",
                    "param_desc": "Name or description of timer to cancel",
                    "trigger": "cancel timer",
                    "example": "Cancel the pasta timer",
                    "outcome": "Timer cancelled",
                },
                {
                    "name": "list_timers",
                    "description": "List all active timers",
                    "param": "filter",
                    "type": "string",
                    "param_desc": "Optional filter for timer names",
                    "trigger": "show timers",
                    "example": "What timers do I have?",
                    "outcome": "Active timers listed",
                },
            ]

        if "calendar" in domain_lower or "schedule" in domain_lower:
            return [
                {
                    "name": "get_events",
                    "description": "Get calendar events for a time period",
                    "param": "time_range",
                    "type": "string",
                    "param_desc": "Time range like 'today', 'tomorrow', 'this week'",
                    "trigger": "what's on my calendar",
                    "example": "What do I have scheduled today?",
                    "outcome": "Events listed for today",
                },
                {
                    "name": "create_event",
                    "description": "Create a new calendar event",
                    "param": "event_details",
                    "type": "string",
                    "param_desc": "Event details including title, time, and optional location",
                    "trigger": "add event",
                    "example": "Add a meeting at 3pm tomorrow",
                    "outcome": "Event created",
                },
            ]

        if "smart home" in domain_lower or "home" in domain_lower or "light" in domain_lower:
            return [
                {
                    "name": "control_device",
                    "description": "Control a smart home device",
                    "param": "device_action",
                    "type": "string",
                    "param_desc": "Device and action like 'turn on living room lights'",
                    "trigger": "turn on/off",
                    "example": "Turn on the bedroom lights",
                    "outcome": "Lights turned on",
                },
                {
                    "name": "get_device_state",
                    "description": "Get the current state of a device",
                    "param": "device",
                    "type": "string",
                    "param_desc": "Device to check",
                    "trigger": "is the",
                    "example": "Is the front door locked?",
                    "outcome": "Door status returned",
                },
            ]

        # Default generic actions
        return [
            {
                "name": "perform_action",
                "description": f"Perform a {domain} related action",
                "param": "action_details",
                "type": "string",
                "param_desc": "Details of what to do",
                "trigger": domain,
                "example": f"Help me with {domain}",
                "outcome": "Action completed",
            },
            {
                "name": "get_info",
                "description": f"Get information about {domain}",
                "param": "query",
                "type": "string",
                "param_desc": "What information is needed",
                "trigger": "what is",
                "example": f"Tell me about {domain}",
                "outcome": "Information provided",
            },
        ]

    def _generate_domain_triggers(self, domain: str) -> list[str]:
        """Generate domain-specific triggers."""
        return [
            f"User asks about {domain}",
            f"Request involves {domain} operations",
            f"Query mentions {domain} related terms",
        ]

    def _generate_domain_constraints(self, domain: str) -> list[str]:
        """Generate domain-specific constraints."""
        return [
            f"Only perform {domain} related actions",
            "Ask for confirmation before destructive operations",
            "Provide clear feedback after each action",
        ]

    def _generate_domain_knowledge(self, domain: str) -> str:
        """Generate domain-specific knowledge."""
        return f"""This behavior specializes in {domain}. Key considerations:
- Users may use natural language to describe their needs
- Context from previous interactions should inform responses
- When uncertain, ask clarifying questions
- Prioritize safety and user confirmation for important actions"""

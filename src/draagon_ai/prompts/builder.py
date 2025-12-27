"""Dynamic prompt builder.

Constructs prompts dynamically from:
- Core templates
- Available actions from behaviors, extensions, MCPs
- Enabled capabilities
- App-specific overrides

This enables apps to customize prompts without hardcoding in the framework.

Example:
    from draagon_ai.prompts.builder import PromptBuilder

    builder = PromptBuilder()

    # Add core actions
    builder.add_action("answer", "Respond directly from context")
    builder.add_action("clarify", "Ask for clarification")

    # Add capability actions (from extensions)
    builder.add_capability_actions("calendar", [
        ("calendar_query", "Check user's calendar/schedule"),
        ("calendar_create", "Add event to calendar"),
    ])

    # Add MCP actions
    builder.add_mcp_actions("fetch", [
        ("fetch_url", "Fetch content from a URL"),
    ])

    # Build the decision prompt
    prompt = builder.build_decision_prompt(
        template=DECISION_TEMPLATE,
        assistant_name="Assistant",
        assistant_intro="A helpful voice assistant",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionDef:
    """Definition of an available action."""

    name: str
    description: str
    source: str = "core"  # core, capability, extension, mcp
    args_schema: dict[str, Any] | None = None


@dataclass
class CapabilityDef:
    """Definition of a capability (group of related actions)."""

    name: str
    actions: list[ActionDef] = field(default_factory=list)
    fast_route_actions: list[ActionDef] = field(default_factory=list)
    tool_selection_guide: str = ""  # Guidance for when to use these actions


class PromptBuilder:
    """Dynamically builds prompts from components.

    Instead of hardcoding available actions in prompts, this builder
    allows apps to compose prompts from:
    - Core actions (always available)
    - Capability actions (from enabled capabilities)
    - Extension actions (from loaded extensions)
    - MCP actions (from connected MCP servers)

    Example:
        builder = PromptBuilder()
        builder.add_core_actions([
            ActionDef("answer", "Respond directly from context"),
            ActionDef("get_time", "Get current time"),
        ])
        builder.add_capability("calendar", CALENDAR_CAPABILITY)

        prompt = builder.build_decision_prompt(DECISION_TEMPLATE)
    """

    def __init__(self) -> None:
        """Initialize the prompt builder."""
        self._core_actions: list[ActionDef] = []
        self._capabilities: dict[str, CapabilityDef] = {}
        self._extension_actions: dict[str, list[ActionDef]] = {}
        self._mcp_actions: dict[str, list[ActionDef]] = {}
        self._custom_sections: dict[str, str] = {}

    def add_core_action(self, name: str, description: str) -> "PromptBuilder":
        """Add a core action that's always available."""
        self._core_actions.append(ActionDef(name, description, "core"))
        return self

    def add_core_actions(self, actions: list[ActionDef]) -> "PromptBuilder":
        """Add multiple core actions."""
        for action in actions:
            action.source = "core"
            self._core_actions.append(action)
        return self

    def add_capability(self, name: str, capability: CapabilityDef) -> "PromptBuilder":
        """Add a capability with its actions."""
        self._capabilities[name] = capability
        return self

    def add_capability_actions(
        self,
        capability_name: str,
        actions: list[tuple[str, str]]
    ) -> "PromptBuilder":
        """Add actions for a capability (simple tuple format)."""
        if capability_name not in self._capabilities:
            self._capabilities[capability_name] = CapabilityDef(name=capability_name)

        for name, desc in actions:
            self._capabilities[capability_name].actions.append(
                ActionDef(name, desc, f"capability:{capability_name}")
            )
        return self

    def add_extension_actions(
        self,
        extension_name: str,
        actions: list[tuple[str, str]]
    ) -> "PromptBuilder":
        """Add actions from an extension."""
        if extension_name not in self._extension_actions:
            self._extension_actions[extension_name] = []

        for name, desc in actions:
            self._extension_actions[extension_name].append(
                ActionDef(name, desc, f"extension:{extension_name}")
            )
        return self

    def add_mcp_actions(
        self,
        server_name: str,
        actions: list[tuple[str, str]]
    ) -> "PromptBuilder":
        """Add actions from an MCP server."""
        if server_name not in self._mcp_actions:
            self._mcp_actions[server_name] = []

        for name, desc in actions:
            self._mcp_actions[server_name].append(
                ActionDef(name, desc, f"mcp:{server_name}")
            )
        return self

    def add_custom_section(self, key: str, content: str) -> "PromptBuilder":
        """Add a custom section that can be injected into templates."""
        self._custom_sections[key] = content
        return self

    def get_all_actions(self) -> list[ActionDef]:
        """Get all registered actions."""
        actions = list(self._core_actions)

        for cap in self._capabilities.values():
            actions.extend(cap.actions)

        for ext_actions in self._extension_actions.values():
            actions.extend(ext_actions)

        for mcp_actions in self._mcp_actions.values():
            actions.extend(mcp_actions)

        return actions

    def format_actions_section(self) -> str:
        """Format all actions as a prompt section."""
        lines = ["AVAILABLE ACTIONS:"]

        # Core actions
        for action in self._core_actions:
            lines.append(f"- {action.name}: {action.description}")

        # Capability actions (grouped)
        for cap_name, cap in self._capabilities.items():
            if cap.actions:
                lines.append(f"\n{cap_name.upper()} ACTIONS:")
                for action in cap.actions:
                    lines.append(f"- {action.name}: {action.description}")

        # Extension actions
        for ext_name, ext_actions in self._extension_actions.items():
            if ext_actions:
                lines.append(f"\n{ext_name.upper()} EXTENSION:")
                for action in ext_actions:
                    lines.append(f"- {action.name}: {action.description}")

        # MCP actions
        for server_name, mcp_actions in self._mcp_actions.items():
            if mcp_actions:
                lines.append(f"\n{server_name.upper()} MCP SERVER:")
                for action in mcp_actions:
                    lines.append(f"- {action.name}: {action.description}")

        return "\n".join(lines)

    def format_tool_selection_guide(self) -> str:
        """Format tool selection guidance from capabilities."""
        lines = ["TOOL SELECTION GUIDE:"]

        for cap in self._capabilities.values():
            if cap.tool_selection_guide:
                lines.append(cap.tool_selection_guide)

        return "\n".join(lines) if len(lines) > 1 else ""

    def format_fast_route_actions(self) -> str:
        """Format fast-path actions for routing prompt."""
        lines = ["FAST-PATH ACTIONS (handle immediately, only when CLEARLY matching):"]

        idx = 1
        # Core fast actions
        for action in self._core_actions:
            lines.append(f"{idx}. {action.name} - {action.description}")
            idx += 1

        # Capability fast actions
        for cap in self._capabilities.values():
            for action in cap.fast_route_actions:
                lines.append(f"{idx}. {action.name} - {action.description}")
                idx += 1

        return "\n".join(lines)

    def build(
        self,
        template: str,
        **kwargs: Any,
    ) -> str:
        """Build a prompt from template with dynamic sections.

        The template can include these placeholders:
        - {available_actions} - Formatted list of all actions
        - {tool_selection_guide} - Guidance for selecting tools
        - {fast_route_actions} - Fast-path actions for routing
        - Plus any custom sections added via add_custom_section
        - Plus any kwargs passed directly
        """
        # Build dynamic sections
        sections = {
            "available_actions": self.format_actions_section(),
            "tool_selection_guide": self.format_tool_selection_guide(),
            "fast_route_actions": self.format_fast_route_actions(),
            **self._custom_sections,
            **kwargs,
        }

        # Replace placeholders
        result = template
        for key, value in sections.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))

        return result


# Pre-built capability definitions
CORE_ACTIONS = [
    ActionDef("answer", "Respond directly (context has the answer, or say \"I don't have that info\")"),
    ActionDef("clarify", "Genuinely ambiguous (use rarely)"),
    ActionDef("get_time", "Current local time/date"),
    ActionDef("get_weather", "Current local weather (use search_web for other locations or forecasts)"),
    ActionDef("get_location", "Assistant's physical location (room, address)"),
    ActionDef("search_web", "External/current info (news, other cities, forecasts, events, research)"),
    ActionDef("search_knowledge", "Search stored knowledge and memories"),
]

CALENDAR_CAPABILITY = CapabilityDef(
    name="calendar",
    actions=[
        ActionDef("calendar_query", "Check user's calendar/schedule"),
        ActionDef("calendar_create", "Add event to calendar"),
        ActionDef("calendar_delete", "Remove event from calendar"),
    ],
    tool_selection_guide="- Calendar events → calendar_query/calendar_create/calendar_delete",
)

HOME_ASSISTANT_CAPABILITY = CapabilityDef(
    name="home_assistant",
    actions=[
        ActionDef("home_assistant", "Smart home control (lights, switches, sensors, climate)"),
    ],
    fast_route_actions=[
        ActionDef(
            "home_assistant",
            "Direct smart home device control: lights, switches, plugs, fans "
            "(turn on/off, set color/brightness, dim). NOT for containers, servers, VMs!"
        ),
    ],
    tool_selection_guide="- Control lights/switches → home_assistant",
)

COMMANDS_CAPABILITY = CapabilityDef(
    name="commands",
    actions=[
        ActionDef("execute_command", "Run shell command (system info, disk space, processes)"),
    ],
    tool_selection_guide="- System info (disk, processes) → execute_command",
)

TIMERS_CAPABILITY = CapabilityDef(
    name="timers",
    actions=[
        ActionDef("use_tools", "Timers, scheduled jobs, interests, pending events"),
    ],
    tool_selection_guide="- Timers, jobs, interests → use_tools",
)

CODE_DOCS_CAPABILITY = CapabilityDef(
    name="code_docs",
    actions=[
        ActionDef("get_code_docs", "Programming library documentation"),
        ActionDef("search_code", "Search source code by pattern"),
        ActionDef("read_code", "Read source code file"),
        ActionDef("list_code_files", "List files in directory"),
    ],
    tool_selection_guide="""- Programming questions → get_code_docs
- Source code exploration → search_code/read_code""",
)


def create_default_builder(
    capabilities: list[str] | None = None,
) -> PromptBuilder:
    """Create a builder with common defaults.

    Args:
        capabilities: List of capability names to enable.
            Defaults to all: ["calendar", "home_assistant", "commands", "timers", "code_docs"]

    Returns:
        Configured PromptBuilder.
    """
    if capabilities is None:
        capabilities = ["calendar", "home_assistant", "commands", "timers", "code_docs"]

    builder = PromptBuilder()
    builder.add_core_actions(CORE_ACTIONS)

    capability_map = {
        "calendar": CALENDAR_CAPABILITY,
        "home_assistant": HOME_ASSISTANT_CAPABILITY,
        "commands": COMMANDS_CAPABILITY,
        "timers": TIMERS_CAPABILITY,
        "code_docs": CODE_DOCS_CAPABILITY,
    }

    for cap_name in capabilities:
        if cap_name in capability_map:
            builder.add_capability(cap_name, capability_map[cap_name])

    return builder

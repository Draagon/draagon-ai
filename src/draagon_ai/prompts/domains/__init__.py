"""Prompt domains for draagon-ai.

Each domain contains related prompts for a functional area.

Prompt Architecture:
- **Core domains** (routing, decision, synthesis, memory, quality, conversation):
  Generic prompts that work with any AI assistant. These use template
  variables like {assistant_name}, {query}, {history} for customization.

- **Capability domains** (home_assistant, calendar, commands):
  Prompts for optional capabilities. Enable these when your app integrates
  with Home Assistant, Google Calendar, or shell command execution.

- **Extension domains**: Extensions can provide additional prompt domains
  via Extension.get_prompt_domains(). These are merged at runtime.

- **App overrides**: Apps can override any prompt by providing their own
  version with the same domain:name key.

Template Variables (use these in prompts):
- {assistant_name}: The AI assistant's name
- {query}: User's current query
- {history}: Recent conversation history
- {user_id}: Current user identifier
- {area}: Current room/area (for home control)
"""

from .routing import ROUTING_PROMPTS
from .decision import DECISION_PROMPTS
from .home_assistant import HOME_ASSISTANT_PROMPTS
from .calendar import CALENDAR_PROMPTS
from .commands import COMMANDS_PROMPTS
from .synthesis import SYNTHESIS_PROMPTS
from .memory import MEMORY_PROMPTS
from .quality import QUALITY_PROMPTS
from .conversation import CONVERSATION_PROMPTS
from .templates import (
    TEMPLATE_PROMPTS,
    DECISION_TEMPLATE,
    FAST_ROUTE_TEMPLATE,
    INTENT_CLASSIFICATION_TEMPLATE,
    SYNTHESIS_TEMPLATE,
    DEFAULT_DECISION_EXAMPLES,
    DEFAULT_FAST_ROUTE_EXAMPLES,
)

# Core prompts - generic, always loaded
CORE_PROMPTS = {
    "routing": ROUTING_PROMPTS,
    "decision": DECISION_PROMPTS,
    "synthesis": SYNTHESIS_PROMPTS,
    "memory": MEMORY_PROMPTS,
    "quality": QUALITY_PROMPTS,
    "conversation": CONVERSATION_PROMPTS,
}

# Capability prompts - loaded when capability is enabled
CAPABILITY_PROMPTS = {
    "home_assistant": HOME_ASSISTANT_PROMPTS,
    "calendar": CALENDAR_PROMPTS,
    "commands": COMMANDS_PROMPTS,
}

# All prompts - core + capabilities
ALL_PROMPTS = {**CORE_PROMPTS, **CAPABILITY_PROMPTS}

__all__ = [
    # Individual domain exports
    "ROUTING_PROMPTS",
    "DECISION_PROMPTS",
    "HOME_ASSISTANT_PROMPTS",
    "CALENDAR_PROMPTS",
    "COMMANDS_PROMPTS",
    "SYNTHESIS_PROMPTS",
    "MEMORY_PROMPTS",
    "QUALITY_PROMPTS",
    "CONVERSATION_PROMPTS",
    # Template exports (for dynamic building)
    "TEMPLATE_PROMPTS",
    "DECISION_TEMPLATE",
    "FAST_ROUTE_TEMPLATE",
    "INTENT_CLASSIFICATION_TEMPLATE",
    "SYNTHESIS_TEMPLATE",
    "DEFAULT_DECISION_EXAMPLES",
    "DEFAULT_FAST_ROUTE_EXAMPLES",
    # Grouped exports
    "CORE_PROMPTS",
    "CAPABILITY_PROMPTS",
    "ALL_PROMPTS",
]

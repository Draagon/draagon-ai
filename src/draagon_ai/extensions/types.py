"""Extension types and protocols for draagon-ai.

This module defines the base types and protocols for extensions.
Extensions are optional modules that can be installed and configured
independently of the core draagon-ai package.

Extensions can provide:
- **Behaviors**: Complete personality packages (prompts + actions + triggers)
- **Prompt Domains**: Collections of LLM prompts for specific capabilities
- **Tools**: Function definitions for agent capabilities
- **MCP Servers**: Model Context Protocol server configurations
- **Services**: Shared service instances (caches, clients, etc.)

Example:
    from draagon_ai.extensions import Extension, ExtensionInfo

    class MyExtension(Extension):
        @property
        def info(self) -> ExtensionInfo:
            return ExtensionInfo(
                name="my-extension",
                version="1.0.0",
                description="My custom extension",
                author="me",
                requires_core=">=0.2.0",
                provides_prompt_domains=["my_domain"],
            )

        def initialize(self, config: dict) -> None:
            self._my_setting = config.get("my_setting", "default")

        def get_behaviors(self) -> list:
            from .behaviors import MY_BEHAVIOR
            return [MY_BEHAVIOR]

        def get_prompt_domains(self) -> dict[str, dict[str, str]]:
            return {
                "my_domain": {
                    "MY_PROMPT": "You are a helpful assistant...",
                }
            }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from draagon_ai.behaviors import Behavior


class MCPTransport(Enum):
    """MCP server transport type."""

    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP (Model Context Protocol) server.

    MCP servers provide tools that the agent can use. They run as
    subprocess (stdio) or connect via HTTP/WebSocket.

    Example:
        config = MCPServerConfig(
            name="calendar",
            command="google-calendar-mcp",
            env={"GOOGLE_OAUTH_CREDENTIALS": "/path/to/creds.json"},
        )
    """

    name: str
    """Unique server name (used as tool prefix, e.g., 'calendar.list_events')."""

    command: str
    """Command to run (for stdio transport) or URL (for http/websocket)."""

    transport: MCPTransport = MCPTransport.STDIO
    """Transport type for MCP communication."""

    args: list[str] = field(default_factory=list)
    """Command arguments (for stdio transport)."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables for the server process."""

    enabled: bool = True
    """Whether this server is enabled."""

    timeout_seconds: int = 30
    """Connection/request timeout in seconds."""


@dataclass
class ExtensionInfo:
    """Extension metadata.

    This dataclass contains all the information about an extension
    that is needed for discovery, dependency resolution, and display.
    """

    # Required fields
    name: str
    """Unique extension name (e.g., 'storytelling', 'architect')."""

    version: str
    """Semantic version string (e.g., '1.0.0')."""

    description: str
    """Human-readable description of what the extension provides."""

    author: str
    """Author or organization name."""

    requires_core: str
    """Core version requirement (e.g., '>=0.2.0')."""

    # Optional fields
    requires_extensions: list[str] = field(default_factory=list)
    """List of extension names this extension depends on."""

    provides_services: list[str] = field(default_factory=list)
    """List of service names this extension provides."""

    provides_behaviors: list[str] = field(default_factory=list)
    """List of behavior IDs this extension provides."""

    provides_tools: list[str] = field(default_factory=list)
    """List of tool names this extension provides."""

    provides_prompt_domains: list[str] = field(default_factory=list)
    """List of prompt domain names this extension provides."""

    provides_mcp_servers: list[str] = field(default_factory=list)
    """List of MCP server names this extension provides."""

    config_schema: dict[str, Any] | None = None
    """JSON Schema for validating extension config."""

    homepage: str = ""
    """URL to extension homepage or documentation."""

    license: str = ""
    """License identifier (e.g., 'MIT', 'Apache-2.0')."""


class Extension(ABC):
    """Base class for all draagon-ai extensions.

    Extensions provide optional functionality that can be enabled or
    disabled via configuration. They can provide:
    - Behaviors (personality templates)
    - Services (business logic)
    - Tools (capabilities like MCP servers)

    Lifecycle:
    1. Extension is discovered via entry points
    2. Extension class is instantiated
    3. initialize() is called with config
    4. Extension provides its components via get_* methods
    5. shutdown() is called on cleanup

    Example:
        class StorytellingExtension(Extension):
            @property
            def info(self) -> ExtensionInfo:
                return ExtensionInfo(
                    name="storytelling",
                    version="1.0.0",
                    description="Interactive fiction",
                    author="draagon-ai",
                    requires_core=">=0.2.0",
                    provides_behaviors=["story_teller"],
                )

            def initialize(self, config: dict) -> None:
                self._drama_intensity = config.get("drama_intensity", 0.7)

            def get_behaviors(self) -> list[Behavior]:
                from .behavior import create_story_teller
                return [create_story_teller(self._drama_intensity)]
    """

    @property
    @abstractmethod
    def info(self) -> ExtensionInfo:
        """Return extension metadata.

        This is called during discovery to get information about
        the extension before it is initialized.

        Returns:
            ExtensionInfo with name, version, dependencies, etc.
        """
        ...

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize extension with configuration.

        Called after the extension is instantiated, with the
        configuration from the draagon.yaml file.

        Args:
            config: Extension-specific configuration dict.

        Raises:
            ExtensionError: If initialization fails.
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources on shutdown.

        Called when the extension manager is shutting down.
        Override this to clean up any resources (connections,
        file handles, etc.).
        """
        pass

    def get_services(self) -> dict[str, Any]:
        """Return services provided by this extension.

        Returns:
            Dict mapping service names to service instances.
        """
        return {}

    def get_behaviors(self) -> list["Behavior"]:
        """Return behaviors provided by this extension.

        Returns:
            List of Behavior instances.
        """
        return []

    def get_tools(self) -> list[Any]:
        """Return tools provided by this extension.

        Returns:
            List of tool instances (e.g., MCP tool configs).
        """
        return []

    def get_prompt_domains(self) -> dict[str, dict[str, str]]:
        """Return prompt domains provided by this extension.

        Prompt domains are collections of related LLM prompts for
        specific capabilities. Each domain contains named prompts.

        Returns:
            Dict mapping domain names to dicts of prompt_name: prompt_content.

        Example:
            return {
                "home_assistant": {
                    "DEVICE_RESOLUTION_PROMPT": "Given a user request...",
                    "ENTITY_SEARCH_PROMPT": "Search for entities...",
                },
                "calendar": {
                    "EVENT_CREATION_PROMPT": "Create a calendar event...",
                },
            }
        """
        return {}

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Return MCP server configurations provided by this extension.

        MCP servers are external tools that the agent can use via
        the Model Context Protocol.

        Returns:
            List of MCPServerConfig instances.

        Example:
            return [
                MCPServerConfig(
                    name="calendar",
                    command="google-calendar-mcp",
                    env={"GOOGLE_OAUTH_CREDENTIALS": "..."},
                ),
            ]
        """
        return []


class ExtensionError(Exception):
    """Base exception for extension-related errors."""

    pass


class ExtensionNotFoundError(ExtensionError):
    """Raised when a requested extension is not found."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Extension not found: {name}")


class ExtensionLoadError(ExtensionError):
    """Raised when an extension fails to load."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Failed to load extension '{name}': {reason}")


class ExtensionDependencyError(ExtensionError):
    """Raised when extension dependencies are not satisfied."""

    def __init__(self, name: str, missing: list[str]) -> None:
        self.name = name
        self.missing = missing
        super().__init__(
            f"Extension '{name}' has missing dependencies: {', '.join(missing)}"
        )

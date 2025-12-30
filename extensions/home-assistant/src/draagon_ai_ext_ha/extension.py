"""Home Assistant Extension for draagon-ai.

This extension provides smart home control capabilities via Home Assistant.
It is designed to be generic and not tied to any specific application.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from draagon_ai.extensions import Extension, ExtensionInfo
from draagon_ai.orchestration.registry import Tool, ToolParameter

from .client import HomeAssistantClient, EntityState
from .resolver import EntityResolver

logger = logging.getLogger(__name__)


class HomeAssistantExtension(Extension):
    """Home Assistant smart home control extension.

    Provides tools for:
    - get_entity: Query entity state
    - search_entities: Search for entities
    - call_service: Control devices
    - get_weather: Get weather from HA
    - get_location: Get home location info

    Configuration (draagon.yaml):
        extensions:
          home_assistant:
            enabled: true
            config:
              url: "http://192.168.168.206:8123"
              token: "${HA_TOKEN}"
              cache_ttl_seconds: 300

    Example:
        ext = HomeAssistantExtension()
        ext.initialize({
            "url": "http://ha:8123",
            "token": "your-token",
        })
        tools = ext.get_tools()
    """

    def __init__(self) -> None:
        """Initialize the extension."""
        self._client: HomeAssistantClient | None = None
        self._resolver: EntityResolver | None = None
        self._config: dict[str, Any] = {}
        self._initialized: bool = False

    @property
    def info(self) -> ExtensionInfo:
        """Return extension metadata."""
        return ExtensionInfo(
            name="home_assistant",
            version="0.1.0",
            description="Home Assistant smart home integration",
            author="draagon-ai",
            requires_core=">=0.1.0",
            provides_behaviors=[],
            provides_tools=[
                "get_entity",
                "search_entities",
                "call_service",
                "get_weather",
                "get_location",
            ],
            provides_prompt_domains=["home_assistant"],
            provides_mcp_servers=[],
            config_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Home Assistant URL",
                        "default": "http://localhost:8123",
                    },
                    "token": {
                        "type": "string",
                        "description": "Long-lived access token",
                    },
                    "cache_ttl_seconds": {
                        "type": "integer",
                        "description": "Device cache TTL in seconds",
                        "default": 300,
                    },
                },
                "required": ["token"],
            },
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Extension configuration
        """
        self._config = config

        # Get URL and token from config or environment
        url = config.get("url") or os.getenv("HA_URL", "http://localhost:8123")
        token = config.get("token") or os.getenv("HA_TOKEN", "")
        cache_ttl = config.get("cache_ttl_seconds", 300)

        if not token:
            logger.warning("Home Assistant token not configured")
            return

        self._client = HomeAssistantClient(
            url=url,
            token=token,
            cache_ttl_seconds=cache_ttl,
        )
        self._resolver = EntityResolver(self._client)
        self._initialized = True

        logger.info(f"HomeAssistantExtension initialized with {url}")

    def shutdown(self) -> None:
        """Clean up resources."""
        self._client = None
        self._resolver = None
        self._initialized = False

    async def _get_entity(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get entity state.

        Args:
            args: Tool arguments (entity_id)
            context: Execution context

        Returns:
            Entity state dict
        """
        if not self._client or not self._resolver:
            return {"error": "Home Assistant not configured"}

        entity_id = args.get("entity_id", "")
        if not entity_id:
            return {"error": "entity_id required"}

        # Try direct lookup first
        state = await self._client.get_entity(entity_id)
        if state:
            return state.to_dict()

        # Try fuzzy resolution
        resolved = await self._resolver.resolve(entity_id)
        if resolved:
            state = await self._client.get_entity(resolved.entity_id)
            if state:
                return state.to_dict()

        return {"error": f"Entity not found: {entity_id}"}

    async def _search_entities(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for entities.

        Args:
            args: Tool arguments (filter)
            context: Execution context

        Returns:
            List of matching entity dicts
        """
        if not self._client:
            return [{"error": "Home Assistant not configured"}]

        filter_text = args.get("filter", "")
        states = await self._client.get_states(filter_text or None)

        # Limit results
        states = states[:20]
        return [s.to_dict() for s in states]

    async def _call_service(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a Home Assistant service.

        Args:
            args: Tool arguments (domain, service, data)
            context: Execution context

        Returns:
            Service result dict
        """
        if not self._client or not self._resolver:
            return {"error": "Home Assistant not configured"}

        domain = args.get("domain", "light")
        service = args.get("service", "turn_on")
        data = args.get("data", {})

        # Handle natural language entity resolution
        entity_id = data.get("entity_id", "")
        if entity_id and "." not in entity_id:
            resolved = await self._resolver.resolve(entity_id, domain=domain)
            if resolved:
                data["entity_id"] = resolved.entity_id

        result = await self._client.call_service(domain, service, data)
        return result.to_dict()

    async def _get_weather(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get weather from Home Assistant.

        Args:
            args: Tool arguments (unused)
            context: Execution context

        Returns:
            Weather data dict
        """
        if not self._client:
            return {"error": "Home Assistant not configured"}

        entity_id = args.get("entity_id", "weather.home")
        return await self._client.get_weather(entity_id)

    async def _get_location(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get home location info.

        Args:
            args: Tool arguments (unused)
            context: Execution context (may contain area_id, device_id)

        Returns:
            Location dict
        """
        if not self._client:
            return {"error": "Home Assistant not configured"}

        context = context or {}
        result: dict[str, Any] = {}

        # Get area info from context
        area_id = context.get("area_id")
        if area_id:
            result["room"] = area_id
            result["area_id"] = area_id

        # Get home location
        location = await self._client.get_home_location()
        if location:
            result.update(location)

        return result if result else {"error": "Location not available"}

    def get_tools(self) -> list[Tool]:
        """Return tools provided by this extension."""
        return [
            Tool(
                name="get_entity",
                description="Get the state of a Home Assistant entity (light, switch, sensor, etc.)",
                handler=self._get_entity,
                parameters=[
                    ToolParameter(
                        name="entity_id",
                        type="string",
                        description="Entity ID (e.g., 'light.bedroom') or natural language (e.g., 'bedroom lights')",
                        required=True,
                    ),
                ],
            ),
            Tool(
                name="search_entities",
                description="Search Home Assistant entities by name or type (solar, battery, lights, temperature)",
                handler=self._search_entities,
                parameters=[
                    ToolParameter(
                        name="filter",
                        type="string",
                        description="Filter text to match entity_id or friendly_name",
                        required=False,
                    ),
                ],
            ),
            Tool(
                name="call_service",
                description="Control a Home Assistant device (lights, switches, climate, etc.)",
                handler=self._call_service,
                parameters=[
                    ToolParameter(
                        name="domain",
                        type="string",
                        description="Service domain (light, switch, climate, media_player, cover)",
                        required=True,
                    ),
                    ToolParameter(
                        name="service",
                        type="string",
                        description="Service to call (turn_on, turn_off, toggle, set_temperature)",
                        required=True,
                    ),
                    ToolParameter(
                        name="data",
                        type="object",
                        description="Service data: {entity_id, brightness, rgb_color, temperature}",
                        required=False,
                    ),
                ],
            ),
            Tool(
                name="get_weather",
                description="Get current weather conditions at home",
                handler=self._get_weather,
                parameters=[
                    ToolParameter(
                        name="entity_id",
                        type="string",
                        description="Weather entity ID (default: weather.home)",
                        required=False,
                    ),
                ],
            ),
            Tool(
                name="get_location",
                description="Get home location and room information",
                handler=self._get_location,
                parameters=[],
            ),
        ]

    def get_prompt_domains(self) -> dict[str, dict[str, str]]:
        """Return prompt domains for this extension."""
        return {
            "home_assistant": {
                "HA_DEVICE_RESOLUTION_PROMPT": """Given a user's natural language reference to a smart home device, resolve it to a Home Assistant entity.

User said: "{query}"
Available entities:
{entities}

Output the best matching entity_id, or "unknown" if no good match.""",
            },
        }

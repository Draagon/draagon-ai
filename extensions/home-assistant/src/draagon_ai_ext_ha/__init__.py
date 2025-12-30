"""Home Assistant Extension for draagon-ai.

This extension provides smart home control capabilities via Home Assistant.
It uses pluggable backends for the actual HA API calls, making it adaptable
to different HA setups.

Features:
- Entity state queries (lights, sensors, switches, etc.)
- Device control (turn on/off, brightness, color, temperature)
- Entity search and discovery
- Weather from Home Assistant
- Location awareness (which room the voice device is in)

Configuration (draagon.yaml):
    extensions:
      home_assistant:
        enabled: true
        config:
          url: "http://192.168.168.206:8123"
          token: "${HA_TOKEN}"
          cache_ttl_seconds: 300

Usage:
    from draagon_ai_ext_ha import HomeAssistantExtension

    ext = HomeAssistantExtension()
    ext.initialize({"url": "http://ha:8123", "token": "..."})
    tools = ext.get_tools()
"""

from .extension import HomeAssistantExtension
from .client import HomeAssistantClient, EntityState, ServiceResult
from .resolver import EntityResolver, ResolvedEntity

__all__ = [
    "HomeAssistantExtension",
    "HomeAssistantClient",
    "EntityState",
    "ServiceResult",
    "EntityResolver",
    "ResolvedEntity",
]

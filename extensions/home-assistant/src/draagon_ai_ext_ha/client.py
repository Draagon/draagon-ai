"""Home Assistant API client.

This module provides a generic HTTP client for the Home Assistant REST API.
It handles authentication, caching, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EntityState:
    """State of a Home Assistant entity."""

    entity_id: str
    state: str
    attributes: dict[str, Any] = field(default_factory=dict)
    last_changed: str = ""
    last_updated: str = ""
    friendly_name: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "EntityState":
        """Create from HA API response."""
        attrs = data.get("attributes", {})
        return cls(
            entity_id=data.get("entity_id", ""),
            state=data.get("state", "unknown"),
            attributes=attrs,
            last_changed=data.get("last_changed", ""),
            last_updated=data.get("last_updated", ""),
            friendly_name=attrs.get("friendly_name", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "state": self.state,
            "friendly_name": self.friendly_name or self.entity_id,
            "attributes": self.attributes,
        }


@dataclass
class ServiceResult:
    """Result of calling a Home Assistant service."""

    success: bool
    message: str = ""
    entities_affected: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"success": self.success}
        if self.message:
            result["message"] = self.message
        if self.entities_affected:
            result["entities_affected"] = self.entities_affected
        if self.error:
            result["error"] = self.error
        return result


class HomeAssistantClient:
    """HTTP client for Home Assistant REST API.

    This client handles:
    - Authentication via long-lived access token
    - Entity state caching with configurable TTL
    - Error handling and retries

    Example:
        client = HomeAssistantClient(
            url="http://192.168.168.206:8123",
            token="your-token-here",
        )

        states = await client.get_states()
        result = await client.call_service("light", "turn_on", {"entity_id": "light.bedroom"})
    """

    def __init__(
        self,
        url: str,
        token: str,
        cache_ttl_seconds: int = 300,
    ) -> None:
        """Initialize the client.

        Args:
            url: Home Assistant base URL (e.g., http://192.168.168.206:8123)
            token: Long-lived access token
            cache_ttl_seconds: How long to cache entity states (default: 5 minutes)
        """
        self.url = url.rstrip("/")
        self.token = token
        self.cache_ttl = cache_ttl_seconds
        self._states_cache: list[EntityState] = []
        self._cache_time: float = 0
        self._areas_cache: dict[str, str] = {}
        self._devices_cache: dict[str, dict] = {}

    async def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict | list | None:
        """Make an authenticated request to HA API.

        Args:
            method: HTTP method (GET, POST)
            path: API path (e.g., /api/states)
            data: Request body for POST
            timeout: Request timeout in seconds

        Returns:
            JSON response or None on error
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not installed")
            return None

        url = f"{self.url}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        logger.warning(f"HA API error: {resp.status} - {await resp.text()}")
                        return None
                elif method == "POST":
                    async with session.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status in (200, 201):
                            return await resp.json()
                        logger.warning(f"HA API error: {resp.status} - {await resp.text()}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"HA API timeout: {path}")
            return None
        except Exception as e:
            logger.error(f"HA API error: {e}")
            return None

        return None

    async def get_states(
        self,
        filter_text: str | None = None,
        use_cache: bool = True,
    ) -> list[EntityState]:
        """Get all entity states.

        Args:
            filter_text: Optional filter for entity_id or friendly_name
            use_cache: Whether to use cached states

        Returns:
            List of EntityState objects
        """
        now = time.time()

        # Check cache
        if use_cache and self._states_cache and (now - self._cache_time) < self.cache_ttl:
            states = self._states_cache
        else:
            # Fetch from API
            data = await self._request("GET", "/api/states")
            if not data or not isinstance(data, list):
                return []

            states = [EntityState.from_api(s) for s in data]
            self._states_cache = states
            self._cache_time = now

        # Apply filter
        if filter_text:
            filter_lower = filter_text.lower()
            states = [
                s for s in states
                if filter_lower in s.entity_id.lower()
                or filter_lower in s.friendly_name.lower()
            ]

        return states

    async def get_entity(self, entity_id: str) -> EntityState | None:
        """Get a specific entity's state.

        Args:
            entity_id: The entity ID (e.g., light.bedroom)

        Returns:
            EntityState or None if not found
        """
        data = await self._request("GET", f"/api/states/{entity_id}")
        if data and isinstance(data, dict):
            return EntityState.from_api(data)
        return None

    async def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
    ) -> ServiceResult:
        """Call a Home Assistant service.

        Args:
            domain: Service domain (e.g., light, switch)
            service: Service name (e.g., turn_on, turn_off)
            data: Service data including entity_id

        Returns:
            ServiceResult with success status
        """
        data = data or {}
        result = await self._request("POST", f"/api/services/{domain}/{service}", data)

        if result is None:
            return ServiceResult(
                success=False,
                error="Failed to call service",
            )

        entities = []
        if isinstance(result, list):
            entities = [r.get("entity_id", "") for r in result if "entity_id" in r]

        entity_id = data.get("entity_id", "device")
        return ServiceResult(
            success=True,
            message=f"Called {domain}.{service} on {entity_id}",
            entities_affected=entities,
        )

    async def get_config(self) -> dict[str, Any]:
        """Get Home Assistant configuration.

        Returns:
            Config dict with location, units, etc.
        """
        result = await self._request("GET", "/api/config")
        return result if isinstance(result, dict) else {}

    async def get_areas(self) -> dict[str, str]:
        """Get area ID to name mapping.

        Returns:
            Dict mapping area_id to area name
        """
        if self._areas_cache:
            return self._areas_cache

        # Use websocket API via REST template
        # This is a simplified version - in production use websocket
        config = await self.get_config()
        # Areas not directly in config, would need websocket

        return self._areas_cache

    async def get_weather(self, entity_id: str = "weather.home") -> dict[str, Any]:
        """Get weather from Home Assistant.

        Args:
            entity_id: Weather entity ID

        Returns:
            Weather data dict
        """
        state = await self.get_entity(entity_id)
        if not state:
            # Try to find any weather entity
            states = await self.get_states("weather.")
            if states:
                state = states[0]

        if not state:
            return {"error": "No weather entity found"}

        attrs = state.attributes
        return {
            "condition": state.state,
            "temperature": attrs.get("temperature"),
            "humidity": attrs.get("humidity"),
            "wind_speed": attrs.get("wind_speed"),
            "forecast": attrs.get("forecast", [])[:3],  # Next 3 periods
        }

    async def get_home_location(self) -> dict[str, Any] | None:
        """Get home location from HA config.

        Returns:
            Location dict with address, coordinates, etc.
        """
        config = await self.get_config()
        if not config:
            return None

        return {
            "name": config.get("location_name", "Home"),
            "latitude": config.get("latitude"),
            "longitude": config.get("longitude"),
            "elevation": config.get("elevation"),
            "unit_system": config.get("unit_system", {}).get("length", "metric"),
        }

    def invalidate_cache(self) -> None:
        """Clear all caches."""
        self._states_cache = []
        self._cache_time = 0
        self._areas_cache = {}
        self._devices_cache = {}

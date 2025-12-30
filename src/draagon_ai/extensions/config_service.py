"""Extension configuration service with hybrid YAML + memory support.

This service manages extension configuration from multiple sources:
1. YAML files (draagon.yaml) - Infrastructure config, rarely changes
2. Memory (Qdrant) - User preferences, learned context
3. Defaults - Sensible fallbacks

Configuration flows:
- On startup: Load YAML, merge with memory preferences
- On voice command: Store in memory, override YAML
- On query: Check memory first, fall back to YAML, then defaults

Example usage:
    config_service = ExtensionConfigService(memory, yaml_config)

    # Get a config value (checks memory → YAML → default)
    interval = await config_service.get("security-monitor", "check_interval_seconds", 300)

    # Set via voice command (stores in memory)
    await config_service.set("security-monitor", "quiet_hours.start", "22:00", source="voice")

    # Get network context (merges YAML + learned services)
    context = await config_service.get_network_context("security-monitor")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MemoryService(Protocol):
    """Protocol for memory service interface."""

    async def search(
        self,
        query: str,
        user_id: str = "system",
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[Any]:
        """Search memories."""
        ...

    async def store(
        self,
        content: str,
        user_id: str = "system",
        memory_type: str = "fact",
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory."""
        ...


@dataclass
class ConfigValue:
    """A configuration value with metadata."""

    value: Any
    source: str  # "yaml", "memory", "default"
    updated_at: datetime | None = None
    updated_by: str | None = None  # user_id who set it


@dataclass
class NetworkService:
    """A known network service."""

    name: str
    ip: str
    ports: list[int] = field(default_factory=list)
    protocols: list[str] = field(default_factory=list)
    notes: str = ""
    learned: bool = False  # True if learned via voice, False if from YAML


@dataclass
class NetworkContext:
    """Full network context for security analysis."""

    home_net: str
    known_services: list[NetworkService]
    trusted_ips: list[str] = field(default_factory=list)
    external_services: list[str] = field(default_factory=list)  # e.g., "Cloudflare", "Google"


class ExtensionConfigService:
    """Manages extension configuration from multiple sources.

    Priority order:
    1. Memory (user preferences, learned context)
    2. YAML (infrastructure config)
    3. Defaults (sensible fallbacks)

    Memory keys are stored as: "extension:{ext_name}:config:{key}"
    """

    # Memory type for extension config
    CONFIG_MEMORY_TYPE = "preference"
    NETWORK_MEMORY_TYPE = "fact"

    def __init__(
        self,
        memory_service: MemoryService | None = None,
        yaml_config: dict[str, Any] | None = None,
    ):
        """Initialize the config service.

        Args:
            memory_service: Memory service for preference storage.
            yaml_config: Parsed YAML configuration dict.
        """
        self._memory = memory_service
        self._yaml = yaml_config or {}
        self._cache: dict[str, ConfigValue] = {}

    def set_memory_service(self, memory_service: MemoryService) -> None:
        """Set memory service (for late binding)."""
        self._memory = memory_service

    def set_yaml_config(self, yaml_config: dict[str, Any]) -> None:
        """Set YAML config (for late binding)."""
        self._yaml = yaml_config

    async def get(
        self,
        extension: str,
        key: str,
        default: Any = None,
        use_cache: bool = True,
    ) -> Any:
        """Get a configuration value.

        Checks memory first, then YAML, then returns default.

        Args:
            extension: Extension name (e.g., "security-monitor").
            key: Config key, supports dot notation (e.g., "monitors.suricata.enabled").
            default: Default value if not found.
            use_cache: Whether to use cached values.

        Returns:
            Configuration value.
        """
        cache_key = f"{extension}:{key}"

        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].value

        # Check memory first (user preferences)
        if self._memory:
            memory_val = await self._get_from_memory(extension, key)
            if memory_val is not None:
                self._cache[cache_key] = ConfigValue(
                    value=memory_val,
                    source="memory",
                )
                return memory_val

        # Check YAML
        yaml_val = self._get_from_yaml(extension, key)
        if yaml_val is not None:
            self._cache[cache_key] = ConfigValue(
                value=yaml_val,
                source="yaml",
            )
            return yaml_val

        # Return default
        return default

    async def get_with_source(
        self,
        extension: str,
        key: str,
        default: Any = None,
    ) -> ConfigValue:
        """Get a configuration value with source metadata.

        Args:
            extension: Extension name.
            key: Config key.
            default: Default value.

        Returns:
            ConfigValue with value and source info.
        """
        value = await self.get(extension, key, default)
        cache_key = f"{extension}:{key}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        return ConfigValue(value=default, source="default")

    async def set(
        self,
        extension: str,
        key: str,
        value: Any,
        source: str = "api",
        user_id: str = "system",
    ) -> None:
        """Set a configuration value in memory.

        This stores the preference in memory, overriding YAML for this key.

        Args:
            extension: Extension name.
            key: Config key.
            value: Value to set.
            source: How this was set ("voice", "api", "ui").
            user_id: User who set it.
        """
        if not self._memory:
            logger.warning("No memory service, config change not persisted")
            return

        # Store in memory
        content = f"{key}: {value}"
        entities = [f"extension:{extension}", f"config:{key}"]

        await self._memory.store(
            content=content,
            user_id="system",  # Extension config is system-level
            memory_type=self.CONFIG_MEMORY_TYPE,
            entities=entities,
            metadata={
                "extension": extension,
                "key": key,
                "value": value,
                "source": source,
                "set_by": user_id,
            },
        )

        # Update cache
        cache_key = f"{extension}:{key}"
        self._cache[cache_key] = ConfigValue(
            value=value,
            source="memory",
            updated_at=datetime.now(),
            updated_by=user_id,
        )

        logger.info(f"Config set: {extension}.{key} = {value} (source: {source})")

    async def get_network_context(self, extension: str = "security-monitor") -> NetworkContext:
        """Build network context from YAML + learned services.

        Args:
            extension: Extension to get network config for.

        Returns:
            NetworkContext with merged YAML + memory data.
        """
        # Start with YAML config
        ext_config = self._yaml.get("extensions", {}).get(extension, {}).get("config", {})
        network_config = ext_config.get("network", {})

        home_net = network_config.get("home_net", "192.168.0.0/24")
        known_services: list[NetworkService] = []

        # Add services from YAML
        for svc in network_config.get("known_services", []):
            known_services.append(
                NetworkService(
                    name=svc.get("name", "Unknown"),
                    ip=svc.get("ip", ""),
                    ports=svc.get("ports", []),
                    protocols=svc.get("protocols", []),
                    notes=svc.get("notes", ""),
                    learned=False,
                )
            )

        # Add learned services from memory
        if self._memory:
            learned = await self._memory.search(
                query="known service network device IP address",
                user_id="system",
                memory_type=self.NETWORK_MEMORY_TYPE,
                limit=50,
            )

            for item in learned:
                parsed = self._parse_network_service(item)
                if parsed and not self._service_exists(parsed.ip, known_services):
                    parsed.learned = True
                    known_services.append(parsed)

        return NetworkContext(
            home_net=home_net,
            known_services=known_services,
            trusted_ips=network_config.get("trusted_ips", []),
            external_services=network_config.get("external_services", []),
        )

    async def add_network_service(
        self,
        name: str,
        ip: str,
        ports: list[int] | None = None,
        protocols: list[str] | None = None,
        notes: str = "",
        user_id: str = "system",
    ) -> None:
        """Add a learned network service.

        Args:
            name: Service name (e.g., "NAS", "Plex").
            ip: IP address.
            ports: Open ports.
            protocols: Protocols (SMB, HTTP, etc).
            notes: Additional notes.
            user_id: User who added this.
        """
        if not self._memory:
            logger.warning("No memory service, network service not stored")
            return

        content = f"{name} is at {ip}"
        if ports:
            content += f" with ports {', '.join(map(str, ports))}"
        if protocols:
            content += f" running {', '.join(protocols)}"

        await self._memory.store(
            content=content,
            user_id="system",
            memory_type=self.NETWORK_MEMORY_TYPE,
            entities=["network", "service", name.lower(), ip],
            metadata={
                "name": name,
                "ip": ip,
                "ports": ports or [],
                "protocols": protocols or [],
                "notes": notes,
                "added_by": user_id,
            },
        )

        logger.info(f"Network service added: {name} at {ip}")

    async def get_all_config(self, extension: str) -> dict[str, ConfigValue]:
        """Get all configuration for an extension.

        Args:
            extension: Extension name.

        Returns:
            Dict of key → ConfigValue for all known config.
        """
        result: dict[str, ConfigValue] = {}

        # Get from YAML
        ext_config = self._yaml.get("extensions", {}).get(extension, {}).get("config", {})
        for key, value in self._flatten_dict(ext_config).items():
            result[key] = ConfigValue(value=value, source="yaml")

        # Override with memory values
        if self._memory:
            memories = await self._memory.search(
                query=f"extension:{extension} config",
                user_id="system",
                memory_type=self.CONFIG_MEMORY_TYPE,
                limit=100,
            )

            for mem in memories:
                if hasattr(mem, "metadata") and mem.metadata:
                    key = mem.metadata.get("key")
                    value = mem.metadata.get("value")
                    if key:
                        result[key] = ConfigValue(
                            value=value,
                            source="memory",
                            updated_by=mem.metadata.get("set_by"),
                        )

        return result

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()

    # Private methods

    async def _get_from_memory(self, extension: str, key: str) -> Any | None:
        """Get a value from memory."""
        if not self._memory:
            return None

        results = await self._memory.search(
            query=f"extension:{extension} config:{key}",
            user_id="system",
            memory_type=self.CONFIG_MEMORY_TYPE,
            limit=1,
        )

        if results and hasattr(results[0], "metadata"):
            metadata = results[0].metadata or {}
            if metadata.get("key") == key:
                return metadata.get("value")

        return None

    def _get_from_yaml(self, extension: str, key: str) -> Any | None:
        """Get a value from YAML config using dot notation."""
        ext_config = self._yaml.get("extensions", {}).get(extension, {}).get("config", {})

        # Navigate nested keys
        parts = key.split(".")
        current = ext_config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _flatten_dict(self, d: dict, parent_key: str = "") -> dict[str, Any]:
        """Flatten a nested dict with dot notation keys."""
        items: list[tuple[str, Any]] = []

        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def _parse_network_service(self, memory_item: Any) -> NetworkService | None:
        """Parse a network service from a memory item."""
        if hasattr(memory_item, "metadata") and memory_item.metadata:
            meta = memory_item.metadata
            if "ip" in meta:
                return NetworkService(
                    name=meta.get("name", "Unknown"),
                    ip=meta["ip"],
                    ports=meta.get("ports", []),
                    protocols=meta.get("protocols", []),
                    notes=meta.get("notes", ""),
                )

        # Try parsing from content
        if hasattr(memory_item, "content"):
            content = memory_item.content
            # Try to extract "X is at IP" pattern
            match = re.search(r"(\w+)\s+is\s+at\s+(\d+\.\d+\.\d+\.\d+)", content)
            if match:
                return NetworkService(
                    name=match.group(1),
                    ip=match.group(2),
                )

        return None

    def _service_exists(self, ip: str, services: list[NetworkService]) -> bool:
        """Check if a service with this IP already exists."""
        return any(s.ip == ip for s in services)


# Singleton instance
_config_service: ExtensionConfigService | None = None


def get_extension_config_service() -> ExtensionConfigService:
    """Get or create the extension config service singleton."""
    global _config_service
    if _config_service is None:
        _config_service = ExtensionConfigService()
    return _config_service

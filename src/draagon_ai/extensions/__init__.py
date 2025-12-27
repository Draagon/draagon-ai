"""Extension system for draagon-ai.

This module provides the extension infrastructure for draagon-ai,
allowing optional functionality to be packaged as separate installable
extensions. Extensions are discovered via Python entry points and
configured via draagon.yaml files.

Key components:
- Extension: Base class for all extensions
- ExtensionInfo: Metadata about an extension
- ExtensionManager: Manages extension lifecycle
- discover_extensions: Find installed extensions

Example:
    from draagon_ai.extensions import ExtensionManager

    # Create manager and load enabled extensions
    manager = ExtensionManager()
    manager.discover_and_load()

    # Get all behaviors from extensions
    for behavior in manager.get_all_behaviors():
        print(f"Loaded behavior: {behavior.behavior_id}")

    # Get a specific service
    service = manager.get_service("drama_manager")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import (
    Extension,
    ExtensionInfo,
    ExtensionError,
    ExtensionNotFoundError,
    ExtensionLoadError,
    ExtensionDependencyError,
)
from .discovery import (
    discover_extensions,
    discover_extension_info,
    load_extension,
    EXTENSION_GROUP,
)
from .config import (
    load_config,
    DraagonExtensionConfig,
    ExtensionConfig,
)

if TYPE_CHECKING:
    from draagon_ai.behaviors import Behavior

logger = logging.getLogger(__name__)

# Global extension manager instance
_extension_manager: "ExtensionManager | None" = None


class ExtensionManager:
    """Manages extension lifecycle and provides access to extension components.

    The ExtensionManager is responsible for:
    - Discovering installed extensions via entry points
    - Loading configuration from draagon.yaml
    - Initializing enabled extensions
    - Providing access to extension-provided components
    - Shutting down extensions cleanly

    Example:
        from draagon_ai.extensions import ExtensionManager

        # Create and initialize
        manager = ExtensionManager(config_path="draagon.yaml")
        manager.discover_and_load()

        # Access extension components
        behaviors = manager.get_all_behaviors()
        services = manager.get_all_services()

        # Shutdown when done
        manager.shutdown()
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        config: DraagonExtensionConfig | None = None,
    ):
        """Initialize the extension manager.

        Args:
            config_path: Path to draagon.yaml config file.
            config: Pre-loaded configuration (alternative to config_path).
        """
        self._extensions: dict[str, Extension] = {}
        self._config = config or load_config(config_path)
        self._initialized = False

    @property
    def config(self) -> DraagonExtensionConfig:
        """Get the loaded configuration."""
        return self._config

    @property
    def loaded_extensions(self) -> list[str]:
        """Get names of loaded extensions."""
        return list(self._extensions.keys())

    def discover_and_load(self) -> None:
        """Discover and load all enabled extensions.

        This method:
        1. Discovers installed extensions via entry points
        2. Checks which are enabled in configuration
        3. Resolves dependencies
        4. Initializes enabled extensions

        Raises:
            ExtensionDependencyError: If dependencies are not satisfied.
            ExtensionLoadError: If an extension fails to load.
        """
        # Discover available extensions
        available = discover_extensions()
        logger.info(f"Discovered {len(available)} extension(s)")

        # Determine which to load
        to_load: list[str] = []
        for name, ext_class in available.items():
            if self._config.is_extension_enabled(name):
                to_load.append(name)
            else:
                logger.debug(f"Extension '{name}' is disabled")

        # Check dependencies and order by them
        ordered = self._resolve_dependencies(to_load, available)

        # Load extensions in dependency order
        for name in ordered:
            ext_class = available[name]
            ext_config = self._config.get_extension_config(name)
            self._load_extension(name, ext_class, ext_config)

        self._initialized = True
        logger.info(f"Loaded {len(self._extensions)} extension(s)")

    def _resolve_dependencies(
        self,
        to_load: list[str],
        available: dict[str, type[Extension]],
    ) -> list[str]:
        """Resolve extension dependencies and return load order.

        Args:
            to_load: Extensions to load.
            available: All available extensions.

        Returns:
            List of extension names in dependency order.

        Raises:
            ExtensionDependencyError: If dependencies cannot be satisfied.
        """
        # Build dependency graph
        deps: dict[str, list[str]] = {}
        for name in to_load:
            ext_class = available[name]
            try:
                # Instantiate temporarily to get info
                ext = ext_class()
                deps[name] = ext.info.requires_extensions
            except Exception as e:
                logger.warning(f"Failed to get info for '{name}': {e}")
                deps[name] = []

        # Check all dependencies are available
        for name, requires in deps.items():
            missing = [r for r in requires if r not in to_load]
            if missing:
                raise ExtensionDependencyError(name, missing)

        # Simple topological sort
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in deps.get(name, []):
                visit(dep)
            ordered.append(name)

        for name in to_load:
            visit(name)

        return ordered

    def _load_extension(
        self,
        name: str,
        ext_class: type[Extension],
        config: dict[str, Any],
    ) -> None:
        """Load and initialize a single extension.

        Args:
            name: Extension name.
            ext_class: Extension class to instantiate.
            config: Extension configuration.

        Raises:
            ExtensionLoadError: If initialization fails.
        """
        try:
            ext = ext_class()
            ext.initialize(config)
            self._extensions[name] = ext
            logger.info(f"Loaded extension: {name} v{ext.info.version}")
        except Exception as e:
            raise ExtensionLoadError(name, str(e)) from e

    def shutdown(self) -> None:
        """Shutdown all loaded extensions.

        Calls shutdown() on each extension in reverse load order.
        """
        for name in reversed(list(self._extensions.keys())):
            try:
                self._extensions[name].shutdown()
                logger.debug(f"Shutdown extension: {name}")
            except Exception as e:
                logger.warning(f"Error shutting down '{name}': {e}")

        self._extensions.clear()
        self._initialized = False

    def get_extension(self, name: str) -> Extension:
        """Get a loaded extension by name.

        Args:
            name: Extension name.

        Returns:
            The extension instance.

        Raises:
            ExtensionNotFoundError: If extension is not loaded.
        """
        if name not in self._extensions:
            raise ExtensionNotFoundError(name)
        return self._extensions[name]

    def get_all_services(self) -> dict[str, Any]:
        """Aggregate services from all loaded extensions.

        Returns:
            Dict mapping service names to service instances.
        """
        services: dict[str, Any] = {}
        for ext in self._extensions.values():
            ext_services = ext.get_services()
            for svc_name, svc in ext_services.items():
                if svc_name in services:
                    logger.warning(f"Service '{svc_name}' provided by multiple extensions")
                services[svc_name] = svc
        return services

    def get_service(self, name: str) -> Any | None:
        """Get a specific service from extensions.

        Args:
            name: Service name.

        Returns:
            Service instance if found, None otherwise.
        """
        for ext in self._extensions.values():
            services = ext.get_services()
            if name in services:
                return services[name]
        return None

    def get_all_behaviors(self) -> list["Behavior"]:
        """Aggregate behaviors from all loaded extensions.

        Returns:
            List of all Behavior instances from extensions.
        """
        behaviors: list[Behavior] = []
        for ext in self._extensions.values():
            behaviors.extend(ext.get_behaviors())
        return behaviors

    def get_all_tools(self) -> list[Any]:
        """Aggregate tools from all loaded extensions.

        Returns:
            List of all tool instances from extensions.
        """
        tools: list[Any] = []
        for ext in self._extensions.values():
            tools.extend(ext.get_tools())
        return tools


def get_extension_manager() -> ExtensionManager:
    """Get the global extension manager instance.

    Creates and initializes the manager if not already done.

    Returns:
        Global ExtensionManager instance.
    """
    global _extension_manager
    if _extension_manager is None:
        _extension_manager = ExtensionManager()
        _extension_manager.discover_and_load()
    return _extension_manager


def reset_extension_manager() -> None:
    """Reset the global extension manager.

    Shuts down the current manager and clears the global instance.
    Useful for testing.
    """
    global _extension_manager
    if _extension_manager is not None:
        _extension_manager.shutdown()
        _extension_manager = None


__all__ = [
    # Types
    "Extension",
    "ExtensionInfo",
    "ExtensionConfig",
    "DraagonExtensionConfig",
    # Exceptions
    "ExtensionError",
    "ExtensionNotFoundError",
    "ExtensionLoadError",
    "ExtensionDependencyError",
    # Discovery
    "discover_extensions",
    "discover_extension_info",
    "load_extension",
    "EXTENSION_GROUP",
    # Config
    "load_config",
    # Manager
    "ExtensionManager",
    "get_extension_manager",
    "reset_extension_manager",
]

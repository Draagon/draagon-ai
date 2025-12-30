"""Extension discovery via Python entry points and built-ins.

This module provides discovery of installed extensions using:
1. Built-in extensions bundled with draagon-ai
2. Python's entry point mechanism for third-party extensions

Built-in extensions are always available. Third-party extensions
register themselves via pyproject.toml entry points.

Example pyproject.toml for an extension:
    [project.entry-points."draagon_ai.extensions"]
    home_assistant = "draagon_ai_ext_ha:HomeAssistantExtension"

Usage:
    from draagon_ai.extensions.discovery import discover_extensions

    # Discover all extensions (builtins + entry points)
    extensions = discover_extensions()
    for name, ext_class in extensions.items():
        print(f"Found extension: {name}")
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Extension

logger = logging.getLogger(__name__)

# Entry point group name for draagon-ai extensions
EXTENSION_GROUP = "draagon_ai.extensions"


def discover_extensions(include_builtins: bool = True) -> dict[str, type["Extension"]]:
    """Discover installed extensions via builtins and entry points.

    This function discovers extensions from two sources:
    1. Built-in extensions bundled with draagon-ai (command_security, storyteller)
    2. Third-party extensions registered via entry points in pyproject.toml

    The entry points are registered in pyproject.toml like:
        [project.entry-points."draagon_ai.extensions"]
        home_assistant = "draagon_ai_ext_ha:HomeAssistantExtension"

    Args:
        include_builtins: Whether to include built-in extensions (default True)

    Returns:
        Dict mapping extension names to Extension subclasses.

    Example:
        extensions = discover_extensions()
        # {'command_security': <class 'CommandSecurityExtension'>,
        #  'storyteller': <class 'StorytellerExtension'>,
        #  'home_assistant': <class 'HomeAssistantExtension'>, ...}
    """
    extensions: dict[str, type[Extension]] = {}

    # 1. Load built-in extensions first
    if include_builtins:
        try:
            from draagon_ai.extensions.builtins import discover_builtin_extensions

            builtin_exts = discover_builtin_extensions()
            extensions.update(builtin_exts)
            for name in builtin_exts:
                logger.debug(f"Discovered built-in extension: {name}")
        except ImportError as e:
            logger.warning(f"Could not load built-in extensions: {e}")

    # 2. Load entry point extensions (can override builtins if needed)
    eps = entry_points(group=EXTENSION_GROUP)

    for ep in eps:
        try:
            # Load the extension class
            ext_class = ep.load()
            extensions[ep.name] = ext_class
            logger.debug(f"Discovered extension: {ep.name}")
        except Exception as e:
            logger.warning(f"Failed to load extension '{ep.name}': {e}")

    return extensions


def discover_extension_info() -> dict[str, dict]:
    """Discover extensions and get their metadata without loading.

    This is a lightweight version that only gets entry point metadata,
    useful for listing available extensions without importing them.

    Returns:
        Dict mapping extension names to entry point metadata.

    Example:
        info = discover_extension_info()
        for name, meta in info.items():
            print(f"{name}: {meta['module']}")
    """
    info: dict[str, dict] = {}

    eps = entry_points(group=EXTENSION_GROUP)

    for ep in eps:
        info[ep.name] = {
            "name": ep.name,
            "module": ep.value,
            "group": ep.group,
        }

    return info


def load_extension(name: str) -> type["Extension"] | None:
    """Load a specific extension by name.

    Checks built-in extensions first, then entry points.

    Args:
        name: Extension name to load.

    Returns:
        Extension class if found, None otherwise.

    Example:
        ext_class = load_extension("command_security")
        if ext_class:
            ext = ext_class()
            ext.initialize(config)
    """
    # Check built-ins first
    try:
        from draagon_ai.extensions.builtins import get_builtin_extension

        builtin = get_builtin_extension(name)
        if builtin is not None:
            return builtin
    except ImportError:
        pass

    # Check entry points
    eps = entry_points(group=EXTENSION_GROUP)

    for ep in eps:
        if ep.name == name:
            try:
                return ep.load()
            except Exception as e:
                logger.error(f"Failed to load extension '{name}': {e}")
                return None

    return None

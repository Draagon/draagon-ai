"""Extension discovery via Python entry points.

This module provides discovery of installed extensions using Python's
entry point mechanism (Airflow-style). Extensions register themselves
via pyproject.toml entry points.

Example pyproject.toml for an extension:
    [project.entry-points."draagon_ai.extensions"]
    storytelling = "draagon_ai_ext_storytelling:StorytellingExtension"

Usage:
    from draagon_ai.extensions.discovery import discover_extensions

    # Discover all installed extensions
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


def discover_extensions() -> dict[str, type["Extension"]]:
    """Discover installed extensions via entry points.

    This function scans Python's entry points for the draagon_ai.extensions
    group and returns a dict mapping extension names to their classes.

    The entry points are registered in pyproject.toml like:
        [project.entry-points."draagon_ai.extensions"]
        storytelling = "draagon_ai_ext_storytelling:StorytellingExtension"

    Returns:
        Dict mapping extension names to Extension subclasses.

    Example:
        extensions = discover_extensions()
        # {'storytelling': <class 'StorytellingExtension'>, ...}
    """
    extensions: dict[str, type[Extension]] = {}

    # Get entry points for our group
    # Python 3.10+ returns a SelectableGroups object
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

    Args:
        name: Extension name to load.

    Returns:
        Extension class if found, None otherwise.

    Example:
        ext_class = load_extension("storytelling")
        if ext_class:
            ext = ext_class()
            ext.initialize(config)
    """
    eps = entry_points(group=EXTENSION_GROUP)

    for ep in eps:
        if ep.name == name:
            try:
                return ep.load()
            except Exception as e:
                logger.error(f"Failed to load extension '{name}': {e}")
                return None

    return None

"""Built-in extensions for draagon-ai.

These extensions are bundled with draagon-ai and don't require
separate pip installation. They provide common functionality
that many AI agents need.

Built-in Extensions:
- command_security: Secure shell command execution with classification
- storyteller: Interactive storytelling and text adventures

Usage:
    Built-in extensions are automatically discovered and can be enabled
    in draagon.yaml like any other extension:

    extensions:
      command_security:
        enabled: true
        config:
          passcode: "1234"
          backend: "http"

      storyteller:
        enabled: true
        config:
          drama_intensity: 0.7

Third-party extensions (installed via pip) use entry points for discovery.
Built-ins are loaded directly from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from draagon_ai.extensions.types import Extension

# Registry of built-in extensions
# Maps extension name to the class (lazy import to avoid circular deps)
BUILTIN_EXTENSIONS: dict[str, str] = {
    "command_security": "draagon_ai.extensions.builtins.command_security:CommandSecurityExtension",
    "storyteller": "draagon_ai.extensions.builtins.storyteller:StorytellerExtension",
}


def get_builtin_extension(name: str) -> type["Extension"] | None:
    """Get a built-in extension class by name.

    Args:
        name: Extension name (e.g., "command_security")

    Returns:
        Extension class if found, None otherwise.
    """
    if name not in BUILTIN_EXTENSIONS:
        return None

    module_path = BUILTIN_EXTENSIONS[name]
    module_name, class_name = module_path.rsplit(":", 1)

    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to load built-in extension '{name}': {e}"
        )
        return None


def discover_builtin_extensions() -> dict[str, type["Extension"]]:
    """Discover all built-in extensions.

    Returns:
        Dict mapping extension names to their classes.
    """
    extensions: dict[str, type[Extension]] = {}

    for name in BUILTIN_EXTENSIONS:
        ext_class = get_builtin_extension(name)
        if ext_class is not None:
            extensions[name] = ext_class

    return extensions


__all__ = [
    "BUILTIN_EXTENSIONS",
    "get_builtin_extension",
    "discover_builtin_extensions",
]

"""Extension configuration loading.

This module handles loading extension configuration from draagon.yaml files.
The config file controls which extensions are enabled and their settings.

Example draagon.yaml:
    extensions:
      storytelling:
        enabled: true
        config:
          drama_intensity: 0.7

      architect:
        enabled: false

    core:
      evolution:
        min_interactions: 50
        auto_apply: false

    app:
      name: "my-app"
      behaviors:
        - my_behavior

Usage:
    from draagon_ai.extensions.config import load_config

    config = load_config("draagon.yaml")
    if config.is_extension_enabled("storytelling"):
        ext_config = config.get_extension_config("storytelling")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default config file names to search for
DEFAULT_CONFIG_FILES = [
    "draagon.yaml",
    "draagon.yml",
    ".draagon.yaml",
    ".draagon.yml",
]


@dataclass
class ExtensionConfig:
    """Configuration for a single extension."""

    name: str
    """Extension name."""

    enabled: bool = True
    """Whether the extension is enabled."""

    config: dict[str, Any] = field(default_factory=dict)
    """Extension-specific configuration."""


@dataclass
class DraagonExtensionConfig:
    """Full draagon extension configuration.

    This class holds all configuration loaded from draagon.yaml,
    including extension settings, core service settings, and app settings.
    """

    extensions: dict[str, ExtensionConfig] = field(default_factory=dict)
    """Extension configurations by name."""

    core: dict[str, Any] = field(default_factory=dict)
    """Core service configurations (evolution, etc.)."""

    app: dict[str, Any] = field(default_factory=dict)
    """App-specific settings."""

    source_path: Path | None = None
    """Path to the config file that was loaded."""

    def is_extension_enabled(self, name: str) -> bool:
        """Check if an extension is enabled.

        Args:
            name: Extension name.

        Returns:
            True if extension is enabled (or not configured, defaults to True).
        """
        if name not in self.extensions:
            # Extensions are enabled by default if not configured
            return True
        return self.extensions[name].enabled

    def get_extension_config(self, name: str) -> dict[str, Any]:
        """Get configuration for an extension.

        Args:
            name: Extension name.

        Returns:
            Extension-specific config dict (empty if not configured).
        """
        if name not in self.extensions:
            return {}
        return self.extensions[name].config

    def get_core_config(self, service: str) -> dict[str, Any]:
        """Get configuration for a core service.

        Args:
            service: Core service name (e.g., 'evolution').

        Returns:
            Service-specific config dict (empty if not configured).
        """
        return self.core.get(service, {})


def load_config(path: Path | str | None = None) -> DraagonExtensionConfig:
    """Load extension configuration from a YAML file.

    If no path is provided, searches for default config files in the
    current directory and parent directories.

    Args:
        path: Optional path to config file.

    Returns:
        Loaded configuration.

    Example:
        # Load from explicit path
        config = load_config("my-config.yaml")

        # Auto-discover config file
        config = load_config()
    """
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}")
            return DraagonExtensionConfig()
    else:
        config_path = _find_config_file()
        if config_path is None:
            logger.debug("No config file found, using defaults")
            return DraagonExtensionConfig()

    return _load_yaml_config(config_path)


def _find_config_file() -> Path | None:
    """Search for a config file in current and parent directories.

    Returns:
        Path to config file if found, None otherwise.
    """
    # Start from current working directory
    current = Path.cwd()

    # Search up to 5 levels up
    for _ in range(5):
        for filename in DEFAULT_CONFIG_FILES:
            config_path = current / filename
            if config_path.exists():
                logger.debug(f"Found config file: {config_path}")
                return config_path

        # Move to parent directory
        parent = current.parent
        if parent == current:
            # Reached root
            break
        current = parent

    return None


def _load_yaml_config(path: Path) -> DraagonExtensionConfig:
    """Load and parse a YAML config file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, cannot load config files")
        return DraagonExtensionConfig()

    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config file {path}: {e}")
        return DraagonExtensionConfig()

    return _parse_config(raw, path)


def _parse_config(raw: dict, source_path: Path | None = None) -> DraagonExtensionConfig:
    """Parse raw YAML dict into structured config.

    Args:
        raw: Raw dict from YAML parsing.
        source_path: Optional source file path.

    Returns:
        Structured configuration.
    """
    config = DraagonExtensionConfig(source_path=source_path)

    # Parse extensions section
    extensions_raw = raw.get("extensions", {})
    for name, ext_data in extensions_raw.items():
        if isinstance(ext_data, bool):
            # Simple enabled/disabled flag
            config.extensions[name] = ExtensionConfig(
                name=name,
                enabled=ext_data,
            )
        elif isinstance(ext_data, dict):
            # Full configuration
            config.extensions[name] = ExtensionConfig(
                name=name,
                enabled=ext_data.get("enabled", True),
                config=_expand_env_vars(ext_data.get("config", {})),
            )
        else:
            logger.warning(f"Invalid extension config for '{name}': {ext_data}")

    # Parse core section
    config.core = _expand_env_vars(raw.get("core", {}))

    # Parse app section
    config.app = _expand_env_vars(raw.get("app", {}))

    return config


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in config values.

    Supports ${VAR} and $VAR syntax.

    Args:
        data: Config data (dict, list, or scalar).

    Returns:
        Data with environment variables expanded.
    """
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Expand ${VAR} syntax
        return os.path.expandvars(data)
    else:
        return data

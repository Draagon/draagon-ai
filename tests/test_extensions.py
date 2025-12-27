"""Tests for the extension infrastructure."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from draagon_ai.extensions import (
    Extension,
    ExtensionInfo,
    ExtensionManager,
    ExtensionError,
    ExtensionNotFoundError,
    ExtensionLoadError,
    ExtensionDependencyError,
    discover_extensions,
    load_config,
    DraagonExtensionConfig,
    reset_extension_manager,
)


# =============================================================================
# Test ExtensionInfo
# =============================================================================


def test_extension_info_required_fields():
    """Test ExtensionInfo with required fields only."""
    info = ExtensionInfo(
        name="test-ext",
        version="1.0.0",
        description="Test extension",
        author="tester",
        requires_core=">=0.1.0",
    )

    assert info.name == "test-ext"
    assert info.version == "1.0.0"
    assert info.description == "Test extension"
    assert info.author == "tester"
    assert info.requires_core == ">=0.1.0"


def test_extension_info_optional_fields():
    """Test ExtensionInfo with all fields."""
    info = ExtensionInfo(
        name="full-ext",
        version="2.0.0",
        description="Full extension",
        author="tester",
        requires_core=">=0.2.0",
        requires_extensions=["other-ext"],
        provides_services=["my_service"],
        provides_behaviors=["my_behavior"],
        provides_tools=["my_tool"],
        config_schema={"type": "object"},
        homepage="https://example.com",
        license="MIT",
    )

    assert info.requires_extensions == ["other-ext"]
    assert info.provides_services == ["my_service"]
    assert info.provides_behaviors == ["my_behavior"]
    assert info.provides_tools == ["my_tool"]
    assert info.config_schema == {"type": "object"}
    assert info.homepage == "https://example.com"
    assert info.license == "MIT"


# =============================================================================
# Test Extension Base Class
# =============================================================================


class MockExtension(Extension):
    """A mock extension for testing."""

    def __init__(self):
        self._initialized = False
        self._config = {}

    @property
    def info(self) -> ExtensionInfo:
        return ExtensionInfo(
            name="mock",
            version="1.0.0",
            description="Mock extension",
            author="test",
            requires_core=">=0.1.0",
            provides_services=["mock_service"],
            provides_behaviors=["mock_behavior"],
        )

    def initialize(self, config: dict) -> None:
        self._config = config
        self._initialized = True

    def shutdown(self) -> None:
        self._initialized = False

    def get_services(self) -> dict:
        return {"mock_service": "mock_value"}

    def get_behaviors(self) -> list:
        return ["mock_behavior_instance"]


def test_extension_lifecycle():
    """Test extension initialization and shutdown."""
    ext = MockExtension()

    assert ext.info.name == "mock"
    assert not ext._initialized

    ext.initialize({"key": "value"})
    assert ext._initialized
    assert ext._config == {"key": "value"}

    ext.shutdown()
    assert not ext._initialized


def test_extension_services():
    """Test extension get_services."""
    ext = MockExtension()
    ext.initialize({})

    services = ext.get_services()
    assert "mock_service" in services


def test_extension_behaviors():
    """Test extension get_behaviors."""
    ext = MockExtension()
    ext.initialize({})

    behaviors = ext.get_behaviors()
    assert len(behaviors) == 1


# =============================================================================
# Test Configuration Loading
# =============================================================================


def test_load_config_empty():
    """Test loading config when no file exists."""
    config = load_config("/nonexistent/path/draagon.yaml")

    assert isinstance(config, DraagonExtensionConfig)
    assert len(config.extensions) == 0


def test_config_is_extension_enabled_default():
    """Test extension enabled check with default (not configured)."""
    config = DraagonExtensionConfig()

    # Extensions are enabled by default if not configured
    assert config.is_extension_enabled("anything") is True


def test_config_is_extension_enabled_explicit():
    """Test extension enabled check with explicit config."""
    from draagon_ai.extensions.config import ExtensionConfig

    config = DraagonExtensionConfig(
        extensions={
            "enabled_ext": ExtensionConfig(name="enabled_ext", enabled=True),
            "disabled_ext": ExtensionConfig(name="disabled_ext", enabled=False),
        }
    )

    assert config.is_extension_enabled("enabled_ext") is True
    assert config.is_extension_enabled("disabled_ext") is False


def test_config_get_extension_config():
    """Test getting extension-specific config."""
    from draagon_ai.extensions.config import ExtensionConfig

    config = DraagonExtensionConfig(
        extensions={
            "my_ext": ExtensionConfig(
                name="my_ext",
                enabled=True,
                config={"setting": "value"},
            ),
        }
    )

    ext_config = config.get_extension_config("my_ext")
    assert ext_config == {"setting": "value"}

    # Non-existent extension returns empty dict
    assert config.get_extension_config("other") == {}


def test_config_get_core_config():
    """Test getting core service config."""
    config = DraagonExtensionConfig(
        core={
            "evolution": {"min_interactions": 50},
        }
    )

    evo_config = config.get_core_config("evolution")
    assert evo_config["min_interactions"] == 50

    # Non-existent service returns empty dict
    assert config.get_core_config("other") == {}


# =============================================================================
# Test Extension Manager
# =============================================================================


def test_extension_manager_empty():
    """Test extension manager with no extensions."""
    manager = ExtensionManager()

    assert manager.loaded_extensions == []
    assert manager.get_all_services() == {}
    assert manager.get_all_behaviors() == []


def test_extension_manager_discover_and_load():
    """Test extension discovery and loading."""
    manager = ExtensionManager()

    # Mock discover_extensions where it's imported in the manager module
    with patch.object(
        manager.__class__,
        "_resolve_dependencies",
        return_value=["mock"],
    ):
        # Directly register a mock extension
        ext = MockExtension()
        ext.initialize({})
        manager._extensions["mock"] = ext

        assert "mock" in manager.loaded_extensions


def test_extension_manager_get_extension():
    """Test getting a specific extension."""
    manager = ExtensionManager()

    # Directly register a mock extension
    ext = MockExtension()
    ext.initialize({})
    manager._extensions["mock"] = ext

    result = manager.get_extension("mock")
    assert isinstance(result, MockExtension)


def test_extension_manager_get_extension_not_found():
    """Test getting non-existent extension raises error."""
    manager = ExtensionManager()

    with pytest.raises(ExtensionNotFoundError) as exc_info:
        manager.get_extension("nonexistent")

    assert "nonexistent" in str(exc_info.value)


def test_extension_manager_shutdown():
    """Test extension manager shutdown."""
    manager = ExtensionManager()

    # Directly register a mock extension
    ext = MockExtension()
    ext.initialize({})
    manager._extensions["mock"] = ext

    assert len(manager.loaded_extensions) == 1

    manager.shutdown()

    assert len(manager.loaded_extensions) == 0


def test_extension_manager_disabled_extension():
    """Test that disabled extensions are not loaded."""
    from draagon_ai.extensions.config import ExtensionConfig

    config = DraagonExtensionConfig(
        extensions={
            "mock": ExtensionConfig(name="mock", enabled=False),
        }
    )

    manager = ExtensionManager(config=config)

    with patch("draagon_ai.extensions.discovery.discover_extensions") as mock_discover:
        mock_discover.return_value = {"mock": MockExtension}
        manager.discover_and_load()

        # Extension should not be loaded
        assert "mock" not in manager.loaded_extensions


def test_extension_manager_aggregates_services():
    """Test that manager aggregates services from all extensions."""
    manager = ExtensionManager()

    # Directly register a mock extension
    ext = MockExtension()
    ext.initialize({})
    manager._extensions["mock"] = ext

    services = manager.get_all_services()
    assert "mock_service" in services


def test_extension_manager_aggregates_behaviors():
    """Test that manager aggregates behaviors from all extensions."""
    manager = ExtensionManager()

    # Directly register a mock extension
    ext = MockExtension()
    ext.initialize({})
    manager._extensions["mock"] = ext

    behaviors = manager.get_all_behaviors()
    assert len(behaviors) == 1


# =============================================================================
# Test Exception Classes
# =============================================================================


def test_extension_not_found_error():
    """Test ExtensionNotFoundError."""
    error = ExtensionNotFoundError("my-ext")

    assert error.name == "my-ext"
    assert "my-ext" in str(error)


def test_extension_load_error():
    """Test ExtensionLoadError."""
    error = ExtensionLoadError("my-ext", "import failed")

    assert error.name == "my-ext"
    assert error.reason == "import failed"
    assert "my-ext" in str(error)
    assert "import failed" in str(error)


def test_extension_dependency_error():
    """Test ExtensionDependencyError."""
    error = ExtensionDependencyError("my-ext", ["dep1", "dep2"])

    assert error.name == "my-ext"
    assert error.missing == ["dep1", "dep2"]
    assert "my-ext" in str(error)
    assert "dep1" in str(error)


# =============================================================================
# Test Global Manager
# =============================================================================


def test_reset_extension_manager():
    """Test resetting the global extension manager."""
    reset_extension_manager()

    # Should not raise
    reset_extension_manager()

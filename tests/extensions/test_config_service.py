"""Tests for ExtensionConfigService - hybrid YAML + memory configuration.

These tests focus on REAL behavior:
- YAML traversal and parsing logic
- Configuration priority and fallback
- Network context building
- Edge cases

Integration tests with real Qdrant are in test_config_service_integration.py
"""

import pytest
from unittest.mock import AsyncMock
from dataclasses import dataclass

from draagon_ai.extensions.config_service import (
    ExtensionConfigService,
    ConfigValue,
    NetworkContext,
    NetworkService,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def yaml_config():
    """Sample YAML configuration."""
    return {
        "extensions": {
            "security-monitor": {
                "config": {
                    "check_interval_seconds": 300,
                    "notifications": {
                        "voice": {
                            "enabled": True,
                            "min_severity": "high",
                            "quiet_hours": {
                                "enabled": True,
                                "start": "22:00",
                                "end": "08:00",
                            },
                        },
                    },
                    "network": {
                        "home_net": "192.168.168.0/24",
                        "known_services": [
                            {"name": "Plex", "ip": "192.168.168.204", "ports": [32400]},
                            {"name": "Home Assistant", "ip": "192.168.168.206"},
                        ],
                        "trusted_ips": ["192.168.168.1"],
                        "external_services": ["Cloudflare"],
                    },
                },
            },
        },
    }


@pytest.fixture
def config_service(yaml_config):
    """Create a config service with sample config."""
    return ExtensionConfigService(yaml_config=yaml_config)


# =============================================================================
# Test YAML Traversal Logic
# =============================================================================


@pytest.mark.asyncio
async def test_get_simple_key(config_service):
    """Test getting a top-level config value."""
    value = await config_service.get("security-monitor", "check_interval_seconds")
    assert value == 300


@pytest.mark.asyncio
async def test_get_nested_key_with_dot_notation(config_service):
    """Test dot notation traversal into nested config."""
    value = await config_service.get(
        "security-monitor",
        "notifications.voice.min_severity",
    )
    assert value == "high"

    # Deeper nesting
    value = await config_service.get(
        "security-monitor",
        "notifications.voice.quiet_hours.start",
    )
    assert value == "22:00"


@pytest.mark.asyncio
async def test_get_returns_default_for_missing_key(config_service):
    """Test that missing keys return the default."""
    value = await config_service.get(
        "security-monitor",
        "nonexistent.deeply.nested.key",
        default="fallback",
    )
    assert value == "fallback"


@pytest.mark.asyncio
async def test_get_returns_default_for_missing_extension(config_service):
    """Test that missing extensions return the default."""
    value = await config_service.get(
        "unknown-extension",
        "any.key",
        default="default_value",
    )
    assert value == "default_value"


@pytest.mark.asyncio
async def test_get_with_empty_yaml():
    """Test behavior with empty YAML config."""
    service = ExtensionConfigService(yaml_config={})
    value = await service.get("any-ext", "any.key", default="empty")
    assert value == "empty"


# =============================================================================
# Test Cache Behavior
# =============================================================================


@pytest.mark.asyncio
async def test_cache_bypass_returns_fresh_value(config_service):
    """Test that use_cache=False bypasses cache."""
    # Prime the cache
    await config_service.get("security-monitor", "check_interval_seconds")

    # Manually corrupt cache
    config_service._cache["security-monitor:check_interval_seconds"] = ConfigValue(
        value=999,
        source="cache",
    )

    # With cache: returns corrupted value
    cached = await config_service.get("security-monitor", "check_interval_seconds")
    assert cached == 999

    # Without cache: returns real value
    fresh = await config_service.get(
        "security-monitor",
        "check_interval_seconds",
        use_cache=False,
    )
    assert fresh == 300


# =============================================================================
# Test Network Context Building
# =============================================================================


@pytest.mark.asyncio
async def test_network_context_from_yaml(config_service):
    """Test building network context from YAML."""
    context = await config_service.get_network_context("security-monitor")

    assert context.home_net == "192.168.168.0/24"
    assert len(context.known_services) == 2
    assert context.known_services[0].name == "Plex"
    assert context.known_services[0].ip == "192.168.168.204"
    assert context.known_services[0].ports == [32400]
    assert context.known_services[0].learned is False
    assert context.trusted_ips == ["192.168.168.1"]
    assert "Cloudflare" in context.external_services


@pytest.mark.asyncio
async def test_network_context_deduplicates_by_ip():
    """Test that duplicate IPs from memory don't create duplicates."""
    @dataclass
    class MockMemory:
        content: str
        metadata: dict

    mock_memory = AsyncMock()
    # Memory returns a service with same IP as YAML
    mock_memory.search.return_value = [
        MockMemory(
            content="Media Server is at 192.168.168.204",
            metadata={
                "name": "Media Server",
                "ip": "192.168.168.204",  # Same as Plex in YAML
            },
        )
    ]

    yaml_config = {
        "extensions": {
            "security-monitor": {
                "config": {
                    "network": {
                        "home_net": "192.168.168.0/24",
                        "known_services": [
                            {"name": "Plex", "ip": "192.168.168.204"},
                        ],
                    },
                },
            },
        },
    }

    service = ExtensionConfigService(
        memory_service=mock_memory,
        yaml_config=yaml_config,
    )

    context = await service.get_network_context("security-monitor")

    # Should have only 1 service (deduped by IP)
    assert len(context.known_services) == 1
    ips = [s.ip for s in context.known_services]
    assert ips.count("192.168.168.204") == 1


@pytest.mark.asyncio
async def test_network_context_uses_default_home_net():
    """Test default home_net when not configured."""
    service = ExtensionConfigService(yaml_config={})
    context = await service.get_network_context("security-monitor")

    assert context.home_net == "192.168.0.0/24"  # Default
    assert context.known_services == []


@pytest.mark.asyncio
async def test_network_context_parses_content_string():
    """Test parsing network service from content when metadata is missing."""
    @dataclass
    class MockMemory:
        content: str
        metadata: dict | None = None

    mock_memory = AsyncMock()
    mock_memory.search.return_value = [
        MockMemory(
            content="Printer is at 192.168.168.230",
            metadata=None,
        )
    ]

    service = ExtensionConfigService(
        memory_service=mock_memory,
        yaml_config={"extensions": {"security-monitor": {"config": {"network": {}}}}},
    )

    context = await service.get_network_context("security-monitor")
    names = [s.name for s in context.known_services]
    assert "Printer" in names


# =============================================================================
# Test Config Flattening
# =============================================================================


@pytest.mark.asyncio
async def test_get_all_config_flattens_nested(config_service):
    """Test that get_all_config flattens nested keys."""
    all_config = await config_service.get_all_config("security-monitor")

    # Should have flattened keys
    assert "check_interval_seconds" in all_config
    assert all_config["check_interval_seconds"].value == 300

    assert "notifications.voice.min_severity" in all_config
    assert all_config["notifications.voice.min_severity"].value == "high"

    assert "notifications.voice.quiet_hours.enabled" in all_config
    assert all_config["notifications.voice.quiet_hours.enabled"].value is True

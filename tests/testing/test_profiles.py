"""Tests for app profiles (TASK-007)."""

from __future__ import annotations

import pytest

from draagon_ai.testing.profiles import (
    AppProfile,
    ToolSet,
    DEFAULT_PROFILE,
    RESEARCHER_PROFILE,
    ASSISTANT_PROFILE,
    MINIMAL_PROFILE,
)


# =============================================================================
# Test ToolSet Enum
# =============================================================================


class TestToolSet:
    """Tests for ToolSet enum."""

    def test_minimal_value(self):
        """MINIMAL has correct value."""
        assert ToolSet.MINIMAL.value == "minimal"

    def test_basic_value(self):
        """BASIC has correct value."""
        assert ToolSet.BASIC.value == "basic"

    def test_full_value(self):
        """FULL has correct value."""
        assert ToolSet.FULL.value == "full"


# =============================================================================
# Test AppProfile
# =============================================================================


class TestAppProfile:
    """Tests for AppProfile dataclass."""

    def test_create_minimal_profile(self):
        """Can create profile with minimal args."""
        profile = AppProfile(
            name="test",
            personality="Test personality",
        )

        assert profile.name == "test"
        assert profile.personality == "Test personality"
        assert profile.tool_set == ToolSet.BASIC  # Default
        assert profile.memory_config == {}  # Default
        assert profile.llm_model_tier == "standard"  # Default

    def test_create_full_profile(self):
        """Can create profile with all args."""
        profile = AppProfile(
            name="full",
            personality="Full personality",
            tool_set=ToolSet.FULL,
            memory_config={"working_ttl": 300},
            llm_model_tier="advanced",
            extra_config={"custom": "value"},
        )

        assert profile.tool_set == ToolSet.FULL
        assert profile.memory_config == {"working_ttl": 300}
        assert profile.llm_model_tier == "advanced"
        assert profile.extra_config == {"custom": "value"}

    def test_profile_is_immutable_style(self):
        """Profile behaves like immutable config."""
        profile = AppProfile(
            name="test",
            personality="Test",
        )

        # Can access but not typically modify
        assert profile.name == "test"


# =============================================================================
# Test Pre-defined Profiles
# =============================================================================


class TestPreDefinedProfiles:
    """Tests for pre-defined profiles."""

    def test_default_profile(self):
        """DEFAULT_PROFILE is valid."""
        assert DEFAULT_PROFILE.name == "default"
        assert DEFAULT_PROFILE.tool_set == ToolSet.BASIC
        assert "helpful" in DEFAULT_PROFILE.personality.lower()

    def test_researcher_profile(self):
        """RESEARCHER_PROFILE is configured for research."""
        assert RESEARCHER_PROFILE.name == "researcher"
        assert RESEARCHER_PROFILE.tool_set == ToolSet.FULL
        assert RESEARCHER_PROFILE.llm_model_tier == "advanced"
        assert "research" in RESEARCHER_PROFILE.personality.lower()

    def test_assistant_profile(self):
        """ASSISTANT_PROFILE is configured for daily tasks."""
        assert ASSISTANT_PROFILE.name == "assistant"
        assert ASSISTANT_PROFILE.tool_set == ToolSet.BASIC
        assert ASSISTANT_PROFILE.memory_config.get("working_ttl") == 600
        assert "assistant" in ASSISTANT_PROFILE.personality.lower()

    def test_minimal_profile(self):
        """MINIMAL_PROFILE is configured for no tools."""
        assert MINIMAL_PROFILE.name == "minimal"
        assert MINIMAL_PROFILE.tool_set == ToolSet.MINIMAL
        assert MINIMAL_PROFILE.llm_model_tier == "fast"


# =============================================================================
# Test Profile Usage Patterns
# =============================================================================


class TestProfilePatterns:
    """Tests for common profile usage patterns."""

    def test_can_override_profile_values(self):
        """Can create modified version of profile."""
        from dataclasses import replace

        # Use dataclasses.replace to create modified copy
        modified = replace(
            DEFAULT_PROFILE,
            name="modified_default",
            tool_set=ToolSet.FULL,
        )

        assert modified.name == "modified_default"
        assert modified.tool_set == ToolSet.FULL
        # Original unchanged
        assert DEFAULT_PROFILE.tool_set == ToolSet.BASIC

    def test_can_merge_memory_configs(self):
        """Can merge memory configs from base profile."""
        base_config = {"working_ttl": 300, "episodic_ttl": 3600}
        override_config = {"working_ttl": 600}

        merged = {**base_config, **override_config}

        assert merged["working_ttl"] == 600  # Overridden
        assert merged["episodic_ttl"] == 3600  # Kept from base

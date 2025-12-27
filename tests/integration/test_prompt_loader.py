"""Integration tests for PromptLoader.

These tests verify the prompt loading system works correctly with Qdrant.
"""

import pytest
from uuid import uuid4

from draagon_ai.prompts import (
    PromptLoader,
    PromptDomain,
    PromptStatus,
)
from draagon_ai.prompts.domains import ALL_PROMPTS


# Use a unique collection for each test run to avoid conflicts
TEST_COLLECTION = f"test_prompts_{uuid4().hex[:8]}"


@pytest.fixture
async def loader():
    """Create a PromptLoader for testing."""
    loader = PromptLoader(
        qdrant_url="http://192.168.168.216:6333",
        collection_name=TEST_COLLECTION,
    )
    await loader.initialize()
    yield loader
    # Cleanup: delete the test collection
    try:
        await loader._registry._client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass
    await loader.close()


class TestPromptLoading:
    """Tests for loading prompts into Qdrant."""

    @pytest.mark.asyncio
    async def test_load_single_prompt(self, loader: PromptLoader):
        """Test loading a single prompt."""
        test_prompts = {
            "test": {
                "TEST_PROMPT": "Hello {name}, this is a test prompt.",
            }
        }

        stats = await loader.load_prompts(test_prompts)

        assert stats["test"] == 1

        # Verify it was stored
        prompt = await loader.get_prompt("test", "TEST_PROMPT")
        assert prompt is not None
        assert prompt.name == "TEST_PROMPT"
        assert "Hello {name}" in prompt.content

    @pytest.mark.asyncio
    async def test_load_multiple_prompts(self, loader: PromptLoader):
        """Test loading multiple prompts in a domain."""
        test_prompts = {
            "routing": {
                "PROMPT_A": "First prompt {var1}",
                "PROMPT_B": "Second prompt {var2}",
                "PROMPT_C": "Third prompt",
            }
        }

        stats = await loader.load_prompts(test_prompts)

        assert stats["routing"] == 3

        # Verify all stored
        prompts = await loader.list_prompts(domain="routing")
        assert len(prompts) == 3

    @pytest.mark.asyncio
    async def test_skip_unchanged_prompts(self, loader: PromptLoader):
        """Test that unchanged prompts are not reloaded."""
        test_prompts = {
            "decision": {
                "STABLE_PROMPT": "This content won't change",
            }
        }

        # First load
        stats1 = await loader.load_prompts(test_prompts)
        assert stats1["decision"] == 1

        # Second load (same content)
        stats2 = await loader.load_prompts(test_prompts)
        assert stats2["decision"] == 0  # No new loads

    @pytest.mark.asyncio
    async def test_force_reload(self, loader: PromptLoader):
        """Test force reload even if unchanged."""
        test_prompts = {
            "decision": {
                "FORCE_PROMPT": "Original content",
            }
        }

        # First load
        await loader.load_prompts(test_prompts)

        # Force reload
        stats = await loader.load_prompts(test_prompts, force=True)
        assert stats["decision"] == 1

    @pytest.mark.asyncio
    async def test_load_changed_prompt(self, loader: PromptLoader):
        """Test loading a prompt with changed content."""
        # Initial load
        initial = {"decision": {"CHANGING_PROMPT": "Version 1"}}
        await loader.load_prompts(initial)

        # Changed content
        updated = {"decision": {"CHANGING_PROMPT": "Version 2"}}
        stats = await loader.load_prompts(updated)
        assert stats["decision"] == 1

        # Verify new content
        prompt = await loader.get_prompt("decision", "CHANGING_PROMPT")
        assert "Version 2" in prompt.content


class TestPromptRetrieval:
    """Tests for retrieving prompts."""

    @pytest.mark.asyncio
    async def test_get_prompt_content(self, loader: PromptLoader):
        """Test getting prompt content directly."""
        test_prompts = {
            "synthesis": {
                "CONTENT_PROMPT": "Direct content access",
            }
        }
        await loader.load_prompts(test_prompts)

        content = await loader.get_prompt_content("synthesis", "CONTENT_PROMPT")
        assert content == "Direct content access"

    @pytest.mark.asyncio
    async def test_get_nonexistent_prompt(self, loader: PromptLoader):
        """Test getting a prompt that doesn't exist."""
        prompt = await loader.get_prompt("fake", "FAKE_PROMPT")
        assert prompt is None

    @pytest.mark.asyncio
    async def test_fill_prompt(self, loader: PromptLoader):
        """Test filling prompt template variables."""
        template = "Hello {name}, you asked: {question}"

        filled = loader.fill_prompt(
            template,
            name="Doug",
            question="What time is it?",
        )

        assert filled == "Hello Doug, you asked: What time is it?"

    @pytest.mark.asyncio
    async def test_fill_prompt_partial(self, loader: PromptLoader):
        """Test partial fill when some variables missing."""
        template = "Hello {name}, {unknown_var}"

        filled = loader.fill_prompt(template, name="Doug")

        assert "Hello Doug" in filled
        assert "{unknown_var}" in filled  # Unfilled variable preserved


class TestShadowVersions:
    """Tests for shadow version management."""

    @pytest.mark.asyncio
    async def test_create_shadow_version(self, loader: PromptLoader):
        """Test creating a shadow version for A/B testing."""
        # Create base prompt
        test_prompts = {"decision": {"EVOLVING_PROMPT": "Original version"}}
        await loader.load_prompts(test_prompts)

        # Create shadow
        shadow = await loader.create_shadow_version(
            domain="decision",
            name="EVOLVING_PROMPT",
            content="Evolved version",
            parent_version="1.0.0",
        )

        assert shadow is not None
        # Shadow versions don't auto-activate - current_version stays at 1.0.0
        assert shadow.current_version == "1.0.0"
        # But the shadow version exists in versions
        assert "1.1.0" in shadow.versions
        assert shadow.versions["1.1.0"].status == PromptStatus.SHADOW


class TestDomainListing:
    """Tests for listing domains and prompts."""

    @pytest.mark.asyncio
    async def test_list_domains(self, loader: PromptLoader):
        """Test listing all domains."""
        test_prompts = {
            "routing": {"R1": "Route 1"},
            "decision": {"D1": "Decision 1"},
            "synthesis": {"S1": "Synth 1"},
        }
        await loader.load_prompts(test_prompts)

        domains = await loader.list_domains()

        assert "routing" in domains
        assert "decision" in domains
        assert "synthesis" in domains

    @pytest.mark.asyncio
    async def test_list_prompts_all(self, loader: PromptLoader):
        """Test listing all prompts."""
        test_prompts = {
            "routing": {"R1": "Route 1", "R2": "Route 2"},
            "decision": {"D1": "Decision 1"},
        }
        await loader.load_prompts(test_prompts)

        all_prompts = await loader.list_prompts()

        assert len(all_prompts) == 3

    @pytest.mark.asyncio
    async def test_list_prompts_filtered(self, loader: PromptLoader):
        """Test listing prompts by domain."""
        test_prompts = {
            "routing": {"R1": "Route 1", "R2": "Route 2"},
            "decision": {"D1": "Decision 1"},
        }
        await loader.load_prompts(test_prompts)

        routing_prompts = await loader.list_prompts(domain="routing")

        assert len(routing_prompts) == 2
        assert all(p.domain == PromptDomain.ROUTING for p in routing_prompts)


class TestRoxyPromptLoading:
    """Tests for loading actual Roxy prompts."""

    @pytest.mark.asyncio
    async def test_load_roxy_routing_prompts(self, loader: PromptLoader):
        """Test loading Roxy routing prompts."""
        routing_only = {"routing": ALL_PROMPTS["routing"]}
        stats = await loader.load_prompts(routing_only)

        assert stats["routing"] >= 2  # INTENT_CLASSIFICATION, FAST_ROUTE

        # Verify specific prompt
        fast_route = await loader.get_prompt("routing", "FAST_ROUTE_PROMPT")
        assert fast_route is not None
        assert "FAST-PATH ACTIONS" in fast_route.content

    @pytest.mark.asyncio
    async def test_load_roxy_decision_prompt(self, loader: PromptLoader):
        """Test loading Roxy decision prompt."""
        decision_only = {"decision": ALL_PROMPTS["decision"]}
        stats = await loader.load_prompts(decision_only)

        assert stats["decision"] == 1

        decision = await loader.get_prompt("decision", "DECISION_PROMPT")
        assert decision is not None
        assert "AVAILABLE ACTIONS" in decision.content
        assert "memory_update" in decision.content

    @pytest.mark.asyncio
    async def test_load_all_roxy_prompts(self, loader: PromptLoader):
        """Test loading all Roxy prompt domains."""
        stats = await loader.load_prompts(ALL_PROMPTS)

        # Verify all domains loaded
        assert "routing" in stats
        assert "decision" in stats
        assert "synthesis" in stats
        assert "memory" in stats
        assert "quality" in stats

        # Verify counts (some prompts are dicts, not strings)
        assert stats["routing"] >= 2
        assert stats["decision"] >= 1
        assert stats["synthesis"] >= 2

    @pytest.mark.asyncio
    async def test_roxy_prompt_variables(self, loader: PromptLoader):
        """Test that Roxy prompts have expected variables."""
        decision_only = {"decision": ALL_PROMPTS["decision"]}
        await loader.load_prompts(decision_only)

        decision = await loader.get_prompt("decision", "DECISION_PROMPT")

        # Check for expected template variables
        assert "{question}" in decision.content
        assert "{user_id}" in decision.content
        assert "{context}" in decision.content

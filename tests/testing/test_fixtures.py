"""Tests for integration testing fixtures (TASK-003)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from draagon_ai.testing.fixtures import (
    SeedApplicator,
    seed_factory,
    seed,
)
from draagon_ai.testing.seeds import (
    SeedFactory,
    SeedItem,
    SeedSet,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_global_registry():
    """Clear global registry before and after each test."""
    SeedFactory.clear_global_registry()
    yield
    SeedFactory.clear_global_registry()


@pytest.fixture
def mock_provider():
    """Create a mock MemoryProvider."""
    provider = MagicMock()
    provider.store = AsyncMock(side_effect=lambda *args, **kwargs: f"mem_{id(args)}")
    provider.delete = AsyncMock(return_value=True)
    return provider


# =============================================================================
# Test SeedApplicator
# =============================================================================


class TestSeedApplicator:
    """Tests for SeedApplicator helper class."""

    def test_creates_with_factory(self):
        """SeedApplicator can be created with just a factory."""
        factory = SeedFactory()
        applicator = SeedApplicator(factory)

        assert applicator.factory is factory
        assert applicator.default_provider is None

    def test_creates_with_default_provider(self, mock_provider):
        """SeedApplicator can have a default provider."""
        factory = SeedFactory()
        applicator = SeedApplicator(factory, mock_provider)

        assert applicator.default_provider is mock_provider

    @pytest.mark.asyncio
    async def test_apply_uses_default_provider(self, mock_provider):
        """apply() uses default provider when none specified."""

        @SeedFactory.register("test_seed")
        class TestSeed(SeedItem):
            async def create(self, provider, **deps):
                await provider.store("test content")
                return "mem_test"

        factory = SeedFactory()
        applicator = SeedApplicator(factory, mock_provider)
        seed_set = SeedSet("test", ["test_seed"])

        results = await applicator.apply(seed_set)

        assert "test_seed" in results
        mock_provider.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_uses_override_provider(self, mock_provider):
        """apply() uses override provider when specified."""

        @SeedFactory.register("test_seed")
        class TestSeed(SeedItem):
            async def create(self, provider, **deps):
                await provider.store("test content")
                return "mem_test"

        # Create with default provider, but override
        default_provider = MagicMock()
        default_provider.store = AsyncMock(return_value="default_mem")

        factory = SeedFactory()
        applicator = SeedApplicator(factory, default_provider)
        seed_set = SeedSet("test", ["test_seed"])

        await applicator.apply(seed_set, provider=mock_provider)

        # Override should be used, not default
        mock_provider.store.assert_called_once()
        default_provider.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_raises_without_provider(self):
        """apply() raises if no provider available."""

        @SeedFactory.register("test_seed")
        class TestSeed(SeedItem):
            async def create(self, provider, **deps):
                return "mem_test"

        factory = SeedFactory()
        applicator = SeedApplicator(factory)  # No default provider
        seed_set = SeedSet("test", ["test_seed"])

        with pytest.raises(ValueError, match="No MemoryProvider available"):
            await applicator.apply(seed_set)

    @pytest.mark.asyncio
    async def test_cleanup_uses_factory_cleanup(self, mock_provider):
        """cleanup() delegates to factory.cleanup()."""

        @SeedFactory.register("test_seed")
        class TestSeed(SeedItem):
            async def create(self, provider, **deps):
                return "mem_test"

        factory = SeedFactory()
        applicator = SeedApplicator(factory, mock_provider)
        seed_set = SeedSet("test", ["test_seed"])

        # Apply seeds
        await applicator.apply(seed_set)

        # Cleanup
        deleted = await applicator.cleanup()

        assert deleted == 1
        mock_provider.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_without_provider(self):
        """cleanup() returns 0 if no provider."""
        factory = SeedFactory()
        applicator = SeedApplicator(factory)  # No default provider

        deleted = await applicator.cleanup()

        assert deleted == 0


# =============================================================================
# Test seed_factory Fixture
# =============================================================================


class TestSeedFactoryFixture:
    """Tests for seed_factory fixture."""

    def test_returns_fresh_factory(self, seed_factory):
        """seed_factory returns a fresh SeedFactory instance."""
        assert isinstance(seed_factory, SeedFactory)

    def test_factory_has_global_seeds(self, seed_factory):
        """Factory includes globally registered seeds."""

        @SeedFactory.register("global_test")
        class GlobalTestSeed(SeedItem):
            async def create(self, provider, **deps):
                return "global"

        # Need a new factory to pick up the global seed
        factory = SeedFactory()
        assert "global_test" in factory.list_seeds()

    def test_factories_are_isolated(self, seed_factory):
        """Each factory instance is isolated."""
        # Register a local seed on the fixture's factory
        class LocalSeed(SeedItem):
            async def create(self, provider, **deps):
                return "local"

        seed_factory.register_instance("local_only", LocalSeed())

        # Create a new factory - shouldn't have the local seed
        new_factory = SeedFactory()
        assert "local_only" in seed_factory.list_seeds()
        assert "local_only" not in new_factory.list_seeds()


# =============================================================================
# Test seed Fixture
# =============================================================================


class TestSeedFixture:
    """Tests for seed fixture."""

    def test_returns_applicator(self, seed):
        """seed fixture returns a SeedApplicator."""
        assert isinstance(seed, SeedApplicator)

    def test_applicator_has_no_default_provider(self, seed):
        """Applicator from fixture has no default provider."""
        assert seed.default_provider is None

    @pytest.mark.asyncio
    async def test_can_apply_with_explicit_provider(self, seed, mock_provider):
        """Can apply seeds by passing provider explicitly."""

        @SeedFactory.register("explicit_test")
        class ExplicitTestSeed(SeedItem):
            async def create(self, provider, **deps):
                return "explicit_mem"

        # Register in the fixture's factory
        seed.factory.register_instance(
            "explicit_test",
            ExplicitTestSeed()
        )

        seed_set = SeedSet("test", ["explicit_test"])
        results = await seed.apply(seed_set, provider=mock_provider)

        assert "explicit_test" in results


# =============================================================================
# Test Parallel Safety
# =============================================================================


class TestParallelSafety:
    """Tests for parallel test execution safety."""

    def test_multiple_factories_isolated(self):
        """Multiple factory instances don't share state."""

        @SeedFactory.register("shared_global")
        class SharedGlobalSeed(SeedItem):
            async def create(self, provider, **deps):
                return "shared"

        factory1 = SeedFactory()
        factory2 = SeedFactory()

        # Register local to factory1
        class Local1(SeedItem):
            async def create(self, provider, **deps):
                return "local1"

        factory1.register_instance("local1", Local1())

        # factory2 should not see local1
        assert "local1" in factory1.list_seeds()
        assert "local1" not in factory2.list_seeds()

        # Both should have global
        assert "shared_global" in factory1.list_seeds()
        assert "shared_global" in factory2.list_seeds()

    @pytest.mark.asyncio
    async def test_applicators_isolated(self, mock_provider):
        """Multiple applicators don't share tracked memories."""

        @SeedFactory.register("tracking_test")
        class TrackingSeed(SeedItem):
            async def create(self, provider, **deps):
                return "tracked_mem"

        factory1 = SeedFactory()
        factory2 = SeedFactory()

        applicator1 = SeedApplicator(factory1, mock_provider)
        applicator2 = SeedApplicator(factory2, mock_provider)

        seed_set = SeedSet("test", ["tracking_test"])

        # Apply in applicator1
        await applicator1.apply(seed_set)

        # applicator2 should have nothing to cleanup
        deleted1 = await applicator1.cleanup()
        deleted2 = await applicator2.cleanup()

        assert deleted1 == 1
        assert deleted2 == 0

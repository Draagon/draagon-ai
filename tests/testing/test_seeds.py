"""Tests for seed factory and base classes (TASK-001)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from draagon_ai.testing.seeds import (
    CircularDependencyError,
    SeedFactory,
    SeedItem,
    SeedNotFoundError,
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
# Test SeedItem Base Class
# =============================================================================


class TestSeedItem:
    """Tests for SeedItem abstract base class."""

    def test_seed_item_is_abstract(self):
        """SeedItem cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            SeedItem()

    def test_seed_item_requires_create_method(self):
        """Subclasses must implement create()."""

        class IncompleteSeed(SeedItem):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteSeed()

    def test_seed_item_dependencies_default_empty(self):
        """Default dependencies is empty list."""

        class SimpleSeed(SeedItem):
            async def create(self, provider, **deps):
                return "test"

        seed = SimpleSeed()
        assert seed.dependencies == []

    def test_seed_item_can_declare_dependencies(self):
        """Seeds can declare dependencies."""

        class DependentSeed(SeedItem):
            dependencies = ["parent_seed"]

            async def create(self, provider, **deps):
                return "test"

        seed = DependentSeed()
        assert seed.dependencies == ["parent_seed"]


# =============================================================================
# Test SeedSet
# =============================================================================


class TestSeedSet:
    """Tests for SeedSet dataclass."""

    def test_create_seed_set(self):
        """Can create a SeedSet."""
        seed_set = SeedSet("my_scenario", ["seed_a", "seed_b"])
        assert seed_set.name == "my_scenario"
        assert seed_set.seed_ids == ["seed_a", "seed_b"]

    def test_seed_set_with_description(self):
        """SeedSet can have a description."""
        seed_set = SeedSet(
            "user_with_cats",
            ["user_doug", "cat_names"],
            description="User profile with cat information",
        )
        assert seed_set.description == "User profile with cat information"

    def test_seed_set_rejects_duplicates(self):
        """SeedSet rejects duplicate seed IDs."""
        with pytest.raises(ValueError, match="Duplicate seeds"):
            SeedSet("bad_set", ["seed_a", "seed_b", "seed_a"])


# =============================================================================
# Test SeedFactory Registration
# =============================================================================


class TestSeedFactoryRegistration:
    """Tests for SeedFactory registration methods."""

    def test_register_decorator(self):
        """@SeedFactory.register() registers seed classes globally."""

        @SeedFactory.register("test_seed")
        class TestSeed(SeedItem):
            async def create(self, provider, **deps):
                return "created"

        # Verify global registration
        assert "test_seed" in SeedFactory._global_seed_classes
        assert SeedFactory._global_seed_classes["test_seed"] is TestSeed

    def test_register_decorator_requires_seed_item(self):
        """@SeedFactory.register() requires SeedItem subclass."""
        with pytest.raises(TypeError, match="SeedItem subclass"):

            @SeedFactory.register("bad_seed")
            class NotASeed:
                pass

    def test_factory_instance_gets_global_seeds(self):
        """Factory instances get copies of global seeds."""

        @SeedFactory.register("global_seed")
        class GlobalSeed(SeedItem):
            async def create(self, provider, **deps):
                return "global"

        factory = SeedFactory()
        assert "global_seed" in factory.list_seeds()
        assert factory.get_seed("global_seed") is not None

    def test_register_instance_is_local(self):
        """register_instance() only affects that factory instance."""

        class LocalSeed(SeedItem):
            async def create(self, provider, **deps):
                return "local"

        factory1 = SeedFactory()
        factory2 = SeedFactory()

        factory1.register_instance("local_seed", LocalSeed())

        assert "local_seed" in factory1.list_seeds()
        assert "local_seed" not in factory2.list_seeds()

    def test_register_instance_requires_seed_item(self):
        """register_instance() requires SeedItem instance."""
        factory = SeedFactory()

        with pytest.raises(TypeError, match="Expected SeedItem"):
            factory.register_instance("bad", "not a seed")


# =============================================================================
# Test Dependency Resolution
# =============================================================================


class TestDependencyResolution:
    """Tests for topological sort of seed dependencies."""

    def test_simple_dependency_order(self):
        """Dependencies are sorted correctly."""

        @SeedFactory.register("parent")
        class ParentSeed(SeedItem):
            async def create(self, provider, **deps):
                return "parent"

        @SeedFactory.register("child")
        class ChildSeed(SeedItem):
            dependencies = ["parent"]

            async def create(self, provider, **deps):
                return "child"

        factory = SeedFactory()
        sorted_ids = factory._topological_sort(["child", "parent"])

        # Parent must come before child
        assert sorted_ids.index("parent") < sorted_ids.index("child")

    def test_transitive_dependencies(self):
        """Transitive dependencies are included and ordered."""

        @SeedFactory.register("grandparent")
        class GrandparentSeed(SeedItem):
            async def create(self, provider, **deps):
                return "grandparent"

        @SeedFactory.register("parent")
        class ParentSeed(SeedItem):
            dependencies = ["grandparent"]

            async def create(self, provider, **deps):
                return "parent"

        @SeedFactory.register("child")
        class ChildSeed(SeedItem):
            dependencies = ["parent"]

            async def create(self, provider, **deps):
                return "child"

        factory = SeedFactory()
        # Only request child, but grandparent and parent should be included
        sorted_ids = factory._topological_sort(["child"])

        assert "grandparent" in sorted_ids
        assert "parent" in sorted_ids
        assert sorted_ids.index("grandparent") < sorted_ids.index("parent")
        assert sorted_ids.index("parent") < sorted_ids.index("child")

    def test_circular_dependency_detected(self):
        """Circular dependencies raise CircularDependencyError."""

        @SeedFactory.register("seed_a")
        class SeedA(SeedItem):
            dependencies = ["seed_b"]

            async def create(self, provider, **deps):
                return "a"

        @SeedFactory.register("seed_b")
        class SeedB(SeedItem):
            dependencies = ["seed_a"]

            async def create(self, provider, **deps):
                return "b"

        factory = SeedFactory()

        with pytest.raises(CircularDependencyError) as exc_info:
            factory._topological_sort(["seed_a"])

        # Cycle should mention both seeds
        assert "seed_a" in str(exc_info.value)
        assert "seed_b" in str(exc_info.value)

    def test_missing_seed_error(self):
        """Missing seeds raise SeedNotFoundError."""

        @SeedFactory.register("existing")
        class ExistingSeed(SeedItem):
            async def create(self, provider, **deps):
                return "exists"

        factory = SeedFactory()

        with pytest.raises(SeedNotFoundError) as exc_info:
            factory._topological_sort(["nonexistent"])

        assert exc_info.value.seed_id == "nonexistent"
        assert "existing" in exc_info.value.available

    def test_missing_dependency_error(self):
        """Missing dependencies raise SeedNotFoundError."""

        @SeedFactory.register("orphan")
        class OrphanSeed(SeedItem):
            dependencies = ["missing_parent"]

            async def create(self, provider, **deps):
                return "orphan"

        factory = SeedFactory()

        with pytest.raises(SeedNotFoundError) as exc_info:
            factory._topological_sort(["orphan"])

        assert exc_info.value.seed_id == "missing_parent"


# =============================================================================
# Test Parallel Safety (Instance Isolation)
# =============================================================================


class TestParallelSafety:
    """Tests for instance-based registry isolation."""

    def test_instance_isolation(self):
        """Factory instances don't interfere with each other."""

        class LocalSeed(SeedItem):
            async def create(self, provider, **deps):
                return "local"

        factory1 = SeedFactory()
        factory2 = SeedFactory()

        factory1.register_instance("unique_to_f1", LocalSeed())
        factory2.register_instance("unique_to_f2", LocalSeed())

        assert "unique_to_f1" in factory1.list_seeds()
        assert "unique_to_f1" not in factory2.list_seeds()
        assert "unique_to_f2" in factory2.list_seeds()
        assert "unique_to_f2" not in factory1.list_seeds()

    def test_global_seeds_are_copied_not_shared(self):
        """Each factory gets its own instance of global seeds."""

        @SeedFactory.register("shared_class")
        class SharedClassSeed(SeedItem):
            state = None

            async def create(self, provider, **deps):
                return "shared"

        factory1 = SeedFactory()
        factory2 = SeedFactory()

        seed1 = factory1.get_seed("shared_class")
        seed2 = factory2.get_seed("shared_class")

        # Different instances
        assert seed1 is not seed2

        # Modifying one doesn't affect the other
        seed1.state = "modified"
        assert seed2.state is None


# =============================================================================
# Test create_all() Execution
# =============================================================================


class TestCreateAll:
    """Tests for create_all() execution."""

    @pytest.mark.asyncio
    async def test_create_all_basic(self, mock_provider):
        """create_all() creates seeds in order."""
        call_order = []

        @SeedFactory.register("first")
        class FirstSeed(SeedItem):
            async def create(self, provider, **deps):
                call_order.append("first")
                return "mem_first"

        @SeedFactory.register("second")
        class SecondSeed(SeedItem):
            dependencies = ["first"]

            async def create(self, provider, **deps):
                call_order.append("second")
                return "mem_second"

        factory = SeedFactory()
        results = await factory.create_all(["second", "first"], mock_provider)

        assert call_order == ["first", "second"]
        assert results["first"] == "mem_first"
        assert results["second"] == "mem_second"

    @pytest.mark.asyncio
    async def test_create_all_passes_dependency_results(self, mock_provider):
        """Dependency results are passed to dependent seeds."""

        @SeedFactory.register("parent")
        class ParentSeed(SeedItem):
            async def create(self, provider, **deps):
                return "parent_mem_id"

        @SeedFactory.register("child")
        class ChildSeed(SeedItem):
            dependencies = ["parent"]
            received_deps = None

            async def create(self, provider, **deps):
                ChildSeed.received_deps = deps
                return "child_mem_id"

        factory = SeedFactory()
        await factory.create_all(["child"], mock_provider)

        assert ChildSeed.received_deps == {"parent": "parent_mem_id"}

    @pytest.mark.asyncio
    async def test_create_all_empty_list(self, mock_provider):
        """create_all() with empty list returns empty dict."""
        factory = SeedFactory()
        results = await factory.create_all([], mock_provider)
        assert results == {}

    @pytest.mark.asyncio
    async def test_create_all_dict_result(self, mock_provider):
        """Seeds can return dict of multiple memory IDs."""

        @SeedFactory.register("multi")
        class MultiSeed(SeedItem):
            async def create(self, provider, **deps):
                return {"primary": "mem_1", "secondary": "mem_2"}

        factory = SeedFactory()
        results = await factory.create_all(["multi"], mock_provider)

        assert results["multi"] == {"primary": "mem_1", "secondary": "mem_2"}


# =============================================================================
# Test Rollback on Failure
# =============================================================================


class TestRollback:
    """Tests for rollback when seed creation fails."""

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, mock_provider):
        """Failed seed creation rolls back previously created memories."""
        created_ids = []

        @SeedFactory.register("succeeds_1")
        class Succeeds1(SeedItem):
            async def create(self, provider, **deps):
                mem_id = "mem_succeeds_1"
                created_ids.append(mem_id)
                return mem_id

        @SeedFactory.register("succeeds_2")
        class Succeeds2(SeedItem):
            dependencies = ["succeeds_1"]

            async def create(self, provider, **deps):
                mem_id = "mem_succeeds_2"
                created_ids.append(mem_id)
                return mem_id

        @SeedFactory.register("fails")
        class Fails(SeedItem):
            dependencies = ["succeeds_2"]

            async def create(self, provider, **deps):
                raise RuntimeError("Seed creation failed!")

        factory = SeedFactory()

        with pytest.raises(RuntimeError, match="Seed creation failed"):
            await factory.create_all(["fails"], mock_provider)

        # Both successful seeds should have been rolled back
        assert mock_provider.delete.call_count == 2

        # Verify rollback was in reverse order
        delete_calls = [call.args[0] for call in mock_provider.delete.call_args_list]
        assert delete_calls == ["mem_succeeds_2", "mem_succeeds_1"]

    @pytest.mark.asyncio
    async def test_rollback_handles_delete_errors(self, mock_provider):
        """Rollback continues even if some deletes fail."""
        mock_provider.delete = AsyncMock(side_effect=Exception("Delete failed"))

        @SeedFactory.register("good")
        class GoodSeed(SeedItem):
            async def create(self, provider, **deps):
                return "mem_good"

        @SeedFactory.register("bad")
        class BadSeed(SeedItem):
            dependencies = ["good"]

            async def create(self, provider, **deps):
                raise RuntimeError("Bad seed!")

        factory = SeedFactory()

        # Original error should still be raised, not the delete error
        with pytest.raises(RuntimeError, match="Bad seed"):
            await factory.create_all(["bad"], mock_provider)

    @pytest.mark.asyncio
    async def test_rollback_handles_dict_results(self, mock_provider):
        """Rollback handles seeds that return dict of IDs."""

        @SeedFactory.register("multi")
        class MultiSeed(SeedItem):
            async def create(self, provider, **deps):
                return {"a": "mem_a", "b": "mem_b"}

        @SeedFactory.register("fails_after")
        class FailsAfter(SeedItem):
            dependencies = ["multi"]

            async def create(self, provider, **deps):
                raise RuntimeError("Failed!")

        factory = SeedFactory()

        with pytest.raises(RuntimeError):
            await factory.create_all(["fails_after"], mock_provider)

        # Both IDs from dict should be deleted
        delete_calls = {call.args[0] for call in mock_provider.delete.call_args_list}
        assert "mem_a" in delete_calls
        assert "mem_b" in delete_calls


# =============================================================================
# Test Cleanup
# =============================================================================


class TestCleanup:
    """Tests for cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_created_memories(self, mock_provider):
        """cleanup() deletes all memories created by this factory."""

        @SeedFactory.register("seed1")
        class Seed1(SeedItem):
            async def create(self, provider, **deps):
                return "mem_1"

        @SeedFactory.register("seed2")
        class Seed2(SeedItem):
            async def create(self, provider, **deps):
                return "mem_2"

        factory = SeedFactory()
        await factory.create_all(["seed1", "seed2"], mock_provider)

        # Reset mock to count cleanup deletes
        mock_provider.delete.reset_mock()

        deleted = await factory.cleanup(mock_provider)

        assert deleted == 2
        assert mock_provider.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_clears_tracking(self, mock_provider):
        """cleanup() clears the tracked memory IDs."""

        @SeedFactory.register("seed")
        class Seed(SeedItem):
            async def create(self, provider, **deps):
                return "mem_id"

        factory = SeedFactory()
        await factory.create_all(["seed"], mock_provider)

        await factory.cleanup(mock_provider)

        # Second cleanup should delete nothing
        mock_provider.delete.reset_mock()
        deleted = await factory.cleanup(mock_provider)

        assert deleted == 0
        assert mock_provider.delete.call_count == 0

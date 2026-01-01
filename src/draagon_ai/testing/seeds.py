"""
Seed data factories for integration testing.

Seeds provide declarative, reusable test data creation with dependency resolution.
Seeds use the REAL MemoryProvider API - no wrapper abstractions.

Critical Design Decision: Instance-Based Registry
------------------------------------------------
Each test gets its own SeedFactory instance to enable parallel test execution.
Global seeds registered via @SeedFactory.register() are copied to each instance.

Example:
    @SeedFactory.register("user_doug")
    class DougUserSeed(SeedItem):
        async def create(self, provider: MemoryProvider) -> str:
            return await provider.store(
                content="User profile: Doug",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER,
            )

    USER_WITH_PREFS = SeedSet("user_with_prefs", ["user_doug", "user_preferences"])

    # In test:
    async def test_something(seed, memory_provider):
        await seed.apply(USER_WITH_PREFS, memory_provider)
        # Test with seeded data...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from draagon_ai.memory.base import MemoryProvider


class CircularDependencyError(Exception):
    """Raised when seed dependencies form a cycle."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}")


class SeedNotFoundError(Exception):
    """Raised when a referenced seed is not registered."""

    def __init__(self, seed_id: str, available: list[str]):
        self.seed_id = seed_id
        self.available = available
        available_str = ", ".join(sorted(available)[:10])
        if len(available) > 10:
            available_str += f", ... ({len(available) - 10} more)"
        super().__init__(
            f"Seed '{seed_id}' not found. Available seeds: {available_str}"
        )


class SeedItem(ABC):
    """Abstract base class for seed data items.

    Each seed creates one or more memories in the database.
    Seeds can declare dependencies on other seeds.

    CRITICAL: Seeds use the REAL MemoryProvider API directly.
    Do not create wrapper methods - use provider.store(), provider.search(), etc.

    Example:
        @SeedFactory.register("user_preferences")
        class UserPreferencesSeed(SeedItem):
            dependencies = ["user_doug"]  # Depends on user being created first

            async def create(
                self,
                provider: MemoryProvider,
                user_doug: str = None,  # Receives memory ID from dependency
            ) -> str:
                return await provider.store(
                    content="User preferences: dark mode",
                    memory_type=MemoryType.PREFERENCE,
                    scope=MemoryScope.USER,
                )
    """

    # Class-level dependencies (override in subclasses)
    dependencies: ClassVar[list[str]] = []

    @abstractmethod
    async def create(
        self,
        provider: "MemoryProvider",
        **dependency_results: Any,
    ) -> str | dict[str, str] | None:
        """Create the seed data.

        Args:
            provider: The REAL MemoryProvider instance (not a wrapper)
            **dependency_results: Results from dependency seeds (memory IDs)

        Returns:
            Memory ID(s) created, or None if no tracking needed.
            Can return a single string ID or a dict of named IDs.
        """
        ...


@dataclass
class SeedSet:
    """A named collection of seeds that form a test scenario.

    Example:
        USER_WITH_CATS = SeedSet("user_with_cats", ["user_doug", "cat_names"])

        async def test_recall_cats(seed, memory_provider):
            await seed.apply(USER_WITH_CATS, memory_provider)
            # Test...
    """

    name: str
    seed_ids: list[str]
    description: str = ""

    def __post_init__(self):
        # Validate no duplicates
        if len(self.seed_ids) != len(set(self.seed_ids)):
            duplicates = [
                s for s in self.seed_ids if self.seed_ids.count(s) > 1
            ]
            raise ValueError(f"Duplicate seeds in SeedSet '{self.name}': {duplicates}")


@dataclass
class SeedFactory:
    """Registry and executor for seed items.

    IMPORTANT: Uses instance-based registry for parallel test safety.
    Each test gets its own factory instance with a copy of global seeds.

    Class-level registry (_global_seed_classes) stores decorator-registered seeds.
    Instance-level registry (_registry) gets a copy on __init__().

    Example:
        # Global registration (via decorator)
        @SeedFactory.register("user_doug")
        class DougUserSeed(SeedItem):
            ...

        # Per-test instance
        factory = SeedFactory()
        factory.register_instance("temp_seed", TempSeed())

        # Execute seeds
        results = await factory.create_all(["user_doug", "temp_seed"], provider)
    """

    # Global registry for decorator-registered seed classes
    _global_seed_classes: ClassVar[dict[str, type[SeedItem]]] = {}

    # Instance-level registry - each test gets its own copy
    _registry: dict[str, SeedItem] = field(default_factory=dict)

    # Track created memory IDs for rollback
    _created_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Copy global seeds to instance registry."""
        for seed_id, seed_class in self._global_seed_classes.items():
            if seed_id not in self._registry:
                self._registry[seed_id] = seed_class()

    @classmethod
    def register(cls, seed_id: str):
        """Decorator to register a seed class globally.

        The class will be instantiated for each SeedFactory instance.

        Example:
            @SeedFactory.register("my_seed")
            class MySeed(SeedItem):
                async def create(self, provider):
                    return await provider.store(...)
        """
        def decorator(seed_class: type[SeedItem]) -> type[SeedItem]:
            if not issubclass(seed_class, SeedItem):
                raise TypeError(
                    f"@SeedFactory.register requires a SeedItem subclass, "
                    f"got {seed_class.__name__}"
                )
            cls._global_seed_classes[seed_id] = seed_class
            return seed_class
        return decorator

    @classmethod
    def clear_global_registry(cls):
        """Clear global seed registry. Useful for testing the factory itself."""
        cls._global_seed_classes.clear()

    def register_instance(self, seed_id: str, seed: SeedItem) -> None:
        """Register a seed instance for this factory only.

        Use for test-specific seeds that shouldn't be global.

        Args:
            seed_id: Unique identifier for the seed
            seed: The SeedItem instance
        """
        if not isinstance(seed, SeedItem):
            raise TypeError(f"Expected SeedItem, got {type(seed).__name__}")
        self._registry[seed_id] = seed

    def get_seed(self, seed_id: str) -> SeedItem | None:
        """Get a seed by ID."""
        return self._registry.get(seed_id)

    def list_seeds(self) -> list[str]:
        """List all registered seed IDs."""
        return list(self._registry.keys())

    def _topological_sort(self, seed_ids: list[str]) -> list[str]:
        """Sort seed IDs by dependencies using Kahn's algorithm.

        Args:
            seed_ids: Seed IDs to sort

        Returns:
            Sorted list where dependencies come before dependents

        Raises:
            SeedNotFoundError: If a seed ID is not registered
            CircularDependencyError: If dependencies form a cycle
        """
        # Validate all seeds exist
        for seed_id in seed_ids:
            if seed_id not in self._registry:
                raise SeedNotFoundError(seed_id, self.list_seeds())

        # Build dependency graph (only for requested seeds)
        # We need to include transitive dependencies
        all_seeds = set(seed_ids)
        to_process = list(seed_ids)

        while to_process:
            seed_id = to_process.pop()
            seed = self._registry[seed_id]
            for dep_id in seed.dependencies:
                if dep_id not in self._registry:
                    raise SeedNotFoundError(dep_id, self.list_seeds())
                if dep_id not in all_seeds:
                    all_seeds.add(dep_id)
                    to_process.append(dep_id)

        # Build in-degree map
        in_degree: dict[str, int] = defaultdict(int)
        dependents: dict[str, list[str]] = defaultdict(list)

        for seed_id in all_seeds:
            seed = self._registry[seed_id]
            for dep_id in seed.dependencies:
                if dep_id in all_seeds:
                    dependents[dep_id].append(seed_id)
                    in_degree[seed_id] += 1
            # Ensure all seeds are in in_degree (even if 0)
            if seed_id not in in_degree:
                in_degree[seed_id] = 0

        # Kahn's algorithm
        queue = [s for s in all_seeds if in_degree[s] == 0]
        result = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            seed_id = queue.pop(0)
            result.append(seed_id)

            for dependent in dependents[seed_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(result) != len(all_seeds):
            # Find the cycle for error message
            remaining = all_seeds - set(result)
            cycle = self._find_cycle(remaining)
            raise CircularDependencyError(cycle)

        return result

    def _find_cycle(self, seeds: set[str]) -> list[str]:
        """Find a cycle in the remaining seeds (for error message)."""
        # Simple DFS to find any cycle
        visited = set()
        path = []

        def dfs(seed_id: str) -> list[str] | None:
            if seed_id in path:
                cycle_start = path.index(seed_id)
                return path[cycle_start:] + [seed_id]
            if seed_id in visited:
                return None

            visited.add(seed_id)
            path.append(seed_id)

            seed = self._registry.get(seed_id)
            if seed:
                for dep_id in seed.dependencies:
                    if dep_id in seeds:
                        result = dfs(dep_id)
                        if result:
                            return result

            path.pop()
            return None

        for seed_id in seeds:
            cycle = dfs(seed_id)
            if cycle:
                return cycle

        return list(seeds)[:3] + ["..."]  # Fallback

    async def create_all(
        self,
        seed_ids: list[str],
        provider: "MemoryProvider",
    ) -> dict[str, Any]:
        """Create all specified seeds with dependency resolution.

        Seeds are created in topological order. If a seed fails,
        all previously created memories are rolled back (deleted).

        Args:
            seed_ids: List of seed IDs to create
            provider: The REAL MemoryProvider instance

        Returns:
            Dict mapping seed_id to creation result (memory IDs)

        Raises:
            SeedNotFoundError: If a seed ID is not registered
            CircularDependencyError: If dependencies form a cycle
        """
        if not seed_ids:
            return {}

        # Sort by dependencies
        sorted_ids = self._topological_sort(seed_ids)

        # Track results and created IDs for rollback
        results: dict[str, Any] = {}
        created_memory_ids: list[str] = []

        try:
            for seed_id in sorted_ids:
                seed = self._registry[seed_id]

                # Gather dependency results
                dep_results = {}
                for dep_id in seed.dependencies:
                    if dep_id in results:
                        dep_results[dep_id] = results[dep_id]

                # Create the seed
                result = await seed.create(provider, **dep_results)
                results[seed_id] = result

                # Track memory IDs for rollback
                if isinstance(result, str) and result:
                    created_memory_ids.append(result)
                elif isinstance(result, dict):
                    for v in result.values():
                        if isinstance(v, str) and v:
                            created_memory_ids.append(v)

            self._created_ids = created_memory_ids
            return results

        except Exception as e:
            # Rollback: delete all created memories in reverse order
            for memory_id in reversed(created_memory_ids):
                try:
                    await provider.delete(memory_id)
                except Exception:
                    # Best-effort rollback - don't mask original error
                    pass
            raise

    async def cleanup(self, provider: "MemoryProvider") -> int:
        """Clean up all memories created by this factory.

        Args:
            provider: The MemoryProvider instance

        Returns:
            Number of memories deleted
        """
        deleted = 0
        for memory_id in reversed(self._created_ids):
            try:
                if await provider.delete(memory_id):
                    deleted += 1
            except Exception:
                pass
        self._created_ids.clear()
        return deleted

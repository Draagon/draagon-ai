"""
Test sequence infrastructure for multi-step integration tests.

Test sequences allow multiple test steps to share database state,
enabling tests for agent learning, belief reconciliation, and
other stateful behavior across multiple interactions.

Example:
    from draagon_ai.testing import TestSequence, step

    class TestLearningFlow(TestSequence):
        '''Test agent learning across interactions.'''

        @step(1)
        async def test_initial_unknown(self, agent):
            response = await agent.process("When is my birthday?")
            assert response.confidence < 0.5

        @step(2, depends_on="test_initial_unknown")
        async def test_learn_birthday(self, agent):
            response = await agent.process("My birthday is March 15")
            assert "march 15" in response.answer.lower()

        @step(3, depends_on="test_learn_birthday")
        async def test_recall_birthday(self, agent):
            response = await agent.process("When is my birthday?")
            assert "march 15" in response.answer.lower()

Key concepts:
- Steps execute in order specified by @step(order)
- Database persists across steps within a sequence
- Dependencies can be declared with depends_on
- Use sequence_database fixture for class-scoped database
"""

from __future__ import annotations

import logging
from abc import ABC
from functools import wraps
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)


class StepDependencyError(Exception):
    """Raised when a step dependency is invalid."""

    def __init__(self, step_name: str, depends_on: str, available: list[str]):
        self.step_name = step_name
        self.depends_on = depends_on
        self.available = available
        available_str = ", ".join(sorted(available)) if available else "(none)"
        super().__init__(
            f"Step '{step_name}' depends on '{depends_on}' which does not exist. "
            f"Available steps: {available_str}"
        )


class StepOrderError(Exception):
    """Raised when step ordering is invalid."""

    def __init__(self, step_name: str, depends_on: str, step_order: int, dep_order: int):
        self.step_name = step_name
        self.depends_on = depends_on
        self.step_order = step_order
        self.dep_order = dep_order
        super().__init__(
            f"Step '{step_name}' (order {step_order}) depends on '{depends_on}' "
            f"(order {dep_order}), but dependency must have lower order."
        )


def step(order: int, depends_on: str | None = None) -> Callable:
    """Decorator to mark a method as a test sequence step.

    Steps are executed in order. Database state persists between steps.

    Args:
        order: Execution order (lower numbers run first)
        depends_on: Optional name of step that must complete first

    Returns:
        Decorated function with step metadata

    Example:
        @step(1)
        async def test_first(self, agent):
            ...

        @step(2, depends_on="test_first")
        async def test_second(self, agent):
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._step_order = order
        func._step_depends_on = depends_on
        func._is_sequence_step = True

        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            logger.info(f"Executing step {order}: {func.__name__}")
            return await func(self, *args, **kwargs)

        # Copy metadata to wrapper
        wrapper._step_order = order
        wrapper._step_depends_on = depends_on
        wrapper._is_sequence_step = True

        return wrapper

    return decorator


class TestSequence(ABC):
    """Base class for multi-step test sequences.

    Database persists across steps within a sequence.
    Use @step decorator to define execution order.

    Steps are discovered at class creation time and executed in order.
    Dependencies are validated to ensure they exist and have lower order.

    Example:
        class TestMyFlow(TestSequence):
            @step(1)
            async def test_setup(self, agent):
                # Initial state
                pass

            @step(2, depends_on="test_setup")
            async def test_action(self, agent):
                # Perform action
                pass

            @step(3, depends_on="test_action")
            async def test_verify(self, agent):
                # Verify result
                pass
    """

    # Cache of steps per class
    _step_cache: ClassVar[dict[type, list[tuple[int, str, Callable]]]] = {}

    @classmethod
    def get_steps(cls) -> list[tuple[int, str, Callable]]:
        """Get all steps in execution order.

        Returns:
            List of (order, name, method) tuples sorted by order
        """
        # Check cache
        if cls in cls._step_cache:
            return cls._step_cache[cls]

        steps = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            try:
                method = getattr(cls, name)
            except AttributeError:
                continue
            if callable(method) and getattr(method, "_is_sequence_step", False):
                order = method._step_order
                steps.append((order, name, method))

        # Sort by order
        sorted_steps = sorted(steps, key=lambda x: x[0])

        # Cache result
        cls._step_cache[cls] = sorted_steps

        return sorted_steps

    @classmethod
    def get_step_names(cls) -> list[str]:
        """Get ordered list of step method names.

        Returns:
            List of step method names in execution order
        """
        return [name for _, name, _ in cls.get_steps()]

    @classmethod
    def validate_dependencies(cls) -> None:
        """Validate that all step dependencies are resolvable.

        Raises:
            StepDependencyError: If a dependency references non-existent step
            StepOrderError: If a dependency has higher or equal order
        """
        steps = cls.get_steps()
        step_names = {name for _, name, _ in steps}
        step_orders = {name: order for order, name, _ in steps}

        for order, name, method in steps:
            depends_on = getattr(method, "_step_depends_on", None)

            if depends_on is not None:
                # Check dependency exists
                if depends_on not in step_names:
                    raise StepDependencyError(name, depends_on, list(step_names))

                # Check dependency has lower order
                dep_order = step_orders[depends_on]
                if dep_order >= order:
                    raise StepOrderError(name, depends_on, order, dep_order)

    @classmethod
    def clear_step_cache(cls) -> None:
        """Clear the step cache. Useful for testing."""
        cls._step_cache.clear()

    def __init_subclass__(cls, **kwargs):
        """Validate dependencies when subclass is defined."""
        super().__init_subclass__(**kwargs)

        # Only validate if class has steps (not abstract)
        steps = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            try:
                method = getattr(cls, name)
                if callable(method) and getattr(method, "_is_sequence_step", False):
                    steps.append(name)
            except AttributeError:
                continue

        if steps:
            # Clear cache for this class since it's new/modified
            if cls in cls._step_cache:
                del cls._step_cache[cls]

            # Validate dependencies
            try:
                cls.validate_dependencies()
            except (StepDependencyError, StepOrderError):
                # Re-raise validation errors
                raise

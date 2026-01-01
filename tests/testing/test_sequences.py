"""Tests for test sequence infrastructure (TASK-005)."""

from __future__ import annotations

import pytest

from draagon_ai.testing.sequences import (
    StepDependencyError,
    StepOrderError,
    TestSequence,
    step,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_step_cache():
    """Clear step cache before and after each test."""
    TestSequence.clear_step_cache()
    yield
    TestSequence.clear_step_cache()


# =============================================================================
# Test @step Decorator
# =============================================================================


class TestStepDecorator:
    """Tests for the @step decorator."""

    def test_step_adds_order_metadata(self):
        """@step adds _step_order to function."""

        class MySequence(TestSequence):
            @step(5)
            async def test_step(self):
                pass

        method = MySequence.test_step
        assert hasattr(method, "_step_order")
        assert method._step_order == 5

    def test_step_adds_depends_on_metadata(self):
        """@step adds _step_depends_on to function."""

        class MySequence(TestSequence):
            @step(1)
            async def test_first(self):
                pass

            @step(2, depends_on="test_first")
            async def test_second(self):
                pass

        assert MySequence.test_second._step_depends_on == "test_first"

    def test_step_defaults_depends_on_to_none(self):
        """@step defaults depends_on to None."""

        class MySequence(TestSequence):
            @step(1)
            async def test_step(self):
                pass

        assert MySequence.test_step._step_depends_on is None

    def test_step_marks_as_sequence_step(self):
        """@step marks function as sequence step."""

        class MySequence(TestSequence):
            @step(1)
            async def test_step(self):
                pass

            async def not_a_step(self):
                pass

        assert getattr(MySequence.test_step, "_is_sequence_step", False) is True
        assert getattr(MySequence.not_a_step, "_is_sequence_step", False) is False


# =============================================================================
# Test TestSequence.get_steps()
# =============================================================================


class TestGetSteps:
    """Tests for TestSequence.get_steps()."""

    def test_get_steps_returns_ordered_list(self):
        """get_steps() returns steps in execution order."""

        class MySequence(TestSequence):
            @step(3)
            async def test_third(self):
                pass

            @step(1)
            async def test_first(self):
                pass

            @step(2)
            async def test_second(self):
                pass

        steps = MySequence.get_steps()

        assert len(steps) == 3
        assert steps[0][1] == "test_first"
        assert steps[1][1] == "test_second"
        assert steps[2][1] == "test_third"

    def test_get_steps_returns_tuples(self):
        """get_steps() returns (order, name, method) tuples."""

        class MySequence(TestSequence):
            @step(1)
            async def test_step(self):
                pass

        steps = MySequence.get_steps()

        assert len(steps) == 1
        order, name, method = steps[0]
        assert order == 1
        assert name == "test_step"
        assert callable(method)

    def test_get_steps_excludes_non_steps(self):
        """get_steps() only includes @step decorated methods."""

        class MySequence(TestSequence):
            @step(1)
            async def test_step(self):
                pass

            async def regular_method(self):
                pass

            @classmethod
            def class_method(cls):
                pass

        steps = MySequence.get_steps()

        step_names = [name for _, name, _ in steps]
        assert "test_step" in step_names
        assert "regular_method" not in step_names
        assert "class_method" not in step_names

    def test_get_steps_caches_result(self):
        """get_steps() caches result per class."""

        class MySequence(TestSequence):
            @step(1)
            async def test_step(self):
                pass

        steps1 = MySequence.get_steps()
        steps2 = MySequence.get_steps()

        assert steps1 is steps2  # Same object (cached)

    def test_get_steps_different_classes_not_shared(self):
        """get_steps() caches separately for each class."""

        class Sequence1(TestSequence):
            @step(1)
            async def test_a(self):
                pass

        class Sequence2(TestSequence):
            @step(1)
            async def test_b(self):
                pass

        steps1 = Sequence1.get_steps()
        steps2 = Sequence2.get_steps()

        assert steps1[0][1] == "test_a"
        assert steps2[0][1] == "test_b"


# =============================================================================
# Test TestSequence.get_step_names()
# =============================================================================


class TestGetStepNames:
    """Tests for TestSequence.get_step_names()."""

    def test_returns_ordered_names(self):
        """get_step_names() returns names in order."""

        class MySequence(TestSequence):
            @step(2)
            async def test_second(self):
                pass

            @step(1)
            async def test_first(self):
                pass

        names = MySequence.get_step_names()

        assert names == ["test_first", "test_second"]


# =============================================================================
# Test TestSequence.validate_dependencies()
# =============================================================================


class TestValidateDependencies:
    """Tests for dependency validation."""

    def test_valid_dependencies_pass(self):
        """Valid dependencies don't raise."""

        class MySequence(TestSequence):
            @step(1)
            async def test_first(self):
                pass

            @step(2, depends_on="test_first")
            async def test_second(self):
                pass

        # Should not raise
        MySequence.validate_dependencies()

    def test_nonexistent_dependency_raises(self):
        """Non-existent dependency raises StepDependencyError."""
        with pytest.raises(StepDependencyError) as exc_info:

            class BadSequence(TestSequence):
                @step(1, depends_on="nonexistent")
                async def test_step(self):
                    pass

        assert exc_info.value.step_name == "test_step"
        assert exc_info.value.depends_on == "nonexistent"

    def test_dependency_with_higher_order_raises(self):
        """Dependency with higher order raises StepOrderError."""
        with pytest.raises(StepOrderError) as exc_info:

            class BadSequence(TestSequence):
                @step(2)
                async def test_later(self):
                    pass

                @step(1, depends_on="test_later")
                async def test_earlier(self):
                    pass

        assert exc_info.value.step_name == "test_earlier"
        assert exc_info.value.depends_on == "test_later"
        assert exc_info.value.step_order == 1
        assert exc_info.value.dep_order == 2

    def test_dependency_with_equal_order_raises(self):
        """Dependency with equal order raises StepOrderError."""
        with pytest.raises(StepOrderError):

            class BadSequence(TestSequence):
                @step(1)
                async def test_a(self):
                    pass

                @step(1, depends_on="test_a")
                async def test_b(self):
                    pass

    def test_no_dependencies_is_valid(self):
        """Steps without dependencies are valid."""

        class MySequence(TestSequence):
            @step(1)
            async def test_first(self):
                pass

            @step(2)
            async def test_second(self):
                pass

        # Should not raise
        MySequence.validate_dependencies()

    def test_chain_dependencies_valid(self):
        """Chain of dependencies is valid."""

        class MySequence(TestSequence):
            @step(1)
            async def test_a(self):
                pass

            @step(2, depends_on="test_a")
            async def test_b(self):
                pass

            @step(3, depends_on="test_b")
            async def test_c(self):
                pass

        # Should not raise
        MySequence.validate_dependencies()


# =============================================================================
# Test Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for helpful error messages."""

    def test_dependency_error_lists_available_steps(self):
        """StepDependencyError lists available steps."""
        with pytest.raises(StepDependencyError) as exc_info:

            class BadSequence(TestSequence):
                @step(1)
                async def test_real_step(self):
                    pass

                @step(2, depends_on="wrong_name")
                async def test_bad_step(self):
                    pass

        error_msg = str(exc_info.value)
        assert "test_real_step" in error_msg
        assert "wrong_name" in error_msg

    def test_order_error_shows_orders(self):
        """StepOrderError shows both step orders."""
        with pytest.raises(StepOrderError) as exc_info:

            class BadSequence(TestSequence):
                @step(5)
                async def test_later(self):
                    pass

                @step(2, depends_on="test_later")
                async def test_earlier(self):
                    pass

        error_msg = str(exc_info.value)
        assert "order 2" in error_msg
        assert "order 5" in error_msg


# =============================================================================
# Test Execution (with async)
# =============================================================================


class TestExecution:
    """Tests for step execution."""

    @pytest.mark.asyncio
    async def test_step_wrapper_preserves_function(self):
        """Step wrapper preserves original function behavior."""

        class MySequence(TestSequence):
            result = None

            @step(1)
            async def test_step(self):
                MySequence.result = "executed"
                return "value"

        seq = MySequence()
        result = await seq.test_step()

        assert MySequence.result == "executed"
        assert result == "value"

    @pytest.mark.asyncio
    async def test_steps_execute_in_order(self):
        """Steps execute in defined order."""
        execution_order = []

        class MySequence(TestSequence):
            @step(3)
            async def test_third(self):
                execution_order.append("third")

            @step(1)
            async def test_first(self):
                execution_order.append("first")

            @step(2)
            async def test_second(self):
                execution_order.append("second")

        seq = MySequence()

        # Execute in get_steps() order
        for order, name, method in MySequence.get_steps():
            await method(seq)

        assert execution_order == ["first", "second", "third"]


# =============================================================================
# Test Inheritance
# =============================================================================


class TestInheritance:
    """Tests for sequence inheritance."""

    def test_subclass_can_add_steps(self):
        """Subclass can add more steps."""

        class BaseSequence(TestSequence):
            @step(1)
            async def test_base(self):
                pass

        class DerivedSequence(BaseSequence):
            @step(2)
            async def test_derived(self):
                pass

        steps = DerivedSequence.get_steps()
        names = [name for _, name, _ in steps]

        assert "test_base" in names
        assert "test_derived" in names

    def test_each_class_has_own_cache(self):
        """Each class maintains its own step cache."""

        class Sequence1(TestSequence):
            @step(1)
            async def test_s1(self):
                pass

        class Sequence2(TestSequence):
            @step(1)
            async def test_s2(self):
                pass

        # Access steps for both
        Sequence1.get_steps()
        Sequence2.get_steps()

        # Each should be cached separately
        assert Sequence1 in TestSequence._step_cache
        assert Sequence2 in TestSequence._step_cache
        assert TestSequence._step_cache[Sequence1] != TestSequence._step_cache[Sequence2]

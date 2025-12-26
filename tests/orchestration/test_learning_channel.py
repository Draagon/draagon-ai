"""Tests for Learning Channel (Phase C.1)."""

import pytest
from datetime import datetime

from draagon_ai.orchestration import (
    Learning,
    LearningType,
    LearningScope,
    StubLearningChannel,
    InMemoryLearningChannel,
    create_learning_channel,
    get_learning_channel,
    set_learning_channel,
    reset_learning_channel,
)


@pytest.fixture(autouse=True)
def reset_channel():
    """Reset channel before each test."""
    reset_learning_channel()
    yield
    reset_learning_channel()


class TestLearning:
    """Tests for the Learning data class."""

    def test_create_learning(self):
        """Test basic learning creation."""
        learning = Learning(
            learning_type=LearningType.FACT,
            content="Doug's birthday is March 15",
            source_agent_id="roxy",
        )

        assert learning.learning_id is not None
        assert learning.learning_type == LearningType.FACT
        assert learning.content == "Doug's birthday is March 15"
        assert learning.source_agent_id == "roxy"
        assert learning.scope == LearningScope.CONTEXT  # Default

    def test_learning_with_entities(self):
        """Test learning with extracted entities."""
        learning = Learning(
            learning_type=LearningType.FACT,
            content="Doug lives in Philadelphia",
            entities=["Doug", "Philadelphia"],
            source_agent_id="roxy",
        )

        assert "Doug" in learning.entities
        assert "Philadelphia" in learning.entities

    def test_learning_with_context(self):
        """Test learning with context scope."""
        learning = Learning(
            learning_type=LearningType.SKILL,
            content="How to restart the service",
            source_agent_id="roxy",
            source_context_id="home",
            scope=LearningScope.GLOBAL,
        )

        assert learning.source_context_id == "home"
        assert learning.scope == LearningScope.GLOBAL

    def test_learning_serialization(self):
        """Test to_dict and from_dict round-trip."""
        original = Learning(
            learning_type=LearningType.INSIGHT,
            content="Users prefer concise responses",
            source_agent_id="roxy",
            entities=["users", "responses"],
            confidence=0.8,
            importance=0.7,
            verified=True,
            metadata={"category": "preference"},
        )

        data = original.to_dict()
        restored = Learning.from_dict(data)

        assert restored.learning_id == original.learning_id
        assert restored.learning_type == original.learning_type
        assert restored.content == original.content
        assert restored.source_agent_id == original.source_agent_id
        assert restored.entities == original.entities
        assert restored.confidence == original.confidence
        assert restored.importance == original.importance
        assert restored.verified == original.verified
        assert restored.metadata == original.metadata


class TestStubLearningChannel:
    """Tests for StubLearningChannel."""

    @pytest.mark.anyio
    async def test_broadcast_logs(self):
        """Test that broadcast logs the learning."""
        channel = StubLearningChannel()

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Test fact",
            source_agent_id="roxy",
        )

        await channel.broadcast(learning)

        log = channel.get_learning_log()
        assert len(log) == 1
        assert log[0].content == "Test fact"

    @pytest.mark.anyio
    async def test_subscribe_creates_subscription(self):
        """Test that subscribe creates a subscription."""
        channel = StubLearningChannel()

        async def handler(learning: Learning):
            pass

        sub_id = await channel.subscribe(
            agent_id="agent1",
            handler=handler,
            context_id="home",
            learning_types={LearningType.FACT},
        )

        assert sub_id is not None

        subs = channel.get_subscriptions()
        assert len(subs) == 1
        assert subs[0].agent_id == "agent1"
        assert subs[0].context_id == "home"
        assert LearningType.FACT in subs[0].learning_types

    @pytest.mark.anyio
    async def test_unsubscribe(self):
        """Test unsubscribing."""
        channel = StubLearningChannel()

        async def handler(learning: Learning):
            pass

        sub_id = await channel.subscribe("agent1", handler)
        assert len(channel.get_subscriptions()) == 1

        result = await channel.unsubscribe(sub_id)
        assert result is True
        assert len(channel.get_subscriptions()) == 0

    @pytest.mark.anyio
    async def test_unsubscribe_nonexistent(self):
        """Test unsubscribing non-existent subscription."""
        channel = StubLearningChannel()

        result = await channel.unsubscribe("nonexistent-id")
        assert result is False

    @pytest.mark.anyio
    async def test_clear(self):
        """Test clearing the channel."""
        channel = StubLearningChannel()

        async def handler(learning: Learning):
            pass

        await channel.subscribe("agent1", handler)
        await channel.broadcast(Learning(
            learning_type=LearningType.FACT,
            content="test",
            source_agent_id="roxy",
        ))

        channel.clear()

        assert len(channel.get_subscriptions()) == 0
        assert len(channel.get_learning_log()) == 0


class TestInMemoryLearningChannel:
    """Tests for InMemoryLearningChannel."""

    @pytest.mark.anyio
    async def test_broadcast_calls_handlers(self):
        """Test that broadcast calls matching handlers."""
        channel = InMemoryLearningChannel()
        received = []

        async def handler(learning: Learning):
            received.append(learning)

        await channel.subscribe("listener", handler, context_id="home")

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Broadcast test",
            source_agent_id="broadcaster",
            source_context_id="home",
            scope=LearningScope.CONTEXT,
        )

        await channel.broadcast(learning)

        assert len(received) == 1
        assert received[0].content == "Broadcast test"

    @pytest.mark.anyio
    async def test_no_self_notification(self):
        """Test that source agent doesn't receive its own learning."""
        channel = InMemoryLearningChannel()
        received = []

        async def handler(learning: Learning):
            received.append(learning)

        await channel.subscribe("roxy", handler)

        learning = Learning(
            learning_type=LearningType.FACT,
            content="My own learning",
            source_agent_id="roxy",  # Same as subscriber
        )

        await channel.broadcast(learning)

        assert len(received) == 0

    @pytest.mark.anyio
    async def test_scope_filtering(self):
        """Test that PRIVATE scope isn't shared."""
        channel = InMemoryLearningChannel()
        received = []

        async def handler(learning: Learning):
            received.append(learning)

        await channel.subscribe("listener", handler)

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Private fact",
            source_agent_id="broadcaster",
            scope=LearningScope.PRIVATE,
        )

        await channel.broadcast(learning)

        assert len(received) == 0

    @pytest.mark.anyio
    async def test_context_filtering(self):
        """Test context-based filtering."""
        channel = InMemoryLearningChannel()
        received = []

        async def handler(learning: Learning):
            received.append(learning)

        # Subscribe to "home" context only
        await channel.subscribe("listener", handler, context_id="home")

        # Broadcast from different context
        learning = Learning(
            learning_type=LearningType.FACT,
            content="From office",
            source_agent_id="broadcaster",
            source_context_id="office",
            scope=LearningScope.CONTEXT,
        )

        await channel.broadcast(learning)

        assert len(received) == 0

    @pytest.mark.anyio
    async def test_type_filtering(self):
        """Test learning type filtering."""
        channel = InMemoryLearningChannel()
        received = []

        async def handler(learning: Learning):
            received.append(learning)

        # Subscribe to SKILL only
        await channel.subscribe(
            "listener",
            handler,
            learning_types={LearningType.SKILL},
        )

        # Broadcast a FACT (wrong type)
        fact = Learning(
            learning_type=LearningType.FACT,
            content="A fact",
            source_agent_id="broadcaster",
        )
        await channel.broadcast(fact)
        assert len(received) == 0

        # Broadcast a SKILL (correct type)
        skill = Learning(
            learning_type=LearningType.SKILL,
            content="A skill",
            source_agent_id="broadcaster",
        )
        await channel.broadcast(skill)
        assert len(received) == 1

    @pytest.mark.anyio
    async def test_global_scope_reaches_all(self):
        """Test that GLOBAL scope reaches all subscribers."""
        channel = InMemoryLearningChannel()
        received = []

        async def handler(learning: Learning):
            received.append(learning)

        await channel.subscribe("listener", handler, context_id="home")

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Global fact",
            source_agent_id="broadcaster",
            source_context_id="office",  # Different context
            scope=LearningScope.GLOBAL,
        )

        await channel.broadcast(learning)

        assert len(received) == 1

    @pytest.mark.anyio
    async def test_handler_error_handling(self):
        """Test that handler errors don't break broadcast."""
        channel = InMemoryLearningChannel()
        call_count = 0

        async def bad_handler(learning: Learning):
            raise Exception("Handler error")

        async def good_handler(learning: Learning):
            nonlocal call_count
            call_count += 1

        await channel.subscribe("bad_agent", bad_handler)
        await channel.subscribe("good_agent", good_handler)

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Test",
            source_agent_id="broadcaster",
        )

        await channel.broadcast(learning)

        # Good handler should still be called
        assert call_count == 1

    @pytest.mark.anyio
    async def test_handlers_run_concurrently(self):
        """Test that handlers run concurrently, not sequentially.

        If handlers ran sequentially and each took 0.1s, two handlers
        would take 0.2s. With concurrent execution, they should complete
        in ~0.1s (plus small overhead).
        """
        import asyncio
        import time

        channel = InMemoryLearningChannel()
        handler_times: list[float] = []
        start_times: list[float] = []

        async def slow_handler(learning: Learning):
            start_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate slow operation
            handler_times.append(time.time())

        # Subscribe multiple slow handlers
        await channel.subscribe("agent1", slow_handler)
        await channel.subscribe("agent2", slow_handler)
        await channel.subscribe("agent3", slow_handler)

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Test concurrent execution",
            source_agent_id="broadcaster",
        )

        start = time.time()
        await channel.broadcast(learning)
        total_time = time.time() - start

        # All 3 handlers should have been called
        assert len(handler_times) == 3

        # If running concurrently, total time should be ~0.1s (not 0.3s)
        # Allow some overhead, but should be well under 0.2s
        assert total_time < 0.2, f"Handlers took {total_time}s - likely running sequentially"

        # Verify all handlers started at approximately the same time
        start_spread = max(start_times) - min(start_times)
        assert start_spread < 0.05, f"Handlers didn't start concurrently (spread: {start_spread}s)"

    @pytest.mark.anyio
    async def test_slow_handler_does_not_block_others(self):
        """Test that a slow handler doesn't block faster handlers."""
        import asyncio

        channel = InMemoryLearningChannel()
        completion_order: list[str] = []

        async def slow_handler(learning: Learning):
            await asyncio.sleep(0.2)
            completion_order.append("slow")

        async def fast_handler(learning: Learning):
            await asyncio.sleep(0.05)
            completion_order.append("fast")

        await channel.subscribe("slow_agent", slow_handler)
        await channel.subscribe("fast_agent", fast_handler)

        learning = Learning(
            learning_type=LearningType.FACT,
            content="Test",
            source_agent_id="broadcaster",
        )

        await channel.broadcast(learning)

        # Both handlers should complete
        assert len(completion_order) == 2

        # Fast handler should complete first (concurrent execution)
        assert completion_order[0] == "fast"
        assert completion_order[1] == "slow"


class TestChannelFactory:
    """Tests for channel factory functions."""

    def test_create_stub_channel(self):
        """Test creating stub channel."""
        channel = create_learning_channel("stub")
        assert isinstance(channel, StubLearningChannel)

    def test_create_memory_channel(self):
        """Test creating in-memory channel."""
        channel = create_learning_channel("memory")
        assert isinstance(channel, InMemoryLearningChannel)

    def test_create_unknown_channel(self):
        """Test creating unknown channel type."""
        with pytest.raises(ValueError):
            create_learning_channel("unknown")


class TestSingletonChannel:
    """Tests for singleton channel pattern."""

    def test_get_default_channel(self):
        """Test getting default channel."""
        channel = get_learning_channel()
        assert channel is not None
        assert isinstance(channel, StubLearningChannel)

    def test_get_returns_same_instance(self):
        """Test that get returns same instance."""
        channel1 = get_learning_channel()
        channel2 = get_learning_channel()
        assert channel1 is channel2

    def test_set_channel(self):
        """Test setting custom channel."""
        custom = InMemoryLearningChannel()
        set_learning_channel(custom)

        channel = get_learning_channel()
        assert channel is custom

    def test_reset_channel(self):
        """Test resetting channel."""
        custom = InMemoryLearningChannel()
        set_learning_channel(custom)

        reset_learning_channel()

        channel = get_learning_channel()
        assert channel is not custom
        assert isinstance(channel, StubLearningChannel)


class TestThreadSafeSingleton:
    """Tests for thread-safe singleton implementation."""

    def test_concurrent_get_channel(self):
        """Test that concurrent get_learning_channel calls return same instance."""
        import concurrent.futures
        import threading

        reset_learning_channel()

        results = []
        errors = []

        def get_channel():
            try:
                channel = get_learning_channel()
                results.append(id(channel))
            except Exception as e:
                errors.append(e)

        # Run many threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_channel) for _ in range(100)]
            concurrent.futures.wait(futures)

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All threads should get the same instance
        assert len(results) == 100
        assert len(set(results)) == 1, f"Multiple instances created: {set(results)}"

    def test_concurrent_set_and_get_channel(self):
        """Test thread-safety when setting and getting concurrently."""
        import concurrent.futures
        import random

        reset_learning_channel()

        errors = []

        def access_channel():
            try:
                if random.choice([True, False]):
                    # Set a new channel
                    set_learning_channel(InMemoryLearningChannel())
                else:
                    # Get existing channel
                    channel = get_learning_channel()
                    assert channel is not None
            except Exception as e:
                errors.append(e)

        # Run many threads doing mixed operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(access_channel) for _ in range(100)]
            concurrent.futures.wait(futures)

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

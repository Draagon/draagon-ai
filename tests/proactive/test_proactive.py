"""Tests for proactive suggestions framework."""

import pytest
import time
import asyncio

from draagon_ai.proactive import (
    Suggestion,
    SuggestionPriority,
    SuggestionCategory,
    AggregatedSuggestions,
    SuggestionProvider,
    DeduplicationCache,
    ProactiveEngine,
)
from draagon_ai.proactive.provider import ProviderContext
from draagon_ai.proactive.engine import ProactiveEngineConfig


class TestSuggestionPriority:
    """Tests for SuggestionPriority enum."""

    def test_priority_weights(self):
        """Test priority weight ordering."""
        assert SuggestionPriority.URGENT.weight > SuggestionPriority.HIGH.weight
        assert SuggestionPriority.HIGH.weight > SuggestionPriority.MEDIUM.weight
        assert SuggestionPriority.MEDIUM.weight > SuggestionPriority.LOW.weight

    def test_priority_values(self):
        """Test priority string values."""
        assert SuggestionPriority.URGENT.value == "urgent"
        assert SuggestionPriority.LOW.value == "low"


class TestSuggestion:
    """Tests for Suggestion class."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = Suggestion(
            id="test-123",
            message="Meeting in 30 minutes",
            priority=SuggestionPriority.HIGH,
            category=SuggestionCategory.CALENDAR,
        )

        assert suggestion.id == "test-123"
        assert suggestion.message == "Meeting in 30 minutes"
        assert suggestion.priority == SuggestionPriority.HIGH
        assert suggestion.category == SuggestionCategory.CALENDAR
        assert not suggestion.actionable

    def test_default_values(self):
        """Test default values."""
        suggestion = Suggestion(id="test", message="Test")

        assert suggestion.priority == SuggestionPriority.MEDIUM
        assert suggestion.category == SuggestionCategory.CUSTOM
        assert suggestion.source == "unknown"
        assert suggestion.expires_at is None
        assert suggestion.metadata == {}

    def test_is_expired_none(self):
        """Test expiration with no expiry set."""
        suggestion = Suggestion(id="test", message="Test")
        assert not suggestion.is_expired()

    def test_is_expired_future(self):
        """Test not expired when in future."""
        suggestion = Suggestion(
            id="test",
            message="Test",
            expires_at=time.time() + 100,
        )
        assert not suggestion.is_expired()

    def test_is_expired_past(self):
        """Test expired when in past."""
        suggestion = Suggestion(
            id="test",
            message="Test",
            expires_at=time.time() - 1,
        )
        assert suggestion.is_expired()

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = Suggestion(
            id="test-123",
            message="Test message",
            priority=SuggestionPriority.HIGH,
            category=SuggestionCategory.SECURITY,
            source="security_provider",
            metadata={"entity": "door.front"},
            actionable=True,
        )

        data = original.to_dict()
        restored = Suggestion.from_dict(data)

        assert restored.id == original.id
        assert restored.message == original.message
        assert restored.priority == original.priority
        assert restored.category == original.category
        assert restored.source == original.source
        assert restored.metadata == original.metadata


class TestAggregatedSuggestions:
    """Tests for AggregatedSuggestions class."""

    def test_empty_aggregated(self):
        """Test empty aggregated suggestions."""
        agg = AggregatedSuggestions()

        assert agg.count == 0
        assert not agg.has_urgent
        assert agg.get_top(3) == []

    def test_count_and_has_urgent(self):
        """Test count and urgent detection."""
        suggestions = [
            Suggestion(id="1", message="Urgent!", priority=SuggestionPriority.URGENT),
            Suggestion(id="2", message="Normal", priority=SuggestionPriority.MEDIUM),
        ]

        agg = AggregatedSuggestions(
            suggestions=suggestions,
            by_priority={
                SuggestionPriority.URGENT: [suggestions[0]],
                SuggestionPriority.MEDIUM: [suggestions[1]],
            },
        )

        assert agg.count == 2
        assert agg.has_urgent

    def test_get_top(self):
        """Test getting top N suggestions."""
        suggestions = [
            Suggestion(id="1", message="First"),
            Suggestion(id="2", message="Second"),
            Suggestion(id="3", message="Third"),
        ]

        agg = AggregatedSuggestions(suggestions=suggestions)

        top = agg.get_top(2)
        assert len(top) == 2
        assert top[0].id == "1"
        assert top[1].id == "2"

    def test_filter_by_category(self):
        """Test filtering by category."""
        cal = Suggestion(id="1", message="Cal", category=SuggestionCategory.CALENDAR)
        sec = Suggestion(id="2", message="Sec", category=SuggestionCategory.SECURITY)

        agg = AggregatedSuggestions(
            suggestions=[cal, sec],
            by_category={
                SuggestionCategory.CALENDAR: [cal],
                SuggestionCategory.SECURITY: [sec],
            },
        )

        calendar_only = agg.filter_by_category(SuggestionCategory.CALENDAR)
        assert len(calendar_only) == 1
        assert calendar_only[0].id == "1"

    def test_filter_by_priority(self):
        """Test filtering by minimum priority."""
        urgent = Suggestion(id="1", message="Urgent", priority=SuggestionPriority.URGENT)
        high = Suggestion(id="2", message="High", priority=SuggestionPriority.HIGH)
        low = Suggestion(id="3", message="Low", priority=SuggestionPriority.LOW)

        agg = AggregatedSuggestions(suggestions=[urgent, high, low])

        high_plus = agg.filter_by_priority(SuggestionPriority.HIGH)
        assert len(high_plus) == 2
        assert all(s.priority.weight >= SuggestionPriority.HIGH.weight for s in high_plus)


class TestDeduplicationCache:
    """Tests for DeduplicationCache class."""

    def test_should_show_new(self):
        """Test new suggestion should be shown."""
        cache = DeduplicationCache()
        assert cache.should_show("sug-1", "user-1")

    def test_should_not_show_after_mark(self):
        """Test suggestion not shown after marking."""
        cache = DeduplicationCache()

        cache.mark_shown("sug-1", "user-1")
        assert not cache.should_show("sug-1", "user-1")

    def test_different_users(self):
        """Test same suggestion for different users."""
        cache = DeduplicationCache()

        cache.mark_shown("sug-1", "user-1")

        assert not cache.should_show("sug-1", "user-1")
        assert cache.should_show("sug-1", "user-2")

    def test_expires_after_ttl(self):
        """Test entry expires after TTL."""
        cache = DeduplicationCache(default_ttl=0.1)  # 100ms

        cache.mark_shown("sug-1", "user-1")
        assert not cache.should_show("sug-1", "user-1")

        time.sleep(0.15)
        assert cache.should_show("sug-1", "user-1")

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        cache = DeduplicationCache(default_ttl=10)  # 10 seconds default

        cache.mark_shown("sug-1", "user-1", ttl=0.1)  # 100ms custom

        time.sleep(0.15)
        assert cache.should_show("sug-1", "user-1")

    def test_clear_for_user(self):
        """Test clearing cache for specific user."""
        cache = DeduplicationCache()

        cache.mark_shown("sug-1", "user-1")
        cache.mark_shown("sug-2", "user-1")
        cache.mark_shown("sug-1", "user-2")

        cleared = cache.clear_for_user("user-1")

        assert cleared == 2
        assert cache.should_show("sug-1", "user-1")
        assert not cache.should_show("sug-1", "user-2")

    def test_clear_all(self):
        """Test clearing entire cache."""
        cache = DeduplicationCache()

        cache.mark_shown("sug-1", "user-1")
        cache.mark_shown("sug-2", "user-2")

        cleared = cache.clear_all()

        assert cleared == 2
        assert cache.should_show("sug-1", "user-1")
        assert cache.should_show("sug-2", "user-2")

    def test_get_stats(self):
        """Test cache statistics."""
        cache = DeduplicationCache(default_ttl=3600)

        cache.mark_shown("sug-1", "user-1")
        cache.mark_shown("sug-2", "user-1")

        stats = cache.get_stats()

        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["default_ttl"] == 3600


class TestProviderContext:
    """Tests for ProviderContext class."""

    def test_create_context(self):
        """Test creating provider context."""
        context = ProviderContext(
            user_id="user-123",
            area_id="living_room",
            timezone="America/New_York",
        )

        assert context.user_id == "user-123"
        assert context.area_id == "living_room"
        assert context.timezone == "America/New_York"

    def test_default_values(self):
        """Test default context values."""
        context = ProviderContext(user_id="user-1")

        assert context.area_id is None
        assert context.device_id is None
        assert context.metadata == {}


class MockProvider(SuggestionProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        suggestions: list[Suggestion] | None = None,
        enabled: bool = True,
        delay: float = 0,
    ):
        self._name = name
        self._suggestions = suggestions or []
        self._enabled = enabled
        self._delay = delay
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def categories(self) -> list[SuggestionCategory]:
        return [SuggestionCategory.CUSTOM]

    async def get_suggestions(self, context: ProviderContext) -> list[Suggestion]:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._suggestions


class TestSuggestionProvider:
    """Tests for SuggestionProvider ABC."""

    @pytest.mark.asyncio
    async def test_mock_provider(self):
        """Test mock provider works."""
        suggestions = [
            Suggestion(id="1", message="Test 1"),
            Suggestion(id="2", message="Test 2"),
        ]
        provider = MockProvider(suggestions=suggestions)

        context = ProviderContext(user_id="user-1")
        result = await provider.get_suggestions(context)

        assert len(result) == 2
        assert provider.call_count == 1

    def test_provider_repr(self):
        """Test provider string representation."""
        provider = MockProvider(name="test_provider")
        assert "test_provider" in repr(provider)


class TestProactiveEngine:
    """Tests for ProactiveEngine class."""

    @pytest.mark.asyncio
    async def test_empty_engine(self):
        """Test engine with no providers."""
        engine = ProactiveEngine()
        context = ProviderContext(user_id="user-1")

        result = await engine.get_suggestions(context)

        assert result.count == 0
        assert result.provider_stats == {}

    @pytest.mark.asyncio
    async def test_single_provider(self):
        """Test engine with single provider."""
        suggestions = [
            Suggestion(id="1", message="Test 1", source="mock"),
        ]
        provider = MockProvider(suggestions=suggestions)

        engine = ProactiveEngine()
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        assert result.count == 1
        assert result.suggestions[0].message == "Test 1"
        assert result.provider_stats["mock"] == 1

    @pytest.mark.asyncio
    async def test_multiple_providers(self):
        """Test engine with multiple providers."""
        provider1 = MockProvider(
            name="provider1",
            suggestions=[Suggestion(id="1", message="P1", source="provider1")],
        )
        provider2 = MockProvider(
            name="provider2",
            suggestions=[Suggestion(id="2", message="P2", source="provider2")],
        )

        engine = ProactiveEngine()
        engine.register_provider(provider1)
        engine.register_provider(provider2)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        assert result.count == 2
        assert "provider1" in result.provider_stats
        assert "provider2" in result.provider_stats

    @pytest.mark.asyncio
    async def test_disabled_provider_skipped(self):
        """Test disabled provider is skipped."""
        provider = MockProvider(enabled=False)

        engine = ProactiveEngine()
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        await engine.get_suggestions(context)

        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_priority_sorting(self):
        """Test suggestions sorted by priority."""
        suggestions = [
            Suggestion(id="low", message="Low", priority=SuggestionPriority.LOW),
            Suggestion(id="urgent", message="Urgent", priority=SuggestionPriority.URGENT),
            Suggestion(id="high", message="High", priority=SuggestionPriority.HIGH),
        ]
        provider = MockProvider(suggestions=suggestions)

        engine = ProactiveEngine()
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        assert result.suggestions[0].id == "urgent"
        assert result.suggestions[1].id == "high"
        assert result.suggestions[2].id == "low"

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test suggestions are deduplicated."""
        suggestions = [
            Suggestion(id="1", message="Same"),
        ]
        provider = MockProvider(suggestions=suggestions)

        engine = ProactiveEngine()
        engine.register_provider(provider)
        context = ProviderContext(user_id="user-1")

        # First call - should show
        result1 = await engine.get_suggestions(context)
        assert result1.count == 1

        # Second call - should be deduplicated
        result2 = await engine.get_suggestions(context)
        assert result2.count == 0

    @pytest.mark.asyncio
    async def test_include_shown(self):
        """Test include_shown bypasses deduplication."""
        suggestions = [Suggestion(id="1", message="Test")]
        provider = MockProvider(suggestions=suggestions)

        engine = ProactiveEngine()
        engine.register_provider(provider)
        context = ProviderContext(user_id="user-1")

        # First call
        await engine.get_suggestions(context)

        # Second call with include_shown
        result = await engine.get_suggestions(context, include_shown=True)
        assert result.count == 1

    @pytest.mark.asyncio
    async def test_max_suggestions(self):
        """Test max suggestions limit."""
        suggestions = [
            Suggestion(id=f"sug-{i}", message=f"Test {i}")
            for i in range(20)
        ]
        provider = MockProvider(suggestions=suggestions)

        config = ProactiveEngineConfig(max_suggestions=5)
        engine = ProactiveEngine(config=config)
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        assert result.count == 5

    @pytest.mark.asyncio
    async def test_min_priority_filter(self):
        """Test minimum priority filter."""
        suggestions = [
            Suggestion(id="1", message="Low", priority=SuggestionPriority.LOW),
            Suggestion(id="2", message="High", priority=SuggestionPriority.HIGH),
        ]
        provider = MockProvider(suggestions=suggestions)

        engine = ProactiveEngine()
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(
            context,
            min_priority=SuggestionPriority.HIGH,
        )

        assert result.count == 1
        assert result.suggestions[0].id == "2"

    @pytest.mark.asyncio
    async def test_expired_filtered(self):
        """Test expired suggestions are filtered."""
        suggestions = [
            Suggestion(id="1", message="Expired", expires_at=time.time() - 1),
            Suggestion(id="2", message="Valid"),
        ]
        provider = MockProvider(suggestions=suggestions)

        engine = ProactiveEngine()
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        assert result.count == 1
        assert result.suggestions[0].id == "2"

    @pytest.mark.asyncio
    async def test_provider_timeout(self):
        """Test provider timeout handling."""
        provider = MockProvider(delay=0.5)  # 500ms delay

        config = ProactiveEngineConfig(provider_timeout=0.1)  # 100ms timeout
        engine = ProactiveEngine(config=config)
        engine.register_provider(provider)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        # Should timeout and return empty
        assert result.count == 0

    def test_register_duplicate_provider(self):
        """Test registering duplicate provider raises error."""
        engine = ProactiveEngine()
        engine.register_provider(MockProvider(name="test"))

        with pytest.raises(ValueError, match="already registered"):
            engine.register_provider(MockProvider(name="test"))

    def test_unregister_provider(self):
        """Test unregistering provider."""
        engine = ProactiveEngine()
        engine.register_provider(MockProvider(name="test"))

        assert engine.unregister_provider("test") is True
        assert engine.unregister_provider("nonexistent") is False
        assert "test" not in engine.list_providers()

    def test_get_provider(self):
        """Test getting provider by name."""
        provider = MockProvider(name="test")
        engine = ProactiveEngine()
        engine.register_provider(provider)

        assert engine.get_provider("test") is provider
        assert engine.get_provider("nonexistent") is None

    def test_list_providers(self):
        """Test listing providers."""
        engine = ProactiveEngine()
        engine.register_provider(MockProvider(name="provider1"))
        engine.register_provider(MockProvider(name="provider2"))

        providers = engine.list_providers()
        assert "provider1" in providers
        assert "provider2" in providers

    def test_clear_cache(self):
        """Test clearing engine cache."""
        engine = ProactiveEngine()
        engine._cache.mark_shown("sug-1", "user-1")
        engine._cache.mark_shown("sug-2", "user-2")

        # Clear for specific user
        cleared = engine.clear_cache("user-1")
        assert cleared == 1

        # Clear all
        engine._cache.mark_shown("sug-3", "user-3")
        cleared = engine.clear_cache()
        assert cleared >= 1

    def test_get_stats(self):
        """Test getting engine stats."""
        engine = ProactiveEngine()
        engine.register_provider(MockProvider(name="test"))

        stats = engine.get_stats()

        assert "providers" in stats
        assert "test" in stats["providers"]
        assert "cache" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_initialize_shutdown(self):
        """Test initialize and shutdown lifecycle."""
        provider = MockProvider()
        engine = ProactiveEngine()
        engine.register_provider(provider)

        await engine.initialize()
        assert engine._initialized

        await engine.shutdown()
        assert not engine._initialized

    @pytest.mark.asyncio
    async def test_sequential_providers(self):
        """Test sequential provider execution."""
        provider1 = MockProvider(name="p1", suggestions=[Suggestion(id="1", message="P1")])
        provider2 = MockProvider(name="p2", suggestions=[Suggestion(id="2", message="P2")])

        config = ProactiveEngineConfig(parallel_providers=False)
        engine = ProactiveEngine(config=config)
        engine.register_provider(provider1)
        engine.register_provider(provider2)

        context = ProviderContext(user_id="user-1")
        result = await engine.get_suggestions(context)

        assert result.count == 2
        assert provider1.call_count == 1
        assert provider2.call_count == 1

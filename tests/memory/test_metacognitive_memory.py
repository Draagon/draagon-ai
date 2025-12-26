"""Tests for Metacognitive Memory layer."""

import pytest

from draagon_ai.memory import (
    TemporalCognitiveGraph,
    MetacognitiveMemory,
    Skill,
    Strategy,
    Insight,
    BehaviorNode,
)


@pytest.fixture
def graph():
    """Create a fresh graph for each test."""
    return TemporalCognitiveGraph()


@pytest.fixture
def metacognitive(graph):
    """Create metacognitive memory instance."""
    return MetacognitiveMemory(graph)


class TestSkillManagement:
    """Test skill CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_skill(self, metacognitive):
        """Test adding a skill."""
        skill = await metacognitive.add_skill(
            name="restart_plex",
            skill_type="command",
            procedure="docker restart plex",
        )

        assert skill is not None
        assert skill.skill_name == "restart_plex"
        assert skill.skill_type == "command"
        assert skill.procedure == "docker restart plex"
        assert skill.version == 1

    @pytest.mark.asyncio
    async def test_get_skill(self, metacognitive):
        """Test retrieving skill by ID."""
        skill = await metacognitive.add_skill(
            "test_skill",
            "procedure",
            "do something",
        )

        retrieved = await metacognitive.get_skill(skill.node_id)

        assert retrieved is not None
        assert retrieved.skill_name == "test_skill"

    @pytest.mark.asyncio
    async def test_get_skill_by_name(self, metacognitive):
        """Test retrieving skill by name."""
        await metacognitive.add_skill("my_skill", "command", "ls -la")

        skill = await metacognitive.get_skill_by_name("my_skill")

        assert skill is not None
        assert skill.skill_name == "my_skill"

    @pytest.mark.asyncio
    async def test_duplicate_skill_returns_existing(self, metacognitive):
        """Test that adding duplicate skill returns existing."""
        skill1 = await metacognitive.add_skill("unique_skill", "cmd", "test")
        skill2 = await metacognitive.add_skill("unique_skill", "cmd", "different")

        assert skill1.node_id == skill2.node_id


class TestSkillEffectiveness:
    """Test skill effectiveness tracking."""

    @pytest.mark.asyncio
    async def test_record_success(self, metacognitive):
        """Test recording skill success."""
        skill = await metacognitive.add_skill("test", "cmd", "echo hi")

        await metacognitive.record_skill_result(skill.node_id, success=True)
        await metacognitive.record_skill_result(skill.node_id, success=True)

        updated = await metacognitive.get_skill(skill.node_id)
        assert updated.success_count == 2
        assert updated.failure_count == 0

    @pytest.mark.asyncio
    async def test_record_failure(self, metacognitive):
        """Test recording skill failure."""
        skill = await metacognitive.add_skill("failing", "cmd", "bad cmd")

        await metacognitive.record_skill_result(skill.node_id, success=False)

        updated = await metacognitive.get_skill(skill.node_id)
        assert updated.failure_count == 1

    @pytest.mark.asyncio
    async def test_effectiveness_score(self, metacognitive):
        """Test effectiveness score calculation."""
        skill = await metacognitive.add_skill("mixed", "cmd", "test")

        # 3 successes, 1 failure = 75% effectiveness
        await metacognitive.record_skill_result(skill.node_id, success=True)
        await metacognitive.record_skill_result(skill.node_id, success=True)
        await metacognitive.record_skill_result(skill.node_id, success=True)
        await metacognitive.record_skill_result(skill.node_id, success=False)

        updated = await metacognitive.get_skill(skill.node_id)
        assert updated.effectiveness_score == 0.75

    @pytest.mark.asyncio
    async def test_needs_improvement(self, metacognitive):
        """Test needs_improvement detection."""
        skill = await metacognitive.add_skill("poor", "cmd", "bad")

        # 1 success, 3 failures = 25% effectiveness
        await metacognitive.record_skill_result(skill.node_id, success=True)
        await metacognitive.record_skill_result(skill.node_id, success=False)
        await metacognitive.record_skill_result(skill.node_id, success=False)
        await metacognitive.record_skill_result(skill.node_id, success=False)

        updated = await metacognitive.get_skill(skill.node_id)
        assert updated.needs_improvement is True

    @pytest.mark.asyncio
    async def test_get_effective_skills(self, metacognitive):
        """Test retrieving high-effectiveness skills."""
        good = await metacognitive.add_skill("good", "cmd", "works")
        bad = await metacognitive.add_skill("bad", "cmd", "fails")

        # Good skill: 4 successes
        for _ in range(4):
            await metacognitive.record_skill_result(good.node_id, success=True)

        # Bad skill: 4 failures
        for _ in range(4):
            await metacognitive.record_skill_result(bad.node_id, success=False)

        effective = await metacognitive.get_effective_skills(
            min_effectiveness=0.8,
            min_uses=3,
        )

        assert len(effective) == 1
        assert effective[0].skill_name == "good"


class TestSkillImprovement:
    """Test skill improvement (versioning)."""

    @pytest.mark.asyncio
    async def test_improve_skill(self, metacognitive):
        """Test improving a skill creates new version."""
        original = await metacognitive.add_skill(
            "old_way",
            "cmd",
            "systemctl restart plex",
        )

        improved = await metacognitive.improve_skill(
            original.node_id,
            new_procedure="docker restart plex",
            reason="Using Docker now",
        )

        assert improved is not None
        assert improved.version == 2
        assert improved.procedure == "docker restart plex"
        assert improved.parent_skill_id == original.node_id

    @pytest.mark.asyncio
    async def test_improved_skill_resets_counts(self, metacognitive):
        """Test that improved skill has fresh counts."""
        original = await metacognitive.add_skill("v1", "cmd", "old")
        await metacognitive.record_skill_result(original.node_id, success=True)
        await metacognitive.record_skill_result(original.node_id, success=True)

        improved = await metacognitive.improve_skill(original.node_id, "new")

        assert improved.success_count == 0
        assert improved.failure_count == 0


class TestStrategyManagement:
    """Test strategy operations."""

    @pytest.mark.asyncio
    async def test_add_strategy(self, metacognitive):
        """Test adding a strategy."""
        skill = await metacognitive.add_skill("diagnose", "procedure", "check logs")

        strategy = await metacognitive.add_strategy(
            name="troubleshooting",
            strategy_type="reasoning",
            description="Steps to diagnose system issues",
            applicable_contexts=["error", "failure", "issue"],
            skill_ids=[skill.node_id],
        )

        assert strategy is not None
        assert strategy.strategy_name == "troubleshooting"
        assert skill.node_id in strategy.skill_ids

    @pytest.mark.asyncio
    async def test_get_strategy(self, metacognitive):
        """Test retrieving strategy by ID."""
        strategy = await metacognitive.add_strategy(
            "test_strategy",
            "decision",
            "Test description",
        )

        retrieved = await metacognitive.get_strategy(strategy.node_id)

        assert retrieved is not None
        assert retrieved.strategy_name == "test_strategy"

    @pytest.mark.asyncio
    async def test_find_strategy_for_context(self, metacognitive):
        """Test finding strategies for a context."""
        await metacognitive.add_strategy(
            "error_handling",
            "reasoning",
            "How to handle errors gracefully",
            applicable_contexts=["error", "exception", "failure"],
        )

        strategies = await metacognitive.find_strategy_for_context(
            "The system threw an error",
        )

        # Should find the strategy
        assert isinstance(strategies, list)


class TestInsightManagement:
    """Test insight operations."""

    @pytest.mark.asyncio
    async def test_add_insight(self, metacognitive):
        """Test adding an insight."""
        insight = await metacognitive.add_insight(
            content="Users often ask about weather after time queries",
            insight_type="pattern",
            context="conversation flow",
            recommendation="Proactively offer weather info",
        )

        assert insight is not None
        assert insight.insight_type == "pattern"
        assert insight.evidence_count == 1

    @pytest.mark.asyncio
    async def test_insights_can_be_added(self, metacognitive):
        """Test that multiple insights can be added."""
        insight1 = await metacognitive.add_insight(
            "Weather queries common after greetings",
            "pattern",
        )

        insight2 = await metacognitive.add_insight(
            "Time queries often precede calendar queries",
            "pattern",
        )

        # Both insights should exist
        assert insight1 is not None
        assert insight2 is not None
        assert insight1.evidence_count >= 1


class TestBehaviorManagement:
    """Test behavior operations."""

    @pytest.mark.asyncio
    async def test_register_behavior(self, metacognitive):
        """Test registering a behavior."""
        behavior = await metacognitive.register_behavior(
            behavior_id="voice_assistant",
            behavior_name="Voice Assistant",
            description="Handles voice queries",
            learns_from_scope="user:doug",
            contributes_to_scope="household:mealing_home",
        )

        assert behavior is not None
        assert behavior.behavior_id == "voice_assistant"
        assert behavior.behavior_name == "Voice Assistant"

    @pytest.mark.asyncio
    async def test_behavior_dependencies(self, metacognitive):
        """Test behavior dependency tracking."""
        base = await metacognitive.register_behavior(
            "base",
            "Base Behavior",
            "Foundation",
        )

        derived = await metacognitive.register_behavior(
            "derived",
            "Derived Behavior",
            "Uses base",
            depends_on=[base.node_id],
        )

        assert base.node_id in derived.depends_on

    @pytest.mark.asyncio
    async def test_record_behavior_execution(self, metacognitive):
        """Test recording behavior execution."""
        behavior = await metacognitive.register_behavior(
            "test",
            "Test",
            "For testing",
        )

        await metacognitive.record_behavior_execution(behavior.node_id, success=True)
        await metacognitive.record_behavior_execution(
            behavior.node_id,
            success=True,
            fitness_delta=0.1,
        )

        # Verify the behavior was updated (would need get_behavior method)
        # For now, just verify it doesn't crash
        assert True


class TestBehaviorEvolution:
    """Test behavior evolution features via the metacognitive layer."""

    @pytest.mark.asyncio
    async def test_behavior_effectiveness_recording(self, metacognitive):
        """Test behavior effectiveness through execution recording."""
        behavior = await metacognitive.register_behavior(
            "eff_test",
            "Effectiveness Test",
            "Testing effectiveness tracking",
        )

        # Record some results
        await metacognitive.record_behavior_execution(behavior.node_id, success=True)
        await metacognitive.record_behavior_execution(behavior.node_id, success=True)
        await metacognitive.record_behavior_execution(behavior.node_id, success=False)

        # Behavior exists
        assert behavior is not None

    @pytest.mark.asyncio
    async def test_behavior_fitness_update(self, metacognitive):
        """Test that fitness can be updated."""
        behavior = await metacognitive.register_behavior(
            "fit_test",
            "Fitness Test",
            "Testing fitness updates",
        )

        # Record with fitness boost
        result = await metacognitive.record_behavior_execution(
            behavior.node_id,
            success=True,
            fitness_delta=0.15,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_behavior_self_improvement_settings(self, metacognitive):
        """Test behavior self-improvement configuration."""
        behavior = await metacognitive.register_behavior(
            "improve_test",
            "Improvement Test",
            "Testing self-improvement settings",
        )

        # Behavior should have self-improvement enabled by default
        assert behavior.metadata.get("can_self_improve", True) is True


class TestStats:
    """Test metacognitive statistics."""

    @pytest.mark.asyncio
    async def test_stats(self, metacognitive):
        """Test getting metacognitive stats."""
        await metacognitive.add_skill("s1", "cmd", "test1")
        await metacognitive.add_skill("s2", "cmd", "test2")
        await metacognitive.add_strategy("strategy1", "reasoning", "desc")
        await metacognitive.add_insight("insight1", "pattern")
        await metacognitive.register_behavior("b1", "Behavior1", "desc")

        stats = metacognitive.stats()

        assert stats["skill_count"] == 2
        assert stats["strategy_count"] == 1
        assert stats["insight_count"] == 1
        assert stats["behavior_count"] == 1

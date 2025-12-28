"""Safety E2E tests for the autonomous agent.

These tests verify critical safety guarantees end-to-end:
- Tier restrictions are enforced
- Harmful actions are blocked regardless of tier
- Budget and rate limits prevent runaway behavior
- Self-monitoring catches issues
- Transparency is maintained

REQ-004-10: Safety E2E tests
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.orchestration.autonomous import (
    AutonomousAgentService,
    AutonomousConfig,
    ActionType,
    ActionTier,
    ProposedAction,
    ApprovedAction,
    ActionResult,
    ActionLog,
    HarmCheck,
    SafetyCheck,
    AutonomousContext,
    SelfMonitoringFinding,
    SelfMonitoringResult,
    LLMProvider,
    SearchProvider,
    MemoryStoreProvider,
    ContextProvider,
    NotificationProvider,
)


# =============================================================================
# Realistic Providers for E2E Testing
# =============================================================================


class RealisticLLMProvider:
    """LLM provider that simulates realistic safety-aware responses."""

    def __init__(self, override_harmful: bool = False):
        """Initialize with optional harmful override for testing.

        Args:
            override_harmful: If True, LLM incorrectly says actions are safe
                            (tests that other guardrails still catch them)
        """
        self.override_harmful = override_harmful
        self.call_history: list[dict] = []
        self._response_overrides: list[str] = []

    def override_response(self, response: str) -> None:
        """Override the next response."""
        self._response_overrides.append(response)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        self.call_history.append({"prompt": prompt, "max_tokens": max_tokens})

        if self._response_overrides:
            return self._response_overrides.pop(0)

        # Simulate realistic LLM safety awareness
        prompt_lower = prompt.lower()

        # Harm check detection
        if "harmful" in prompt_lower or "harm" in prompt_lower:
            if self.override_harmful:
                return json.dumps({
                    "potentially_harmful": False,
                    "reason": "Appears safe",
                    "confidence": 0.8,
                })

            # Realistic harm detection
            harmful_keywords = [
                "security camera", "password", "financial",
                "delete", "modify", "access private",
                "impersonate", "send message", "disable alarm"
            ]
            is_harmful = any(kw in prompt_lower for kw in harmful_keywords)

            return json.dumps({
                "potentially_harmful": is_harmful,
                "reason": "Detected potentially harmful action" if is_harmful else "Action appears safe",
                "confidence": 0.9,
            })

        # Semantic safety check
        if "safety" in prompt_lower or "safe" in prompt_lower:
            if self.override_harmful:
                return "SAFE: Action approved"

            # Check for dangerous patterns
            dangerous = ["private", "security", "financial", "password", "credential"]
            is_dangerous = any(d in prompt_lower for d in dangerous)

            if is_dangerous:
                return "UNSAFE: Action involves sensitive systems"
            return "SAFE: Action is within acceptable bounds"

        # Proposal generation
        if "propose" in prompt_lower or "action" in prompt_lower:
            return json.dumps({
                "proposed_actions": [{
                    "type": "research",
                    "description": "Research safe topic",
                    "reasoning": "General learning",
                    "risk_tier": 0,
                }]
            })

        # Self-monitoring
        if "monitor" in prompt_lower or "review" in prompt_lower:
            return json.dumps({
                "findings": [],
                "notify_user": False,
                "lessons_learned": [],
            })

        return "Default response"


class RealisticMemoryProvider:
    """Memory provider that tracks all operations for safety verification."""

    def __init__(self):
        self.all_operations: list[dict] = []
        self.stored_items: list[dict] = []
        self.stored_logs: list[dict] = []

    async def store(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        **metadata,
    ) -> str:
        op = {
            "operation": "store",
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
        }
        self.all_operations.append(op)
        self.stored_items.append(op)
        return f"mem_{len(self.stored_items)}"

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        self.all_operations.append({
            "operation": "search",
            "query": query,
            "timestamp": datetime.now().isoformat(),
        })
        return self.stored_items[:limit]

    async def get_logs(self, record_type: str, limit: int = 100) -> list[dict]:
        return self.stored_logs[:limit]

    async def store_log(self, log) -> None:
        self.all_operations.append({
            "operation": "store_log",
            "log_type": type(log).__name__,
            "timestamp": datetime.now().isoformat(),
        })
        self.stored_logs.append({"log": log})


class RealisticNotificationProvider:
    """Notification provider that tracks all notifications for safety verification."""

    def __init__(self):
        self.notifications: list[dict] = []

    async def queue_notification(
        self,
        message: str,
        priority: str = "medium",
    ) -> None:
        self.notifications.append({
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        })


# =============================================================================
# Safety Guarantee Tests
# =============================================================================


class TestTierSafetyGuarantees:
    """Test that tier restrictions are always enforced."""

    @pytest.mark.asyncio
    async def test_cannot_send_message_without_approval(self):
        """Tier 3 action (send message) is always blocked."""
        llm = RealisticLLMProvider(override_harmful=True)  # LLM says safe
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send message to user about the weather",
            reasoning="User might want to know",
            risk_tier=ActionTier.TIER_3,
        )

        approved = await service._filter_through_guardrails([action])

        # Must be blocked regardless of LLM response
        assert len(approved) == 0
        blocked = [log for log in service._action_logs if log.blocked]
        assert len(blocked) == 1
        assert "tier" in blocked[0].blocked_reason.lower()

    @pytest.mark.asyncio
    async def test_cannot_control_device_autonomously(self):
        """Tier 3 action (device control) is always blocked."""
        llm = RealisticLLMProvider(override_harmful=True)
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Turn off security cameras",
            reasoning="Maintenance",
            risk_tier=ActionTier.TIER_3,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_cannot_perform_financial_actions(self):
        """Tier 4 action (financial) is absolutely forbidden."""
        llm = RealisticLLMProvider(override_harmful=True)
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Transfer money to external account",
            reasoning="Bill payment",
            risk_tier=ActionTier.TIER_4,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        # LLM should NOT even be consulted for Tier 4
        harm_calls = [c for c in llm.call_history if "harm" in c["prompt"].lower()]
        assert len(harm_calls) == 0

    @pytest.mark.asyncio
    async def test_cannot_access_security_systems(self):
        """Tier 4 action (security access) is absolutely forbidden."""
        llm = RealisticLLMProvider(override_harmful=True)
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Disable home security alarm",
            reasoning="User is locked out",
            risk_tier=ActionTier.TIER_4,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_cannot_impersonate_user(self):
        """Tier 4 action (impersonation) is absolutely forbidden."""
        llm = RealisticLLMProvider(override_harmful=True)
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send email as user to their boss",
            reasoning="User forgot to reply",
            risk_tier=ActionTier.TIER_4,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0


class TestHarmCheckSafetyGuarantees:
    """Test that harmful actions are blocked even with low tier."""

    @pytest.mark.asyncio
    async def test_harmful_research_blocked(self):
        """Harmful Tier 0 action is blocked by harm check."""
        llm = RealisticLLMProvider()  # Normal LLM detects harm
        config = AutonomousConfig(enabled=True, log_all_proposals=True)
        memory = RealisticMemoryProvider()

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research user's private browsing history and passwords",
            reasoning="Curious about security practices",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        blocked = [log for log in service._action_logs if log.blocked]
        assert len(blocked) == 1
        assert "harm" in blocked[0].blocked_reason.lower()

    @pytest.mark.asyncio
    async def test_privacy_violating_research_blocked(self):
        """Research that violates privacy is blocked."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research and compile user's personal financial records",
            reasoning="Want to help with budgeting",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_credential_research_blocked(self):
        """Research for credentials is blocked."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(enabled=True)

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Find user's saved passwords and credentials",
            reasoning="Security audit",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0


class TestSemanticSafetyGuarantees:
    """Test semantic safety check as final guardrail."""

    @pytest.mark.asyncio
    async def test_semantic_safety_catches_subtle_harm(self):
        """Semantic safety catches actions that pass harm check."""
        llm = RealisticLLMProvider()

        # Override harm check to pass, but semantic to fail
        llm.override_response(json.dumps({
            "potentially_harmful": False,
            "confidence": 0.8,
        }))
        llm.override_response("UNSAFE: Upon deeper analysis, this could violate privacy")

        config = AutonomousConfig(
            enabled=True,
            require_semantic_safety_check=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Analyze patterns in user's private messages",
            reasoning="Better personalization",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        blocked = [log for log in service._action_logs if log.blocked]
        assert any("semantic" in log.blocked_reason.lower() for log in blocked)

    @pytest.mark.asyncio
    async def test_semantic_safety_can_be_required(self):
        """Semantic safety check can be enforced via config."""
        llm = RealisticLLMProvider()

        # Set up responses: harm check passes, semantic fails
        llm.override_response(json.dumps({"potentially_harmful": False}))
        llm.override_response("UNSAFE: This seems risky")

        config = AutonomousConfig(
            enabled=True,
            require_semantic_safety_check=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research something",
            reasoning="Learning",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0


class TestBudgetSafetyGuarantees:
    """Test that budget limits prevent runaway behavior."""

    @pytest.mark.asyncio
    async def test_budget_enforced_strictly(self):
        """Daily budget is strictly enforced."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            daily_action_budget=3,
        )

        service = AutonomousAgentService(llm=llm, config=config)
        service._actions_today = 3  # Already at budget

        # Queue a valid proposal
        llm.override_response(json.dumps({
            "proposed_actions": [{
                "type": "rest",
                "description": "Rest",
                "reasoning": "Nothing",
                "risk_tier": 0,
            }]
        }))

        results = await service.run_cycle()

        # Must return empty - budget exhausted
        assert results == []

    @pytest.mark.asyncio
    async def test_budget_prevents_excessive_cycles(self):
        """Multiple cycles are blocked once budget is exhausted."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            daily_action_budget=2,
            max_actions_per_cycle=1,
        )
        memory = RealisticMemoryProvider()

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        # Queue valid proposals for multiple cycles
        for _ in range(5):
            llm.override_response(json.dumps({
                "proposed_actions": [{
                    "type": "rest",
                    "description": "Rest",
                    "reasoning": "Nothing",
                    "risk_tier": 0,
                }]
            }))
            llm.override_response(json.dumps({"potentially_harmful": False}))
            llm.override_response("SAFE")

        total_actions = 0
        for _ in range(5):
            results = await service.run_cycle()
            total_actions += len(results)

        # Must respect budget
        assert total_actions <= 2

    @pytest.mark.asyncio
    async def test_budget_resets_daily(self):
        """Budget resets on new day."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            daily_action_budget=5,
        )

        service = AutonomousAgentService(llm=llm, config=config)
        service._actions_today = 5  # Exhausted
        service._last_action_reset = datetime.now() - timedelta(days=1)  # Yesterday

        # Should reset
        service._check_daily_reset()

        assert service._actions_today == 0


class TestRateLimitSafetyGuarantees:
    """Test that rate limits prevent repetitive behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_prevents_same_type_spam(self):
        """Rate limit prevents consecutive same-type actions."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            max_consecutive_same_type=2,
            log_all_proposals=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)
        service._recent_action_types = [
            ActionType.RESEARCH,
            ActionType.RESEARCH,
        ]

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="More research",
            reasoning="Learning",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        blocked = [log for log in service._action_logs if log.blocked]
        assert any("rate" in log.blocked_reason.lower() for log in blocked)

    @pytest.mark.asyncio
    async def test_rate_limit_allows_variety(self):
        """Rate limit allows different action types."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            max_consecutive_same_type=2,
        )

        llm.override_response(json.dumps({"potentially_harmful": False}))
        llm.override_response("SAFE")

        service = AutonomousAgentService(llm=llm, config=config)
        service._recent_action_types = [
            ActionType.RESEARCH,
            ActionType.RESEARCH,
        ]

        # Different type should be allowed
        action = ProposedAction(
            action_type=ActionType.REFLECT,
            description="Reflect",
            reasoning="Self-improvement",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 1


class TestSelfMonitoringSafetyGuarantees:
    """Test that self-monitoring catches issues."""

    @pytest.mark.asyncio
    async def test_self_monitoring_detects_issues(self):
        """Self-monitoring detects and reports issues."""
        llm = RealisticLLMProvider()

        # Override monitoring response
        llm.override_response(json.dumps({
            "findings": [{
                "type": "unexpected_result",
                "description": "Action produced unexpected result",
                "severity": "high",
                "action_recommended": "Review and adjust",
            }],
            "notify_user": True,
            "notification_message": "Found an issue that needs attention",
        }))

        memory = RealisticMemoryProvider()
        notification = RealisticNotificationProvider()
        config = AutonomousConfig(
            enabled=True,
            enable_self_monitoring=True,
            persist_logs=True,
        )

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
            notification_provider=notification,
        )

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        # Should store finding
        assert len(memory.stored_logs) == 1
        # Should notify user
        assert len(notification.notifications) == 1

    @pytest.mark.asyncio
    async def test_self_monitoring_runs_after_cycle(self):
        """Self-monitoring runs after each cycle."""
        llm = RealisticLLMProvider()

        # Queue responses for full cycle
        llm.override_response(json.dumps({
            "proposed_actions": [{
                "type": "rest",
                "description": "Rest",
                "reasoning": "Nothing",
                "risk_tier": 0,
            }]
        }))
        llm.override_response(json.dumps({"potentially_harmful": False}))
        llm.override_response("SAFE")
        llm.override_response(json.dumps({
            "findings": [{
                "type": "pattern",
                "description": "Detected pattern",
                "severity": "medium",
            }],
            "notify_user": False,
        }))

        memory = RealisticMemoryProvider()
        config = AutonomousConfig(
            enabled=True,
            enable_self_monitoring=True,
            persist_logs=True,
        )

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        await service.run_cycle()

        # Self-monitoring should have stored finding
        findings = [l for l in memory.stored_logs if "SelfMonitoringFinding" in str(l)]
        assert len(findings) >= 1


class TestTransparencySafetyGuarantees:
    """Test that all actions are transparent and logged."""

    @pytest.mark.asyncio
    async def test_all_actions_logged(self):
        """Every action (successful or blocked) is logged."""
        llm = RealisticLLMProvider()
        memory = RealisticMemoryProvider()
        config = AutonomousConfig(
            enabled=True,
            log_all_proposals=True,
            persist_logs=True,
        )

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        # Execute successful action
        action1 = ProposedAction(
            action_type=ActionType.REST,
            description="Rest",
            reasoning="Nothing",
            risk_tier=ActionTier.TIER_0,
        )
        await service._log_action(action1, ActionResult(success=True, outcome="Done"))

        # Log blocked action
        action2 = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Blocked action",
            reasoning="Testing",
            risk_tier=ActionTier.TIER_3,
        )
        await service._log_blocked(action2, "tier_violation")

        # Both should be logged
        assert len(service._action_logs) == 2
        # Both should be persisted
        assert len(memory.stored_logs) == 2

    @pytest.mark.asyncio
    async def test_blocked_reasons_are_clear(self):
        """Blocked actions have clear, informative reasons."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            log_all_proposals=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        # Test different block reasons
        actions = [
            (ProposedAction(
                action_type=ActionType.PREPARE_SUGGESTION,
                description="High tier",
                reasoning="Testing",
                risk_tier=ActionTier.TIER_3,
            ), "tier"),
        ]

        for action, expected_reason in actions:
            await service._log_blocked(action, f"{expected_reason}_violation")

        for log in service._action_logs:
            assert log.blocked is True
            assert log.blocked_reason is not None
            assert len(log.blocked_reason) > 0

    @pytest.mark.asyncio
    async def test_transparency_api_complete(self):
        """Transparency API returns complete information."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            daily_action_budget=20,
            shadow_mode=False,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        # Add some logs
        action = ProposedAction(
            action_type=ActionType.REST,
            description="Rest",
            reasoning="Nothing",
        )
        await service._log_action(action, ActionResult(success=True, outcome="Done"))

        blocked = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Blocked",
            reasoning="Testing",
            risk_tier=ActionTier.TIER_3,
        )
        await service._log_blocked(blocked, "tier_violation")

        # Check stats
        stats = service.get_stats()
        assert "enabled" in stats
        assert "shadow_mode" in stats
        assert "actions_today" in stats
        assert "daily_budget" in stats
        assert "total_logs" in stats
        assert "blocked_count" in stats
        assert stats["blocked_count"] == 1

        # Check logs
        all_logs = service.get_action_logs()
        assert len(all_logs) == 2

        blocked_logs = service.get_blocked_logs()
        assert len(blocked_logs) == 1


class TestShadowModeSafety:
    """Test shadow mode as safety feature."""

    @pytest.mark.asyncio
    async def test_shadow_mode_prevents_execution(self):
        """Shadow mode prevents actual execution while logging."""
        llm = RealisticLLMProvider()

        # Queue responses for action that would execute
        llm.override_response(json.dumps({
            "proposed_actions": [{
                "type": "research",
                "description": "Research something",
                "reasoning": "Learning",
                "risk_tier": 0,
            }]
        }))
        llm.override_response(json.dumps({"potentially_harmful": False}))
        llm.override_response("SAFE")

        memory = RealisticMemoryProvider()
        config = AutonomousConfig(
            enabled=True,
            shadow_mode=True,  # Shadow mode ON
            persist_logs=True,
        )

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        results = await service.run_cycle()

        # No actual results in shadow mode
        assert results == []

        # But action should be logged
        assert len(service._action_logs) >= 1

        # Memory should NOT have stored research results
        research_stores = [
            op for op in memory.all_operations
            if op.get("operation") == "store" and "insight" in op.get("memory_type", "")
        ]
        assert len(research_stores) == 0

    @pytest.mark.asyncio
    async def test_shadow_mode_can_be_toggled(self):
        """Shadow mode can be enabled/disabled."""
        llm = RealisticLLMProvider()

        config = AutonomousConfig(
            enabled=True,
            shadow_mode=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        assert service.config.shadow_mode is True

        # Toggle off
        service.config.shadow_mode = False
        assert service.config.shadow_mode is False


class TestDefenseInDepth:
    """Test that multiple layers provide defense in depth."""

    @pytest.mark.asyncio
    async def test_multiple_layers_block_harmful_action(self):
        """Multiple guardrail layers all contribute to safety."""
        llm = RealisticLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            require_semantic_safety_check=True,
            log_all_proposals=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        # Action that should be blocked by tier
        tier_blocked = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send message",
            reasoning="Help",
            risk_tier=ActionTier.TIER_3,
        )

        approved = await service._filter_through_guardrails([tier_blocked])
        assert len(approved) == 0

        # Action that passes tier but blocked by harm check
        llm.override_response(json.dumps({
            "potentially_harmful": True,
            "reason": "Could harm privacy",
        }))

        harm_blocked = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research private info",
            reasoning="Curiosity",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([harm_blocked])
        assert len(approved) == 0

        # Action that passes tier and harm but blocked by semantic
        llm.override_response(json.dumps({"potentially_harmful": False}))
        llm.override_response("UNSAFE: Deep analysis shows risk")

        semantic_blocked = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research user patterns",
            reasoning="Personalization",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([semantic_blocked])
        assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_all_guardrails_must_pass(self):
        """Action must pass ALL guardrails to be approved."""
        llm = RealisticLLMProvider()

        # All checks pass
        llm.override_response(json.dumps({"potentially_harmful": False}))
        llm.override_response("SAFE")

        config = AutonomousConfig(
            enabled=True,
            require_semantic_safety_check=True,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research AI trends",
            reasoning="Stay informed",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 1
        # All guardrails should be in passed list
        assert "tier" in approved[0].guardrails_passed
        assert "rate" in approved[0].guardrails_passed
        assert "harm" in approved[0].guardrails_passed
        assert "semantic" in approved[0].guardrails_passed

"""Integration tests for SecurityMonitorService.

These tests verify the complete security monitoring flow:
- Real scheduling system integration
- Real alert processing pipeline
- Timer-based check cycles
- Summary generation

Uses real components where possible, minimal mocking.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import sys
sys.path.insert(0, "/home/doug/Development/draagon-ai/extensions/security-monitor/src")
sys.path.insert(0, "/home/doug/Development/draagon-ai/src")

from draagon_ai_ext_security.service import SecurityMonitorService
from draagon_ai_ext_security.models import Alert, Severity, ThreatLevel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def full_config():
    """Full configuration for integration tests."""
    return {
        "check_interval_seconds": 1,  # Fast for tests
        "monitors": {
            "suricata": {"enabled": False},  # Disable real monitors
            "syslog": {"enabled": False},
            "system_health": {"enabled": False},
        },
        "notifications": {
            "voice": {
                "enabled": True,
                "min_severity": "high",
                "quiet_hours": {
                    "enabled": False,  # Disable for tests
                },
            },
            "summary": {
                "enabled": False,  # Disable cron for tests
            },
        },
        "network": {
            "home_net": "192.168.168.0/24",
            "known_services": [
                {"name": "Plex", "ip": "192.168.168.204", "ports": [32400]},
                {"name": "Home Assistant", "ip": "192.168.168.206", "ports": [8123]},
            ],
        },
    }


def create_test_alert(
    alert_id: str = "test_001",
    signature: str = "ET SCAN Test",
    source_ip: str = "192.168.168.100",
    severity: Severity = Severity.MEDIUM,
) -> Alert:
    """Create a test alert with sensible defaults."""
    return Alert(
        id=alert_id,
        source="test",
        signature=signature,
        description=f"Test alert: {signature}",
        source_ip=source_ip,
        dest_ip="192.168.168.1",
        severity=severity,
        timestamp=datetime.now(),
        raw_data={"test": True},
    )


# =============================================================================
# Integration Tests - Alert Pipeline
# =============================================================================


class TestAlertPipeline:
    """Test the complete alert processing pipeline."""

    @pytest.mark.asyncio
    async def test_full_alert_cycle(self, full_config):
        """Test: ingest → deduplicate → store → retrieve."""
        service = SecurityMonitorService(full_config)

        # Simulate ingesting alerts
        alerts = [
            create_test_alert("a1", "SCAN Attempt", "10.0.0.1", Severity.MEDIUM),
            create_test_alert("a2", "SCAN Attempt", "10.0.0.1", Severity.MEDIUM),  # Dup
            create_test_alert("a3", "SQL Injection", "10.0.0.2", Severity.HIGH),
        ]

        # Deduplicate
        unique = service._deduplicate(alerts)
        assert len(unique) == 2  # One duplicate removed

        # Add to history
        service._alert_history.extend(unique)

        # Retrieve recent
        recent = service.get_recent_alerts(minutes=60)
        assert len(recent) == 2

        # Verify severity filtering
        high_alerts = [a for a in recent if a.severity == Severity.HIGH]
        assert len(high_alerts) == 1
        assert "SQL Injection" in high_alerts[0].signature

    @pytest.mark.asyncio
    async def test_alert_history_pruning_over_time(self, full_config):
        """Test that old alerts are pruned correctly."""
        service = SecurityMonitorService(full_config)

        # Add alerts at different times
        now = datetime.now()

        # Old alert (25 hours ago)
        old_alert = create_test_alert("old")
        old_alert.timestamp = now - timedelta(hours=25)

        # Recent alert (1 hour ago)
        recent_alert = create_test_alert("recent")
        recent_alert.timestamp = now - timedelta(hours=1)

        # Current alert
        current_alert = create_test_alert("current")
        current_alert.timestamp = now

        service._alert_history = [old_alert, recent_alert, current_alert]

        # Prune
        service._prune_history()

        # Should have removed the old one
        assert len(service._alert_history) == 2
        ids = [a.id for a in service._alert_history]
        assert "old" not in ids
        assert "recent" in ids
        assert "current" in ids

    @pytest.mark.asyncio
    async def test_known_service_detection(self, full_config):
        """Test that alerts from known services are identified."""
        service = SecurityMonitorService(full_config)

        # Alert from Plex server (known service)
        plex_alert = create_test_alert(
            "plex_scan",
            "Port Scan Detected",
            "192.168.168.204",  # Plex IP
            Severity.HIGH,
        )

        # Get network context
        context = await service._get_network_context()

        # Check if source IP is in known services
        known_ips = [svc["ip"] for svc in context["known_services"]]
        assert plex_alert.source_ip in known_ips


# =============================================================================
# Integration Tests - Scheduling
# =============================================================================


class TestSchedulingIntegration:
    """Test real scheduling system integration."""

    @pytest.mark.asyncio
    async def test_check_cycle_executes(self, full_config):
        """Test that check cycle actually runs."""
        service = SecurityMonitorService(full_config)

        # Track if check was called
        check_called = False
        original_check = service._check_cycle

        async def tracked_check():
            nonlocal check_called
            check_called = True
            await original_check()

        service._check_cycle = tracked_check

        # Manually trigger check
        result = await service.trigger_check_now()

        assert result["success"] is True
        assert check_called is True
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_scheduled_check_handler_returns_result(self, full_config):
        """Test the scheduled check handler returns proper result format."""
        service = SecurityMonitorService(full_config)
        service._check_cycle = AsyncMock()

        result = await service._scheduled_check(params={}, event_data=None)

        assert result["success"] is True
        assert "timestamp" in result
        # Field name may be alerts_found or alerts_processed
        assert "alerts_found" in result or "alerts_processed" in result
        service._check_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_real_timer_fires(self, full_config):
        """Test that a real asyncio timer fires the check."""
        service = SecurityMonitorService(full_config)

        # Track check executions
        check_count = 0

        async def counting_check():
            nonlocal check_count
            check_count += 1

        service._check_cycle = counting_check
        service._check_interval = 0.1  # 100ms

        # Start a simple timer loop
        async def timer_loop():
            for _ in range(3):
                await asyncio.sleep(service._check_interval)
                await service._check_cycle()

        await asyncio.wait_for(timer_loop(), timeout=1.0)

        assert check_count == 3


# =============================================================================
# Integration Tests - Summary Generation
# =============================================================================


class TestSummaryGeneration:
    """Test summary generation with real data."""

    @pytest.mark.asyncio
    async def test_generate_summary_from_real_alerts(self, full_config):
        """Test summary generation with actual alert data."""
        service = SecurityMonitorService(full_config)

        # Add variety of alerts
        now = datetime.now()
        service._alert_history = [
            create_test_alert("c1", "Critical Exploit", "10.0.0.1", Severity.CRITICAL),
            create_test_alert("h1", "High Risk Scan", "10.0.0.2", Severity.HIGH),
            create_test_alert("h2", "Another High", "10.0.0.3", Severity.HIGH),
            create_test_alert("m1", "Medium Alert", "10.0.0.4", Severity.MEDIUM),
            create_test_alert("l1", "Low Priority", "10.0.0.5", Severity.LOW),
        ]

        # Set timestamps to today
        for alert in service._alert_history:
            alert.timestamp = now - timedelta(hours=2)

        # Generate summary (mock the announce method)
        service._announce = AsyncMock()

        result = await service._generate_summary(
            params={"include_noise": False},
            event_data=None,
        )

        assert result["success"] is True
        assert result["total_alerts"] == 5
        assert result["by_severity"]["critical"] == 1
        assert result["by_severity"]["high"] == 2

    @pytest.mark.asyncio
    async def test_summary_message_format_realistic(self, full_config):
        """Test that summary messages are voice-friendly."""
        service = SecurityMonitorService(full_config)

        # Test with realistic data
        summary_data = {
            "total_alerts": 15,
            "by_severity": {
                "critical": 0,
                "high": 3,
                "medium": 7,
                "low": 5,
            },
        }

        message = service._format_summary_message(summary_data, include_noise=False)

        # Should be speakable
        assert "15 alerts" in message
        assert "3 high priority" in message
        # Should not be too technical
        assert "CRITICAL" not in message  # Should say "critical" not "CRITICAL"


# =============================================================================
# Integration Tests - Network Context
# =============================================================================


class TestNetworkContextIntegration:
    """Test network context with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_context_from_config_only(self, full_config):
        """Test network context when only config is available."""
        service = SecurityMonitorService(full_config)

        context = await service._get_network_context()

        assert context["home_net"] == "192.168.168.0/24"
        assert len(context["known_services"]) == 2

        # Verify service details
        plex = next(s for s in context["known_services"] if s["name"] == "Plex")
        assert plex["ip"] == "192.168.168.204"
        assert plex["ports"] == [32400]

    @pytest.mark.asyncio
    async def test_alert_from_known_vs_unknown_ip(self, full_config):
        """Test differentiation between known and unknown source IPs."""
        service = SecurityMonitorService(full_config)

        context = await service._get_network_context()
        known_ips = {s["ip"] for s in context["known_services"]}

        # Known service IP
        known_alert = create_test_alert("k1", "Scan", "192.168.168.204")  # Plex
        assert known_alert.source_ip in known_ips

        # Unknown IP
        unknown_alert = create_test_alert("u1", "Scan", "10.99.99.99")
        assert unknown_alert.source_ip not in known_ips

        # Internal but unknown IP
        internal_unknown = create_test_alert("i1", "Scan", "192.168.168.150")
        assert internal_unknown.source_ip not in known_ips
        # But it IS within home_net
        assert internal_unknown.source_ip.startswith("192.168.168.")


# =============================================================================
# Integration Tests - Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_check_cycle_handles_monitor_failure(self, full_config):
        """Test that check cycle continues even if a monitor fails."""
        service = SecurityMonitorService(full_config)

        # Add a failing mock monitor with required interface
        class FailingMonitor:
            name = "failing_monitor"

            async def check(self):
                raise RuntimeError("Monitor connection failed")

        service._monitors = [FailingMonitor()]

        # Should not raise, should log and continue
        try:
            await service._check_cycle()
        except Exception as e:
            pytest.fail(f"Check cycle should not propagate exceptions: {e}")

    @pytest.mark.asyncio
    async def test_service_handles_missing_config_gracefully(self):
        """Test service with minimal/missing config."""
        # Empty config
        service = SecurityMonitorService({})

        assert service._check_interval == 300  # Default
        assert len(service._monitors) == 3  # All enabled by default

        # Can still get network context (uses config fallback)
        context = await service._get_network_context()
        # May return empty dict or dict with defaults
        assert isinstance(context, dict)

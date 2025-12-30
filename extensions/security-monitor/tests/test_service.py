"""Tests for SecurityMonitorService.

These tests focus on REAL behavior:
- Monitor initialization from config
- Alert deduplication logic
- History pruning logic
- Quiet hours time calculations
- Summary message formatting

Integration tests with real scheduling/monitors are in test_service_integration.py
"""

import pytest
from datetime import datetime, timedelta

import sys
sys.path.insert(0, "/home/doug/Development/draagon-ai/extensions/security-monitor/src")

from draagon_ai_ext_security.service import SecurityMonitorService
from draagon_ai_ext_security.models import Alert, Severity, ThreatLevel, AnalysisResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Basic configuration for security monitor."""
    return {
        "check_interval_seconds": 60,
        "monitors": {
            "suricata": {"enabled": False},
            "syslog": {"enabled": False},
            "system_health": {"enabled": False},
        },
        "notifications": {
            "voice": {
                "enabled": True,
                "min_severity": "high",
                "quiet_hours": {
                    "enabled": True,
                    "start": "22:00",
                    "end": "08:00",
                },
            },
            "summary": {
                "enabled": True,
                "schedule": "09:00",
                "include_noise": False,
            },
        },
        "network": {
            "home_net": "192.168.168.0/24",
            "known_services": [
                {"name": "Plex", "ip": "192.168.168.204"},
            ],
        },
    }


@pytest.fixture
def service(basic_config):
    """Create a SecurityMonitorService instance."""
    return SecurityMonitorService(basic_config)


@pytest.fixture
def sample_alert():
    """Create a sample alert for testing."""
    return Alert(
        id="alert_001",
        source="suricata",
        signature="ET SCAN Test Alert",
        description="Test scan alert",
        source_ip="192.168.168.100",
        dest_ip="192.168.168.1",
        severity=Severity.MEDIUM,
        timestamp=datetime.now(),
        raw_data={"test": True},
    )


# =============================================================================
# Test Monitor Initialization
# =============================================================================


def test_default_check_interval():
    """Test default check interval when not configured."""
    svc = SecurityMonitorService({})
    assert svc._check_interval == 300  # Default 5 minutes


def test_custom_check_interval(basic_config):
    """Test custom check interval from config."""
    svc = SecurityMonitorService(basic_config)
    assert svc._check_interval == 60


def test_monitors_disabled_creates_none(basic_config):
    """Test that disabled monitors are not created."""
    # All monitors disabled in basic_config
    svc = SecurityMonitorService(basic_config)
    assert len(svc._monitors) == 0


def test_monitors_enabled_creates_all():
    """Test that enabled monitors are created."""
    config = {
        "monitors": {
            "suricata": {"enabled": True},
            "syslog": {"enabled": True},
            "system_health": {"enabled": True},
        },
    }
    svc = SecurityMonitorService(config)
    assert len(svc._monitors) == 3


def test_partial_monitors_enabled():
    """Test enabling only some monitors."""
    config = {
        "monitors": {
            "suricata": {"enabled": True},
            "syslog": {"enabled": False},
            "system_health": {"enabled": True},
        },
    }
    svc = SecurityMonitorService(config)
    assert len(svc._monitors) == 2


# =============================================================================
# Test Alert Deduplication
# =============================================================================


def test_deduplicate_removes_same_signature_and_source(service, sample_alert):
    """Test that alerts with same signature+source_ip are deduplicated."""
    alert1 = sample_alert
    alert2 = Alert(
        id="alert_002",
        source="suricata",
        signature="ET SCAN Test Alert",  # Same signature
        description="Duplicate alert",
        source_ip="192.168.168.100",  # Same source
        dest_ip="192.168.168.2",
        severity=Severity.MEDIUM,
        timestamp=datetime.now(),
        raw_data={},
    )

    unique = service._deduplicate([alert1, alert2])
    assert len(unique) == 1


def test_deduplicate_keeps_different_signatures(service, sample_alert):
    """Test that different signatures are kept."""
    alert1 = sample_alert
    alert2 = Alert(
        id="alert_002",
        source="suricata",
        signature="ET SCAN Different Alert",  # Different signature
        description="Different alert",
        source_ip="192.168.168.100",  # Same source
        dest_ip="192.168.168.1",
        severity=Severity.HIGH,
        timestamp=datetime.now(),
        raw_data={},
    )

    unique = service._deduplicate([alert1, alert2])
    assert len(unique) == 2


def test_deduplicate_keeps_different_sources(service, sample_alert):
    """Test that same signature from different IPs are kept."""
    alert1 = sample_alert
    alert2 = Alert(
        id="alert_002",
        source="suricata",
        signature="ET SCAN Test Alert",  # Same signature
        description="From different source",
        source_ip="192.168.168.200",  # Different source
        dest_ip="192.168.168.1",
        severity=Severity.MEDIUM,
        timestamp=datetime.now(),
        raw_data={},
    )

    unique = service._deduplicate([alert1, alert2])
    assert len(unique) == 2


# =============================================================================
# Test History Pruning
# =============================================================================


def test_prune_history_removes_old_alerts(service, sample_alert):
    """Test that alerts older than 24 hours are pruned."""
    old_alert = Alert(
        id="old_alert",
        source="suricata",
        signature="Old Alert",
        description="Old alert",
        source_ip="192.168.168.100",
        dest_ip="192.168.168.1",
        severity=Severity.LOW,
        timestamp=datetime.now() - timedelta(hours=25),
        raw_data={},
    )

    service._alert_history = [old_alert, sample_alert]
    service._prune_history()

    assert len(service._alert_history) == 1
    assert service._alert_history[0] == sample_alert


def test_prune_history_keeps_recent_alerts(service, sample_alert):
    """Test that recent alerts are kept."""
    recent_alert = Alert(
        id="recent",
        source="suricata",
        signature="Recent Alert",
        description="Recent alert",
        source_ip="192.168.168.100",
        dest_ip="192.168.168.1",
        severity=Severity.LOW,
        timestamp=datetime.now() - timedelta(hours=12),
        raw_data={},
    )

    service._alert_history = [recent_alert, sample_alert]
    service._prune_history()

    assert len(service._alert_history) == 2


# =============================================================================
# Test Quiet Hours Logic
# =============================================================================


def test_quiet_hours_disabled_returns_false(basic_config):
    """Test quiet hours check when disabled."""
    basic_config["notifications"]["voice"]["quiet_hours"]["enabled"] = False
    svc = SecurityMonitorService(basic_config)
    assert svc._is_quiet_hours() is False


def test_quiet_hours_during_night():
    """Test quiet hours during configured night time (22:00-08:00)."""
    config = {
        "notifications": {
            "voice": {
                "quiet_hours": {
                    "enabled": True,
                    "start": "22:00",
                    "end": "08:00",
                },
            },
        },
    }
    svc = SecurityMonitorService(config)

    # Test times that should be quiet
    from unittest.mock import patch

    # 23:00 - should be quiet
    with patch("draagon_ai_ext_security.service.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = datetime.strptime("23:00", "%H:%M").time()
        mock_dt.strptime = datetime.strptime
        assert svc._is_quiet_hours() is True

    # 03:00 - should be quiet (after midnight)
    with patch("draagon_ai_ext_security.service.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = datetime.strptime("03:00", "%H:%M").time()
        mock_dt.strptime = datetime.strptime
        assert svc._is_quiet_hours() is True


def test_quiet_hours_during_day():
    """Test quiet hours during daytime (should not be quiet)."""
    config = {
        "notifications": {
            "voice": {
                "quiet_hours": {
                    "enabled": True,
                    "start": "22:00",
                    "end": "08:00",
                },
            },
        },
    }
    svc = SecurityMonitorService(config)

    from unittest.mock import patch

    # 12:00 - should NOT be quiet
    with patch("draagon_ai_ext_security.service.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = datetime.strptime("12:00", "%H:%M").time()
        mock_dt.strptime = datetime.strptime
        assert svc._is_quiet_hours() is False

    # 09:00 - should NOT be quiet (just after end)
    with patch("draagon_ai_ext_security.service.datetime") as mock_dt:
        mock_dt.now.return_value.time.return_value = datetime.strptime("09:00", "%H:%M").time()
        mock_dt.strptime = datetime.strptime
        assert svc._is_quiet_hours() is False


# =============================================================================
# Test Recent Alerts Filtering
# =============================================================================


def test_get_recent_alerts_filters_by_time(service, sample_alert):
    """Test getting alerts within time window."""
    old_alert = Alert(
        id="old_alert",
        source="suricata",
        signature="Old Alert",
        description="Old alert",
        source_ip="192.168.168.100",
        dest_ip="192.168.168.1",
        severity=Severity.LOW,
        timestamp=datetime.now() - timedelta(hours=2),
        raw_data={},
    )

    service._alert_history = [old_alert, sample_alert]

    # Get last 60 minutes - should only get sample_alert
    recent = service.get_recent_alerts(minutes=60)
    assert len(recent) == 1
    assert recent[0] == sample_alert

    # Get last 180 minutes - should get both
    recent = service.get_recent_alerts(minutes=180)
    assert len(recent) == 2


# =============================================================================
# Test Summary Message Formatting
# =============================================================================


def test_format_summary_no_alerts(service):
    """Test formatting empty summary."""
    message = service._format_summary_message(
        {"total_alerts": 0, "by_severity": {}},
        include_noise=False,
    )

    assert "No security alerts" in message
    assert "All quiet" in message


def test_format_summary_with_critical(service):
    """Test formatting summary with critical alerts."""
    message = service._format_summary_message(
        {
            "total_alerts": 5,
            "by_severity": {"critical": 1, "high": 2, "medium": 2},
        },
        include_noise=False,
    )

    assert "5 alerts" in message
    assert "1 critical" in message
    assert "2 high priority" in message


def test_format_summary_with_low_priority(service):
    """Test formatting summary including low priority alerts."""
    message = service._format_summary_message(
        {
            "total_alerts": 10,
            "by_severity": {"high": 2, "medium": 3, "low": 5},
        },
        include_noise=True,
    )

    assert "10 alerts" in message
    assert "8 lower priority alerts" in message  # 3 medium + 5 low


def test_format_summary_high_only(service):
    """Test formatting summary with only high priority."""
    message = service._format_summary_message(
        {
            "total_alerts": 3,
            "by_severity": {"high": 3},
        },
        include_noise=False,
    )

    assert "3 alerts" in message
    assert "3 high priority" in message


# =============================================================================
# Test Network Context Fallback
# =============================================================================


@pytest.mark.asyncio
async def test_get_network_context_uses_config(service):
    """Test network context falls back to init config."""
    # No config service set
    service._config_service = None

    context = await service._get_network_context()

    # Should use config from init
    assert context["home_net"] == "192.168.168.0/24"
    assert len(context["known_services"]) == 1
    assert context["known_services"][0]["name"] == "Plex"

"""Security-specific tools for the agentic analyzer."""

from typing import Any
import logging

logger = logging.getLogger(__name__)

# Global reference to config service (set by extension)
_config_service = None


def set_config_service(config_service: Any) -> None:
    """Set the config service for tools to use."""
    global _config_service
    _config_service = config_service


def get_security_tools(config: dict[str, Any]) -> list:
    """Get security-specific tools.

    Args:
        config: Extension configuration.

    Returns:
        List of Tool objects.
    """
    # Import here to avoid circular imports
    from draagon_ai.tools import Tool, ToolParameter

    return [
        Tool(
            name="check_suricata_alerts",
            description="Get recent Suricata IDS alerts, optionally filtered by severity or time",
            handler=_check_suricata_alerts,
            parameters=[
                ToolParameter(
                    name="minutes",
                    type="integer",
                    description="How many minutes back to check (default: 60)",
                    required=False,
                    default=60,
                ),
                ToolParameter(
                    name="min_severity",
                    type="string",
                    description="Minimum severity level",
                    required=False,
                    enum=["critical", "high", "medium", "low"],
                    default="low",
                ),
            ],
            returns="List of recent alerts with details",
        ),
        Tool(
            name="search_threat_intel",
            description="Search for threat intelligence about an IP address, domain, or file hash",
            handler=_search_threat_intel,
            parameters=[
                ToolParameter(
                    name="indicator",
                    type="string",
                    description="IP address, domain, or hash to research",
                    required=True,
                ),
            ],
            returns="Threat intelligence findings",
        ),
        Tool(
            name="check_ip_reputation",
            description="Check if an IP address is known to be malicious using reputation services",
            handler=_check_ip_reputation,
            parameters=[
                ToolParameter(
                    name="ip",
                    type="string",
                    description="IP address to check",
                    required=True,
                ),
            ],
            returns="IP reputation data including abuse reports and risk score",
        ),
        Tool(
            name="query_pihole_logs",
            description="Check Pi-hole DNS logs for queries to a specific domain",
            handler=_query_pihole_logs,
            parameters=[
                ToolParameter(
                    name="domain",
                    type="string",
                    description="Domain to search for in DNS logs",
                    required=True,
                ),
                ToolParameter(
                    name="hours",
                    type="integer",
                    description="How many hours back to search (default: 24)",
                    required=False,
                    default=24,
                ),
            ],
            returns="DNS query history for the domain",
        ),
        Tool(
            name="recall_security_memory",
            description="Search memory for past security investigations and findings",
            handler=_recall_security_memory,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="What to search for (IP, domain, signature, etc.)",
                    required=True,
                ),
            ],
            returns="Past investigations and learnings related to the query",
        ),
        Tool(
            name="get_network_context",
            description="Get current network context including known services and recent traffic patterns",
            handler=_get_network_context,
            parameters=[],
            returns="Network context information",
        ),
        Tool(
            name="security_status",
            description="Get overall security status summary for a time period",
            handler=_security_status,
            parameters=[
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Time period to summarize",
                    required=False,
                    enum=["1h", "24h", "7d"],
                    default="24h",
                ),
                ToolParameter(
                    name="include_noise",
                    type="boolean",
                    description="Include low-priority/noise alerts in summary",
                    required=False,
                    default=False,
                ),
            ],
            returns="Security status summary",
        ),
        # =========================================================================
        # VOICE CONFIGURATION TOOLS
        # These allow users to configure security settings via voice commands
        # =========================================================================
        Tool(
            name="add_network_device",
            description="Add a known network device/service to reduce false positives",
            handler=_add_network_device,
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Name of the device (e.g., 'NAS', 'Ring Doorbell', 'Smart TV')",
                    required=True,
                ),
                ToolParameter(
                    name="ip",
                    type="string",
                    description="IP address of the device",
                    required=True,
                ),
                ToolParameter(
                    name="ports",
                    type="array",
                    description="Known ports the device uses (optional)",
                    required=False,
                ),
                ToolParameter(
                    name="protocols",
                    type="array",
                    description="Protocols the device uses like 'http', 'smb', 'mqtt' (optional)",
                    required=False,
                ),
                ToolParameter(
                    name="notes",
                    type="string",
                    description="Additional notes about the device",
                    required=False,
                ),
            ],
            returns="Confirmation that the device was added",
        ),
        Tool(
            name="remove_network_device",
            description="Remove a device from the known network devices list",
            handler=_remove_network_device,
            parameters=[
                ToolParameter(
                    name="name_or_ip",
                    type="string",
                    description="Name or IP address of the device to remove",
                    required=True,
                ),
            ],
            returns="Confirmation that the device was removed",
        ),
        Tool(
            name="list_network_devices",
            description="List all known network devices",
            handler=_list_network_devices,
            parameters=[],
            returns="List of known network devices with their details",
        ),
        Tool(
            name="set_quiet_hours",
            description="Set quiet hours when only critical alerts will be announced",
            handler=_set_quiet_hours,
            parameters=[
                ToolParameter(
                    name="start",
                    type="string",
                    description="Start time in HH:MM format (e.g., '22:00')",
                    required=True,
                ),
                ToolParameter(
                    name="end",
                    type="string",
                    description="End time in HH:MM format (e.g., '07:00')",
                    required=True,
                ),
                ToolParameter(
                    name="enabled",
                    type="boolean",
                    description="Whether quiet hours are enabled (default: true)",
                    required=False,
                    default=True,
                ),
            ],
            returns="Confirmation of quiet hours setting",
        ),
        Tool(
            name="set_alert_severity",
            description="Set the minimum severity for voice alerts",
            handler=_set_alert_severity,
            parameters=[
                ToolParameter(
                    name="min_severity",
                    type="string",
                    description="Minimum severity level for alerts",
                    required=True,
                    enum=["critical", "high", "medium", "low"],
                ),
            ],
            returns="Confirmation of severity setting",
        ),
        Tool(
            name="set_check_interval",
            description="Set how often to check for security issues",
            handler=_set_check_interval,
            parameters=[
                ToolParameter(
                    name="minutes",
                    type="integer",
                    description="Check interval in minutes (5-60)",
                    required=True,
                ),
            ],
            returns="Confirmation of check interval setting",
        ),
        Tool(
            name="get_security_settings",
            description="Get current security monitor settings",
            handler=_get_security_settings,
            parameters=[],
            returns="Current security settings including quiet hours, severity, and check interval",
        ),
    ]


# Tool handler implementations


async def _check_suricata_alerts(
    minutes: int = 60, min_severity: str = "low", **kwargs: Any
) -> dict:
    """Check Suricata for recent alerts."""
    # TODO: Implement actual Suricata log reading
    return {
        "alerts": [],
        "count": 0,
        "timeframe_minutes": minutes,
        "min_severity": min_severity,
    }


async def _search_threat_intel(indicator: str, **kwargs: Any) -> dict:
    """Search for threat intelligence."""
    # TODO: Implement SearXNG search for threat intel
    return {
        "indicator": indicator,
        "type": _detect_indicator_type(indicator),
        "findings": [],
        "risk_level": "unknown",
    }


async def _check_ip_reputation(ip: str, **kwargs: Any) -> dict:
    """Check IP reputation."""
    # TODO: Implement AbuseIPDB or similar lookup
    return {
        "ip": ip,
        "reputation_score": None,
        "abuse_reports": 0,
        "categories": [],
        "last_seen": None,
    }


async def _query_pihole_logs(domain: str, hours: int = 24, **kwargs: Any) -> dict:
    """Query Pi-hole DNS logs."""
    # TODO: Implement Pi-hole API query
    return {
        "domain": domain,
        "queries_found": 0,
        "blocked": False,
        "clients": [],
    }


async def _recall_security_memory(query: str, **kwargs: Any) -> dict:
    """Search security memory."""
    # TODO: Implement Qdrant search
    return {
        "query": query,
        "matches": [],
        "past_investigations": [],
    }


async def _get_network_context(**kwargs: Any) -> dict:
    """Get network context."""
    return {
        "home_net": "192.168.168.0/24",
        "known_services": [
            {"name": "Plex", "ip": "192.168.168.204"},
            {"name": "Home Assistant", "ip": "192.168.168.206"},
            {"name": "Pi-hole", "ip": "192.168.168.208"},
            {"name": "NAS", "ip": "192.168.168.202"},
        ],
        "external_services": [
            "Tailscale VPN",
            "Cloudflare DNS",
        ],
    }


async def _security_status(
    timeframe: str = "24h", include_noise: bool = False, **kwargs: Any
) -> dict:
    """Get security status summary."""
    # TODO: Aggregate from all monitors
    return {
        "timeframe": timeframe,
        "status": "ok",
        "total_alerts": 0,
        "by_severity": {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "noise": 0,
        },
        "action_required": False,
        "summary": "No security issues detected.",
    }


def _detect_indicator_type(indicator: str) -> str:
    """Detect the type of indicator (IP, domain, hash)."""
    import re

    # IPv4
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", indicator):
        return "ip"

    # Hash (MD5, SHA1, SHA256)
    if re.match(r"^[a-fA-F0-9]{32,64}$", indicator):
        if len(indicator) == 32:
            return "md5"
        elif len(indicator) == 40:
            return "sha1"
        else:
            return "sha256"

    # Assume domain
    return "domain"


# =============================================================================
# VOICE CONFIGURATION TOOL HANDLERS
# =============================================================================


async def _add_network_device(
    name: str,
    ip: str,
    ports: list[int] | None = None,
    protocols: list[str] | None = None,
    notes: str = "",
    **kwargs: Any,
) -> dict:
    """Add a known network device."""
    global _config_service

    if _config_service is None:
        return {
            "success": False,
            "error": "Configuration service not available",
        }

    try:
        await _config_service.add_network_service(
            name=name,
            ip=ip,
            ports=ports or [],
            protocols=protocols or [],
            notes=notes,
            user_id=kwargs.get("user_id", "system"),
        )

        return {
            "success": True,
            "message": f"Added {name} at {ip} to known devices",
            "device": {
                "name": name,
                "ip": ip,
                "ports": ports or [],
                "protocols": protocols or [],
                "notes": notes,
            },
        }
    except Exception as e:
        logger.error(f"Failed to add network device: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _remove_network_device(name_or_ip: str, **kwargs: Any) -> dict:
    """Remove a device from known devices."""
    global _config_service

    if _config_service is None:
        return {
            "success": False,
            "error": "Configuration service not available",
        }

    # Note: This would need to be implemented in the config service
    # For now, we can mark the device as removed in memory
    try:
        # Get current network context
        context = await _config_service.get_network_context("security-monitor")

        # Find and mark device for removal
        found = None
        for service in context.known_services:
            if service.name.lower() == name_or_ip.lower() or service.ip == name_or_ip:
                found = service
                break

        if found:
            # Store a "removal" record in memory
            await _config_service.set(
                extension="security-monitor",
                key=f"removed_device:{found.ip}",
                value=True,
                source="voice",
                user_id=kwargs.get("user_id", "system"),
            )
            return {
                "success": True,
                "message": f"Removed {found.name} ({found.ip}) from known devices",
            }
        else:
            return {
                "success": False,
                "error": f"Device '{name_or_ip}' not found in known devices",
            }
    except Exception as e:
        logger.error(f"Failed to remove network device: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _list_network_devices(**kwargs: Any) -> dict:
    """List all known network devices."""
    global _config_service

    if _config_service is None:
        # Fall back to static list
        return {
            "devices": [
                {"name": "Plex", "ip": "192.168.168.204", "source": "yaml"},
                {"name": "Home Assistant", "ip": "192.168.168.206", "source": "yaml"},
                {"name": "Pi-hole", "ip": "192.168.168.208", "source": "yaml"},
                {"name": "NAS", "ip": "192.168.168.202", "source": "yaml"},
            ],
            "count": 4,
            "source": "fallback",
        }

    try:
        context = await _config_service.get_network_context("security-monitor")

        devices = [
            {
                "name": s.name,
                "ip": s.ip,
                "ports": s.ports,
                "protocols": s.protocols,
                "notes": s.notes,
                "source": "learned" if s.learned else "yaml",
            }
            for s in context.known_services
        ]

        return {
            "devices": devices,
            "count": len(devices),
            "home_net": context.home_net,
        }
    except Exception as e:
        logger.error(f"Failed to list network devices: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _set_quiet_hours(
    start: str, end: str, enabled: bool = True, **kwargs: Any
) -> dict:
    """Set quiet hours for notifications."""
    global _config_service

    if _config_service is None:
        return {
            "success": False,
            "error": "Configuration service not available",
        }

    try:
        # Validate time format
        import re
        time_pattern = r"^([01]?[0-9]|2[0-3]):([0-5][0-9])$"

        if not re.match(time_pattern, start):
            return {
                "success": False,
                "error": f"Invalid start time format: {start}. Use HH:MM format.",
            }

        if not re.match(time_pattern, end):
            return {
                "success": False,
                "error": f"Invalid end time format: {end}. Use HH:MM format.",
            }

        # Store settings
        await _config_service.set(
            extension="security-monitor",
            key="notifications.voice.quiet_hours.start",
            value=start,
            source="voice",
            user_id=kwargs.get("user_id", "system"),
        )
        await _config_service.set(
            extension="security-monitor",
            key="notifications.voice.quiet_hours.end",
            value=end,
            source="voice",
            user_id=kwargs.get("user_id", "system"),
        )
        await _config_service.set(
            extension="security-monitor",
            key="notifications.voice.quiet_hours.enabled",
            value=enabled,
            source="voice",
            user_id=kwargs.get("user_id", "system"),
        )

        if enabled:
            return {
                "success": True,
                "message": f"Quiet hours set from {start} to {end}",
                "settings": {
                    "start": start,
                    "end": end,
                    "enabled": enabled,
                },
            }
        else:
            return {
                "success": True,
                "message": "Quiet hours disabled",
                "settings": {
                    "enabled": False,
                },
            }
    except Exception as e:
        logger.error(f"Failed to set quiet hours: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _set_alert_severity(min_severity: str, **kwargs: Any) -> dict:
    """Set minimum severity for voice alerts."""
    global _config_service

    valid_severities = ["critical", "high", "medium", "low"]

    if min_severity.lower() not in valid_severities:
        return {
            "success": False,
            "error": f"Invalid severity: {min_severity}. Must be one of: {', '.join(valid_severities)}",
        }

    if _config_service is None:
        return {
            "success": False,
            "error": "Configuration service not available",
        }

    try:
        await _config_service.set(
            extension="security-monitor",
            key="notifications.voice.min_severity",
            value=min_severity.lower(),
            source="voice",
            user_id=kwargs.get("user_id", "system"),
        )

        severity_descriptions = {
            "critical": "Only critical security threats will be announced",
            "high": "High and critical threats will be announced",
            "medium": "Medium, high, and critical threats will be announced",
            "low": "All security alerts will be announced",
        }

        return {
            "success": True,
            "message": severity_descriptions[min_severity.lower()],
            "settings": {
                "min_severity": min_severity.lower(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to set alert severity: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _set_check_interval(minutes: int, **kwargs: Any) -> dict:
    """Set how often to check for security issues."""
    global _config_service

    # Validate range
    if minutes < 5:
        return {
            "success": False,
            "error": "Check interval must be at least 5 minutes",
        }

    if minutes > 60:
        return {
            "success": False,
            "error": "Check interval cannot exceed 60 minutes",
        }

    if _config_service is None:
        return {
            "success": False,
            "error": "Configuration service not available",
        }

    try:
        await _config_service.set(
            extension="security-monitor",
            key="check_interval_seconds",
            value=minutes * 60,
            source="voice",
            user_id=kwargs.get("user_id", "system"),
        )

        return {
            "success": True,
            "message": f"Security checks will run every {minutes} minutes",
            "settings": {
                "check_interval_minutes": minutes,
                "check_interval_seconds": minutes * 60,
            },
        }
    except Exception as e:
        logger.error(f"Failed to set check interval: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _get_security_settings(**kwargs: Any) -> dict:
    """Get current security settings."""
    global _config_service

    if _config_service is None:
        # Return defaults
        return {
            "quiet_hours": {
                "enabled": True,
                "start": "22:00",
                "end": "08:00",
            },
            "min_severity": "high",
            "check_interval_minutes": 5,
            "source": "defaults",
        }

    try:
        # Get settings with sources
        quiet_enabled = await _config_service.get_with_source(
            "security-monitor", "notifications.voice.quiet_hours.enabled", True
        )
        quiet_start = await _config_service.get_with_source(
            "security-monitor", "notifications.voice.quiet_hours.start", "22:00"
        )
        quiet_end = await _config_service.get_with_source(
            "security-monitor", "notifications.voice.quiet_hours.end", "08:00"
        )
        min_severity = await _config_service.get_with_source(
            "security-monitor", "notifications.voice.min_severity", "high"
        )
        check_interval = await _config_service.get_with_source(
            "security-monitor", "check_interval_seconds", 300
        )

        return {
            "quiet_hours": {
                "enabled": quiet_enabled.value,
                "start": quiet_start.value,
                "end": quiet_end.value,
                "source": quiet_enabled.source,
            },
            "min_severity": {
                "value": min_severity.value,
                "source": min_severity.source,
            },
            "check_interval_minutes": {
                "value": check_interval.value // 60,
                "source": check_interval.source,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get security settings: {e}")
        return {
            "success": False,
            "error": str(e),
        }

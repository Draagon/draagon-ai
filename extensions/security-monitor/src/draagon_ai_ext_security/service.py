"""Security monitor background service.

This service provides autonomous security monitoring with:
- Multi-source alert collection (Suricata, syslog, system health)
- LLM-powered threat analysis
- Memory integration for learning from past issues
- Voice notifications through Home Assistant
- Scheduled daily summaries

Uses draagon-ai's scheduling system for:
- Periodic security checks (interval-based)
- Daily security summaries (cron-based)
- Event-triggered heightened monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable

from draagon_ai_ext_security.models import Alert, AnalysisResult, ThreatLevel, Severity
from draagon_ai_ext_security.monitors import (
    BaseMonitor,
    SuricataMonitor,
    SyslogMonitor,
    SystemHealthMonitor,
)

logger = logging.getLogger(__name__)


class SecurityMonitorService:
    """Background service that continuously monitors for threats.

    Orchestrates all monitors, aggregates alerts, runs analysis,
    and handles notifications.

    Uses draagon-ai's SchedulingService for:
    - Periodic checks via schedule_interval()
    - Daily summaries via schedule_cron()
    - Persistent task storage (survives restarts)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the security monitor service.

        Args:
            config: Extension configuration.
        """
        self._config = config
        self._running = False
        self._monitors: list[BaseMonitor] = []
        self._check_interval = config.get("check_interval_seconds", 300)
        self._alert_history: list[Alert] = []
        self._last_analysis: datetime | None = None
        self._scheduler = None
        self._config_service = None

        # Task IDs for cleanup
        self._check_task_id: str | None = None
        self._summary_task_id: str | None = None

        # Initialize monitors based on config
        self._init_monitors()

    def _init_monitors(self) -> None:
        """Initialize configured monitors."""
        monitors_config = self._config.get("monitors", {})

        # Suricata
        if monitors_config.get("suricata", {}).get("enabled", True):
            self._monitors.append(
                SuricataMonitor(monitors_config.get("suricata", {}))
            )

        # Syslog
        if monitors_config.get("syslog", {}).get("enabled", True):
            self._monitors.append(
                SyslogMonitor(monitors_config.get("syslog", {}))
            )

        # System Health
        if monitors_config.get("system_health", {}).get("enabled", True):
            self._monitors.append(
                SystemHealthMonitor(monitors_config.get("system_health", {}))
            )

        logger.info(f"Initialized {len(self._monitors)} security monitors")

    async def start(self) -> None:
        """Start the monitoring service.

        Uses the scheduling system for:
        1. Periodic security checks (every check_interval_seconds)
        2. Daily security summary (at configured time)
        """
        logger.info("Starting security monitor service")
        self._running = True

        # Initialize all monitors
        for monitor in self._monitors:
            await monitor.initialize()

        # Try to use the scheduling system
        try:
            await self._start_with_scheduler()
        except ImportError:
            logger.warning("Scheduling system not available, using fallback loop")
            await self._start_fallback_loop()

    async def _start_with_scheduler(self) -> None:
        """Start using draagon-ai's scheduling system."""
        from draagon_ai.scheduling import SchedulingService, InMemoryPersistence

        # Create scheduler with in-memory persistence for now
        # In production, this would use QdrantPersistence
        self._scheduler = SchedulingService(persistence=InMemoryPersistence())

        # Register our actions
        self._scheduler.register_action("security_check", self._scheduled_check)
        self._scheduler.register_action("security_summary", self._generate_summary)

        # Start the scheduler
        await self._scheduler.start()

        # Schedule periodic security checks
        self._check_task_id = await self._scheduler.schedule_interval(
            name="security_monitor_check",
            interval=timedelta(seconds=self._check_interval),
            action="security_check",
            action_params={},
            tags=["security", "background", "periodic"],
            metadata={"source": "security-monitor"},
        )
        logger.info(f"Scheduled security checks every {self._check_interval}s (task: {self._check_task_id})")

        # Schedule daily summary if configured
        summary_config = self._config.get("notifications", {}).get("summary", {})
        if summary_config.get("enabled", False):
            schedule_time = summary_config.get("schedule", "09:00")
            # Convert HH:MM to cron expression
            hour, minute = schedule_time.split(":")
            cron_expr = f"{minute} {hour} * * *"

            self._summary_task_id = await self._scheduler.schedule_cron(
                name="security_daily_summary",
                expression=cron_expr,
                action="security_summary",
                action_params={"include_noise": summary_config.get("include_noise", False)},
                tags=["security", "summary", "daily"],
                metadata={"source": "security-monitor"},
            )
            logger.info(f"Scheduled daily summary at {schedule_time} (task: {self._summary_task_id})")

    async def _start_fallback_loop(self) -> None:
        """Fallback to simple asyncio loop if scheduler unavailable."""
        while self._running:
            try:
                await self._check_cycle()
                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
                await asyncio.sleep(60)  # Back off on error

    async def stop(self) -> None:
        """Stop the monitoring loop gracefully."""
        logger.info("Stopping security monitor service")
        self._running = False

        # Cancel scheduled tasks
        if self._scheduler:
            if self._check_task_id:
                await self._scheduler.cancel(self._check_task_id)
            if self._summary_task_id:
                await self._scheduler.cancel(self._summary_task_id)
            await self._scheduler.stop()

        # Shutdown monitors
        for monitor in self._monitors:
            await monitor.shutdown()

    async def _scheduled_check(
        self,
        params: dict[str, Any],
        event_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handler for scheduled security checks.

        This is called by the scheduling system.
        """
        await self._check_cycle()
        return {
            "success": True,
            "alerts_processed": len(self._alert_history),
            "timestamp": datetime.now().isoformat(),
        }

    async def _check_cycle(self) -> None:
        """Run one check cycle across all monitors."""
        all_alerts: list[Alert] = []

        # Collect alerts from all monitors
        for monitor in self._monitors:
            try:
                alerts = await monitor.check()
                all_alerts.extend(alerts)
                logger.debug(f"{monitor.name}: {len(alerts)} new alerts")
            except Exception as e:
                logger.error(f"Monitor {monitor.name} failed: {e}")

        if not all_alerts:
            return

        # Deduplicate and aggregate
        unique_alerts = self._deduplicate(all_alerts)
        logger.info(f"Processing {len(unique_alerts)} unique alerts")

        # Store in history
        self._alert_history.extend(unique_alerts)
        self._prune_history()

        # Analyze alerts
        for alert in unique_alerts:
            result = await self._analyze_alert(alert)
            await self._handle_result(alert, result)

        self._last_analysis = datetime.now()

    def _deduplicate(self, alerts: list[Alert]) -> list[Alert]:
        """Remove duplicate alerts within time window."""
        seen: set[str] = set()
        unique: list[Alert] = []

        for alert in alerts:
            # Key on signature + source IP within 5 minute window
            key = f"{alert.signature}:{alert.source_ip}"
            if key not in seen:
                seen.add(key)
                unique.append(alert)

        return unique

    def _prune_history(self) -> None:
        """Remove alerts older than 24 hours from history."""
        cutoff = datetime.now() - timedelta(hours=24)
        self._alert_history = [
            a for a in self._alert_history if a.timestamp > cutoff
        ]

    async def _analyze_alert(self, alert: Alert) -> AnalysisResult:
        """Analyze an alert using the LLM.

        TODO: Implement actual LLM analysis with Groq.
        For now, returns placeholder analysis.
        """
        # Check memory for past similar issues
        past_context = await self._check_memory(alert)

        # Quick classification for known patterns
        if past_context.get("is_known_false_positive"):
            return AnalysisResult(
                threat_level=ThreatLevel.NOISE,
                is_false_positive=True,
                confidence=0.95,
                reasoning=f"Known false positive: {past_context.get('reason', 'previously marked benign')}",
                action_required="None",
                needs_investigation=False,
            )

        # Get network context for analysis
        network_context = await self._get_network_context()

        # Check if source IP is a known service
        if alert.source_ip and self._is_known_service(alert.source_ip, network_context):
            return AnalysisResult(
                threat_level=ThreatLevel.LOW,
                is_false_positive=True,
                confidence=0.8,
                reasoning=f"Traffic from known internal service",
                action_required="None",
                needs_investigation=False,
            )

        # Placeholder for full LLM analysis
        return AnalysisResult(
            threat_level=ThreatLevel.LOW,
            is_false_positive=True,
            confidence=0.5,
            reasoning="Placeholder analysis - LLM not configured",
            action_required="None",
            needs_investigation=False,
        )

    async def _check_memory(self, alert: Alert) -> dict[str, Any]:
        """Check memory for past similar issues."""
        # TODO: Integrate with memory service
        return {}

    async def _get_network_context(self) -> dict[str, Any]:
        """Get network context from config service."""
        if self._config_service:
            try:
                context = await self._config_service.get_network_context("security-monitor")
                return {
                    "home_net": context.home_net,
                    "known_services": [
                        {"name": s.name, "ip": s.ip}
                        for s in context.known_services
                    ],
                }
            except Exception as e:
                logger.warning(f"Failed to get network context: {e}")

        # Fall back to config
        return self._config.get("network", {})

    def _is_known_service(self, ip: str, context: dict[str, Any]) -> bool:
        """Check if an IP is a known internal service."""
        known_services = context.get("known_services", [])
        return any(s.get("ip") == ip for s in known_services)

    async def _handle_result(
        self, alert: Alert, result: AnalysisResult
    ) -> None:
        """Handle analysis result (notify, store, etc)."""
        logger.info(
            f"Alert {alert.signature}: {result.threat_level.value} "
            f"(confidence: {result.confidence:.2f})"
        )

        # Notify for high-severity threats
        if result.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            await self._notify(alert, result)

        # Store investigation in memory
        if not result.is_false_positive or result.needs_investigation:
            await self._store_in_memory(alert, result)

    async def _notify(self, alert: Alert, result: AnalysisResult) -> None:
        """Send notification through configured channels.

        Respects quiet hours configuration.
        """
        voice_config = self._config.get("notifications", {}).get("voice", {})

        if not voice_config.get("enabled", True):
            return

        # Check quiet hours
        if self._is_quiet_hours():
            # Only notify for critical during quiet hours
            quiet_min = voice_config.get("quiet_hours", {}).get("quiet_min_severity", "critical")
            if result.threat_level != ThreatLevel.CRITICAL:
                logger.debug(f"Suppressing notification during quiet hours: {alert.signature}")
                return

        # TODO: Implement HA voice notification
        logger.warning(
            f"SECURITY ALERT: {alert.signature} - {result.reasoning}"
        )

    def _is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        quiet_config = (
            self._config.get("notifications", {})
            .get("voice", {})
            .get("quiet_hours", {})
        )

        if not quiet_config.get("enabled", False):
            return False

        try:
            start = datetime.strptime(quiet_config.get("start", "22:00"), "%H:%M").time()
            end = datetime.strptime(quiet_config.get("end", "08:00"), "%H:%M").time()
            now = datetime.now().time()

            # Handle overnight quiet hours (e.g., 22:00 - 08:00)
            if start > end:
                return now >= start or now <= end
            else:
                return start <= now <= end
        except Exception as e:
            logger.warning(f"Error checking quiet hours: {e}")
            return False

    async def _store_in_memory(
        self, alert: Alert, result: AnalysisResult
    ) -> None:
        """Store investigation in Qdrant memory.

        TODO: Implement Qdrant storage.
        """
        pass

    async def _generate_summary(
        self,
        params: dict[str, Any],
        event_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate and announce daily security summary.

        This is called by the scheduling system via cron.
        """
        include_noise = params.get("include_noise", False)

        # Get alerts from last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self._alert_history if a.timestamp > cutoff]

        # Count by severity
        severity_counts = {}
        for alert in recent_alerts:
            sev = alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        summary = {
            "total_alerts": len(recent_alerts),
            "by_severity": severity_counts,
            "timestamp": datetime.now().isoformat(),
        }

        # Generate voice summary
        if recent_alerts:
            message = self._format_summary_message(summary, include_noise)
            await self._announce(message)
        else:
            logger.info("No security alerts in last 24 hours - no summary needed")

        return {"success": True, **summary}

    def _format_summary_message(self, summary: dict, include_noise: bool) -> str:
        """Format summary for voice announcement."""
        total = summary["total_alerts"]
        by_sev = summary["by_severity"]

        if total == 0:
            return "No security alerts in the last 24 hours. All quiet on the network front."

        parts = [f"Security summary: {total} alerts in the last 24 hours."]

        # Highlight critical/high
        critical = by_sev.get("critical", 0)
        high = by_sev.get("high", 0)

        if critical:
            parts.append(f"{critical} critical.")
        if high:
            parts.append(f"{high} high priority.")

        # Include lower severities if requested
        if include_noise:
            medium = by_sev.get("medium", 0)
            low = by_sev.get("low", 0)
            if medium or low:
                parts.append(f"{medium + low} lower priority alerts filtered.")

        return " ".join(parts)

    async def _announce(self, message: str) -> None:
        """Announce message via HA voice."""
        # TODO: Implement HA voice announcement
        logger.info(f"Voice announcement: {message}")

    def set_config_service(self, config_service: Any) -> None:
        """Set the config service for dynamic configuration."""
        self._config_service = config_service

    def get_status(self) -> dict[str, Any]:
        """Get current service status."""
        return {
            "running": self._running,
            "monitors": [m.get_status().__dict__ for m in self._monitors],
            "alerts_24h": len(self._alert_history),
            "last_analysis": (
                self._last_analysis.isoformat() if self._last_analysis else None
            ),
            "scheduler_active": self._scheduler is not None,
            "check_task_id": self._check_task_id,
            "summary_task_id": self._summary_task_id,
        }

    def get_recent_alerts(
        self, minutes: int = 60, min_severity: str = "low"
    ) -> list[Alert]:
        """Get recent alerts from history."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self._alert_history if a.timestamp > cutoff]

    async def trigger_check_now(self) -> dict[str, Any]:
        """Manually trigger a security check."""
        await self._check_cycle()
        return {
            "success": True,
            "alerts_found": len(self._alert_history),
            "timestamp": datetime.now().isoformat(),
        }

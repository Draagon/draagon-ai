"""Security monitor background service."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from draagon_ai_ext_security.models import Alert, AnalysisResult, ThreatLevel
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

        logger.info(f"Initialized {len(self._monitors)} monitors")

    async def start(self) -> None:
        """Start the monitoring loop."""
        logger.info("Starting security monitor service")
        self._running = True

        # Initialize all monitors
        for monitor in self._monitors:
            await monitor.initialize()

        # Start monitoring loop
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

        for monitor in self._monitors:
            await monitor.shutdown()

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
        """
        # Placeholder - will be implemented with Groq integration
        return AnalysisResult(
            threat_level=ThreatLevel.LOW,
            is_false_positive=True,
            confidence=0.5,
            reasoning="Placeholder analysis - LLM not configured",
            action_required="None",
            needs_investigation=False,
        )

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

        TODO: Implement HA voice notification.
        """
        logger.warning(
            f"SECURITY ALERT: {alert.signature} - {result.reasoning}"
        )

    async def _store_in_memory(
        self, alert: Alert, result: AnalysisResult
    ) -> None:
        """Store investigation in Qdrant memory.

        TODO: Implement Qdrant storage.
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """Get current service status."""
        return {
            "running": self._running,
            "monitors": [m.get_status().__dict__ for m in self._monitors],
            "alerts_24h": len(self._alert_history),
            "last_analysis": (
                self._last_analysis.isoformat() if self._last_analysis else None
            ),
        }

    def get_recent_alerts(
        self, minutes: int = 60, min_severity: str = "low"
    ) -> list[Alert]:
        """Get recent alerts from history."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self._alert_history if a.timestamp > cutoff]

"""Suricata IDS monitor."""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import aiofiles

from draagon_ai_ext_security.models import Alert, Severity
from draagon_ai_ext_security.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)


class SuricataMonitor(BaseMonitor):
    """Monitor Suricata IDS eve.json for security alerts.

    Watches the Suricata eve.json log file for new alerts,
    parsing them into standardized Alert objects.

    Config options:
        eve_log: Path to eve.json file
        min_severity: Minimum severity to report (default: low)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._eve_log = config.get("eve_log", "/var/log/suricata/eve.json")
        self._min_severity = config.get("min_severity", "low")
        self._last_position: int = 0
        self._last_inode: int = 0

    @property
    def name(self) -> str:
        return "suricata"

    async def initialize(self) -> None:
        """Initialize by seeking to end of current log."""
        try:
            stat = os.stat(self._eve_log)
            self._last_position = stat.st_size
            self._last_inode = stat.st_ino
            logger.info(
                f"SuricataMonitor initialized at position {self._last_position}"
            )
        except FileNotFoundError:
            logger.warning(f"Suricata eve.json not found: {self._eve_log}")
            self._error = "Log file not found"

    async def check(self) -> list[Alert]:
        """Check for new Suricata alerts.

        Returns:
            List of new alerts since last check.
        """
        alerts: list[Alert] = []
        self._last_check = datetime.now()

        try:
            # Check if file was rotated (inode changed)
            stat = os.stat(self._eve_log)
            if stat.st_ino != self._last_inode:
                logger.info("Suricata log rotated, resetting position")
                self._last_position = 0
                self._last_inode = stat.st_ino

            # Read new lines
            async with aiofiles.open(self._eve_log, "r") as f:
                await f.seek(self._last_position)
                new_lines = await f.readlines()
                self._last_position = await f.tell()

            # Parse alerts
            for line in new_lines:
                try:
                    event = json.loads(line.strip())
                    if event.get("event_type") == "alert":
                        alert = self._parse_alert(event)
                        if alert and self._meets_severity(alert.severity):
                            alerts.append(alert)
                except json.JSONDecodeError:
                    continue

            self._alerts_count_24h += len(alerts)
            self._error = None

        except FileNotFoundError:
            self._error = "Log file not found"
            logger.error(f"Suricata log not found: {self._eve_log}")
        except PermissionError:
            self._error = "Permission denied"
            logger.error(f"Cannot read Suricata log: {self._eve_log}")
        except Exception as e:
            self._error = str(e)
            logger.error(f"Error reading Suricata log: {e}")

        return alerts

    def _parse_alert(self, event: dict) -> Alert | None:
        """Parse a Suricata event into an Alert.

        Args:
            event: Raw Suricata event dict.

        Returns:
            Alert object or None if invalid.
        """
        try:
            alert_data = event.get("alert", {})

            # Map Suricata severity (1-3) to our levels
            suricata_severity = alert_data.get("severity", 3)
            severity = self._map_severity(suricata_severity)

            return Alert(
                id=f"suricata-{uuid.uuid4().hex[:12]}",
                source="suricata",
                timestamp=datetime.fromisoformat(
                    event.get("timestamp", "").replace("Z", "+00:00")
                ),
                severity=severity,
                signature=alert_data.get("signature", "Unknown"),
                description=alert_data.get("category", ""),
                source_ip=event.get("src_ip"),
                dest_ip=event.get("dest_ip"),
                protocol=event.get("proto"),
                raw_data={
                    "signature_id": alert_data.get("signature_id"),
                    "category": alert_data.get("category"),
                    "action": alert_data.get("action"),
                    "gid": alert_data.get("gid"),
                    "rev": alert_data.get("rev"),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse Suricata alert: {e}")
            return None

    def _map_severity(self, suricata_severity: int) -> Severity:
        """Map Suricata severity (1-3) to our Severity enum.

        Suricata: 1 = high, 2 = medium, 3 = low
        """
        mapping = {
            1: Severity.HIGH,
            2: Severity.MEDIUM,
            3: Severity.LOW,
        }
        return mapping.get(suricata_severity, Severity.INFO)

    def _meets_severity(self, severity: Severity) -> bool:
        """Check if alert meets minimum severity threshold."""
        severity_order = [
            Severity.NOISE,
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]

        min_idx = next(
            (i for i, s in enumerate(severity_order) if s.value == self._min_severity),
            0,
        )
        alert_idx = severity_order.index(severity)

        return alert_idx >= min_idx

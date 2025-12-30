"""Syslog monitor for security events."""

import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any

import aiofiles

from draagon_ai_ext_security.models import Alert, Severity
from draagon_ai_ext_security.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)


# Patterns for security-relevant log entries
SECURITY_PATTERNS = [
    {
        "pattern": re.compile(r"Failed password for .* from ([\d.]+)"),
        "severity": Severity.MEDIUM,
        "signature": "SSH Failed Password",
        "extract_ip": 1,
    },
    {
        "pattern": re.compile(r"authentication failure.*rhost=([\d.]+)"),
        "severity": Severity.MEDIUM,
        "signature": "Authentication Failure",
        "extract_ip": 1,
    },
    {
        "pattern": re.compile(r"BREAK-IN ATTEMPT"),
        "severity": Severity.HIGH,
        "signature": "Break-in Attempt Detected",
        "extract_ip": None,
    },
    {
        "pattern": re.compile(r"Accepted publickey for (\w+) from ([\d.]+)"),
        "severity": Severity.INFO,
        "signature": "SSH Login Success",
        "extract_ip": 2,
    },
    {
        "pattern": re.compile(r"sudo:.*COMMAND=(.*)"),
        "severity": Severity.INFO,
        "signature": "Sudo Command Executed",
        "extract_ip": None,
    },
]


class SyslogMonitor(BaseMonitor):
    """Monitor system logs for security events.

    Watches auth.log and syslog for security-relevant entries
    like failed logins, authentication failures, etc.

    Config options:
        paths: List of log files to monitor
        patterns: Additional regex patterns to match
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._paths = config.get("paths", ["/var/log/auth.log"])
        self._file_positions: dict[str, int] = {}
        self._file_inodes: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "syslog"

    async def initialize(self) -> None:
        """Initialize by seeking to end of each log file."""
        for path in self._paths:
            try:
                stat = os.stat(path)
                self._file_positions[path] = stat.st_size
                self._file_inodes[path] = stat.st_ino
            except FileNotFoundError:
                logger.warning(f"Log file not found: {path}")

    async def check(self) -> list[Alert]:
        """Check for new security events in logs.

        Returns:
            List of new alerts.
        """
        alerts: list[Alert] = []
        self._last_check = datetime.now()

        for path in self._paths:
            try:
                new_alerts = await self._check_file(path)
                alerts.extend(new_alerts)
            except Exception as e:
                logger.error(f"Error checking {path}: {e}")

        self._alerts_count_24h += len(alerts)
        return alerts

    async def _check_file(self, path: str) -> list[Alert]:
        """Check a single log file for new entries."""
        alerts: list[Alert] = []

        try:
            stat = os.stat(path)

            # Check for rotation
            if stat.st_ino != self._file_inodes.get(path, 0):
                self._file_positions[path] = 0
                self._file_inodes[path] = stat.st_ino

            async with aiofiles.open(path, "r") as f:
                await f.seek(self._file_positions.get(path, 0))
                new_lines = await f.readlines()
                self._file_positions[path] = await f.tell()

            for line in new_lines:
                alert = self._parse_line(line.strip(), path)
                if alert:
                    alerts.append(alert)

        except FileNotFoundError:
            pass
        except PermissionError:
            self._error = f"Permission denied: {path}"

        return alerts

    def _parse_line(self, line: str, source_file: str) -> Alert | None:
        """Parse a log line for security events."""
        for pattern_info in SECURITY_PATTERNS:
            match = pattern_info["pattern"].search(line)
            if match:
                source_ip = None
                if pattern_info["extract_ip"]:
                    try:
                        source_ip = match.group(pattern_info["extract_ip"])
                    except IndexError:
                        pass

                return Alert(
                    id=f"syslog-{uuid.uuid4().hex[:12]}",
                    source="syslog",
                    timestamp=datetime.now(),  # TODO: parse from log line
                    severity=pattern_info["severity"],
                    signature=pattern_info["signature"],
                    description=line[:200],
                    source_ip=source_ip,
                    raw_data={
                        "file": source_file,
                        "line": line,
                    },
                )

        return None

"""System health monitor for hardware/OS alerts."""

import logging
import subprocess
import uuid
from datetime import datetime
from typing import Any

from draagon_ai_ext_security.models import Alert, Severity
from draagon_ai_ext_security.monitors.base import BaseMonitor

logger = logging.getLogger(__name__)


class SystemHealthMonitor(BaseMonitor):
    """Monitor system health (NVMe temp, disk usage, etc).

    Config options:
        nvme_warning_celsius: Temperature warning threshold (default: 60)
        nvme_critical_celsius: Temperature critical threshold (default: 70)
        disk_warning_percent: Disk usage warning threshold (default: 85)
        disk_critical_percent: Disk usage critical threshold (default: 95)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._nvme_warning = config.get("nvme_warning_celsius", 60)
        self._nvme_critical = config.get("nvme_critical_celsius", 70)
        self._disk_warning = config.get("disk_warning_percent", 85)
        self._disk_critical = config.get("disk_critical_percent", 95)

    @property
    def name(self) -> str:
        return "system_health"

    async def check(self) -> list[Alert]:
        """Check system health metrics.

        Returns:
            List of alerts for any concerning metrics.
        """
        alerts: list[Alert] = []
        self._last_check = datetime.now()

        # Check NVMe temperatures
        nvme_alerts = self._check_nvme_temps()
        alerts.extend(nvme_alerts)

        # Check disk usage
        disk_alerts = self._check_disk_usage()
        alerts.extend(disk_alerts)

        self._alerts_count_24h += len(alerts)
        return alerts

    def _check_nvme_temps(self) -> list[Alert]:
        """Check NVMe drive temperatures."""
        alerts: list[Alert] = []

        try:
            # Check both nvme0 and nvme1
            for device in ["/dev/nvme0", "/dev/nvme1"]:
                result = subprocess.run(
                    ["sudo", "smartctl", "-A", device],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    continue

                for line in result.stdout.split("\n"):
                    if "Temperature:" in line and "Celsius" in line:
                        # Parse "Temperature:                        35 Celsius"
                        parts = line.split()
                        try:
                            temp_idx = parts.index("Celsius") - 1
                            temp = int(parts[temp_idx])

                            if temp >= self._nvme_critical:
                                alerts.append(
                                    self._create_temp_alert(
                                        device, temp, Severity.CRITICAL
                                    )
                                )
                            elif temp >= self._nvme_warning:
                                alerts.append(
                                    self._create_temp_alert(
                                        device, temp, Severity.HIGH
                                    )
                                )
                        except (ValueError, IndexError):
                            pass
                        break

        except subprocess.TimeoutExpired:
            logger.warning("smartctl timed out")
        except FileNotFoundError:
            logger.warning("smartctl not found")
        except Exception as e:
            logger.error(f"Error checking NVMe temps: {e}")

        return alerts

    def _create_temp_alert(
        self, device: str, temp: int, severity: Severity
    ) -> Alert:
        """Create a temperature alert."""
        return Alert(
            id=f"temp-{uuid.uuid4().hex[:12]}",
            source="system_health",
            timestamp=datetime.now(),
            severity=severity,
            signature=f"NVMe Temperature {'Critical' if severity == Severity.CRITICAL else 'Warning'}",
            description=f"{device} is at {temp}°C",
            raw_data={
                "device": device,
                "temperature_celsius": temp,
                "threshold_warning": self._nvme_warning,
                "threshold_critical": self._nvme_critical,
            },
        )

    def _check_disk_usage(self) -> list[Alert]:
        """Check disk usage on key filesystems."""
        alerts: list[Alert] = []

        try:
            result = subprocess.run(
                ["df", "-h", "/", "/home"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            for line in result.stdout.split("\n")[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 5:
                    usage_str = parts[4].rstrip("%")
                    try:
                        usage = int(usage_str)
                        mount = parts[5] if len(parts) > 5 else parts[0]

                        if usage >= self._disk_critical:
                            alerts.append(
                                self._create_disk_alert(
                                    mount, usage, Severity.CRITICAL
                                )
                            )
                        elif usage >= self._disk_warning:
                            alerts.append(
                                self._create_disk_alert(mount, usage, Severity.HIGH)
                            )
                    except ValueError:
                        pass

        except Exception as e:
            logger.error(f"Error checking disk usage: {e}")

        return alerts

    def _create_disk_alert(
        self, mount: str, usage: int, severity: Severity
    ) -> Alert:
        """Create a disk usage alert."""
        return Alert(
            id=f"disk-{uuid.uuid4().hex[:12]}",
            source="system_health",
            timestamp=datetime.now(),
            severity=severity,
            signature=f"Disk Usage {'Critical' if severity == Severity.CRITICAL else 'Warning'}",
            description=f"{mount} is at {usage}% capacity",
            raw_data={
                "mount": mount,
                "usage_percent": usage,
                "threshold_warning": self._disk_warning,
                "threshold_critical": self._disk_critical,
            },
        )

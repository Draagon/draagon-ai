"""Base class for security monitors."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from draagon_ai_ext_security.models import Alert, MonitorStatus


class BaseMonitor(ABC):
    """Abstract base class for all security monitors.

    Each monitor is responsible for:
    1. Watching a specific data source (log file, API, etc.)
    2. Detecting security-relevant events
    3. Converting them to standardized Alert objects
    4. Tracking its own health status
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize monitor with configuration.

        Args:
            config: Monitor-specific configuration.
        """
        self._config = config
        self._last_check: datetime | None = None
        self._alerts_count_24h: int = 0
        self._error: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the monitor name."""
        pass

    @abstractmethod
    async def check(self) -> list[Alert]:
        """Check for new alerts since last check.

        Returns:
            List of new alerts detected.
        """
        pass

    def get_status(self) -> MonitorStatus:
        """Get current monitor health status.

        Returns:
            MonitorStatus with current health info.
        """
        return MonitorStatus(
            name=self.name,
            healthy=self._error is None,
            last_check=self._last_check,
            alerts_count_24h=self._alerts_count_24h,
            error=self._error,
        )

    async def initialize(self) -> None:
        """Initialize the monitor (optional override)."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the monitor (optional override)."""
        pass

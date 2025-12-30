"""Data models for the security monitor extension."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    NOISE = "noise"


class ThreatLevel(str, Enum):
    """Threat level classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOISE = "noise"


@dataclass
class Alert:
    """A security alert from any monitor source."""

    id: str
    source: str  # "suricata", "syslog", "system_health"
    timestamp: datetime
    severity: Severity
    signature: str
    description: str
    source_ip: str | None = None
    dest_ip: str | None = None
    protocol: str | None = None
    raw_data: dict = field(default_factory=dict)
    enrichments: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "signature": self.signature,
            "description": self.description,
            "source_ip": self.source_ip,
            "dest_ip": self.dest_ip,
            "protocol": self.protocol,
            "enrichments": self.enrichments,
        }


@dataclass
class AnalysisResult:
    """Result of analyzing an alert or batch of alerts."""

    threat_level: ThreatLevel
    is_false_positive: bool
    confidence: float
    reasoning: str
    action_required: str
    needs_investigation: bool = False
    investigation_plan: list[str] = field(default_factory=list)
    voice_message: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResult":
        """Create from dictionary (LLM output)."""
        return cls(
            threat_level=ThreatLevel(data.get("threat_level", "noise")),
            is_false_positive=data.get("is_false_positive", False),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            action_required=data.get("action_required", "None"),
            needs_investigation=data.get("needs_investigation", False),
            investigation_plan=data.get("investigation_plan", []),
        )


@dataclass
class Investigation:
    """Result of an agentic investigation."""

    alert_ids: list[str]
    threat_level: ThreatLevel
    summary: str
    detailed_findings: str
    tools_used: list[str]
    memory_references: list[str]
    action_required: str
    voice_message: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class MonitorStatus:
    """Health status of a monitor."""

    name: str
    healthy: bool
    last_check: datetime | None
    alerts_count_24h: int
    error: str | None = None

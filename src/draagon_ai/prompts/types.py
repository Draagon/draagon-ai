"""Prompt types and data structures.

This module defines the core types for the prompt management system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PromptStatus(str, Enum):
    """Status of a prompt version."""

    ACTIVE = "active"  # Currently in use
    SHADOW = "shadow"  # Being A/B tested
    ARCHIVED = "archived"  # No longer in use
    CANDIDATE = "candidate"  # Proposed by evolution


class PromptDomain(str, Enum):
    """Prompt domains for categorization.

    Each domain represents a functional area of the assistant.
    """

    ROUTING = "routing"  # Intent classification, fast routing
    DECISION = "decision"  # Core action selection
    HOME_ASSISTANT = "home_assistant"  # Smart home control
    CALENDAR = "calendar"  # Event creation/parsing
    COMMANDS = "commands"  # Shell command generation
    SYNTHESIS = "synthesis"  # Response generation
    MEMORY = "memory"  # Episode summaries, graph queries
    QUALITY = "quality"  # Pre-response reflection
    CONVERSATION = "conversation"  # Mode detection
    LEARNING = "learning"  # Skill/fact extraction
    COGNITIVE = "cognitive"  # Beliefs, opinions, personality
    CUSTOM = "custom"  # User-defined prompts


@dataclass
class PromptMetadata:
    """Metadata for a prompt.

    Attributes:
        domain: Functional domain of the prompt
        version: Semantic version string
        created_at: When this version was created
        created_by: Who/what created it (user, evolution, etc.)
        parent_version: Previous version this evolved from
        fitness_score: Performance score from evaluation
        usage_count: Number of times this prompt was used
        success_rate: Rate of successful outcomes
        avg_latency_ms: Average response latency
        tags: Categorization tags
        notes: Human-readable notes
    """

    domain: PromptDomain
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    parent_version: str | None = None
    fitness_score: float | None = None
    usage_count: int = 0
    success_rate: float | None = None
    avg_latency_ms: float | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "domain": self.domain.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "parent_version": self.parent_version,
            "fitness_score": self.fitness_score,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptMetadata":
        """Create from dictionary."""
        return cls(
            domain=PromptDomain(data.get("domain", "custom")),
            version=data.get("version", "1.0.0"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            created_by=data.get("created_by", "system"),
            parent_version=data.get("parent_version"),
            fitness_score=data.get("fitness_score"),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate"),
            avg_latency_ms=data.get("avg_latency_ms"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


@dataclass
class PromptVersion:
    """A specific version of a prompt.

    Attributes:
        version: Semantic version string
        content: The prompt text
        status: Current status (active, shadow, etc.)
        metadata: Version metadata
    """

    version: str
    content: str
    status: PromptStatus = PromptStatus.ACTIVE
    metadata: PromptMetadata | None = None


@dataclass
class Prompt:
    """A named prompt with version history.

    Attributes:
        name: Unique prompt identifier (e.g., "DECISION_PROMPT")
        domain: Functional domain
        description: Human-readable description
        current_version: Currently active version string
        versions: Version history
        variables: Template variables used in the prompt
    """

    name: str
    domain: PromptDomain
    description: str = ""
    current_version: str = "1.0.0"
    versions: dict[str, PromptVersion] = field(default_factory=dict)
    variables: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Get current version's content."""
        if self.current_version in self.versions:
            return self.versions[self.current_version].content
        return ""

    def get_version(self, version: str) -> PromptVersion | None:
        """Get a specific version."""
        return self.versions.get(version)

    def add_version(
        self,
        content: str,
        version: str,
        status: PromptStatus = PromptStatus.CANDIDATE,
        metadata: PromptMetadata | None = None,
    ) -> PromptVersion:
        """Add a new version.

        Args:
            content: Prompt text
            version: Version string
            status: Initial status
            metadata: Version metadata

        Returns:
            The created version
        """
        prompt_version = PromptVersion(
            version=version,
            content=content,
            status=status,
            metadata=metadata
            or PromptMetadata(
                domain=self.domain,
                version=version,
                parent_version=self.current_version,
            ),
        )
        self.versions[version] = prompt_version
        return prompt_version

    def activate_version(self, version: str) -> bool:
        """Activate a specific version.

        Args:
            version: Version to activate

        Returns:
            True if successful
        """
        if version not in self.versions:
            return False

        # Archive current active
        if self.current_version in self.versions:
            self.versions[self.current_version].status = PromptStatus.ARCHIVED

        # Activate new
        self.versions[version].status = PromptStatus.ACTIVE
        self.current_version = version
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "domain": self.domain.value,
            "description": self.description,
            "current_version": self.current_version,
            "variables": self.variables,
            "versions": {
                v: {
                    "version": pv.version,
                    "content": pv.content,
                    "status": pv.status.value,
                    "metadata": pv.metadata.to_dict() if pv.metadata else None,
                }
                for v, pv in self.versions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prompt":
        """Create from dictionary."""
        prompt = cls(
            name=data["name"],
            domain=PromptDomain(data.get("domain", "custom")),
            description=data.get("description", ""),
            current_version=data.get("current_version", "1.0.0"),
            variables=data.get("variables", []),
        )

        for v, pv_data in data.get("versions", {}).items():
            prompt.versions[v] = PromptVersion(
                version=pv_data["version"],
                content=pv_data["content"],
                status=PromptStatus(pv_data.get("status", "active")),
                metadata=PromptMetadata.from_dict(pv_data["metadata"])
                if pv_data.get("metadata")
                else None,
            )

        return prompt

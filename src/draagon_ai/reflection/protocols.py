"""Protocol definitions for the reflection module."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import DiscoveredIssue


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used in reflection.

    The reflection module uses LLMs for:
    - Quality evaluation
    - Issue discovery
    - Issue classification
    - Improvement planning
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific options

        Returns:
            Dict with 'content' key containing the response
        """
        ...

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a chat completion and parse as JSON.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific options

        Returns:
            Dict with 'parsed' key containing parsed JSON, or None if parse failed
        """
        ...


@runtime_checkable
class IssueStore(Protocol):
    """Protocol for storing and retrieving discovered issues.

    Implementations might use a vector database, file storage, or API.
    """

    async def store_issue(self, issue: DiscoveredIssue) -> None:
        """Store a discovered issue.

        Args:
            issue: The issue to store
        """
        ...

    async def get_issue(self, issue_id: str) -> DiscoveredIssue | None:
        """Get an issue by ID.

        Args:
            issue_id: The issue ID

        Returns:
            The issue if found, None otherwise
        """
        ...

    async def list_issues(
        self,
        status: str | None = None,
        severity: str | None = None,
        issue_type: str | None = None,
        limit: int = 100,
    ) -> list[DiscoveredIssue]:
        """List issues with optional filtering.

        Args:
            status: Filter by status
            severity: Filter by severity
            issue_type: Filter by type
            limit: Maximum number to return

        Returns:
            List of matching issues
        """
        ...

    async def update_issue(self, issue: DiscoveredIssue) -> None:
        """Update an existing issue.

        Args:
            issue: The updated issue
        """
        ...

    async def delete_issue(self, issue_id: str) -> bool:
        """Delete an issue.

        Args:
            issue_id: The issue ID

        Returns:
            True if deleted, False if not found
        """
        ...


class InMemoryIssueStore:
    """Simple in-memory implementation of IssueStore for testing."""

    def __init__(self):
        self.issues: dict[str, DiscoveredIssue] = {}

    async def store_issue(self, issue: DiscoveredIssue) -> None:
        self.issues[issue.id] = issue

    async def get_issue(self, issue_id: str) -> DiscoveredIssue | None:
        return self.issues.get(issue_id)

    async def list_issues(
        self,
        status: str | None = None,
        severity: str | None = None,
        issue_type: str | None = None,
        limit: int = 100,
    ) -> list[DiscoveredIssue]:
        result = list(self.issues.values())

        if status:
            result = [i for i in result if i.status.value == status]
        if severity:
            result = [i for i in result if i.severity.value == severity]
        if issue_type:
            result = [i for i in result if i.issue_type.value == issue_type]

        return result[:limit]

    async def update_issue(self, issue: DiscoveredIssue) -> None:
        self.issues[issue.id] = issue

    async def delete_issue(self, issue_id: str) -> bool:
        if issue_id in self.issues:
            del self.issues[issue_id]
            return True
        return False

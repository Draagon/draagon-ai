"""Tests for ReflectionService."""

import pytest
import json

from draagon_ai.reflection import (
    ReflectionService,
    ReflectionConfig,
    ReflectionResult,
    DiscoveredIssue,
    IssueType,
    IssueSeverity,
    IssueStatus,
)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: dict | str | None = None):
        if isinstance(response, dict):
            self.response = response
        elif isinstance(response, str):
            self.response = {"content": response, "parsed": None}
        else:
            self.response = {
                "parsed": {
                    "quality_score": 4,
                    "no_issues": True,
                    "user_sentiment": "positive",
                    "issues": [],
                }
            }
        self.chat_calls = []

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        self.chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        if "parsed" in self.response:
            return {"content": json.dumps(self.response["parsed"])}
        return self.response

    async def chat_json(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        self.chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response


class TestReflectionService:
    """Tests for ReflectionService class."""

    @pytest.mark.asyncio
    async def test_reflect_basic(self):
        """Test basic reflection returns ReflectionResult."""
        llm = MockLLMProvider()
        service = ReflectionService(llm)

        result = await service.reflect(
            interaction_id="test123",
            query="What time is it?",
            response="It's 3:00 PM.",
            action="answer",
            tool_calls=["get_time"],
        )

        assert isinstance(result, ReflectionResult)
        assert result.interaction_id == "test123"
        assert result.quality_score == 4
        assert result.no_issues is True

    @pytest.mark.asyncio
    async def test_reflect_with_issues(self):
        """Test reflection that finds issues."""
        llm = MockLLMProvider({
            "parsed": {
                "quality_score": 2,
                "no_issues": False,
                "user_sentiment": "negative",
                "issues": [
                    {
                        "description": "Used wrong tool",
                        "prompt_blamed": "DECISION_PROMPT",
                        "root_cause": "Misunderstood the query",
                        "severity": "HIGH",
                        "suggested_fix": "Add instruction for this case",
                    }
                ],
            }
        })
        service = ReflectionService(llm)

        result = await service.reflect(
            interaction_id="test456",
            query="Turn off the lights",
            response="I searched the web for lights.",
            action="web_search",
            tool_calls=["web_search"],
        )

        assert result.quality_score == 2
        assert result.no_issues is False
        assert len(result.issues) == 1
        assert result.issues[0].description == "Used wrong tool"
        assert result.issues[0].severity == IssueSeverity.HIGH

    @pytest.mark.asyncio
    async def test_reflect_no_llm(self):
        """Test reflection without LLM returns safe default."""
        service = ReflectionService(llm_provider=None)

        result = await service.reflect(
            interaction_id="test789",
            query="test",
            response="test",
            action="answer",
        )

        assert result.quality_score == 3
        assert result.no_issues is True
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_reflect_with_conversation_history(self):
        """Test reflection includes conversation history."""
        llm = MockLLMProvider()
        service = ReflectionService(llm)

        history = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "What's the weather?", "assistant": "It's sunny."},
        ]

        await service.reflect(
            interaction_id="test",
            query="And tomorrow?",
            response="Tomorrow will be rainy.",
            action="answer",
            conversation_history=history,
        )

        # Check that history was included in prompt
        prompt = llm.chat_calls[0]["messages"][1]["content"]
        assert "Hello" in prompt
        assert "sunny" in prompt

    @pytest.mark.asyncio
    async def test_issue_classification(self):
        """Test that issues get classified."""
        llm = MockLLMProvider({
            "parsed": {
                "quality_score": 2,
                "no_issues": False,
                "user_sentiment": "neutral",
                "issues": [
                    {
                        "description": "Wrong tool used",
                        "root_cause": "Missing instruction",
                        "severity": "MEDIUM",
                        "suggested_fix": "Add instruction",
                    }
                ],
            }
        })

        # Override classification response for second call
        original_chat_json = llm.chat_json

        async def chat_json_with_classification(*args, **kwargs):
            result = await original_chat_json(*args, **kwargs)
            # Second call is classification
            if len(llm.chat_calls) > 1:
                return {
                    "parsed": {
                        "issue_type": "PROMPT",
                        "fixable_by_agent": True,
                        "reasoning": "Can be fixed by changing prompt",
                        "suggested_approach": "Update DECISION_PROMPT",
                        "files_involved": ["prompts.py"],
                    }
                }
            return result

        llm.chat_json = chat_json_with_classification

        service = ReflectionService(llm)
        result = await service.reflect(
            interaction_id="test",
            query="test",
            response="test",
            action="answer",
        )

        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.issue_type == IssueType.PROMPT
        assert issue.fixable_by_agent is True


class TestReflectionConfig:
    """Tests for ReflectionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReflectionConfig()

        assert config.max_history_turns == 5
        assert config.max_response_length == 1000
        assert config.use_fast_model is True
        assert len(config.prompts_that_can_be_blamed) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReflectionConfig(
            max_history_turns=10,
            max_response_length=500,
            use_fast_model=False,
        )

        assert config.max_history_turns == 10
        assert config.max_response_length == 500
        assert config.use_fast_model is False


class TestReflectionResult:
    """Tests for ReflectionResult."""

    def test_is_good(self):
        """Test is_good property."""
        result = ReflectionResult(
            quality_score=4,
            issues=[],
            no_issues=True,
            interaction_id="test",
        )
        assert result.is_good is True

        result.quality_score = 3
        assert result.is_good is False

        result.quality_score = 5
        result.no_issues = False
        assert result.is_good is False

    def test_to_dict(self):
        """Test serialization."""
        result = ReflectionResult(
            quality_score=4,
            issues=[],
            no_issues=True,
            interaction_id="test123",
            user_sentiment="positive",
        )

        data = result.to_dict()

        assert data["quality_score"] == 4
        assert data["no_issues"] is True
        assert data["interaction_id"] == "test123"
        assert data["user_sentiment"] == "positive"
        assert "reflected_at" in data


class TestDiscoveredIssue:
    """Tests for DiscoveredIssue."""

    def test_to_dict(self):
        """Test serialization."""
        from datetime import datetime

        issue = DiscoveredIssue(
            id="issue123",
            timestamp=datetime.now(),
            interaction_id="int456",
            query="test query",
            response="test response",
            action_taken="answer",
            tool_calls=["get_time"],
            description="Test issue",
            root_cause="Test cause",
            severity=IssueSeverity.MEDIUM,
            suggested_fix="Test fix",
        )

        data = issue.to_dict()

        assert data["id"] == "issue123"
        assert data["description"] == "Test issue"
        assert data["severity"] == "medium"
        assert data["record_type"] == "discovered_issue"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "issue123",
            "timestamp": "2025-01-01T00:00:00",
            "interaction_id": "int456",
            "query": "test query",
            "response": "test response",
            "action_taken": "answer",
            "tool_calls": ["get_time"],
            "description": "Test issue",
            "root_cause": "Test cause",
            "severity": "medium",
            "suggested_fix": "Test fix",
        }

        issue = DiscoveredIssue.from_dict(data)

        assert issue.id == "issue123"
        assert issue.description == "Test issue"
        assert issue.severity == IssueSeverity.MEDIUM


class TestIssueModels:
    """Tests for issue-related models."""

    def test_issue_severity_values(self):
        """Test IssueSeverity enum values."""
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.HIGH.value == "high"
        assert IssueSeverity.MEDIUM.value == "medium"
        assert IssueSeverity.LOW.value == "low"

    def test_issue_type_values(self):
        """Test IssueType enum values."""
        assert IssueType.PROMPT.value == "prompt"
        assert IssueType.KNOWLEDGE.value == "knowledge"
        assert IssueType.TOOL.value == "tool"
        assert IssueType.BUG.value == "bug"
        assert IssueType.FEATURE.value == "feature"
        assert IssueType.EXTERNAL.value == "external"

    def test_issue_status_values(self):
        """Test IssueStatus enum values."""
        assert IssueStatus.OPEN.value == "open"
        assert IssueStatus.PENDING_FIX.value == "pending"
        assert IssueStatus.FIXED.value == "fixed"
        assert IssueStatus.WONT_FIX.value == "wont_fix"
        assert IssueStatus.DUPLICATE.value == "duplicate"


class TestInMemoryIssueStore:
    """Tests for InMemoryIssueStore."""

    @pytest.mark.asyncio
    async def test_store_and_get(self):
        """Test storing and retrieving issues."""
        from datetime import datetime
        from draagon_ai.reflection.protocols import InMemoryIssueStore

        store = InMemoryIssueStore()

        issue = DiscoveredIssue(
            id="test123",
            timestamp=datetime.now(),
            interaction_id="int456",
            query="test",
            response="test",
            action_taken="answer",
            tool_calls=[],
            description="Test",
            root_cause="Test",
            severity=IssueSeverity.MEDIUM,
            suggested_fix="Test",
        )

        await store.store_issue(issue)
        retrieved = await store.get_issue("test123")

        assert retrieved is not None
        assert retrieved.id == "test123"

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        """Test listing issues with filters."""
        from datetime import datetime
        from draagon_ai.reflection.protocols import InMemoryIssueStore

        store = InMemoryIssueStore()

        # Add issues with different severities
        for i, sev in enumerate([IssueSeverity.HIGH, IssueSeverity.LOW, IssueSeverity.HIGH]):
            issue = DiscoveredIssue(
                id=f"issue{i}",
                timestamp=datetime.now(),
                interaction_id="test",
                query="test",
                response="test",
                action_taken="answer",
                tool_calls=[],
                description="Test",
                root_cause="Test",
                severity=sev,
                suggested_fix="Test",
            )
            await store.store_issue(issue)

        # List all
        all_issues = await store.list_issues()
        assert len(all_issues) == 3

        # Filter by severity
        high_issues = await store.list_issues(severity="high")
        assert len(high_issues) == 2

        low_issues = await store.list_issues(severity="low")
        assert len(low_issues) == 1

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting issues."""
        from datetime import datetime
        from draagon_ai.reflection.protocols import InMemoryIssueStore

        store = InMemoryIssueStore()

        issue = DiscoveredIssue(
            id="todelete",
            timestamp=datetime.now(),
            interaction_id="test",
            query="test",
            response="test",
            action_taken="answer",
            tool_calls=[],
            description="Test",
            root_cause="Test",
            severity=IssueSeverity.LOW,
            suggested_fix="Test",
        )

        await store.store_issue(issue)
        assert await store.get_issue("todelete") is not None

        result = await store.delete_issue("todelete")
        assert result is True
        assert await store.get_issue("todelete") is None

        # Try to delete non-existent
        result = await store.delete_issue("nonexistent")
        assert result is False

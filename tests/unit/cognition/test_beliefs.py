"""Unit tests for Belief Reconciliation Service.

These tests verify the belief reconciliation logic works correctly
with the draagon-ai LLM and memory protocols.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from draagon_ai.llm import MockLLM, ChatResponse
from draagon_ai.memory import Memory, MemoryType, MemoryScope, SearchResult
from draagon_ai.cognition.beliefs import (
    BeliefReconciliationService,
    ReconciliationResult,
    BeliefType,
    ObservationScope,
    UserObservation,
    AgentBelief,
)


class TestBeliefReconciliation:
    """Tests for BeliefReconciliationService."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM service that returns JSON responses."""
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory service."""
        memory = MagicMock()
        memory.store = AsyncMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def service(self, mock_llm, mock_memory):
        """Create service with mocked dependencies."""
        return BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_create_observation_extracts_content(self, service, mock_llm):
        """Test that observation creation extracts content via LLM."""
        json_response = json.dumps({
            "content": "The WiFi password is hunter2",
            "scope": "household",
            "confidence_expressed": 0.9,
            "entities": ["wifi", "password"],
            "is_correction": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        obs = await service.create_observation(
            statement="Hey, the WiFi password is hunter2",
            user_id="doug",
            context="casual conversation",
        )

        assert obs.content == "The WiFi password is hunter2"
        assert obs.source_user_id == "doug"
        assert obs.scope == ObservationScope.HOUSEHOLD
        assert obs.confidence_expressed == 0.9

    @pytest.mark.asyncio
    async def test_create_observation_handles_correction(self, service, mock_llm, mock_memory):
        """Test that corrections trigger immediate reconciliation."""
        obs_response = json.dumps({
            "content": "The WiFi password is newpass123",
            "scope": "household",
            "confidence_expressed": 0.95,
            "entities": ["wifi", "password"],
            "is_correction": True,
            "corrects_topic": "WiFi password",
        })
        belief_response = json.dumps({
            "content": "The WiFi password is newpass123",
            "belief_type": "household_fact",
            "confidence": 0.95,
            "has_conflict": False,
            "needs_clarification": False,
            "clarification_priority": 0.0,
            "reasoning": "User correction",
        })
        mock_llm.responses = [
            f"```json\n{obs_response}\n```",
            f"```json\n{belief_response}\n```"
        ]

        mock_memory.search.return_value = []

        obs = await service.create_observation(
            statement="Actually, the WiFi password changed to newpass123",
            user_id="doug",
        )

        assert obs.content == "The WiFi password is newpass123"
        assert mock_memory.search.called

    @pytest.mark.asyncio
    async def test_reconcile_observations_forms_belief(self, service, mock_llm, mock_memory):
        """Test that reconciling observations creates a belief."""
        json_response = json.dumps({
            "content": "The household has 6 cats",
            "belief_type": "household_fact",
            "confidence": 0.9,
            "has_conflict": False,
            "needs_clarification": False,
            "clarification_priority": 0.0,
            "reasoning": "Multiple observations agree",
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        observations = [
            UserObservation(
                observation_id="obs1",
                content="We have 6 cats",
                source_user_id="doug",
                scope=ObservationScope.HOUSEHOLD,
                timestamp=datetime.now(),
                confidence_expressed=0.9,
            ),
            UserObservation(
                observation_id="obs2",
                content="There are 6 cats in the house",
                source_user_id="lisa",
                scope=ObservationScope.HOUSEHOLD,
                timestamp=datetime.now(),
                confidence_expressed=0.85,
            ),
        ]

        result = await service.reconcile_observations(observations)

        assert result is not None
        assert result.belief.content == "The household has 6 cats"
        assert result.belief.belief_type == BeliefType.HOUSEHOLD_FACT
        assert result.belief.confidence == 0.9
        assert result.action == "created"

    @pytest.mark.asyncio
    async def test_reconcile_detects_conflict(self, service, mock_llm, mock_memory):
        """Test that conflicting observations are detected."""
        json_response = json.dumps({
            "content": "The household has 5-6 cats (conflicting reports)",
            "belief_type": "household_fact",
            "confidence": 0.6,
            "has_conflict": True,
            "conflict_description": "Doug says 6, Lisa says 5",
            "needs_clarification": True,
            "clarification_priority": 0.8,
            "reasoning": "Observations disagree on count",
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        observations = [
            UserObservation(
                observation_id="obs1",
                content="We have 6 cats",
                source_user_id="doug",
                scope=ObservationScope.HOUSEHOLD,
                timestamp=datetime.now(),
                confidence_expressed=0.9,
            ),
        ]

        existing = [
            {
                "content": "We have 5 cats",
                "source_user_id": "lisa",
                "metadata": {
                    "record_type": "user_observation",
                    "observation_id": "obs0",
                    "confidence_expressed": 0.85,
                },
            }
        ]

        result = await service.reconcile_observations(observations, existing)

        assert result is not None
        assert result.action == "conflict_detected"
        assert result.conflict_info is not None
        assert result.belief.needs_clarification is True

    @pytest.mark.asyncio
    async def test_get_belief_returns_stored_belief(self, service, mock_memory):
        """Test retrieving an existing belief."""
        mock_memory_obj = Memory(
            id="mem123",
            content="Doug's birthday is March 15",
            memory_type=MemoryType.BELIEF,
            scope=MemoryScope.AGENT,
            user_id="testbot",
            agent_id="testbot",
            importance=0.95,
            confidence=0.95,
        )
        mock_memory_obj.record_type = "agent_belief"
        mock_memory_obj.belief_id = "belief123"
        mock_memory_obj.belief_type = "household_fact"
        mock_memory_obj.verified = True
        mock_memory_obj.supporting_observations = ["obs1"]
        mock_memory_obj.conflicting_observations = []
        mock_memory_obj.needs_clarification = False
        mock_memory_obj.clarification_priority = 0.0

        mock_result = SearchResult(memory=mock_memory_obj, score=0.9)
        mock_memory.search.return_value = [mock_result]

        belief = await service.get_belief("Doug birthday")

        assert belief is not None
        assert belief.content == "Doug's birthday is March 15"
        assert belief.belief_type == BeliefType.HOUSEHOLD_FACT
        assert belief.verified is True


class TestObservationScopes:
    """Tests for observation scope handling."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.store = AsyncMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def service(self, mock_llm, mock_memory):
        return BeliefReconciliationService(
            llm=mock_llm,
            memory=mock_memory,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_private_scope(self, service, mock_llm):
        """Test private scope observation."""
        json_response = json.dumps({
            "content": "Personal password",
            "scope": "private",
            "confidence_expressed": 0.9,
            "entities": ["password"],
            "is_correction": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        obs = await service.create_observation(
            statement="My password is secret",
            user_id="doug",
        )
        assert obs.scope == ObservationScope.PRIVATE

    @pytest.mark.asyncio
    async def test_personal_scope(self, service, mock_llm):
        """Test personal scope observation."""
        json_response = json.dumps({
            "content": "Doug's birthday",
            "scope": "personal",
            "confidence_expressed": 0.9,
            "entities": ["birthday"],
            "is_correction": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        obs = await service.create_observation(
            statement="My birthday is in March",
            user_id="doug",
        )
        assert obs.scope == ObservationScope.PERSONAL

    @pytest.mark.asyncio
    async def test_household_scope(self, service, mock_llm):
        """Test household scope observation."""
        json_response = json.dumps({
            "content": "WiFi password",
            "scope": "household",
            "confidence_expressed": 0.9,
            "entities": ["wifi"],
            "is_correction": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        obs = await service.create_observation(
            statement="The WiFi password is abc123",
            user_id="doug",
        )
        assert obs.scope == ObservationScope.HOUSEHOLD

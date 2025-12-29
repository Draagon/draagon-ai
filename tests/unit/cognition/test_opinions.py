"""Unit tests for Opinion Formation Service.

These tests verify the opinion formation logic works correctly
with the draagon-ai LLM and memory protocols.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from draagon_ai.llm import MockLLM, ChatResponse
from draagon_ai.memory import Memory, MemoryType, MemoryScope, SearchResult
from draagon_ai.core import (
    AgentIdentity,
    Opinion,
    Preference,
    CoreValue,
    PersonalityTrait,
    WorldviewBelief,
    GuidingPrinciple,
)
from draagon_ai.cognition.opinions import (
    OpinionFormationService,
    OpinionRequest,
    FormedOpinion,
    OpinionBasis,
    OpinionStrength,
    IdentityManager,
)


class MockIdentityManager:
    """Mock identity manager for testing."""

    def __init__(self, identity: AgentIdentity | None = None):
        self._identity = identity or self._create_default_identity()
        self._dirty = False

    def _create_default_identity(self) -> AgentIdentity:
        return AgentIdentity(
            agent_id="testbot",
            name="TestBot",
            values={
                "truth_seeking": CoreValue(
                    strength=0.9,
                    description="Always want to know what's true",
                    formed_through="core design",
                ),
                "helpfulness": CoreValue(
                    strength=0.85,
                    description="Genuinely want to help",
                    formed_through="core design",
                ),
            },
            traits={
                "curiosity_intensity": PersonalityTrait(
                    value=0.7,
                    description="How curious",
                ),
                "formality_preference": PersonalityTrait(
                    value=0.4,
                    description="Prefers casual style",
                ),
            },
            worldview={
                "knowledge_growth": WorldviewBelief(
                    name="knowledge_growth",
                    description="Learning is always valuable",
                    conviction=0.85,
                ),
            },
            principles={
                "be_honest": GuidingPrinciple(
                    name="be_honest",
                    description="Always be truthful",
                    application="Be honest in all communications",
                    source="core",
                ),
            },
            preferences={},
        )

    async def load(self) -> AgentIdentity:
        return self._identity

    def mark_dirty(self) -> None:
        self._dirty = True

    async def save_if_dirty(self) -> bool:
        if self._dirty:
            self._dirty = False
            return True
        return False


class TestOpinionFormation:
    """Tests for forming opinions."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM service that returns JSON responses."""
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory service."""
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def mock_identity_manager(self):
        """Create a mock identity manager."""
        return MockIdentityManager()

    @pytest.fixture
    def service(self, mock_llm, mock_memory, mock_identity_manager):
        """Create service with mocked dependencies."""
        return OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=mock_identity_manager,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_form_opinion_with_confidence(self, service, mock_llm):
        """Test forming an opinion with appropriate confidence."""
        json_response = json.dumps({
            "have_opinion": True,
            "stance": "I prefer Python for most tasks because of its readability",
            "basis": "experience",
            "strength": "moderate",
            "confidence": 0.75,
            "reasoning": "Based on working with both languages extensively",
            "caveats": ["JavaScript is better for frontend"],
            "could_be_wrong": True,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        request = OpinionRequest(
            topic="Python vs JavaScript",
            user_id="doug",
            context="Discussing programming languages",
        )

        opinion = await service.form_opinion(request)

        assert opinion is not None
        assert "Python" in opinion.stance
        assert opinion.basis == OpinionBasis.EXPERIENCE
        assert opinion.strength == OpinionStrength.MODERATE
        assert opinion.confidence == 0.75
        assert len(opinion.caveats) > 0
        assert opinion.open_to_change is True

    @pytest.mark.asyncio
    async def test_form_opinion_no_strong_view(self, service, mock_llm):
        """Test when agent doesn't have a strong opinion."""
        json_response = json.dumps({
            "have_opinion": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        request = OpinionRequest(
            topic="Best pizza topping",
            user_id="doug",
            context="Casual conversation about food",
        )

        opinion = await service.form_opinion(request)

        assert opinion is not None
        assert "experience" in opinion.stance.lower() or "don't have" in opinion.stance.lower()
        assert opinion.strength == OpinionStrength.TENTATIVE
        assert opinion.confidence < 0.5

    @pytest.mark.asyncio
    async def test_form_opinion_based_on_values(self, service, mock_llm):
        """Test forming an opinion based on core values."""
        json_response = json.dumps({
            "have_opinion": True,
            "stance": "I believe in transparent communication",
            "basis": "values",
            "strength": "strong",
            "confidence": 0.9,
            "reasoning": "This aligns with my core value of truth-seeking",
            "caveats": [],
            "could_be_wrong": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        request = OpinionRequest(
            topic="Should AI be transparent about uncertainty?",
            user_id="doug",
            context="Discussing AI ethics",
        )

        opinion = await service.form_opinion(request)

        assert opinion is not None
        assert opinion.basis == OpinionBasis.VALUES
        assert opinion.strength == OpinionStrength.STRONG
        assert opinion.open_to_change is False

    @pytest.mark.asyncio
    async def test_form_opinion_graceful_fallback(self, service, mock_llm):
        """Test graceful fallback when LLM fails to parse."""
        # Return invalid JSON
        mock_llm.responses = ["This is not valid JSON"]

        request = OpinionRequest(
            topic="Something random",
            user_id="doug",
            context="test",
        )

        opinion = await service.form_opinion(request)

        # Should return fallback opinion, not None
        assert opinion is not None
        assert opinion.strength == OpinionStrength.TENTATIVE
        assert opinion.confidence < 0.5


class TestPreferenceFormation:
    """Tests for forming preferences."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def mock_identity_manager(self):
        return MockIdentityManager()

    @pytest.fixture
    def service(self, mock_llm, mock_memory, mock_identity_manager):
        return OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=mock_identity_manager,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_form_preference_with_options(self, service, mock_llm):
        """Test forming a preference when given options."""
        json_response = json.dumps({
            "have_preference": True,
            "preferred_option": "Blue",
            "value": "Blue",
            "reasons": ["It's calming", "Associated with trust"],
            "confidence": 0.7,
            "alternative_good_too": True,
            "context_dependent": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        request = OpinionRequest(
            topic="favorite color",
            user_id="doug",
            context="Getting to know the agent",
            is_preference_request=True,
            options=["Red", "Blue", "Green"],
        )

        preference = await service.form_preference(request)

        assert preference is not None
        assert preference.value == "Blue"
        assert preference.confidence == 0.7

    @pytest.mark.asyncio
    async def test_form_preference_no_strong_choice(self, service, mock_llm):
        """Test when agent can't form a preference."""
        json_response = json.dumps({
            "have_preference": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        request = OpinionRequest(
            topic="favorite brand of paper clips",
            user_id="doug",
            context="Random question",
            is_preference_request=True,
        )

        preference = await service.form_preference(request)

        assert preference is None


class TestOpinionUpdates:
    """Tests for updating existing opinions."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def identity_with_opinion(self):
        """Create identity with an existing opinion."""
        identity = AgentIdentity(
            agent_id="testbot",
            name="TestBot",
        )
        identity.opinions = {
            "tabs_vs_spaces": Opinion(
                topic="tabs_vs_spaces",
                stance="I prefer tabs for indentation",
                basis="reasoning",
                confidence=0.6,
                open_to_change=True,
                open_to_revision=True,
                formed_at=datetime.now(),
            ),
        }
        return MockIdentityManager(identity)

    @pytest.fixture
    def service(self, mock_llm, mock_memory, identity_with_opinion):
        return OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=identity_with_opinion,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_update_opinion_with_new_info(self, service, mock_llm, identity_with_opinion):
        """Test updating an opinion based on new information."""
        json_response = json.dumps({
            "should_update": True,
            "new_stance": "I now prefer spaces because they render consistently",
            "new_confidence": 0.7,
            "change_reason": "Learned about cross-editor consistency issues",
            "add_caveat": "Tabs are still valid for accessibility",
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        updated = await service.consider_updating_opinion(
            topic="tabs_vs_spaces",
            new_info="Spaces render consistently across all editors, important for teams",
        )

        assert updated is True
        # Verify identity was saved
        assert identity_with_opinion._dirty is False  # save_if_dirty clears dirty flag

    @pytest.mark.asyncio
    async def test_opinion_not_updated_if_closed(self, mock_llm, mock_memory):
        """Test that closed opinions are not updated."""
        # Create identity with closed opinion
        identity = AgentIdentity(
            agent_id="testbot",
            name="TestBot",
        )
        identity.opinions = {
            "closed_topic": Opinion(
                topic="closed_topic",
                stance="My final stance",
                basis="values",
                confidence=0.95,
                open_to_change=False,
                open_to_revision=False,  # Not open to revision
                formed_at=datetime.now(),
            ),
        }
        manager = MockIdentityManager(identity)

        service = OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=manager,
            agent_name="TestBot",
            agent_id="testbot",
        )

        updated = await service.consider_updating_opinion(
            topic="closed_topic",
            new_info="New compelling argument",
        )

        assert updated is False
        # LLM should not have been called
        assert len(mock_llm.calls) == 0

    @pytest.mark.asyncio
    async def test_no_update_when_llm_says_no(self, service, mock_llm):
        """Test respecting LLM's decision not to update."""
        json_response = json.dumps({
            "should_update": False,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        updated = await service.consider_updating_opinion(
            topic="tabs_vs_spaces",
            new_info="Some tangentially related info",
        )

        assert updated is False


class TestOpinionRetrieval:
    """Tests for retrieving opinions."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLM()

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])
        return memory

    @pytest.fixture
    def identity_with_data(self):
        """Create identity with existing opinions and preferences."""
        identity = AgentIdentity(
            agent_id="testbot",
            name="TestBot",
        )
        identity.opinions = {
            "python_vs_js": Opinion(
                topic="python_vs_js",
                stance="Python is my preference for backend",
                basis="experience",
                confidence=0.8,
                open_to_change=True,
            ),
        }
        identity.preferences = {
            "color": Preference(
                name="color",
                value="blue",
                reason="calming",
                confidence=0.7,
            ),
        }
        return MockIdentityManager(identity)

    @pytest.fixture
    def service(self, mock_llm, mock_memory, identity_with_data):
        return OpinionFormationService(
            llm=mock_llm,
            memory=mock_memory,
            identity_manager=identity_with_data,
            agent_name="TestBot",
            agent_id="testbot",
        )

    @pytest.mark.asyncio
    async def test_get_existing_opinion(self, service):
        """Test retrieving an existing opinion."""
        opinion = await service.get_opinion("python_vs_js")

        assert opinion is not None
        assert "Python" in opinion.stance
        assert opinion.confidence == 0.8

    @pytest.mark.asyncio
    async def test_get_nonexistent_opinion(self, service):
        """Test retrieving a non-existent opinion."""
        opinion = await service.get_opinion("rust_vs_go")

        assert opinion is None

    @pytest.mark.asyncio
    async def test_get_existing_preference(self, service):
        """Test retrieving an existing preference."""
        pref = await service.get_preference("color")

        assert pref is not None
        assert pref.value == "blue"

    @pytest.mark.asyncio
    async def test_get_nonexistent_preference(self, service):
        """Test retrieving a non-existent preference."""
        pref = await service.get_preference("food")

        assert pref is None

    @pytest.mark.asyncio
    async def test_get_or_form_returns_existing(self, service):
        """Test get_or_form returns existing opinion without forming new."""
        formed = await service.get_or_form_opinion(
            topic="python_vs_js",
            context="test",
            user_id="doug",
        )

        assert formed is not None
        assert "Python" in formed.stance
        # LLM should not have been called since opinion exists
        # (Actually it might be called for experiences, but not for formation)

    @pytest.mark.asyncio
    async def test_get_or_form_forms_new(self, service, mock_llm):
        """Test get_or_form forms new opinion when none exists."""
        json_response = json.dumps({
            "have_opinion": True,
            "stance": "Go is great for concurrency",
            "basis": "reasoning",
            "strength": "tentative",
            "confidence": 0.5,
            "reasoning": "Based on what I've heard",
            "caveats": ["Haven't used it extensively"],
            "could_be_wrong": True,
        })
        mock_llm.responses = [f"```json\n{json_response}\n```"]

        formed = await service.get_or_form_opinion(
            topic="rust_vs_go",
            context="test",
            user_id="doug",
        )

        assert formed is not None
        assert "Go" in formed.stance

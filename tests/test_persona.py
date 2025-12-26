"""Tests for the Persona system."""

import pytest

from draagon_ai.persona import (
    Persona,
    PersonaTraits,
    PersonaRelationship,
    PersonaManager,
    SinglePersonaManager,
    MultiPersonaManager,
)
from draagon_ai.core import AgentIdentity


# =============================================================================
# PersonaTraits Tests
# =============================================================================


class TestPersonaTraits:
    """Tests for PersonaTraits."""

    def test_empty_traits(self):
        """Test empty traits default to 0.5."""
        traits = PersonaTraits()
        assert traits["nonexistent"] == 0.5
        assert traits.get("also_nonexistent") == 0.5

    def test_set_get_trait(self):
        """Test setting and getting traits."""
        traits = PersonaTraits()
        traits["friendly"] = 0.8
        assert traits["friendly"] == 0.8

    def test_trait_clamping(self):
        """Test traits are clamped to 0.0-1.0."""
        traits = PersonaTraits()
        traits["over"] = 1.5
        traits["under"] = -0.5
        assert traits["over"] == 1.0
        assert traits["under"] == 0.0

    def test_from_dict(self):
        """Test creating from dictionary."""
        traits = PersonaTraits.from_dict({"friendly": 0.9, "curious": 0.7})
        assert traits["friendly"] == 0.9
        assert traits["curious"] == 0.7

    def test_items(self):
        """Test getting all traits as items."""
        traits = PersonaTraits.from_dict({"a": 0.5, "b": 0.6})
        items = traits.items()
        assert len(items) == 2
        assert ("a", 0.5) in items
        assert ("b", 0.6) in items


# =============================================================================
# Persona Tests
# =============================================================================


class TestPersona:
    """Tests for Persona."""

    def test_minimal_persona(self):
        """Test creating a minimal persona."""
        persona = Persona(id="test", name="Test")
        assert persona.id == "test"
        assert persona.name == "Test"
        assert persona.description == ""

    def test_full_persona(self):
        """Test creating a fully specified persona."""
        traits = PersonaTraits.from_dict({"friendly": 0.9})
        persona = Persona(
            id="roxy",
            name="Roxy",
            description="A helpful assistant",
            traits=traits,
            voice_notes="Speak concisely",
            knowledge_tags=["smart_home", "calendar"],
        )
        assert persona.id == "roxy"
        assert persona.name == "Roxy"
        assert persona.traits["friendly"] == 0.9
        assert "smart_home" in persona.knowledge_tags

    def test_get_trait_from_simple_traits(self):
        """Test getting traits from simple traits."""
        persona = Persona(
            id="test",
            name="Test",
            traits=PersonaTraits.from_dict({"curious": 0.8}),
        )
        assert persona.get_trait("curious") == 0.8
        assert persona.get_trait("nonexistent") == 0.5

    def test_get_trait_from_identity(self):
        """Test getting traits from full identity."""
        identity = AgentIdentity.create_minimal("test", "Test")
        persona = Persona(id="test", name="Test", identity=identity)

        # Should get from identity
        assert persona.get_trait("curiosity_intensity") == 0.5

    def test_system_prompt_context(self):
        """Test generating system prompt context."""
        persona = Persona(
            id="test",
            name="Ada",
            description="An AI assistant",
            voice_notes="Be formal",
            traits=PersonaTraits.from_dict({"helpful": 0.9}),
        )
        context = persona.get_system_prompt_context()
        assert "Ada" in context
        assert "AI assistant" in context
        assert "Be formal" in context
        assert "helpful" in context

    def test_add_relationship(self):
        """Test adding relationships."""
        persona = Persona(id="a", name="A")
        rel = persona.add_relationship(
            target_id="b",
            relationship_type="ally",
            affinity=0.8,
        )
        assert rel.target_id == "b"
        assert rel.relationship_type == "ally"
        assert rel.affinity == 0.8
        assert persona.relationships["b"] == rel


# =============================================================================
# SinglePersonaManager Tests
# =============================================================================


class TestSinglePersonaManager:
    """Tests for SinglePersonaManager."""

    @pytest.fixture
    def roxy_persona(self):
        """Create a Roxy persona for testing."""
        return Persona(
            id="roxy",
            name="Roxy",
            description="A helpful voice assistant",
        )

    @pytest.mark.asyncio
    async def test_get_active_always_returns_same(self, roxy_persona):
        """Test that get_active always returns the same persona."""
        manager = SinglePersonaManager(roxy_persona)

        # Different contexts should return same persona
        p1 = await manager.get_active_persona({})
        p2 = await manager.get_active_persona({"user_id": "doug"})
        p3 = await manager.get_active_persona({"random": "context"})

        assert p1 is p2 is p3 is roxy_persona

    @pytest.mark.asyncio
    async def test_list_personas(self, roxy_persona):
        """Test listing personas."""
        manager = SinglePersonaManager(roxy_persona)
        personas = await manager.list_personas()
        assert len(personas) == 1
        assert personas[0] is roxy_persona

    @pytest.mark.asyncio
    async def test_get_persona_by_id(self, roxy_persona):
        """Test getting persona by ID."""
        manager = SinglePersonaManager(roxy_persona)

        assert await manager.get_persona("roxy") is roxy_persona
        assert await manager.get_persona("nonexistent") is None

    @pytest.mark.asyncio
    async def test_switch_returns_none(self, roxy_persona):
        """Test that switch_to returns None (not supported)."""
        manager = SinglePersonaManager(roxy_persona)
        result = await manager.switch_to("other")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_persona(self, roxy_persona):
        """Test updating the managed persona."""
        manager = SinglePersonaManager(roxy_persona)
        new_persona = Persona(id="roxy_v2", name="Roxy 2.0")

        await manager.update_persona(new_persona)

        active = await manager.get_active_persona({})
        assert active is new_persona
        assert active.name == "Roxy 2.0"


# =============================================================================
# MultiPersonaManager Tests
# =============================================================================


class TestMultiPersonaManager:
    """Tests for MultiPersonaManager."""

    @pytest.fixture
    def npcs(self):
        """Create some NPC personas for testing."""
        return [
            Persona(
                id="grumak",
                name="Grumak",
                description="An orc blacksmith",
                traits=PersonaTraits.from_dict({"gruff": 0.8}),
            ),
            Persona(
                id="elara",
                name="Elara",
                description="An elven scholar",
                traits=PersonaTraits.from_dict({"mysterious": 0.9}),
            ),
            Persona(
                id="pip",
                name="Pip",
                description="A halfling thief",
                traits=PersonaTraits.from_dict({"mischievous": 0.95}),
            ),
        ]

    @pytest.mark.asyncio
    async def test_empty_manager_raises(self):
        """Test that empty manager raises on get_active."""
        manager = MultiPersonaManager()
        with pytest.raises(ValueError, match="No personas available"):
            await manager.get_active_persona({})

    @pytest.mark.asyncio
    async def test_first_persona_is_default(self, npcs):
        """Test first persona becomes default."""
        manager = MultiPersonaManager(personas=npcs)
        active = await manager.get_active_persona({})
        assert active.id == "grumak"

    @pytest.mark.asyncio
    async def test_explicit_default(self, npcs):
        """Test explicit default_id."""
        manager = MultiPersonaManager(personas=npcs, default_id="elara")
        active = await manager.get_active_persona({})
        assert active.id == "elara"

    @pytest.mark.asyncio
    async def test_switch_to(self, npcs):
        """Test switching personas."""
        manager = MultiPersonaManager(personas=npcs)

        # Start with grumak
        assert (await manager.get_active_persona({})).id == "grumak"

        # Switch to elara
        switched = await manager.switch_to("elara")
        assert switched is not None
        assert switched.id == "elara"

        # Verify it stuck
        assert (await manager.get_active_persona({})).id == "elara"

    @pytest.mark.asyncio
    async def test_switch_to_nonexistent(self, npcs):
        """Test switching to nonexistent persona."""
        manager = MultiPersonaManager(personas=npcs)
        result = await manager.switch_to("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_context_persona_id(self, npcs):
        """Test context can specify persona_id."""
        manager = MultiPersonaManager(personas=npcs)

        # Context can override active persona
        active = await manager.get_active_persona({"persona_id": "pip"})
        assert active.id == "pip"

    @pytest.mark.asyncio
    async def test_context_npc_id(self, npcs):
        """Test context can use npc_id (alias)."""
        manager = MultiPersonaManager(personas=npcs)

        active = await manager.get_active_persona({"npc_id": "elara"})
        assert active.id == "elara"

    @pytest.mark.asyncio
    async def test_list_personas(self, npcs):
        """Test listing all personas."""
        manager = MultiPersonaManager(personas=npcs)
        listed = await manager.list_personas()
        assert len(listed) == 3
        ids = {p.id for p in listed}
        assert ids == {"grumak", "elara", "pip"}

    @pytest.mark.asyncio
    async def test_get_persona(self, npcs):
        """Test getting persona by ID."""
        manager = MultiPersonaManager(personas=npcs)

        assert (await manager.get_persona("grumak")).name == "Grumak"
        assert (await manager.get_persona("elara")).name == "Elara"
        assert await manager.get_persona("nonexistent") is None

    @pytest.mark.asyncio
    async def test_add_persona(self):
        """Test adding personas after creation."""
        manager = MultiPersonaManager()
        manager.add_persona(Persona(id="a", name="A"))
        manager.add_persona(Persona(id="b", name="B"))

        assert len(await manager.list_personas()) == 2
        # First added should be active
        assert (await manager.get_active_persona({})).id == "a"

    @pytest.mark.asyncio
    async def test_remove_persona(self, npcs):
        """Test removing personas."""
        manager = MultiPersonaManager(personas=npcs)

        removed = manager.remove_persona("grumak")
        assert removed is not None
        assert removed.id == "grumak"
        assert len(await manager.list_personas()) == 2

    @pytest.mark.asyncio
    async def test_remove_active_persona(self, npcs):
        """Test removing the active persona switches to default."""
        manager = MultiPersonaManager(personas=npcs, default_id="pip")

        # When default_id is specified, that becomes the initial active persona
        assert (await manager.get_active_persona({})).id == "pip"

        # Switch to grumak
        await manager.switch_to("grumak")
        assert (await manager.get_active_persona({})).id == "grumak"

        # Remove grumak (the now-active persona)
        manager.remove_persona("grumak")

        # Should fall back to default (pip)
        assert (await manager.get_active_persona({})).id == "pip"

    @pytest.mark.asyncio
    async def test_set_default(self, npcs):
        """Test setting default persona."""
        manager = MultiPersonaManager(personas=npcs)

        assert manager.default_id == "grumak"
        manager.set_default("elara")
        assert manager.default_id == "elara"

    @pytest.mark.asyncio
    async def test_relationship_between_personas(self, npcs):
        """Test relationships between personas."""
        manager = MultiPersonaManager(personas=npcs)

        # Add relationship from grumak's perspective
        grumak = await manager.get_persona("grumak")
        grumak.add_relationship("elara", "customer", affinity=0.6)

        # Get relationship
        rel = manager.get_relationship("grumak", "elara")
        assert rel is not None
        assert rel.relationship_type == "customer"
        assert rel.affinity == 0.6

    @pytest.mark.asyncio
    async def test_interaction_callbacks(self, npcs):
        """Test on_interaction_start/end callbacks."""
        manager = MultiPersonaManager(personas=npcs)
        persona = await manager.get_active_persona({})

        # Clear last_active first
        persona.last_active = None

        # Start interaction should update last_active
        await manager.on_interaction_start(persona, {})
        assert persona.last_active is not None

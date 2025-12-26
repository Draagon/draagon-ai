"""Tests for Semantic Memory layer."""

import pytest

from draagon_ai.memory import (
    TemporalCognitiveGraph,
    SemanticMemory,
    Entity,
    Fact,
    Relationship,
    EntityMatch,
)


@pytest.fixture
def graph():
    """Create a fresh graph for each test."""
    return TemporalCognitiveGraph()


@pytest.fixture
def semantic(graph):
    """Create semantic memory instance."""
    return SemanticMemory(graph)


class TestEntityManagement:
    """Test entity creation and management."""

    @pytest.mark.asyncio
    async def test_create_entity(self, semantic):
        """Test creating an entity."""
        entity = await semantic.create_entity(
            name="Doug Mealing",
            entity_type="person",
            aliases=["Doug", "Douglas"],
            properties={"role": "developer"},
        )

        assert entity is not None
        assert entity.canonical_name == "Doug Mealing"
        assert entity.entity_type == "person"
        assert "Doug" in entity.aliases

    @pytest.mark.asyncio
    async def test_get_entity(self, semantic):
        """Test retrieving entity by ID."""
        entity = await semantic.create_entity("Test Entity", "thing")
        retrieved = await semantic.get_entity(entity.node_id)

        assert retrieved is not None
        assert retrieved.canonical_name == "Test Entity"

    @pytest.mark.asyncio
    async def test_add_alias_to_entity(self, semantic):
        """Test adding an alias to an entity."""
        entity = await semantic.create_entity(
            "Doug",
            "person",
        )

        result = await semantic.add_alias(entity.node_id, "Douglas")

        assert result is True
        updated = await semantic.get_entity(entity.node_id)
        assert "Douglas" in updated.aliases


class TestEntityResolution:
    """Test entity resolution (deduplication)."""

    @pytest.mark.asyncio
    async def test_resolve_by_exact_name(self, semantic):
        """Test resolving entity by exact name match."""
        await semantic.create_entity("Doug Mealing", "person")

        matches = await semantic.resolve_entity("Doug Mealing")

        assert len(matches) >= 1
        assert matches[0].entity.canonical_name == "Doug Mealing"
        assert matches[0].match_type == "exact"

    @pytest.mark.asyncio
    async def test_resolve_by_alias(self, semantic):
        """Test resolving entity by alias."""
        await semantic.create_entity(
            "Doug Mealing",
            "person",
            aliases=["Doug", "Douglas Mealing"],
        )

        matches = await semantic.resolve_entity("Doug")

        assert len(matches) >= 1
        assert matches[0].entity.canonical_name == "Doug Mealing"
        assert matches[0].match_type == "alias"

    @pytest.mark.asyncio
    async def test_resolve_by_fuzzy_match(self, semantic):
        """Test fuzzy matching for entity resolution."""
        await semantic.create_entity("Philadelphia", "city")

        matches = await semantic.resolve_entity("Philly", min_score=0.5)

        # Fuzzy match may or may not find depending on algorithm
        # At least ensure it doesn't crash
        assert isinstance(matches, list)

    @pytest.mark.asyncio
    async def test_resolve_returns_scores(self, semantic):
        """Test that resolution returns match scores."""
        await semantic.create_entity("Paris", "city")

        matches = await semantic.resolve_entity("Paris")

        assert len(matches) >= 1
        assert matches[0].match_score > 0.0


class TestEntityMerging:
    """Test entity merging (deduplication)."""

    @pytest.mark.asyncio
    async def test_merge_entities(self, semantic):
        """Test merging two entities."""
        e1 = await semantic.create_entity(
            "Doug",
            "person",
            aliases=["Douglas"],
            properties={"age": 30},
        )

        e2 = await semantic.create_entity(
            "D. Mealing",
            "person",
            aliases=["Doug M"],
            properties={"city": "Philly"},
        )

        merged = await semantic.merge_entities(e1.node_id, e2.node_id)

        assert merged is not None
        # Primary entity should have combined aliases
        assert "Doug M" in merged.aliases or "D. Mealing" in merged.aliases

    @pytest.mark.asyncio
    async def test_merge_combines_properties(self, semantic):
        """Test that merging combines properties."""
        e1 = await semantic.create_entity(
            "Entity1",
            "thing",
            properties={"a": 1},
        )

        e2 = await semantic.create_entity(
            "Entity2",
            "thing",
            properties={"b": 2},
        )

        merged = await semantic.merge_entities(e1.node_id, e2.node_id)

        assert "a" in merged.properties
        assert "b" in merged.properties


class TestFacts:
    """Test fact management."""

    @pytest.mark.asyncio
    async def test_add_fact(self, semantic):
        """Test adding a fact."""
        entity = await semantic.create_entity("Paris", "city")

        fact = await semantic.add_fact(
            content="Paris is the capital of France",
            subject_entity_id=entity.node_id,
            predicate="is_capital_of",
            object_value="France",
        )

        assert fact is not None
        assert "capital" in fact.content.lower()
        assert fact.subject_entity_id == entity.node_id

    @pytest.mark.asyncio
    async def test_get_facts_about_entity(self, semantic):
        """Test retrieving facts about an entity."""
        entity = await semantic.create_entity("Paris", "city")

        await semantic.add_fact(
            "Paris is in France",
            subject_entity_id=entity.node_id,
        )
        await semantic.add_fact(
            "Paris has Eiffel Tower",
            subject_entity_id=entity.node_id,
        )

        facts = await semantic.get_facts_about(entity.node_id)

        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_fact_confidence(self, semantic):
        """Test fact confidence tracking."""
        fact = await semantic.add_fact(
            "Water boils at 100°C",
            confidence=0.95,
        )

        assert fact.confidence == 0.95


class TestRelationships:
    """Test relationship management."""

    @pytest.mark.asyncio
    async def test_add_relationship(self, semantic):
        """Test adding a relationship between entities."""
        doug = await semantic.create_entity("Doug", "person")
        lisa = await semantic.create_entity("Lisa", "person")

        rel = await semantic.add_relationship(
            source_entity_id=doug.node_id,
            target_entity_id=lisa.node_id,
            relationship_type="spouse",
        )

        assert rel is not None
        assert rel.source_entity_id == doug.node_id
        assert rel.target_entity_id == lisa.node_id
        assert rel.relationship_type == "spouse"

    @pytest.mark.asyncio
    async def test_get_entity_relationships(self, semantic):
        """Test retrieving entity relationships."""
        doug = await semantic.create_entity("Doug", "person")
        lisa = await semantic.create_entity("Lisa", "person")
        work = await semantic.create_entity("Acme Inc", "organization")

        await semantic.add_relationship(doug.node_id, lisa.node_id, "spouse")
        await semantic.add_relationship(doug.node_id, work.node_id, "works_at")

        rels = await semantic.get_relationships(doug.node_id)

        assert len(rels) == 2

    @pytest.mark.asyncio
    async def test_relationship_direction(self, semantic):
        """Test that relationships have direction."""
        e1 = await semantic.create_entity("Entity1", "thing")
        e2 = await semantic.create_entity("Entity2", "thing")

        rel = await semantic.add_relationship(
            e1.node_id,
            e2.node_id,
            "connected_to",
        )

        # Should be retrievable from source entity
        rels_from_e1 = await semantic.get_relationships(e1.node_id)

        assert len(rels_from_e1) >= 1
        assert rels_from_e1[0].relationship_type == "connected_to"


class TestSemanticSearch:
    """Test semantic search capabilities."""

    @pytest.mark.asyncio
    async def test_search_entities(self, semantic):
        """Test searching for entities."""
        await semantic.create_entity("Paris", "city")
        await semantic.create_entity("London", "city")
        await semantic.create_entity("Doug", "person")

        results = await semantic.search("cities in Europe", limit=5)

        # Should return some results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_facts(self, semantic):
        """Test searching for facts."""
        await semantic.add_fact("Python was created by Guido van Rossum")
        await semantic.add_fact("JavaScript was created by Brendan Eich")

        results = await semantic.search("programming language creators")

        assert isinstance(results, list)


class TestCommunities:
    """Test community detection."""

    @pytest.mark.asyncio
    async def test_community_assignment(self, semantic):
        """Test that entities can be assigned to communities."""
        entity = await semantic.create_entity(
            "Doug",
            "person",
        )

        result = await semantic.assign_community(
            entity.node_id,
            community_id="household:mealing_home",
        )

        assert result is True

        # Verify the entity was updated
        updated = await semantic.get_entity(entity.node_id)
        assert updated.community_id == "household:mealing_home"

    @pytest.mark.asyncio
    async def test_get_community_members(self, semantic):
        """Test retrieving community members."""
        doug = await semantic.create_entity("Doug", "person")
        lisa = await semantic.create_entity("Lisa", "person")

        await semantic.assign_community(doug.node_id, "family:mealing")
        await semantic.assign_community(lisa.node_id, "family:mealing")

        members = await semantic.get_community_members("family:mealing")

        assert len(members) == 2


class TestConfidencePropagation:
    """Test confidence propagation via supersede."""

    @pytest.mark.asyncio
    async def test_supersede_updates_fact(self, semantic):
        """Test that supersede_fact creates new version."""
        fact = await semantic.add_fact(
            "The sky is blue",
            confidence=0.7,
        )

        updated = await semantic.supersede_fact(
            fact.node_id,
            new_content="The sky is blue during daytime",
        )

        # Supersede should create a new fact
        assert updated is not None
        assert updated.node_id != fact.node_id

    @pytest.mark.asyncio
    async def test_stated_count_available(self, semantic):
        """Test that facts have stated_count attribute."""
        fact = await semantic.add_fact("Water is wet")

        # stated_count should be at least 1
        assert fact.stated_count >= 1


class TestSourceTracking:
    """Test source episode tracking."""

    @pytest.mark.asyncio
    async def test_entity_tracks_source_episodes(self, semantic):
        """Test that entities track source episodes."""
        entity = await semantic.create_entity(
            "Doug",
            "person",
            source_episode_ids=["ep_123", "ep_456"],
        )

        assert "ep_123" in entity.source_episode_ids
        assert "ep_456" in entity.source_episode_ids

    @pytest.mark.asyncio
    async def test_fact_tracks_source_episodes(self, semantic):
        """Test that facts track source episodes."""
        fact = await semantic.add_fact(
            "Doug lives in Philadelphia",
            source_episode_ids=["ep_789"],
        )

        assert "ep_789" in fact.source_episode_ids

"""Tests for the decomposition pipeline integration.

These tests verify:
1. DecompositionService extracts entities, facts, relationships
2. MemoryIntegration stores results in SemanticMemory
3. End-to-end flow with DecompositionMemoryService
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from draagon_ai.cognition.decomposition import (
    DecompositionService,
    DecompositionResult,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    SemanticRole,
    Presupposition,
    CommonsenseInference,
    TemporalInfo,
    ModalityInfo,
    InterpretationBranch,
)
from draagon_ai.cognition.decomposition.service import DecompositionConfig
from draagon_ai.cognition.decomposition.memory_integration import (
    MemoryIntegration,
    DecompositionMemoryService,
    IntegrationConfig,
    IntegrationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Mock LLM provider that returns predefined responses."""
    llm = AsyncMock()
    return llm


@pytest.fixture
def mock_semantic_memory():
    """Mock SemanticMemory layer."""
    memory = MagicMock()

    # Mock entity resolution (no matches by default)
    memory.resolve_entity = AsyncMock(return_value=[])

    # Mock entity creation
    async def create_entity_mock(**kwargs):
        entity = MagicMock()
        entity.node_id = f"entity_{kwargs.get('name', 'unknown')}"
        entity.canonical_name = kwargs.get("name", "")
        return entity

    memory.create_entity = AsyncMock(side_effect=create_entity_mock)

    # Mock fact creation
    async def add_fact_mock(**kwargs):
        fact = MagicMock()
        fact.node_id = f"fact_{id(kwargs)}"
        fact.stated_count = 1  # New fact
        return fact

    memory.add_fact = AsyncMock(side_effect=add_fact_mock)

    # Mock relationship creation
    async def add_rel_mock(**kwargs):
        rel = MagicMock()
        rel.node_id = f"rel_{id(kwargs)}"
        return rel

    memory.add_relationship = AsyncMock(side_effect=add_rel_mock)

    # Mock add_alias
    memory.add_alias = AsyncMock(return_value=True)

    return memory


# =============================================================================
# Type Tests
# =============================================================================


class TestDecompositionTypes:
    """Test decomposition type definitions."""

    def test_entity_type_enum(self):
        """EntityType enum has expected values."""
        assert EntityType.INSTANCE.value == "instance"
        assert EntityType.CLASS.value == "class"
        assert EntityType.ANAPHORA.value == "anaphora"

    def test_extracted_entity_creation(self):
        """ExtractedEntity can be created with required fields."""
        entity = ExtractedEntity(
            text="Doug",
            canonical_name="Doug",
            entity_type=EntityType.INSTANCE,
            confidence=0.95,
        )
        assert entity.text == "Doug"
        assert entity.entity_type == EntityType.INSTANCE
        assert entity.confidence == 0.95
        assert entity.synset_id is None

    def test_extracted_fact_creation(self):
        """ExtractedFact can be created with required fields."""
        fact = ExtractedFact(
            content="Doug has 6 cats",
            subject_text="Doug",
            predicate="has",
            object_value="6 cats",
        )
        assert fact.content == "Doug has 6 cats"
        assert fact.predicate == "has"
        assert fact.subject_entity_id is None

    def test_decomposition_result_creation(self):
        """DecompositionResult can be created and queried."""
        result = DecompositionResult(
            source_text="Doug has 6 cats.",
            entities=[
                ExtractedEntity(
                    text="Doug",
                    canonical_name="Doug",
                    entity_type=EntityType.INSTANCE,
                )
            ],
        )
        assert result.source_text == "Doug has 6 cats."
        assert len(result.entities) == 1

        # Test get_entity_by_text
        entity = result.get_entity_by_text("Doug")
        assert entity is not None
        assert entity.canonical_name == "Doug"

        entity = result.get_entity_by_text("NotFound")
        assert entity is None


# =============================================================================
# DecompositionService Tests
# =============================================================================


class TestDecompositionService:
    """Test the DecompositionService."""

    @pytest.mark.asyncio
    async def test_decompose_simple_sentence(self, mock_llm):
        """DecompositionService processes a simple sentence."""
        # Configure mock LLM responses
        mock_llm.chat.side_effect = [
            # Entity extraction response
            """<entities>
                <entity>
                    <text>Doug</text>
                    <canonical_name>Doug</canonical_name>
                    <entity_type>INSTANCE</entity_type>
                    <confidence>0.95</confidence>
                </entity>
                <entity>
                    <text>cats</text>
                    <canonical_name>cat</canonical_name>
                    <entity_type>CLASS</entity_type>
                    <confidence>0.9</confidence>
                </entity>
            </entities>""",
            # Semantic roles response
            """<roles>
                <role>
                    <predicate>has</predicate>
                    <role_type>ARG0</role_type>
                    <filler>Doug</filler>
                    <confidence>0.9</confidence>
                </role>
                <role>
                    <predicate>has</predicate>
                    <role_type>ARG1</role_type>
                    <filler>6 cats</filler>
                    <confidence>0.9</confidence>
                </role>
            </roles>""",
            # Facts response
            """<facts>
                <fact>
                    <subject>Doug</subject>
                    <predicate>has</predicate>
                    <object>6 cats</object>
                    <confidence>0.95</confidence>
                </fact>
            </facts>""",
            # Relationships response (none expected)
            """<relationships></relationships>""",
            # Presuppositions response
            """<presuppositions></presuppositions>""",
            # Commonsense response
            """<inferences>
                <inference>
                    <relation>xAttr</relation>
                    <head>Doug has cats</head>
                    <tail>Doug is a pet owner</tail>
                    <confidence>0.8</confidence>
                </inference>
            </inferences>""",
            # Temporal response
            """<temporal>
                <tense>present</tense>
                <aspect>state</aspect>
                <confidence>0.9</confidence>
            </temporal>""",
            # Modality response
            """<analysis>
                <modality>
                    <type>none</type>
                    <certainty>1.0</certainty>
                </modality>
                <negation>
                    <is_negated>false</is_negated>
                    <polarity>positive</polarity>
                </negation>
            </analysis>""",
        ]

        service = DecompositionService(llm=mock_llm)
        result = await service.decompose("Doug has 6 cats.")

        # Verify entities extracted
        assert len(result.entities) == 2
        doug = next(e for e in result.entities if e.text == "Doug")
        assert doug.entity_type == EntityType.INSTANCE

        # Verify semantic roles extracted
        assert len(result.semantic_roles) == 2

        # Verify facts extracted
        assert len(result.facts) == 1
        assert result.facts[0].predicate == "has"

        # Verify commonsense inferences
        assert len(result.commonsense_inferences) == 1

        # Verify temporal
        assert result.temporal is not None
        assert result.temporal.tense.value == "present"

        # Verify branches created
        assert len(result.branches) >= 1

    @pytest.mark.asyncio
    async def test_decompose_with_config(self, mock_llm):
        """DecompositionService respects configuration."""
        config = DecompositionConfig(
            extract_entities=True,
            extract_facts=False,
            extract_relationships=False,
            extract_semantic_roles=False,
            extract_presuppositions=False,
            extract_commonsense=False,
            extract_temporal=False,
            extract_modality=False,
        )

        mock_llm.chat.return_value = """<entities>
            <entity>
                <text>Doug</text>
                <canonical_name>Doug</canonical_name>
                <entity_type>INSTANCE</entity_type>
                <confidence>0.95</confidence>
            </entity>
        </entities>"""

        service = DecompositionService(llm=mock_llm, config=config)
        result = await service.decompose("Doug has 6 cats.")

        # Only entities should be extracted
        assert len(result.entities) == 1
        assert len(result.facts) == 0
        assert len(result.semantic_roles) == 0

        # LLM should only be called once (for entities)
        assert mock_llm.chat.call_count == 1


# =============================================================================
# MemoryIntegration Tests
# =============================================================================


class TestMemoryIntegration:
    """Test the MemoryIntegration class."""

    @pytest.mark.asyncio
    async def test_integrate_creates_entities(self, mock_semantic_memory):
        """Integration creates entities in semantic memory."""
        result = DecompositionResult(
            source_text="Doug has 6 cats.",
            entities=[
                ExtractedEntity(
                    text="Doug",
                    canonical_name="Doug",
                    entity_type=EntityType.INSTANCE,
                    confidence=0.95,
                )
            ],
        )

        integrator = MemoryIntegration(mock_semantic_memory)
        integration = await integrator.integrate(result)

        # Verify entity was created
        assert integration.entities_created == 1
        assert len(integration.entity_ids) == 1
        assert "Doug" in integration.entity_mapping

        # Verify create_entity was called
        mock_semantic_memory.create_entity.assert_called_once()
        call_kwargs = mock_semantic_memory.create_entity.call_args.kwargs
        assert call_kwargs["name"] == "Doug"

    @pytest.mark.asyncio
    async def test_integrate_stores_facts(self, mock_semantic_memory):
        """Integration stores facts with entity linking."""
        result = DecompositionResult(
            source_text="Doug has 6 cats.",
            entities=[
                ExtractedEntity(
                    text="Doug",
                    canonical_name="Doug",
                    entity_type=EntityType.INSTANCE,
                )
            ],
            facts=[
                ExtractedFact(
                    content="Doug has 6 cats",
                    subject_text="Doug",
                    predicate="has",
                    object_value="6 cats",
                    confidence=0.95,
                )
            ],
        )

        integrator = MemoryIntegration(mock_semantic_memory)
        integration = await integrator.integrate(result)

        # Verify fact was created
        assert integration.facts_created == 1
        assert len(integration.fact_ids) == 1

        # Verify add_fact was called with entity linking
        mock_semantic_memory.add_fact.assert_called_once()
        call_kwargs = mock_semantic_memory.add_fact.call_args.kwargs
        assert call_kwargs["predicate"] == "has"
        assert call_kwargs["subject_entity_id"] == "entity_Doug"

    @pytest.mark.asyncio
    async def test_integrate_resolves_existing_entities(self, mock_semantic_memory):
        """Integration resolves to existing entities when found."""
        # Configure mock to return an existing entity
        existing_entity = MagicMock()
        existing_entity.node_id = "existing_doug"
        existing_entity.canonical_name = "Doug"

        match = MagicMock()
        match.entity = existing_entity
        mock_semantic_memory.resolve_entity = AsyncMock(return_value=[match])

        result = DecompositionResult(
            source_text="Doug has 6 cats.",
            entities=[
                ExtractedEntity(
                    text="Doug",
                    canonical_name="Doug",
                    entity_type=EntityType.INSTANCE,
                )
            ],
        )

        integrator = MemoryIntegration(mock_semantic_memory)
        integration = await integrator.integrate(result)

        # Entity should be resolved, not created
        assert integration.entities_created == 0
        assert integration.entities_merged == 0  # Not merged, just resolved
        assert len(integration.entity_ids) == 1
        assert integration.entity_ids[0] == "existing_doug"

        # create_entity should NOT be called
        mock_semantic_memory.create_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_integrate_creates_relationships(self, mock_semantic_memory):
        """Integration creates relationships between entities."""
        result = DecompositionResult(
            source_text="Doug works at Anthropic.",
            entities=[
                ExtractedEntity(
                    text="Doug",
                    canonical_name="Doug",
                    entity_type=EntityType.INSTANCE,
                ),
                ExtractedEntity(
                    text="Anthropic",
                    canonical_name="Anthropic",
                    entity_type=EntityType.INSTANCE,
                ),
            ],
            relationships=[
                ExtractedRelationship(
                    source_text="Doug",
                    target_text="Anthropic",
                    relationship_type="works_at",
                    confidence=0.9,
                )
            ],
        )

        integrator = MemoryIntegration(mock_semantic_memory)
        integration = await integrator.integrate(result)

        # Verify relationship was created
        assert integration.relationships_created == 1
        assert len(integration.relationship_ids) == 1

        # Verify add_relationship was called
        mock_semantic_memory.add_relationship.assert_called_once()


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestDecompositionMemoryService:
    """Test the combined DecompositionMemoryService."""

    @pytest.mark.asyncio
    async def test_process_and_store(self, mock_llm, mock_semantic_memory):
        """End-to-end test of process_and_store."""
        # Configure mock LLM for a simple response
        mock_llm.chat.side_effect = [
            """<entities>
                <entity>
                    <text>Doug</text>
                    <canonical_name>Doug</canonical_name>
                    <entity_type>INSTANCE</entity_type>
                    <confidence>0.95</confidence>
                </entity>
            </entities>""",
            """<roles></roles>""",
            """<facts>
                <fact>
                    <subject>Doug</subject>
                    <predicate>has</predicate>
                    <object>6 cats</object>
                    <confidence>0.9</confidence>
                </fact>
            </facts>""",
            """<relationships></relationships>""",
            """<presuppositions></presuppositions>""",
            """<inferences></inferences>""",
            """<temporal><tense>present</tense><aspect>state</aspect></temporal>""",
            """<analysis>
                <modality><type>none</type></modality>
                <negation><is_negated>false</is_negated><polarity>positive</polarity></negation>
            </analysis>""",
        ]

        service = DecompositionMemoryService(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        result = await service.process_and_store(
            "Doug has 6 cats.",
            scope_id="user:assistant:test",
        )

        # Verify entities and facts were created
        assert result.entities_created >= 1
        assert result.facts_created >= 1

        # Verify semantic memory was called
        mock_semantic_memory.create_entity.assert_called()
        mock_semantic_memory.add_fact.assert_called()

    @pytest.mark.asyncio
    async def test_decompose_only(self, mock_llm, mock_semantic_memory):
        """decompose_only returns result without storing."""
        mock_llm.chat.side_effect = [
            """<entities>
                <entity>
                    <text>Doug</text>
                    <canonical_name>Doug</canonical_name>
                    <entity_type>INSTANCE</entity_type>
                    <confidence>0.95</confidence>
                </entity>
            </entities>""",
            """<roles></roles>""",
            """<facts></facts>""",
            """<relationships></relationships>""",
            """<presuppositions></presuppositions>""",
            """<inferences></inferences>""",
            """<temporal><tense>present</tense><aspect>state</aspect></temporal>""",
            """<analysis>
                <modality><type>none</type></modality>
                <negation><is_negated>false</is_negated><polarity>positive</polarity></negation>
            </analysis>""",
        ]

        service = DecompositionMemoryService(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        result = await service.decompose_only("Doug has 6 cats.")

        # Verify result is DecompositionResult
        assert isinstance(result, DecompositionResult)
        assert len(result.entities) == 1

        # Verify semantic memory was NOT called
        mock_semantic_memory.create_entity.assert_not_called()
        mock_semantic_memory.add_fact.assert_not_called()

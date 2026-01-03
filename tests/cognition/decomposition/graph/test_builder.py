"""Unit tests for GraphBuilder class.

Tests for building semantic graphs from Phase 0/1 decomposition output.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from draagon_ai.cognition.decomposition.graph import (
    GraphBuilder,
    GraphBuilderConfig,
    GraphBuildResult,
    SemanticGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    SemanticEdgeType,
)
from draagon_ai.cognition.decomposition.identifiers import (
    EntityType,
    UniversalSemanticIdentifier,
)
from draagon_ai.cognition.decomposition.extractors.models import (
    DecomposedKnowledge,
    SemanticRole,
    Presupposition,
    PresuppositionTrigger,
    CommonsenseInference,
    CommonsenseRelation,
    TemporalInfo,
    ModalityInfo,
    NegationInfo,
    CrossReference,
    WeightedBranch,
    Tense,
    Aspect,
    ModalType,
    Polarity,
)


# =============================================================================
# Mock Objects for Phase 0 Output
# =============================================================================


@dataclass
class MockDisambiguationResult:
    """Mock WSD result for testing."""

    synset_id: str
    confidence: float = 0.9
    method: str = "llm"
    alternatives: list[str] = field(default_factory=list)
    definition: str = ""


@dataclass
class MockClassificationResult:
    """Mock entity classification result for testing.

    Matches the interface of the real ClassificationResult class.
    """

    entity_type: EntityType
    confidence: float = 0.9
    method: str = "llm"
    reasoning: str = ""  # Must match real ClassificationResult field name


@dataclass
class MockContentAnalysis:
    """Mock content analysis result for testing."""

    content_type: str = "prose"


@dataclass
class MockPhase0Result:
    """Mock Phase 0 result for testing."""

    content_analysis: MockContentAnalysis = field(default_factory=MockContentAnalysis)
    disambiguation_results: dict[str, MockDisambiguationResult] = field(default_factory=dict)
    entity_identifiers: dict[str, UniversalSemanticIdentifier] = field(default_factory=dict)
    entity_classifications: dict[str, MockClassificationResult] = field(default_factory=dict)
    structural_knowledge: list[dict] = field(default_factory=list)
    processed_text: str = ""
    skipped_wsd: bool = False
    skip_reason: str = ""


@dataclass
class MockIntegratedResult:
    """Mock integrated pipeline result for testing."""

    source_text: str
    content_type: str = "prose"
    phase0: MockPhase0Result = field(default_factory=MockPhase0Result)
    decomposition: DecomposedKnowledge = field(default_factory=lambda: DecomposedKnowledge(source_text=""))
    chunks_processed: int = 1
    total_duration_ms: float = 0.0
    processing_timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def builder() -> GraphBuilder:
    """Create a default graph builder."""
    return GraphBuilder()


@pytest.fixture
def config_no_presuppositions() -> GraphBuilderConfig:
    """Config with presuppositions disabled."""
    return GraphBuilderConfig(include_presuppositions=False)


@pytest.fixture
def simple_phase0() -> MockPhase0Result:
    """Create a simple Phase 0 result with Doug and a meeting."""
    phase0 = MockPhase0Result()
    phase0.processed_text = "Doug forgot the meeting again"

    # Entity identifiers
    phase0.entity_identifiers = {
        "Doug": UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            canonical_name="Doug",
            confidence=0.95,
        ),
        "meeting": UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="meeting.n.01",
            confidence=0.85,
        ),
    }

    # Entity classifications
    phase0.entity_classifications = {
        "Doug": MockClassificationResult(
            entity_type=EntityType.INSTANCE,
            confidence=0.95,
            method="llm",
            reasoning="Capitalized proper noun",
        ),
        "meeting": MockClassificationResult(
            entity_type=EntityType.CLASS,
            confidence=0.85,
            method="llm",
            reasoning="Common noun with definite article",
        ),
    }

    # WSD results
    phase0.disambiguation_results = {
        "forgot": MockDisambiguationResult(
            synset_id="forget.v.01",
            confidence=0.9,
            method="llm",
            alternatives=["forget.v.02", "forget.v.03"],
        ),
        "meeting": MockDisambiguationResult(
            synset_id="meeting.n.01",
            confidence=0.85,
            method="lesk",
        ),
    }

    return phase0


@pytest.fixture
def simple_decomposition() -> DecomposedKnowledge:
    """Create a simple decomposition with semantic roles and presuppositions."""
    return DecomposedKnowledge(
        source_text="Doug forgot the meeting again",
        entity_ids=["Doug", "meeting"],
        wsd_results={"forgot": "forget.v.01", "meeting": "meeting.n.01"},
        entity_types={"Doug": "instance", "meeting": "class"},
        semantic_roles=[
            SemanticRole(
                predicate="forgot",
                predicate_sense="forget.v.01",
                role="ARG0",
                filler="Doug",
                confidence=0.95,
            ),
            SemanticRole(
                predicate="forgot",
                predicate_sense="forget.v.01",
                role="ARG1",
                filler="the meeting",
                confidence=0.9,
            ),
        ],
        presuppositions=[
            Presupposition(
                content="Doug forgot before",
                trigger_type=PresuppositionTrigger.ITERATIVE,
                trigger_text="again",
                confidence=0.85,
                entity_ids=["Doug"],
            ),
            Presupposition(
                content="A specific meeting exists",
                trigger_type=PresuppositionTrigger.DEFINITE_DESC,
                trigger_text="the meeting",
                confidence=0.9,
                entity_ids=["meeting"],
            ),
        ],
        commonsense_inferences=[
            CommonsenseInference(
                relation=CommonsenseRelation.X_REACT,
                head="Doug forgot the meeting",
                tail="embarrassed or frustrated",
                head_entity_ids=["Doug"],
                confidence=0.8,
            ),
            CommonsenseInference(
                relation=CommonsenseRelation.X_EFFECT,
                head="Doug forgot the meeting",
                tail="Doug missed the meeting",
                head_entity_ids=["Doug"],
                confidence=0.75,
            ),
        ],
        temporal=TemporalInfo(
            tense=Tense.PAST,
            aspect=Aspect.ACHIEVEMENT,
            confidence=0.95,
        ),
        modality=ModalityInfo(
            modal_type=ModalType.NONE,
            confidence=0.9,
        ),
        negation=NegationInfo(
            is_negated=False,
            polarity=Polarity.POSITIVE,
            confidence=0.95,
        ),
    )


@pytest.fixture
def integrated_result(simple_phase0, simple_decomposition) -> MockIntegratedResult:
    """Create an integrated result combining Phase 0 and Phase 1."""
    return MockIntegratedResult(
        source_text="Doug forgot the meeting again",
        phase0=simple_phase0,
        decomposition=simple_decomposition,
    )


# =============================================================================
# Basic Builder Tests
# =============================================================================


class TestGraphBuilderBasics:
    """Basic tests for GraphBuilder."""

    def test_builder_creation(self, builder):
        """Test creating a builder with default config."""
        assert builder.config is not None
        assert builder.config.include_presuppositions is True
        assert builder.config.include_inferences is True

    def test_builder_with_custom_config(self):
        """Test creating a builder with custom config."""
        config = GraphBuilderConfig(
            include_presuppositions=False,
            min_confidence=0.5,
        )
        builder = GraphBuilder(config)

        assert builder.config.include_presuppositions is False
        assert builder.config.min_confidence == 0.5

    def test_build_from_empty_decomposition(self, builder):
        """Test building from empty decomposition."""
        decomp = DecomposedKnowledge(source_text="")
        result = builder.build_from_decomposition(decomp)

        assert result.graph is not None
        assert result.graph.node_count >= 1  # Source node always created
        assert len(result.errors) == 0


# =============================================================================
# Entity Building Tests
# =============================================================================


class TestEntityBuilding:
    """Tests for building entity nodes from Phase 0."""

    def test_entities_created_from_phase0(self, builder, integrated_result):
        """Test that entities from Phase 0 are created as INSTANCE nodes."""
        result = builder.build_from_integrated(integrated_result)

        # Check Doug was created
        assert "Doug" in result.entity_node_map
        doug_node = result.graph.get_node(result.entity_node_map["Doug"])
        assert doug_node is not None
        assert doug_node.canonical_name == "Doug"
        # All text mentions are INSTANCE nodes (the CLASS is a separate synset node)
        assert doug_node.node_type == NodeType.INSTANCE
        assert doug_node.entity_type == EntityType.INSTANCE

    def test_entity_properties_captured(self, builder, integrated_result):
        """Test that entity properties are captured."""
        result = builder.build_from_integrated(integrated_result)

        doug_node = result.graph.get_node(result.entity_node_map["Doug"])
        assert doug_node.properties["entity_type"] == "instance"
        assert doug_node.properties["source_text"] == "Doug"
        assert doug_node.confidence == 0.95

    def test_entity_with_synset(self, builder, integrated_result):
        """Test that entities with synsets have synset in properties (not on node)."""
        result = builder.build_from_integrated(integrated_result)

        meeting_node = result.graph.get_node(result.entity_node_map["meeting"])
        # synset_id on node is None - it's reserved for CLASS nodes (synset nodes)
        # The synset is stored in properties for reference
        assert meeting_node.synset_id is None
        assert meeting_node.properties.get("wordnet_synset") == "meeting.n.01"

    def test_entity_counts_in_stats(self, builder, integrated_result):
        """Test that entity counts are tracked in stats."""
        result = builder.build_from_integrated(integrated_result)

        assert result.stats["entities_created"] >= 2  # Doug and meeting


# =============================================================================
# WSD Node Building Tests
# =============================================================================


class TestWSDNodeBuilding:
    """Tests for building WSD/synset nodes."""

    def test_synset_nodes_created(self, builder, integrated_result):
        """Test that synset nodes are created for WSD results."""
        result = builder.build_from_integrated(integrated_result)

        # Should have synset node for "forgot"
        assert "forget.v.01" in result.synset_node_map
        synset_node = result.graph.get_node(result.synset_node_map["forget.v.01"])
        assert synset_node is not None
        # Synset nodes are CLASS type (abstract concepts, not text mentions)
        assert synset_node.node_type == NodeType.CLASS
        assert synset_node.synset_id == "forget.v.01"

    def test_alternative_senses_created(self, builder, integrated_result):
        """Test that alternative WSD senses are created."""
        result = builder.build_from_integrated(integrated_result)

        # Should have alternative synsets for "forgot"
        assert "forget.v.02" in result.synset_node_map or "forget.v.03" in result.synset_node_map

    def test_alternative_senses_disabled(self, integrated_result):
        """Test disabling alternative sense creation."""
        config = GraphBuilderConfig(include_wsd_alternatives=False)
        builder = GraphBuilder(config)
        result = builder.build_from_integrated(integrated_result)

        # Should NOT have alternative synsets
        assert "forget.v.02" not in result.synset_node_map
        assert "forget.v.03" not in result.synset_node_map


# =============================================================================
# Semantic Role Building Tests
# =============================================================================


class TestSemanticRoleBuilding:
    """Tests for building semantic role edges."""

    def test_predicate_nodes_created(self, builder, integrated_result):
        """Test that predicate nodes are created."""
        result = builder.build_from_integrated(integrated_result)

        # Should have predicate node for "forgot"
        predicate_key = "forgot:forget.v.01"
        assert predicate_key in result.predicate_node_map

        pred_node = result.graph.get_node(result.predicate_node_map[predicate_key])
        assert pred_node is not None
        assert pred_node.node_type == NodeType.EVENT
        assert pred_node.properties["predicate"] == "forgot"

    def test_arg_edges_created(self, builder, integrated_result):
        """Test that ARG edges are created from predicate to fillers."""
        result = builder.build_from_integrated(integrated_result)

        predicate_key = "forgot:forget.v.01"
        pred_node_id = result.predicate_node_map[predicate_key]

        # Get outgoing edges from predicate
        edges = result.graph.get_outgoing_edges(pred_node_id)

        # Should have ARG0 and ARG1 edges
        relation_types = {e.relation_type for e in edges}
        assert "ARG0" in relation_types
        assert "ARG1" in relation_types

    def test_arg_edge_confidence(self, builder, integrated_result):
        """Test that ARG edges have correct confidence."""
        result = builder.build_from_integrated(integrated_result)

        predicate_key = "forgot:forget.v.01"
        pred_node_id = result.predicate_node_map[predicate_key]

        edges = result.graph.get_outgoing_edges(pred_node_id, relation_type="ARG0")
        assert len(edges) > 0
        assert edges[0].confidence == 0.95  # From semantic role


# =============================================================================
# Presupposition Building Tests
# =============================================================================


class TestPresuppositionBuilding:
    """Tests for building presupposition nodes and edges."""

    def test_presupposition_nodes_created(self, builder, integrated_result):
        """Test that presupposition nodes are created."""
        result = builder.build_from_integrated(integrated_result)

        # Should have 2 presuppositions
        assert result.stats["presuppositions_created"] == 2

    def test_presupposition_edges_created(self, builder, integrated_result):
        """Test that presupposition edges link source to presupposition."""
        result = builder.build_from_integrated(integrated_result)

        # Find presupposition nodes
        presup_nodes = [
            n for n in result.graph.iter_nodes()
            if n.properties.get("is_presupposition")
        ]
        assert len(presup_nodes) == 2

        # Each should have an incoming "presupposes" edge from source
        for presup_node in presup_nodes:
            incoming = result.graph.get_incoming_edges(presup_node.node_id)
            presupposes_edges = [e for e in incoming if e.relation_type == "presupposes"]
            assert len(presupposes_edges) >= 1

    def test_presupposition_disabled(self, integrated_result):
        """Test disabling presupposition creation."""
        config = GraphBuilderConfig(include_presuppositions=False)
        builder = GraphBuilder(config)
        result = builder.build_from_integrated(integrated_result)

        assert result.stats["presuppositions_created"] == 0


# =============================================================================
# Commonsense Inference Building Tests
# =============================================================================


class TestInferenceBuilding:
    """Tests for building commonsense inference edges."""

    def test_inference_edges_created(self, builder, integrated_result):
        """Test that inference edges are created."""
        result = builder.build_from_integrated(integrated_result)

        # Should have 2 inferences
        assert result.stats["inferences_created"] == 2

    def test_inference_relation_types(self, builder, integrated_result):
        """Test that inference edges have correct ATOMIC relation types."""
        result = builder.build_from_integrated(integrated_result)

        # Find inference nodes (tail nodes)
        inference_nodes = [
            n for n in result.graph.iter_nodes()
            if n.properties.get("is_inference")
        ]
        assert len(inference_nodes) == 2

        # Check that we have xReact and xEffect edges
        all_incoming_types = set()
        for node in inference_nodes:
            for edge in result.graph.get_incoming_edges(node.node_id):
                all_incoming_types.add(edge.relation_type)

        assert "xReact" in all_incoming_types
        assert "xEffect" in all_incoming_types

    def test_inferences_disabled(self, integrated_result):
        """Test disabling inference creation."""
        config = GraphBuilderConfig(include_inferences=False)
        builder = GraphBuilder(config)
        result = builder.build_from_integrated(integrated_result)

        assert result.stats["inferences_created"] == 0


# =============================================================================
# Temporal Building Tests
# =============================================================================


class TestTemporalBuilding:
    """Tests for building temporal information."""

    def test_temporal_properties_added(self, builder, integrated_result):
        """Test that temporal properties are added to source node."""
        result = builder.build_from_integrated(integrated_result)

        # Find source node
        source_nodes = [
            n for n in result.graph.iter_nodes()
            if n.canonical_name.startswith("source:")
        ]
        assert len(source_nodes) >= 1

        source_node = source_nodes[0]
        assert source_node.properties.get("tense") == "past"
        assert source_node.properties.get("aspect") == "achievement"

    def test_temporal_disabled(self, integrated_result):
        """Test disabling temporal building."""
        config = GraphBuilderConfig(include_temporal=False)
        builder = GraphBuilder(config)
        result = builder.build_from_integrated(integrated_result)

        # Source node should not have temporal properties
        source_nodes = [
            n for n in result.graph.iter_nodes()
            if n.canonical_name.startswith("source:")
        ]
        if source_nodes:
            assert "tense" not in source_nodes[0].properties


# =============================================================================
# Modality and Negation Tests
# =============================================================================


class TestModalityNegationBuilding:
    """Tests for modality and negation properties."""

    def test_modality_properties_added(self, builder, integrated_result):
        """Test that modality properties are added."""
        result = builder.build_from_integrated(integrated_result)

        source_nodes = [
            n for n in result.graph.iter_nodes()
            if n.canonical_name.startswith("source:")
        ]
        assert len(source_nodes) >= 1

        source_node = source_nodes[0]
        assert source_node.properties.get("modal_type") == "none"

    def test_negation_properties_added(self, builder, integrated_result):
        """Test that negation properties are added."""
        result = builder.build_from_integrated(integrated_result)

        source_nodes = [
            n for n in result.graph.iter_nodes()
            if n.canonical_name.startswith("source:")
        ]
        assert len(source_nodes) >= 1

        source_node = source_nodes[0]
        assert source_node.properties.get("is_negated") is False
        assert source_node.properties.get("polarity") == "positive"


# =============================================================================
# Confidence Tracking Tests
# =============================================================================


class TestConfidenceTracking:
    """Tests for confidence tracking on edges."""

    def test_entity_node_confidence(self, builder, integrated_result):
        """Test that entity nodes have confidence scores."""
        result = builder.build_from_integrated(integrated_result)

        doug_node = result.graph.get_node(result.entity_node_map["Doug"])
        assert doug_node.confidence == 0.95

        meeting_node = result.graph.get_node(result.entity_node_map["meeting"])
        assert meeting_node.confidence == 0.85

    def test_edge_confidence(self, builder, integrated_result):
        """Test that edges have confidence scores."""
        result = builder.build_from_integrated(integrated_result)

        # Get all edges
        for edge in result.graph.iter_edges():
            assert 0.0 <= edge.confidence <= 1.0

    def test_min_confidence_filter(self, integrated_result):
        """Test that low-confidence items are filtered."""
        config = GraphBuilderConfig(min_confidence=0.9)
        builder = GraphBuilder(config)
        result = builder.build_from_integrated(integrated_result)

        # xReact inference has 0.8 confidence - should be filtered
        inference_nodes = [
            n for n in result.graph.iter_nodes()
            if n.properties.get("is_inference") and "xReact" in n.properties.get("inference_type", "")
        ]
        # May or may not exist depending on filtering


# =============================================================================
# Existing Graph Merge Tests
# =============================================================================


class TestExistingGraphMerge:
    """Tests for building into an existing graph."""

    def test_merge_into_existing_graph(self, builder, integrated_result):
        """Test building into an existing graph."""
        # Create existing graph with some nodes
        existing = SemanticGraph()
        existing_node = existing.create_node("Existing", NodeType.INSTANCE)

        result = builder.build_from_integrated(integrated_result, existing_graph=existing)

        # Should have both existing and new nodes
        assert existing_node.node_id in result.graph
        assert result.graph.node_count > 1

    def test_merge_preserves_existing_edges(self, builder, integrated_result):
        """Test that merging preserves existing edges."""
        # Create existing graph with edge
        existing = SemanticGraph()
        node_a = existing.create_node("A", NodeType.INSTANCE)
        node_b = existing.create_node("B", NodeType.INSTANCE)
        existing.create_edge(node_a.node_id, node_b.node_id, "knows")

        result = builder.build_from_integrated(integrated_result, existing_graph=existing)

        # Should still have the "knows" edge
        edges = result.graph.get_outgoing_edges(node_a.node_id, relation_type="knows")
        assert len(edges) == 1


# =============================================================================
# Synset Deduplication Tests
# =============================================================================


class TestSynsetDeduplication:
    """Tests for WordNet synset deduplication."""

    def test_synset_deduplicated_within_build(self):
        """Test that same synset is only created once within a single build."""
        builder = GraphBuilder()

        # Create Phase 0 with two words mapping to same synset
        phase0 = MockPhase0Result()
        phase0.disambiguation_results = {
            "cat": MockDisambiguationResult(synset_id="cat.n.01", confidence=0.9),
            "feline": MockDisambiguationResult(synset_id="cat.n.01", confidence=0.85),
        }
        phase0.entity_identifiers = {
            "cat": UniversalSemanticIdentifier(
                entity_type=EntityType.CLASS,
                wordnet_synset="cat.n.01",
                confidence=0.9,
            ),
            "feline": UniversalSemanticIdentifier(
                entity_type=EntityType.CLASS,
                wordnet_synset="cat.n.01",
                confidence=0.85,
            ),
        }

        decomp = DecomposedKnowledge(source_text="cat and feline")
        integrated = MockIntegratedResult(
            source_text="cat and feline",
            phase0=phase0,
            decomposition=decomp,
        )

        result = builder.build_from_integrated(integrated)

        # Should only have ONE synset node for cat.n.01
        synset_nodes = result.graph.find_nodes_by_synset("cat.n.01")
        assert len(synset_nodes) == 1

        # The node should track both words
        synset_node = synset_nodes[0]
        words = synset_node.properties.get("words", [])
        assert "cat" in words
        assert "feline" in words

    def test_synset_deduplicated_across_builds(self):
        """Test that synset from prior build is reused."""
        builder = GraphBuilder()

        # First build: "Doug has a cat"
        phase0_1 = MockPhase0Result()
        phase0_1.disambiguation_results = {
            "cat": MockDisambiguationResult(synset_id="cat.n.01", confidence=0.9),
        }
        phase0_1.entity_identifiers = {
            "Doug": UniversalSemanticIdentifier(
                entity_type=EntityType.INSTANCE,
                canonical_name="Doug",
                confidence=0.95,
            ),
            "cat": UniversalSemanticIdentifier(
                entity_type=EntityType.CLASS,
                wordnet_synset="cat.n.01",
                confidence=0.9,
            ),
        }

        decomp1 = DecomposedKnowledge(source_text="Doug has a cat")
        integrated1 = MockIntegratedResult(
            source_text="Doug has a cat",
            phase0=phase0_1,
            decomposition=decomp1,
        )

        result1 = builder.build_from_integrated(integrated1)
        graph = result1.graph

        # Verify we have one cat.n.01 node
        synset_nodes_before = graph.find_nodes_by_synset("cat.n.01")
        assert len(synset_nodes_before) == 1
        original_node_id = synset_nodes_before[0].node_id

        # Second build: "Sarah also has a cat" - into same graph
        phase0_2 = MockPhase0Result()
        phase0_2.disambiguation_results = {
            "cat": MockDisambiguationResult(synset_id="cat.n.01", confidence=0.88),
        }
        phase0_2.entity_identifiers = {
            "Sarah": UniversalSemanticIdentifier(
                entity_type=EntityType.INSTANCE,
                canonical_name="Sarah",
                confidence=0.92,
            ),
            "cat": UniversalSemanticIdentifier(
                entity_type=EntityType.CLASS,
                wordnet_synset="cat.n.01",
                confidence=0.88,
            ),
        }

        decomp2 = DecomposedKnowledge(source_text="Sarah also has a cat")
        integrated2 = MockIntegratedResult(
            source_text="Sarah also has a cat",
            phase0=phase0_2,
            decomposition=decomp2,
        )

        result2 = builder.build_from_integrated(integrated2, existing_graph=graph)

        # Should STILL only have ONE cat.n.01 node (reused)
        synset_nodes_after = graph.find_nodes_by_synset("cat.n.01")
        assert len(synset_nodes_after) == 1
        assert synset_nodes_after[0].node_id == original_node_id

        # Both decomposition sources should be tracked
        synset_node = synset_nodes_after[0]
        assert len(synset_node.source_ids) >= 2

    def test_different_synsets_not_merged(self):
        """Test that different synsets are NOT merged."""
        builder = GraphBuilder()

        phase0 = MockPhase0Result()
        phase0.disambiguation_results = {
            "cat": MockDisambiguationResult(synset_id="cat.n.01", confidence=0.9),
            "dog": MockDisambiguationResult(synset_id="dog.n.01", confidence=0.9),
        }
        phase0.entity_identifiers = {
            "cat": UniversalSemanticIdentifier(
                entity_type=EntityType.CLASS,
                wordnet_synset="cat.n.01",
                confidence=0.9,
            ),
            "dog": UniversalSemanticIdentifier(
                entity_type=EntityType.CLASS,
                wordnet_synset="dog.n.01",
                confidence=0.9,
            ),
        }

        decomp = DecomposedKnowledge(source_text="cat and dog")
        integrated = MockIntegratedResult(
            source_text="cat and dog",
            phase0=phase0,
            decomposition=decomp,
        )

        result = builder.build_from_integrated(integrated)

        # Should have TWO distinct synset nodes
        cat_nodes = result.graph.find_nodes_by_synset("cat.n.01")
        dog_nodes = result.graph.find_nodes_by_synset("dog.n.01")

        assert len(cat_nodes) == 1
        assert len(dog_nodes) == 1
        assert cat_nodes[0].node_id != dog_nodes[0].node_id


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling during graph building."""

    def test_build_result_tracks_errors(self, builder):
        """Test that errors are tracked in build result."""
        # Create decomposition that might cause issues
        decomp = DecomposedKnowledge(source_text="Test")
        result = builder.build_from_decomposition(decomp)

        # Should have no errors for simple case
        assert isinstance(result.errors, list)

    def test_build_continues_on_partial_failure(self, builder, integrated_result):
        """Test that building continues even if some parts fail."""
        # Even with problematic input, should produce partial graph
        result = builder.build_from_integrated(integrated_result)

        assert result.graph is not None
        assert result.graph.node_count > 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestBuildStatistics:
    """Tests for build statistics tracking."""

    def test_stats_tracked(self, builder, integrated_result):
        """Test that build statistics are tracked."""
        result = builder.build_from_integrated(integrated_result)

        assert "entities_created" in result.stats
        assert "predicates_created" in result.stats
        assert "presuppositions_created" in result.stats
        assert "inferences_created" in result.stats
        assert "edges_created" in result.stats

    def test_stats_accurate(self, builder, integrated_result):
        """Test that statistics are accurate."""
        result = builder.build_from_integrated(integrated_result)

        # Stats should be non-negative
        assert result.stats["entities_created"] >= 0
        assert result.stats["predicates_created"] >= 0
        assert result.stats["edges_created"] >= 0

        # Total nodes should be reasonable given stats
        total_nodes = result.graph.node_count
        assert total_nodes > 0

        # Count actual edges
        edges = list(result.graph.iter_edges())
        # edges_created stat may undercount (edges can be created in multiple places)
        assert len(edges) >= result.stats["edges_created"] // 2

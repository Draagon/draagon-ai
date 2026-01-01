"""Graph Builder for Phase 0/1 Integration.

This module provides the GraphBuilder class that takes the output from the
integrated Phase 0/1 pipeline and builds a semantic graph from it.

The builder maps:
- Entities → GraphNodes (ENTITY, CONCEPT, EVENT types)
- WSD results → Node synset_id metadata
- Semantic roles → Edges (predicate-argument relations)
- Presuppositions → Nodes (PRESUPPOSITION) with edges to triggers
- Commonsense inferences → Edges (cause/effect, intent, reaction)
- Temporal info → Node properties or separate TIMEPOINT nodes
- Modality → Edge properties (certainty, source)
- Negation → Edge properties (polarity)
- Cross-references → Edges to existing nodes
- Weighted branches → Alternative interpretation subgraphs

Philosophy: Store EVERYTHING that could provide context value to the LLM.
We can prune later when we see what doesn't provide value.

Every edge has confidence tracking for how trusted/accurate we think it is.

Example:
    >>> from draagon_ai.cognition.decomposition.graph import GraphBuilder, SemanticGraph
    >>> from draagon_ai.cognition.decomposition.extractors.integrated_pipeline import IntegratedResult
    >>>
    >>> # Process text through Phase 0/1
    >>> result: IntegratedResult = await pipeline.process("Doug forgot the meeting again")
    >>>
    >>> # Build graph from result
    >>> builder = GraphBuilder()
    >>> graph = builder.build(result)
    >>>
    >>> # Graph now contains:
    >>> # - Entity nodes: Doug, "the meeting"
    >>> # - Predicate node: forgot
    >>> # - Presupposition nodes: "Doug forgot before", "A specific meeting exists"
    >>> # - Edges: Doug -[ARG0]-> forgot, forgot -[ARG1]-> meeting
    >>> # - Commonsense edges: forgot -[xReact]-> "frustrated/embarrassed"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import (
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeRelationType,
    MergeResult,
)
from .semantic_graph import SemanticGraph

from ..identifiers import EntityType, UniversalSemanticIdentifier
from ..extractors.models import (
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

# Import Phase0Result and IntegratedResult - use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extractors.integrated_pipeline import IntegratedResult, Phase0Result
    from ..wsd import DisambiguationResult
    from ..entity_classifier import ClassificationResult


logger = logging.getLogger(__name__)


# =============================================================================
# Extended Edge Relation Types for Phase 1
# =============================================================================

# These extend EdgeRelationType for semantic decomposition relationships


class SemanticEdgeType:
    """Extended edge types for semantic decomposition.

    These are string constants that can be used as relation_type on edges.
    We use strings rather than enums for flexibility.
    """

    # Semantic Role edges (predicate-argument structure)
    ARG0 = "ARG0"  # Proto-agent (doer)
    ARG1 = "ARG1"  # Proto-patient (done-to)
    ARG2 = "ARG2"  # Tertiary argument
    ARG3 = "ARG3"  # Quaternary argument
    ARG4 = "ARG4"  # Fifth argument
    ARGM_LOC = "ARGM-LOC"  # Location modifier
    ARGM_TMP = "ARGM-TMP"  # Temporal modifier
    ARGM_MNR = "ARGM-MNR"  # Manner modifier
    ARGM_CAU = "ARGM-CAU"  # Cause modifier
    ARGM_PRP = "ARGM-PRP"  # Purpose modifier
    ARGM_DIR = "ARGM-DIR"  # Direction modifier
    ARGM_EXT = "ARGM-EXT"  # Extent modifier
    ARGM_DIS = "ARGM-DIS"  # Discourse modifier
    ARGM_ADV = "ARGM-ADV"  # Adverbial modifier
    ARGM_NEG = "ARGM-NEG"  # Negation modifier

    # Presupposition edges
    PRESUPPOSES = "presupposes"  # Statement presupposes this
    TRIGGERED_BY = "triggered_by"  # Presupposition triggered by this word

    # Commonsense inference edges (ATOMIC relations)
    X_INTENT = "xIntent"
    X_EFFECT = "xEffect"
    X_REACT = "xReact"
    X_ATTR = "xAttr"
    X_NEED = "xNeed"
    X_WANT = "xWant"
    O_REACT = "oReact"
    O_WANT = "oWant"
    O_EFFECT = "oEffect"
    CAUSES = "Causes"
    IS_BEFORE = "isBefore"
    IS_AFTER = "isAfter"
    HINDERED_BY = "HinderedBy"

    # Temporal edges
    HAPPENED_AT = "happened_at"
    HAS_DURATION = "has_duration"
    HAS_FREQUENCY = "has_frequency"
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"

    # Type/classification edges
    HAS_SENSE = "has_sense"  # Entity has this WordNet sense
    HAS_TYPE = "has_type"  # Entity has this EntityType
    INSTANCE_OF = "instance_of"  # Instance of a class

    # Cross-reference edges
    REFERS_TO = "refers_to"  # Anaphora resolution
    SAME_AS = "same_as"  # Coreference
    PRIOR_INSTANCE = "prior_instance"  # Reference to prior knowledge

    # Interpretation edges
    INTERPRETS_AS = "interprets_as"  # In this branch, X is interpreted as Y

    # WSD edges
    DISAMBIGUATED_TO = "disambiguated_to"  # Word disambiguated to synset
    ALTERNATIVE_SENSE = "alternative_sense"  # Alternative WSD interpretation


# =============================================================================
# Builder Configuration
# =============================================================================


@dataclass
class GraphBuilderConfig:
    """Configuration for the graph builder.

    Attributes:
        include_presuppositions: Store presuppositions as nodes
        include_inferences: Store commonsense inferences as edges
        include_temporal: Store temporal info (nodes or properties)
        include_modality: Store modality as edge properties
        include_negation: Store negation as edge properties
        include_wsd_alternatives: Store alternative WSD senses
        include_branches: Store interpretation branches
        min_confidence: Minimum confidence to include (0.0-1.0)
        temporal_as_nodes: Create TIMEPOINT nodes vs just properties
        presupposition_as_nodes: Create PRESUPPOSITION nodes vs edges only
        inference_as_edges: Create inference edges vs nodes
    """

    include_presuppositions: bool = True
    include_inferences: bool = True
    include_temporal: bool = True
    include_modality: bool = True
    include_negation: bool = True
    include_wsd_alternatives: bool = True
    include_branches: bool = True
    min_confidence: float = 0.0  # Include everything by default
    temporal_as_nodes: bool = True  # Create TIMEPOINT nodes
    presupposition_as_nodes: bool = True  # Create PRESUPPOSITION nodes
    inference_as_edges: bool = True  # Inferences as edges (not nodes)


# =============================================================================
# Build Result
# =============================================================================


@dataclass
class GraphBuildResult:
    """Result of building a graph from decomposition output.

    Attributes:
        graph: The built semantic graph
        source_id: Decomposition source ID for provenance
        entity_node_map: Mapping from entity text → node_id
        predicate_node_map: Mapping from predicate text → node_id
        synset_node_map: Mapping from synset_id → node_id
        stats: Statistics about what was built
        errors: Any errors encountered during building
    """

    graph: SemanticGraph
    source_id: str
    entity_node_map: dict[str, str] = field(default_factory=dict)
    predicate_node_map: dict[str, str] = field(default_factory=dict)
    synset_node_map: dict[str, str] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Graph Builder
# =============================================================================


class GraphBuilder:
    """Builds semantic graphs from Phase 0/1 decomposition output.

    The builder takes an IntegratedResult (or components thereof) and
    constructs a semantic graph with:

    - Entity nodes for all identified entities
    - Predicate/Event nodes for verbs and actions
    - Concept nodes for WordNet synsets
    - Presupposition nodes for implicit assumptions
    - Rich edges capturing all relationships with confidence scores

    Usage:
        >>> builder = GraphBuilder()
        >>> result = builder.build(integrated_result)
        >>> graph = result.graph
        >>> # Query the graph
        >>> doug = graph.find_node("Doug")
        >>> edges = graph.get_outgoing_edges(doug.node_id)
    """

    def __init__(self, config: GraphBuilderConfig | None = None):
        """Initialize the graph builder.

        Args:
            config: Builder configuration
        """
        self.config = config or GraphBuilderConfig()

    def build_from_integrated(
        self,
        result: "IntegratedResult",
        existing_graph: SemanticGraph | None = None,
    ) -> GraphBuildResult:
        """Build a graph from IntegratedResult (Phase 0 + Phase 1).

        This is the main entry point for building from pipeline output.

        Args:
            result: Complete output from IntegratedPipeline
            existing_graph: Optional existing graph to merge into

        Returns:
            GraphBuildResult with the built graph and metadata
        """
        graph = existing_graph or SemanticGraph()
        source_id = result.decomposition.source_id

        build_result = GraphBuildResult(
            graph=graph,
            source_id=source_id,
            stats={
                "entities_created": 0,
                "predicates_created": 0,
                "presuppositions_created": 0,
                "inferences_created": 0,
                "edges_created": 0,
                "wsd_senses_created": 0,
            },
        )

        try:
            # Step 1: Build entity nodes from Phase 0
            self._build_entities(result.phase0, graph, build_result, source_id)

            # Step 2: Build WSD/synset nodes
            self._build_wsd_nodes(result.phase0, graph, build_result, source_id)

            # Step 3: Build from decomposition (Phase 1)
            self._build_from_decomposition(
                result.decomposition, graph, build_result, source_id
            )

        except Exception as e:
            logger.exception(f"Error building graph: {e}")
            build_result.errors.append(str(e))

        return build_result

    def build_from_decomposition(
        self,
        decomposition: DecomposedKnowledge,
        existing_graph: SemanticGraph | None = None,
    ) -> GraphBuildResult:
        """Build a graph from DecomposedKnowledge (Phase 1 only).

        Use this when you only have Phase 1 output without Phase 0.

        Args:
            decomposition: Phase 1 decomposition output
            existing_graph: Optional existing graph to merge into

        Returns:
            GraphBuildResult with the built graph
        """
        graph = existing_graph or SemanticGraph()
        source_id = decomposition.source_id

        build_result = GraphBuildResult(
            graph=graph,
            source_id=source_id,
            stats={
                "entities_created": 0,
                "predicates_created": 0,
                "presuppositions_created": 0,
                "inferences_created": 0,
                "edges_created": 0,
            },
        )

        try:
            self._build_from_decomposition(decomposition, graph, build_result, source_id)
        except Exception as e:
            logger.exception(f"Error building graph: {e}")
            build_result.errors.append(str(e))

        return build_result

    # =========================================================================
    # Phase 0 Building
    # =========================================================================

    def _build_entities(
        self,
        phase0: "Phase0Result",
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
    ) -> None:
        """Build INSTANCE nodes from Phase 0 entity identifiers.

        Every entity mentioned in text is an INSTANCE node - even generic
        references like "cats" (in "cats are cute"). The type information
        (CLASS) is handled by synset nodes linked via HAS_SENSE edges.

        This follows the RDF pattern where:
        - Text mentions → INSTANCE nodes (the specific mention)
        - Synsets/types → CLASS nodes (the abstract concept)
        - HAS_SENSE edges → link instances to their class
        """
        for entity_text, identifier in phase0.entity_identifiers.items():
            # Get classification for additional metadata
            classification = phase0.entity_classifications.get(entity_text)

            # Build properties - store entity_type as metadata, not as node type
            properties: dict[str, Any] = {
                "entity_type": identifier.entity_type.value,
                "source_text": entity_text,
            }

            if identifier.aliases:
                properties["aliases"] = identifier.aliases

            if identifier.domain:
                properties["domain"] = identifier.domain

            if identifier.hypernym_chain:
                properties["hypernym_chain"] = identifier.hypernym_chain

            if identifier.role_relation:
                properties["role_relation"] = identifier.role_relation

            if identifier.anchor_entity_id:
                properties["anchor_entity_id"] = identifier.anchor_entity_id

            if classification:
                properties["classification_method"] = classification.method
                properties["classification_evidence"] = classification.evidence

            # Store synset_id in properties for reference, but don't set on node
            # The synset_id on nodes is reserved for CLASS nodes (synset nodes)
            if identifier.wordnet_synset:
                properties["wordnet_synset"] = identifier.wordnet_synset

            # ALL entity mentions from text are INSTANCE nodes
            # Even "cats" in "cats are cute" is an instance of the class cat.n.01
            # The HAS_SENSE edge will link this instance to its class
            node = graph.create_node(
                canonical_name=identifier.canonical_name or entity_text,
                node_type=NodeType.INSTANCE,  # Always INSTANCE for text mentions
                entity_type=identifier.entity_type,
                properties=properties,
                synset_id=None,  # Don't set synset_id on INSTANCE nodes
                wikidata_qid=identifier.wikidata_qid,
                confidence=identifier.confidence,
                source_id=source_id,
            )

            result.entity_node_map[entity_text] = node.node_id
            result.stats["entities_created"] += 1

    def _build_wsd_nodes(
        self,
        phase0: "Phase0Result",
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
    ) -> None:
        """Build CLASS nodes for WSD results and link INSTANCE nodes to them.

        Creates CLASS nodes for each disambiguated synset (deduplicated) and
        establishes the proper ontological relationships:

        1. HAS_SENSE: Links a word mention to its disambiguated sense
           "cat" (word) --HAS_SENSE--> cat.n.01 (synset)

        2. INSTANCE_OF: Links specific individuals to their class (for INSTANCE entities)
           Whiskers --INSTANCE_OF--> cat.n.01

        The distinction matters for LLM context:
        - HAS_SENSE is linguistic (word sense disambiguation)
        - INSTANCE_OF is ontological (type hierarchy)

        IMPORTANT: We deduplicate synsets by checking:
        1. The synset_node_map (for this build operation)
        2. The existing graph (for previously stored synsets)

        This ensures "cat.n.01" is stored once globally.
        """
        for word, disambiguation in phase0.disambiguation_results.items():
            synset_id = disambiguation.synset_id
            if not synset_id:
                continue

            # Get or create the CLASS node for this synset (deduplicated)
            synset_node_id = self._get_or_create_synset_node(
                graph=graph,
                result=result,
                synset_id=synset_id,
                word=word,
                definition=getattr(disambiguation, 'definition', ''),
                method=disambiguation.method,
                confidence=disambiguation.confidence,
                source_id=source_id,
                is_alternative=False,
            )

            # Link entity to synset if entity exists
            if word in result.entity_node_map:
                entity_node_id = result.entity_node_map[word]
                entity_node = graph.get_node(entity_node_id)

                # Get the entity type to determine which edge to create
                entity_type_str = entity_node.properties.get("entity_type", "")

                # For INSTANCE entities, create INSTANCE_OF edge (ontological)
                # For CLASS/GENERIC entities, create HAS_SENSE edge (linguistic)
                if entity_type_str == "instance":
                    # This is a specific individual - use INSTANCE_OF
                    existing_edges = graph.get_edges_between(
                        entity_node_id, synset_node_id, SemanticEdgeType.INSTANCE_OF
                    )
                    if not existing_edges:
                        graph.create_edge(
                            entity_node_id,
                            synset_node_id,
                            SemanticEdgeType.INSTANCE_OF,
                            properties={"primary": True},
                            confidence=disambiguation.confidence,
                            source_decomposition_id=source_id,
                        )
                        result.stats["edges_created"] += 1
                else:
                    # This is a class reference or generic - use HAS_SENSE
                    existing_edges = graph.get_edges_between(
                        entity_node_id, synset_node_id, SemanticEdgeType.HAS_SENSE
                    )
                    if not existing_edges:
                        graph.create_edge(
                            entity_node_id,
                            synset_node_id,
                            SemanticEdgeType.HAS_SENSE,
                            properties={"primary": True},
                            confidence=disambiguation.confidence,
                            source_decomposition_id=source_id,
                        )
                        result.stats["edges_created"] += 1

            # Create alternative sense nodes if configured
            if self.config.include_wsd_alternatives and disambiguation.alternatives:
                # Distribute remaining confidence
                remaining_conf = max(0.0, 1.0 - disambiguation.confidence)
                num_alts = len(disambiguation.alternatives)
                alt_conf = remaining_conf / num_alts if num_alts > 0 else 0.0

                for alt_synset_id in disambiguation.alternatives:
                    if alt_synset_id == synset_id:
                        continue

                    alt_node_id = self._get_or_create_synset_node(
                        graph=graph,
                        result=result,
                        synset_id=alt_synset_id,
                        word=word,
                        definition="",
                        method=disambiguation.method,
                        confidence=alt_conf,
                        source_id=source_id,
                        is_alternative=True,
                    )

                    # Link entity to alternative sense
                    if word in result.entity_node_map:
                        entity_node_id = result.entity_node_map[word]
                        existing_edges = graph.get_edges_between(
                            entity_node_id, alt_node_id, SemanticEdgeType.ALTERNATIVE_SENSE
                        )
                        if not existing_edges:
                            graph.create_edge(
                                entity_node_id,
                                alt_node_id,
                                SemanticEdgeType.ALTERNATIVE_SENSE,
                                properties={"primary": False},
                                confidence=alt_conf,
                                source_decomposition_id=source_id,
                            )
                            result.stats["edges_created"] += 1

    def _get_or_create_synset_node(
        self,
        graph: SemanticGraph,
        result: GraphBuildResult,
        synset_id: str,
        word: str,
        definition: str,
        method: str,
        confidence: float,
        source_id: str,
        is_alternative: bool,
    ) -> str:
        """Get existing synset node or create a new one.

        This ensures synsets are deduplicated - we only have ONE node
        for "cat.n.01" even if multiple entities link to it.

        Args:
            graph: The semantic graph
            result: Build result with synset_node_map
            synset_id: WordNet synset ID (e.g., "cat.n.01")
            word: The word this synset came from
            definition: Synset definition
            method: Disambiguation method used
            confidence: Confidence in this sense
            source_id: Decomposition source ID
            is_alternative: Whether this is an alternative sense

        Returns:
            Node ID of the synset (existing or newly created)
        """
        # 1. Check our local map first (for this build operation)
        if synset_id in result.synset_node_map:
            # Update the existing node with this source
            existing_node = graph.get_node(result.synset_node_map[synset_id])
            if existing_node:
                existing_node.add_source(source_id)
                # Track additional words that map to this synset
                existing_words = existing_node.properties.get("words", [])
                if word not in existing_words:
                    existing_words.append(word)
                    existing_node.add_property("words", existing_words)
            return result.synset_node_map[synset_id]

        # 2. Check if synset already exists in the graph (from prior builds)
        existing_nodes = graph.find_nodes_by_synset(synset_id)
        if existing_nodes:
            # Reuse existing synset node
            existing_node = existing_nodes[0]
            existing_node.add_source(source_id)
            # Track additional words
            existing_words = existing_node.properties.get("words", [])
            if word not in existing_words:
                existing_words.append(word)
                existing_node.add_property("words", existing_words)
            result.synset_node_map[synset_id] = existing_node.node_id
            return existing_node.node_id

        # 3. Create new synset node (CLASS type - represents the abstract concept)
        synset_node = graph.create_node(
            canonical_name=synset_id,
            node_type=NodeType.CLASS,  # Synsets are CLASS nodes, not INSTANCE
            properties={
                "synset_id": synset_id,
                "definition": definition,
                "words": [word],  # Track all words that map to this synset
                "disambiguation_method": method,
                "is_alternative": is_alternative,
            },
            synset_id=synset_id,
            confidence=confidence,
            source_id=source_id,
        )
        result.synset_node_map[synset_id] = synset_node.node_id
        result.stats["wsd_senses_created"] += 1
        return synset_node.node_id

    # =========================================================================
    # Phase 1 Building
    # =========================================================================

    def _build_from_decomposition(
        self,
        decomposition: DecomposedKnowledge,
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
    ) -> None:
        """Build graph elements from Phase 1 decomposition.

        Handles:
        - Semantic roles (predicate-argument edges)
        - Presuppositions (nodes + edges)
        - Commonsense inferences (edges)
        - Temporal info
        - Modality
        - Negation
        - Cross-references
        - Weighted branches
        """
        # Create source text node as anchor
        source_node = graph.create_node(
            canonical_name=f"source:{source_id[:8]}",
            node_type=NodeType.EVENT,
            properties={
                "source_text": decomposition.source_text,
                "pipeline_version": decomposition.pipeline_version,
            },
            confidence=1.0,
            source_id=source_id,
        )

        # Build semantic roles
        self._build_semantic_roles(
            decomposition.semantic_roles, graph, result, source_id, source_node.node_id
        )

        # Build presuppositions
        if self.config.include_presuppositions:
            self._build_presuppositions(
                decomposition.presuppositions, graph, result, source_id, source_node.node_id
            )

        # Build commonsense inferences
        if self.config.include_inferences:
            self._build_inferences(
                decomposition.commonsense_inferences, graph, result, source_id, source_node.node_id
            )

        # Build temporal info
        if self.config.include_temporal and decomposition.temporal:
            self._build_temporal(
                decomposition.temporal, graph, result, source_id, source_node.node_id
            )

        # Build modality info (as properties on source node)
        if self.config.include_modality and decomposition.modality:
            self._add_modality_properties(
                decomposition.modality, source_node
            )

        # Build negation info
        if self.config.include_negation and decomposition.negation:
            self._add_negation_properties(
                decomposition.negation, source_node
            )

        # Build cross-references
        if decomposition.cross_references:
            self._build_cross_references(
                decomposition.cross_references, graph, result, source_id, source_node.node_id
            )

        # Build weighted branches
        if self.config.include_branches and decomposition.branches:
            self._build_branches(
                decomposition.branches, graph, result, source_id, source_node.node_id
            )

    def _build_semantic_roles(
        self,
        roles: list[SemanticRole],
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
        source_node_id: str,
    ) -> None:
        """Build predicate nodes and argument edges from semantic roles.

        Creates:
        - PREDICATE nodes for each unique predicate
        - ARGn edges from predicate to argument entities
        """
        for role in roles:
            if role.confidence < self.config.min_confidence:
                continue

            # Get or create predicate node
            predicate_key = f"{role.predicate}:{role.predicate_sense or 'unknown'}"
            if predicate_key not in result.predicate_node_map:
                pred_node = graph.create_node(
                    canonical_name=role.predicate,
                    node_type=NodeType.EVENT,
                    properties={
                        "predicate": role.predicate,
                        "sense": role.predicate_sense,
                    },
                    synset_id=role.predicate_sense,
                    confidence=role.confidence,
                    source_id=source_id,
                )
                result.predicate_node_map[predicate_key] = pred_node.node_id
                result.stats["predicates_created"] += 1

            predicate_node_id = result.predicate_node_map[predicate_key]

            # Link source to predicate
            graph.create_edge(
                source_node_id,
                predicate_node_id,
                "has_predicate",
                confidence=role.confidence,
                source_decomposition_id=source_id,
            )
            result.stats["edges_created"] += 1

            # Find or create filler node
            filler_node_id = result.entity_node_map.get(role.filler)
            if not filler_node_id:
                # Create an instance node for this filler (specific entity from text)
                filler_node = graph.create_node(
                    canonical_name=role.filler,
                    node_type=NodeType.INSTANCE,  # Semantic role fillers are instances
                    properties={
                        "from_role": True,
                        "role": role.role,
                    },
                    confidence=role.confidence,
                    source_id=source_id,
                )
                filler_node_id = filler_node.node_id
                result.entity_node_map[role.filler] = filler_node_id
                result.stats["entities_created"] += 1

            # Create edge from predicate to filler with role type
            edge_props: dict[str, Any] = {}
            if role.span:
                edge_props["span"] = list(role.span)

            graph.create_edge(
                predicate_node_id,
                filler_node_id,
                role.role,  # ARG0, ARG1, ARGM-LOC, etc.
                properties=edge_props,
                confidence=role.confidence,
                source_decomposition_id=source_id,
            )
            result.stats["edges_created"] += 1

    def _build_presuppositions(
        self,
        presuppositions: list[Presupposition],
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
        source_node_id: str,
    ) -> None:
        """Build presupposition nodes and edges.

        For each presupposition, creates:
        - A PRESUPPOSITION node (if configured)
        - Edge from source to presupposition
        - Edge from trigger word to presupposition
        """
        for presup in presuppositions:
            if presup.confidence < self.config.min_confidence:
                continue

            if self.config.presupposition_as_nodes:
                # Create presupposition node (CLASS type - abstract proposition)
                presup_node = graph.create_node(
                    canonical_name=presup.content[:50],  # Truncate for display
                    node_type=NodeType.CLASS,  # Presuppositions are abstract propositions
                    properties={
                        "content": presup.content,
                        "trigger_type": presup.trigger_type.value,
                        "trigger_text": presup.trigger_text,
                        "cancellable": presup.cancellable,
                        "is_presupposition": True,
                    },
                    confidence=presup.confidence,
                    source_id=source_id,
                )
                result.stats["presuppositions_created"] += 1

                # Link source to presupposition
                graph.create_edge(
                    source_node_id,
                    presup_node.node_id,
                    SemanticEdgeType.PRESUPPOSES,
                    properties={
                        "trigger_type": presup.trigger_type.value,
                        "trigger_text": presup.trigger_text,
                    },
                    confidence=presup.confidence,
                    source_decomposition_id=source_id,
                )
                result.stats["edges_created"] += 1

                # Link trigger entity to presupposition if trigger is an entity
                trigger_node_id = result.entity_node_map.get(presup.trigger_text)
                if trigger_node_id:
                    graph.create_edge(
                        trigger_node_id,
                        presup_node.node_id,
                        SemanticEdgeType.TRIGGERED_BY,
                        confidence=presup.confidence,
                        source_decomposition_id=source_id,
                    )
                    result.stats["edges_created"] += 1

                # Link presupposition to any mentioned entities
                for entity_id in presup.entity_ids:
                    entity_node_id = result.entity_node_map.get(entity_id)
                    if entity_node_id:
                        graph.create_edge(
                            presup_node.node_id,
                            entity_node_id,
                            "mentions",
                            confidence=presup.confidence,
                            source_decomposition_id=source_id,
                        )
                        result.stats["edges_created"] += 1

    def _build_inferences(
        self,
        inferences: list[CommonsenseInference],
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
        source_node_id: str,
    ) -> None:
        """Build commonsense inference edges.

        For each ATOMIC-style inference, creates an edge from head to tail
        with the relation type (xIntent, xEffect, etc.).
        """
        for inference in inferences:
            if inference.confidence < self.config.min_confidence:
                continue

            # Find head node (usually a predicate or event)
            head_node_id = None
            for entity_id in inference.head_entity_ids:
                if entity_id in result.entity_node_map:
                    head_node_id = result.entity_node_map[entity_id]
                    break

            if not head_node_id:
                # Try to find by predicate name
                head_node_id = result.predicate_node_map.get(f"{inference.head}:unknown")

            if not head_node_id:
                # Use source node as head
                head_node_id = source_node_id

            # Create tail node (the inferred content - abstract proposition)
            tail_node = graph.create_node(
                canonical_name=inference.tail[:50],
                node_type=NodeType.CLASS,  # Inferences are abstract concepts
                properties={
                    "content": inference.tail,
                    "inference_type": inference.relation.value,
                    "source": inference.source,
                    "is_inference": True,
                },
                confidence=inference.confidence,
                source_id=source_id,
            )
            result.stats["inferences_created"] += 1

            # Create edge with the ATOMIC relation type
            graph.create_edge(
                head_node_id,
                tail_node.node_id,
                inference.relation.value,  # xIntent, xEffect, etc.
                properties={
                    "head": inference.head,
                    "inference_source": inference.source,
                },
                confidence=inference.confidence,
                source_decomposition_id=source_id,
            )
            result.stats["edges_created"] += 1

    def _build_temporal(
        self,
        temporal: TemporalInfo,
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
        source_node_id: str,
    ) -> None:
        """Build temporal information nodes/edges.

        Creates TIMEPOINT nodes and links events to them.
        Also adds temporal properties to the source node.
        """
        # Add temporal properties to source node
        source_node = graph.get_node(source_node_id)
        if source_node:
            source_node.add_property("aspect", temporal.aspect.value)
            source_node.add_property("tense", temporal.tense.value)

        if self.config.temporal_as_nodes:
            # Create timepoint node if there's a reference
            if temporal.reference_value:
                time_node = graph.create_node(
                    canonical_name=temporal.reference_value,
                    node_type=NodeType.ATTRIBUTE,
                    properties={
                        "reference_type": temporal.reference_type,
                        "reference_value": temporal.reference_value,
                        "is_temporal": True,
                    },
                    confidence=temporal.confidence,
                    source_id=source_id,
                )

                graph.create_edge(
                    source_node_id,
                    time_node.node_id,
                    SemanticEdgeType.HAPPENED_AT,
                    confidence=temporal.confidence,
                    source_decomposition_id=source_id,
                )
                result.stats["edges_created"] += 1

            # Create duration node
            if temporal.duration:
                duration_node = graph.create_node(
                    canonical_name=temporal.duration,
                    node_type=NodeType.ATTRIBUTE,
                    properties={
                        "duration": temporal.duration,
                        "is_duration": True,
                    },
                    confidence=temporal.confidence,
                    source_id=source_id,
                )

                graph.create_edge(
                    source_node_id,
                    duration_node.node_id,
                    SemanticEdgeType.HAS_DURATION,
                    confidence=temporal.confidence,
                    source_decomposition_id=source_id,
                )
                result.stats["edges_created"] += 1

            # Create frequency node
            if temporal.frequency:
                freq_node = graph.create_node(
                    canonical_name=temporal.frequency,
                    node_type=NodeType.ATTRIBUTE,
                    properties={
                        "frequency": temporal.frequency,
                        "is_frequency": True,
                    },
                    confidence=temporal.confidence,
                    source_id=source_id,
                )

                graph.create_edge(
                    source_node_id,
                    freq_node.node_id,
                    SemanticEdgeType.HAS_FREQUENCY,
                    confidence=temporal.confidence,
                    source_decomposition_id=source_id,
                )
                result.stats["edges_created"] += 1

    def _add_modality_properties(
        self,
        modality: ModalityInfo,
        node: GraphNode,
    ) -> None:
        """Add modality information as node properties."""
        node.add_property("modal_type", modality.modal_type.value)
        if modality.modal_marker:
            node.add_property("modal_marker", modality.modal_marker)
        if modality.certainty is not None:
            node.add_property("certainty", modality.certainty)
        if modality.evidence_source:
            node.add_property("evidence_source", modality.evidence_source)

    def _add_negation_properties(
        self,
        negation: NegationInfo,
        node: GraphNode,
    ) -> None:
        """Add negation information as node properties."""
        node.add_property("is_negated", negation.is_negated)
        node.add_property("polarity", negation.polarity.value)
        if negation.negation_cue:
            node.add_property("negation_cue", negation.negation_cue)
        if negation.negation_scope:
            node.add_property("negation_scope", negation.negation_scope)

    def _build_cross_references(
        self,
        cross_refs: list[CrossReference],
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
        source_node_id: str,
    ) -> None:
        """Build cross-reference edges to prior knowledge."""
        for ref in cross_refs:
            if ref.confidence < self.config.min_confidence:
                continue

            # Create edge to the referenced memory item
            # Note: The reference_id may not exist in this graph - it references
            # external memory. We create an edge anyway for provenance.
            graph.create_edge(
                source_node_id,
                ref.reference_id,  # May be external ID
                ref.reference_type,  # prior_instance, resolves_to, etc.
                properties={
                    "memory_layer": ref.memory_layer,
                    "is_cross_reference": True,
                },
                confidence=ref.confidence,
                source_decomposition_id=source_id,
            )
            result.stats["edges_created"] += 1

    def _build_branches(
        self,
        branches: list[WeightedBranch],
        graph: SemanticGraph,
        result: GraphBuildResult,
        source_id: str,
        source_node_id: str,
    ) -> None:
        """Build interpretation branch nodes/edges.

        Each branch represents an alternative interpretation of the text.
        We create a BRANCH node and link it to entity interpretations.
        """
        for branch in branches:
            if branch.final_weight < self.config.min_confidence:
                continue

            # Create branch node (abstract interpretation concept)
            branch_node = graph.create_node(
                canonical_name=f"branch:{branch.branch_id[:8]}",
                node_type=NodeType.CLASS,  # Interpretations are abstract concepts
                properties={
                    "branch_id": branch.branch_id,
                    "interpretation": branch.interpretation,
                    "base_confidence": branch.confidence,
                    "memory_support": branch.memory_support,
                    "final_weight": branch.final_weight,
                    "supporting_evidence": branch.supporting_evidence,
                    "is_interpretation_branch": True,
                },
                confidence=branch.final_weight,
                source_id=source_id,
            )

            # Link source to branch
            graph.create_edge(
                source_node_id,
                branch_node.node_id,
                "has_interpretation",
                properties={"weight": branch.final_weight},
                confidence=branch.final_weight,
                source_decomposition_id=source_id,
            )
            result.stats["edges_created"] += 1

            # Link branch to entity interpretations
            for entity_text, interpretation in branch.entity_interpretations.items():
                entity_node_id = result.entity_node_map.get(entity_text)
                if entity_node_id:
                    # Create interpretation node for this entity in this branch
                    interp_node = graph.create_node(
                        canonical_name=interpretation[:50],
                        node_type=NodeType.CLASS,  # Interpretation is an abstract concept
                        properties={
                            "interpretation_of": entity_text,
                            "interpretation": interpretation,
                            "branch_id": branch.branch_id,
                        },
                        confidence=branch.final_weight,
                        source_id=source_id,
                    )

                    graph.create_edge(
                        branch_node.node_id,
                        interp_node.node_id,
                        SemanticEdgeType.INTERPRETS_AS,
                        properties={"entity": entity_text},
                        confidence=branch.final_weight,
                        source_decomposition_id=source_id,
                    )
                    result.stats["edges_created"] += 1

    # =========================================================================
    # Helpers
    # =========================================================================

    def _entity_type_to_node_type(self, entity_type: EntityType) -> NodeType:
        """Map EntityType to NodeType.

        EntityType.INSTANCE -> NodeType.INSTANCE (specific individuals)
        EntityType.CLASS -> NodeType.CLASS (abstract types/categories)
        EntityType.ROLE -> NodeType.INSTANCE (roles are filled by instances)
        EntityType.ANAPHORA -> NodeType.INSTANCE (pronouns refer to instances)
        EntityType.GENERIC -> NodeType.CLASS (generic references are class-level)
        """
        mapping = {
            EntityType.INSTANCE: NodeType.INSTANCE,  # Doug, Whiskers, Apple Inc.
            EntityType.CLASS: NodeType.CLASS,        # cat.n.01, meeting.n.01
            EntityType.NAMED_CONCEPT: NodeType.CLASS,  # Named abstract concepts
            EntityType.ROLE: NodeType.INSTANCE,      # "the owner", "my cat"
            EntityType.ANAPHORA: NodeType.INSTANCE,  # "he", "it", "they"
            EntityType.GENERIC: NodeType.CLASS,      # "cats" (in general)
        }
        return mapping.get(entity_type, NodeType.INSTANCE)

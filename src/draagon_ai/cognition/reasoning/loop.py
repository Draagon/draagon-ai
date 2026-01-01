"""Main reasoning loop for probabilistic graph reasoning.

This is the orchestrator that ties together:
1. Phase 0/1 extraction
2. Neo4j storage
3. Probabilistic expansion
4. Context retrieval
5. Response generation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..decomposition.graph import (
    SemanticGraph,
    GraphNode,
    GraphBuilder,
    Neo4jGraphStoreSync,
    NodeType,
)
from ..decomposition.extractors.integrated_pipeline import (
    IntegratedPipeline,
    IntegratedPipelineConfig,
    IntegratedResult,
)

from .context import RecencyWindow, ContextRetriever, RetrievedContext
from .expander import ProbabilisticExpander, ExpansionResult, InterpretationBranch
from .memory import (
    MemoryLayer,
    ContentType,
    MemoryAwareGraphStore,
    VolatileWorkingMemory,
    classify_phase1_content,
)


logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        ...


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning loop."""

    # Neo4j connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    instance_id: str = "default"

    # Expansion settings
    min_branch_probability: float = 0.1
    max_branches: int = 3

    # Context retrieval
    context_depth: int = 2
    max_context_nodes: int = 50

    # Recency window
    recency_window_size: int = 10

    # Phase 0/1 (optional - can be disabled)
    use_phase01_extraction: bool = True
    require_wordnet: bool = False  # False = graceful degradation if WordNet unavailable

    # Beam search (for future)
    beam_width: int = 2

    # Memory settings
    default_memory_layer: MemoryLayer = MemoryLayer.WORKING
    enable_memory_reinforcement: bool = True
    reinforcement_amount: float = 0.1
    volatile_memory_capacity: int = 9  # Miller's Law: 7±2


@dataclass
class ReasoningResult:
    """Result of processing a message through the reasoning loop."""

    # Input
    original_message: str

    # Extraction
    message_graph: SemanticGraph | None
    extraction_time_ms: float = 0.0

    # Expansion
    expansion: ExpansionResult | None = None
    expansion_time_ms: float = 0.0

    # Context
    retrieved_context: RetrievedContext | None = None
    retrieval_time_ms: float = 0.0

    # Best interpretation
    best_interpretation: InterpretationBranch | None = None

    # Storage
    stored_to_graph: bool = False
    nodes_added: int = 0
    edges_added: int = 0

    # Memory tracking
    memory_layer: MemoryLayer = MemoryLayer.WORKING
    nodes_reinforced: int = 0
    content_types_used: list[ContentType] = field(default_factory=list)

    # Additional metadata (e.g., IntegratedResult)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Total time
    total_time_ms: float = 0.0

    def to_summary(self) -> str:
        """Generate a summary for debugging."""
        lines = [
            f"Message: {self.original_message}",
            f"Total time: {self.total_time_ms:.1f}ms",
            "",
        ]

        if self.message_graph:
            lines.append(f"Extracted: {self.message_graph.node_count} nodes, {self.message_graph.edge_count} edges")

        if self.expansion:
            lines.append(f"Expansion: {len(self.expansion.branches)} branches ({self.expansion.ambiguity_type})")
            for b in self.expansion.branches:
                lines.append(f"  - [{b.probability:.0%}] {b.interpretation[:60]}...")

        if self.retrieved_context:
            lines.append(f"Context: {self.retrieved_context.node_count} nodes from graph")

        if self.best_interpretation:
            lines.append(f"Best: {self.best_interpretation.interpretation}")

        if self.stored_to_graph:
            lines.append(f"Stored: +{self.nodes_added} nodes, +{self.edges_added} edges (layer: {self.memory_layer.value})")

        if self.nodes_reinforced > 0:
            lines.append(f"Reinforced: {self.nodes_reinforced} nodes")

        return "\n".join(lines)


class ReasoningLoop:
    """
    Main orchestrator for probabilistic graph reasoning.

    This implements the 9-step pipeline:
    1. Extract semantic graph from message (Phase 0/1)
    2. Get recency context (recent messages)
    3. Expand into interpretation branches
    4. Deepen branches via Phase 0/1 (optional)
    5. Retrieve context from Neo4j
    6. Run beam search (future: ReAct agent)
    7. Reinforce successful paths (future)
    8. Store valuable results
    9. Generate response
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: ReasoningConfig | None = None,
    ):
        self.llm = llm
        self.config = config or ReasoningConfig()

        # Initialize components
        self.recency = RecencyWindow(window_size=self.config.recency_window_size)
        self.expander = ProbabilisticExpander(
            llm=llm,
            min_probability=self.config.min_branch_probability,
            max_branches=self.config.max_branches,
        )
        self.context_retriever = ContextRetriever(
            default_depth=self.config.context_depth,
            max_nodes=self.config.max_context_nodes,
        )

        # Neo4j store (lazy init)
        self._store: Neo4jGraphStoreSync | None = None
        self._memory_store: MemoryAwareGraphStore | None = None

        # Volatile working memory for current session
        self._volatile_memory: VolatileWorkingMemory | None = None

        # Graph builder for Phase 0/1 results
        self.graph_builder = GraphBuilder()

        # Phase 0/1 pipeline (lazy init)
        self._pipeline: IntegratedPipeline | None = None
        self._pipeline_initialized = False

    @property
    def store(self) -> Neo4jGraphStoreSync | None:
        """Lazy-init Neo4j store."""
        if self._store is None:
            try:
                self._store = Neo4jGraphStoreSync(
                    uri=self.config.neo4j_uri,
                    username=self.config.neo4j_user,
                    password=self.config.neo4j_password,
                )
                self.context_retriever.set_store(self._store)
            except Exception as e:
                logger.warning(f"Failed to connect to Neo4j: {e}")
        return self._store

    @property
    def memory_store(self) -> MemoryAwareGraphStore | None:
        """Get memory-aware graph store (lazy init)."""
        if self._memory_store is None and self._store is not None:
            self._memory_store = MemoryAwareGraphStore(
                store=self._store,
                instance_id=self.config.instance_id,
            )
        return self._memory_store

    @property
    def volatile_memory(self) -> VolatileWorkingMemory:
        """Get volatile working memory for current session."""
        if self._volatile_memory is None:
            self._volatile_memory = VolatileWorkingMemory(
                task_id=f"reasoning-{self.config.instance_id}",
                max_items=self.config.volatile_memory_capacity,
            )
        return self._volatile_memory

    @property
    def pipeline(self) -> IntegratedPipeline | None:
        """Get or create the Phase 0/1 pipeline (lazy init)."""
        if not self._pipeline_initialized:
            self._pipeline_initialized = True

            if not self.config.use_phase01_extraction:
                logger.info("Phase 0/1 extraction disabled by config")
                self._pipeline = None
                return None

            try:
                self._pipeline = IntegratedPipeline(
                    config=IntegratedPipelineConfig(
                        require_wordnet=self.config.require_wordnet,
                    ),
                    llm=self.llm,
                )
                logger.info("IntegratedPipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize IntegratedPipeline: {e}")
                self._pipeline = None
        return self._pipeline

    async def _extract_with_pipeline(self, message: str) -> tuple[SemanticGraph, IntegratedResult | None]:
        """Extract semantic graph using the Phase 0/1 pipeline.

        Returns:
            Tuple of (SemanticGraph, IntegratedResult) - result may be None if pipeline unavailable.
        """
        # Try full pipeline if available
        if self.pipeline is not None:
            try:
                integrated_result = await self.pipeline.process(message)
                # Build semantic graph from pipeline result
                graph = self.graph_builder.build_from_integrated(integrated_result)
                logger.debug(f"Pipeline extracted {graph.node_count} nodes, {graph.edge_count} edges")
                return graph, integrated_result
            except Exception as e:
                logger.warning(f"Pipeline extraction failed, falling back to simple: {e}")

        # Fallback to simple extraction
        graph = await self._simple_extraction(message)
        return graph, None

    async def process(self, message: str) -> ReasoningResult:
        """
        Process a message through the full reasoning loop.

        Args:
            message: User message to process

        Returns:
            ReasoningResult with all processing details
        """
        total_start = time.perf_counter()

        result = ReasoningResult(original_message=message, message_graph=None)

        # Step 1: Extract semantic graph via Phase 0/1 pipeline (with fallback)
        extraction_start = time.perf_counter()
        message_graph, integrated_result = await self._extract_with_pipeline(message)
        result.message_graph = message_graph
        result.extraction_time_ms = (time.perf_counter() - extraction_start) * 1000

        # Store integrated result for later use (e.g., in context enrichment)
        if integrated_result:
            result.metadata = {"integrated_result": integrated_result}  # type: ignore

        # Step 2: Recency context is already maintained
        # (recency window is updated at end of each process call)

        # Step 3: Probabilistic expansion
        expansion_start = time.perf_counter()
        expansion = await self.expander.expand(
            message=message,
            message_graph=message_graph,
            recency_context=self.recency,
        )
        result.expansion = expansion
        result.expansion_time_ms = (time.perf_counter() - expansion_start) * 1000

        # Step 4: Skip recursive deepening for MVP

        # Step 5: Context retrieval from Neo4j (only if store already connected)
        if self._store is not None and expansion.branches:
            # Get anchor nodes from top branch
            top_branch = expansion.top_branch
            if top_branch and top_branch.graph.node_count > 0:
                anchors = list(top_branch.graph.iter_nodes())[:5]
            elif message_graph and message_graph.node_count > 0:
                anchors = list(message_graph.iter_nodes())[:5]
            else:
                anchors = []

            if anchors:
                retrieval_start = time.perf_counter()
                retrieved = self.context_retriever.retrieve(
                    instance_id=self.config.instance_id,
                    anchor_nodes=anchors,
                )
                result.retrieved_context = retrieved
                result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 6: Select best interpretation (simple for MVP - just pick highest prob)
        # TODO: Implement beam search ReAct
        result.best_interpretation = expansion.top_branch

        # Step 7: Reinforce context nodes that were used
        if (self._store is not None and
            self.config.enable_memory_reinforcement and
            result.retrieved_context and
            result.retrieved_context.node_count > 0):
            try:
                # Reinforce nodes that were retrieved as context
                context_node_ids = [
                    n.node_id for n in result.retrieved_context.subgraph.iter_nodes()
                ]
                if context_node_ids and self.memory_store:
                    reinforced = self.memory_store.reinforce_nodes(
                        context_node_ids,
                        amount=self.config.reinforcement_amount,
                    )
                    result.nodes_reinforced = reinforced
                    logger.debug(f"Reinforced {reinforced} context nodes")
            except Exception as e:
                logger.warning(f"Failed to reinforce nodes: {e}")

        # Step 8: Store message graph to Neo4j with memory properties
        if self._store is not None and message_graph and message_graph.node_count > 0:
            try:
                # Classify content types for differential decay
                content_type_map = classify_phase1_content(message_graph)
                result.content_types_used = list(set(content_type_map.values()))
                result.memory_layer = self.config.default_memory_layer

                # Use memory-aware store for saving with TTL
                if self.memory_store:
                    save_result = self.memory_store.save_with_memory(
                        message_graph,
                        default_layer=self.config.default_memory_layer,
                        content_type_map=content_type_map,
                    )
                else:
                    # Fallback to basic store
                    save_result = self.store.save(
                        message_graph,
                        self.config.instance_id,
                        clear_existing=False,
                    )

                result.stored_to_graph = True
                result.nodes_added = save_result["nodes"]
                result.edges_added = save_result["edges"]
            except Exception as e:
                logger.warning(f"Failed to store to Neo4j: {e}")

        # Update recency window
        if message_graph:
            self.recency.add(message_graph)

        result.total_time_ms = (time.perf_counter() - total_start) * 1000

        return result

    async def _simple_extraction(self, message: str) -> SemanticGraph:
        """
        Simple extraction without full Phase 0/1 pipeline.

        For MVP, we just extract entities and a basic structure.
        TODO: Replace with IntegratedPipeline integration.
        """
        graph = SemanticGraph()

        # Use LLM for simple entity extraction
        prompt = f"""Extract entities and relationships from this message.

MESSAGE: {message}

Output format:
<entities>
<entity type="person|object|concept|event">Name</entity>
</entities>
<relationships>
<rel source="EntityA" type="relation" target="EntityB"/>
</relationships>"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            # Parse entities
            import re
            entity_pattern = r'<entity type="(\w+)">([^<]+)</entity>'
            entities = {}
            for match in re.finditer(entity_pattern, response):
                etype, name = match.groups()
                node_type = {
                    "person": NodeType.INSTANCE,
                    "object": NodeType.INSTANCE,
                    "concept": NodeType.CLASS,
                    "event": NodeType.EVENT,
                }.get(etype, NodeType.INSTANCE)

                node = graph.create_node(name.strip(), node_type)
                entities[name.strip().lower()] = node

            # Parse relationships
            rel_pattern = r'<rel source="([^"]+)" type="([^"]+)" target="([^"]+)"/>'
            for match in re.finditer(rel_pattern, response):
                source, rel_type, target = match.groups()
                source_node = entities.get(source.strip().lower())
                target_node = entities.get(target.strip().lower())

                if source_node and target_node:
                    graph.create_edge(source_node.node_id, target_node.node_id, rel_type)

        except Exception as e:
            logger.warning(f"Simple extraction failed: {e}")
            # Create minimal graph with just the raw message
            graph.create_node(message[:50], NodeType.INSTANCE, properties={"raw_message": message})

        return graph

    def close(self) -> None:
        """Close connections."""
        if self._store:
            self._store.close()
            self._store = None


# Convenience function for quick testing
async def quick_test(message: str, llm: LLMProvider) -> ReasoningResult:
    """Quick test of the reasoning loop."""
    loop = ReasoningLoop(llm=llm)
    try:
        result = await loop.process(message)
        print(result.to_summary())
        return result
    finally:
        loop.close()

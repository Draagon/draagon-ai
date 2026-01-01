"""Semantic Context Service for enriching agent decisions.

This module bridges the ReasoningLoop with the Agent/DecisionEngine,
providing semantic context from the knowledge graph to inform decisions.

Usage:
    from draagon_ai.orchestration import SemanticContextService

    # Create with existing providers
    semantic = SemanticContextService(
        llm=my_llm,
        neo4j_uri="bolt://localhost:7687",
    )

    # Enrich a query with semantic context
    context = await semantic.enrich(
        query="When is Doug's birthday?",
        user_id="assistant",
    )

    # Use in decision making
    decision = await engine.decide(
        query=query,
        semantic_context=context.to_prompt_context(),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Any:
        ...


@dataclass
class SemanticContext:
    """Semantic context retrieved for a query.

    Contains relevant knowledge from the semantic graph that
    can be used to inform decision making.
    """

    # The original query
    query: str

    # Retrieved facts and relationships
    relevant_facts: list[str] = field(default_factory=list)
    relevant_entities: list[str] = field(default_factory=list)
    related_memories: list[str] = field(default_factory=list)

    # Graph traversal info
    context_nodes_found: int = 0
    retrieval_time_ms: float = 0.0

    # Source info
    used_semantic_graph: bool = False
    used_memory_provider: bool = False

    def to_prompt_context(self) -> str:
        """Format context for injection into LLM prompts.

        Returns:
            Formatted string suitable for prompt injection.
        """
        if not self.relevant_facts and not self.related_memories:
            return ""

        lines = ["## Relevant Context from Memory"]

        if self.relevant_facts:
            lines.append("\n### Known Facts:")
            for fact in self.relevant_facts[:10]:  # Limit to 10
                lines.append(f"- {fact}")

        if self.relevant_entities:
            lines.append("\n### Related Entities:")
            lines.append(", ".join(self.relevant_entities[:20]))

        if self.related_memories:
            lines.append("\n### Related Memories:")
            for memory in self.related_memories[:5]:  # Limit to 5
                lines.append(f"- {memory}")

        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Check if context is empty."""
        return not self.relevant_facts and not self.related_memories


@dataclass
class SemanticContextConfig:
    """Configuration for semantic context service."""

    # Neo4j connection (optional - if not set, uses memory provider only)
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    instance_id: str = "default"

    # Context retrieval settings
    max_facts: int = 10
    max_memories: int = 5
    max_entities: int = 20
    context_depth: int = 2

    # Enable/disable components
    use_reasoning_loop: bool = True
    use_memory_search: bool = True

    # Performance
    timeout_ms: float = 1000.0  # Max time for context retrieval


class SemanticContextService:
    """Service for enriching queries with semantic context.

    This service uses the ReasoningLoop and MemoryProvider to
    retrieve relevant context from the knowledge graph for
    use in agent decision making.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        memory_provider: Any | None = None,
        config: SemanticContextConfig | None = None,
    ):
        """Initialize the semantic context service.

        Args:
            llm: LLM provider for reasoning (optional).
            memory_provider: Memory provider for search (optional).
            config: Service configuration.
        """
        self._llm = llm
        self._memory = memory_provider
        self._config = config or SemanticContextConfig()

        # Lazy-init reasoning loop
        self._reasoning_loop = None
        self._loop_initialized = False

    @property
    def reasoning_loop(self):
        """Get or create the reasoning loop (lazy init)."""
        if not self._loop_initialized:
            self._loop_initialized = True
            if self._llm and self._config.use_reasoning_loop:
                try:
                    from draagon_ai.cognition.reasoning import (
                        ReasoningLoop,
                        ReasoningConfig,
                    )

                    reasoning_config = ReasoningConfig(
                        neo4j_uri=self._config.neo4j_uri or "bolt://localhost:7687",
                        neo4j_user=self._config.neo4j_user,
                        neo4j_password=self._config.neo4j_password,
                        instance_id=self._config.instance_id,
                        context_depth=self._config.context_depth,
                    )
                    self._reasoning_loop = ReasoningLoop(
                        llm=self._llm,
                        config=reasoning_config,
                    )
                    logger.info("ReasoningLoop initialized for semantic context")
                except Exception as e:
                    logger.warning(f"Failed to initialize ReasoningLoop: {e}")
                    self._reasoning_loop = None
        return self._reasoning_loop

    async def enrich(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> SemanticContext:
        """Enrich a query with semantic context.

        Args:
            query: The user query to enrich.
            user_id: Optional user ID for filtering.
            agent_id: Optional agent ID for filtering.

        Returns:
            SemanticContext with relevant facts and memories.
        """
        import time

        start = time.perf_counter()
        context = SemanticContext(query=query)

        # Try reasoning loop for graph-based context
        if self.reasoning_loop is not None:
            try:
                result = await self.reasoning_loop.process(query)

                # Extract relevant facts from retrieved context
                if result.retrieved_context:
                    context.used_semantic_graph = True
                    context.context_nodes_found = result.retrieved_context.node_count

                    # Extract entity names
                    for node in result.retrieved_context.subgraph.iter_nodes():
                        if node.name:
                            context.relevant_entities.append(node.name)

                    # Extract fact content from nodes
                    for node in result.retrieved_context.subgraph.iter_nodes():
                        if hasattr(node, 'properties') and node.properties:
                            content = node.properties.get('content')
                            if content:
                                context.relevant_facts.append(content)

            except Exception as e:
                logger.warning(f"Reasoning loop context retrieval failed: {e}")

        # Try memory provider search
        if self._memory and self._config.use_memory_search:
            try:
                results = await self._memory.search(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,
                    limit=self._config.max_memories,
                )

                context.used_memory_provider = True
                for r in results:
                    if hasattr(r, 'memory') and hasattr(r.memory, 'content'):
                        context.related_memories.append(r.memory.content)
                    elif hasattr(r, 'content'):
                        context.related_memories.append(r.content)

            except Exception as e:
                logger.warning(f"Memory search failed: {e}")

        context.retrieval_time_ms = (time.perf_counter() - start) * 1000
        return context

    def close(self) -> None:
        """Close resources."""
        if self._reasoning_loop:
            self._reasoning_loop.close()
            self._reasoning_loop = None

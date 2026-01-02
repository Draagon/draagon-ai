"""Semantic graph exploration integrated with search orchestration.

This module extends the SearchOrchestrator to use semantic graph traversal
for enhanced research capabilities:

1. Query expansion via graph relationships
2. Multi-hop context retrieval
3. Related concept discovery
4. Knowledge gap detection

Example:
    ```python
    from draagon_ai.orchestration.semantic_search import (
        SemanticSearchOrchestrator,
        SemanticSearchConfig,
    )

    orchestrator = SemanticSearchOrchestrator(
        llm=llm,
        memory=memory,
        search_tool=search_handler,
        semantic_context=semantic_service,
    )

    result = await orchestrator.research_with_graph(
        query="What is Doug's favorite programming language?",
        user_id="assistant",
    )

    # Result includes graph-expanded queries and related concepts
    print(result.answer)
    print(result.related_concepts)
    print(result.graph_context_used)
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
import asyncio
import logging

from .search_orchestration import (
    SearchOrchestrator,
    SearchConfig,
    SearchStrategy,
    SearchResult,
    SearchFinding,
)
from .semantic_context import (
    SemanticContextService,
    SemanticContext,
    SemanticContextConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GraphExpansion:
    """Expanded queries from graph traversal.

    Represents the result of using the semantic graph to expand
    a query into related concepts and deeper branches.
    """

    original_query: str
    expanded_queries: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)
    traversal_depth: int = 0
    nodes_explored: int = 0


@dataclass
class SemanticSearchResult(SearchResult):
    """Extended search result with semantic graph information.

    Inherits from SearchResult and adds graph-specific fields.
    """

    # Graph exploration
    graph_expansion: GraphExpansion | None = None
    semantic_context: SemanticContext | None = None

    # Related concepts discovered during search
    related_concepts: list[str] = field(default_factory=list)

    # Whether graph context was actually used
    graph_context_used: bool = False

    # Knowledge gaps identified (things we couldn't find)
    knowledge_gaps: list[str] = field(default_factory=list)


@dataclass
class SemanticSearchConfig(SearchConfig):
    """Configuration for semantic search orchestration.

    Extends SearchConfig with graph exploration settings.
    """

    # Graph exploration
    enable_graph_expansion: bool = True
    max_expansion_depth: int = 2
    max_related_concepts: int = 10

    # Query expansion
    expand_with_synonyms: bool = True
    expand_with_related: bool = True

    # Knowledge gap detection
    detect_knowledge_gaps: bool = True


# =============================================================================
# Semantic Search Orchestrator
# =============================================================================


class SemanticSearchOrchestrator(SearchOrchestrator):
    """Search orchestrator with semantic graph exploration.

    Extends SearchOrchestrator to use the semantic graph for:
    1. Expanding queries with related concepts
    2. Retrieving multi-hop context before searching
    3. Discovering knowledge gaps
    4. Connecting search results to existing knowledge

    Example:
        ```python
        orchestrator = SemanticSearchOrchestrator(
            llm=llm,
            memory=memory,
            search_tool=search_handler,
            semantic_context=semantic_service,
            config=SemanticSearchConfig(
                strategy=SearchStrategy.HYBRID,
                enable_graph_expansion=True,
            ),
        )

        result = await orchestrator.research_with_graph(
            query="What projects use Python?",
            user_id="user_123",
        )

        # Graph expansion found related concepts
        print(result.related_concepts)  # ["programming", "software", "libraries"]
        ```
    """

    def __init__(
        self,
        llm: Any,
        memory: Any | None = None,
        search_tool: Any | None = None,
        semantic_context: SemanticContextService | None = None,
        config: SemanticSearchConfig | None = None,
    ):
        """Initialize semantic search orchestrator.

        Args:
            llm: LLM provider
            memory: Memory provider for storing findings
            search_tool: Search tool handler
            semantic_context: Semantic context service for graph exploration
            config: Semantic search configuration
        """
        super().__init__(
            llm=llm,
            memory=memory,
            search_tool=search_tool,
            config=config or SemanticSearchConfig(),
        )
        self.semantic_context = semantic_context
        self.semantic_config = config or SemanticSearchConfig()

    async def research_with_graph(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SemanticSearchResult:
        """Execute research with semantic graph exploration.

        This method:
        1. Retrieves semantic context from the knowledge graph
        2. Expands the query using related concepts
        3. Executes search with expanded queries
        4. Identifies knowledge gaps
        5. Connects findings to existing graph

        Args:
            query: The question to research
            user_id: User making the request
            context: Additional context

        Returns:
            SemanticSearchResult with graph-enhanced findings
        """
        start_time = datetime.now()

        # Step 1: Get semantic context from graph
        semantic_ctx = None
        graph_expansion = None

        if self.semantic_context and self.semantic_config.enable_graph_expansion:
            try:
                semantic_ctx = await self.semantic_context.enrich(
                    query=query,
                    user_id=user_id,
                )

                # Step 2: Expand query using graph relationships
                graph_expansion = await self._expand_with_graph(
                    query=query,
                    semantic_context=semantic_ctx,
                )

            except Exception as e:
                logger.warning(f"Semantic context enrichment failed: {e}")

        # Step 3: Build enhanced query set
        queries_to_search = [query]
        if graph_expansion and graph_expansion.expanded_queries:
            queries_to_search.extend(graph_expansion.expanded_queries)

        # Step 4: Execute search (use parent's search logic)
        # We'll do parallel search for the expanded queries
        all_findings: list[SearchFinding] = []
        all_queries: list[str] = []

        if self.search_tool:
            # Limit parallel searches
            search_queries = queries_to_search[: self.config.max_parallel_searches]

            search_tasks = [
                self.search_tool({"query": q}) for q in search_queries
            ]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for q, result in zip(search_queries, search_results):
                if isinstance(result, Exception):
                    logger.warning(f"Search failed for '{q}': {result}")
                    continue
                findings = self._extract_findings(q, result)
                all_findings.extend(findings)
                all_queries.append(q)

        # Step 5: Synthesize answer
        answer = await self._synthesize_with_context(
            query=query,
            findings=all_findings,
            semantic_context=semantic_ctx,
        )

        # Step 6: Identify knowledge gaps
        knowledge_gaps = []
        if self.semantic_config.detect_knowledge_gaps:
            knowledge_gaps = await self._detect_knowledge_gaps(
                query=query,
                findings=all_findings,
                semantic_context=semantic_ctx,
            )

        # Build result
        result = SemanticSearchResult(
            answer=answer,
            success=len(all_findings) > 0 or bool(semantic_ctx and not semantic_ctx.is_empty()),
            findings=all_findings,
            search_queries=all_queries,
            iterations=1,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            # Semantic extensions
            graph_expansion=graph_expansion,
            semantic_context=semantic_ctx,
            related_concepts=graph_expansion.related_concepts if graph_expansion else [],
            graph_context_used=bool(semantic_ctx and not semantic_ctx.is_empty()),
            knowledge_gaps=knowledge_gaps,
        )

        # Store findings in memory
        if self.config.store_findings and self.memory:
            await self._store_findings(result, user_id)

        return result

    async def _expand_with_graph(
        self,
        query: str,
        semantic_context: SemanticContext,
    ) -> GraphExpansion:
        """Expand query using semantic graph relationships.

        Uses the semantic context to find related concepts and
        generate additional search queries.

        Args:
            query: Original query
            semantic_context: Retrieved semantic context

        Returns:
            GraphExpansion with expanded queries and related concepts
        """
        expansion = GraphExpansion(original_query=query)

        # Collect related concepts from context
        if semantic_context.relevant_entities:
            expansion.related_concepts.extend(
                semantic_context.relevant_entities[: self.semantic_config.max_related_concepts]
            )

        # Generate expanded queries using LLM
        if expansion.related_concepts:
            expanded = await self._generate_expanded_queries(
                query=query,
                related_concepts=expansion.related_concepts,
            )
            expansion.expanded_queries = expanded

        # Record graph exploration metrics
        expansion.nodes_explored = semantic_context.context_nodes_found
        expansion.traversal_depth = 2  # Default from config

        return expansion

    async def _generate_expanded_queries(
        self,
        query: str,
        related_concepts: list[str],
    ) -> list[str]:
        """Generate expanded search queries using related concepts.

        Args:
            query: Original query
            related_concepts: Related concepts from graph

        Returns:
            List of expanded queries
        """
        concepts_str = ", ".join(related_concepts[:10])

        prompt = f"""Given this query and related concepts from the knowledge graph,
generate 2-3 additional search queries that explore different angles.

Original query: {query}

Related concepts from knowledge graph:
{concepts_str}

Generate queries that:
1. Use related concepts to explore the topic more broadly
2. Look for connections between concepts
3. Fill in knowledge gaps

Output each query on a separate line, nothing else:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)

            queries = [
                line.strip()
                for line in content.strip().split("\n")
                if line.strip() and not line.strip().startswith("-")
            ]
            return queries[:3]  # Max 3 expanded queries

        except Exception as e:
            logger.warning(f"Failed to generate expanded queries: {e}")
            return []

    async def _synthesize_with_context(
        self,
        query: str,
        findings: list[SearchFinding],
        semantic_context: SemanticContext | None,
    ) -> str:
        """Synthesize answer using both search findings and semantic context.

        Args:
            query: Original query
            findings: Search findings
            semantic_context: Semantic context from graph

        Returns:
            Synthesized answer
        """
        # Build context for synthesis
        context_parts = []

        # Add semantic context if available
        if semantic_context and not semantic_context.is_empty():
            context_parts.append("## Known Information from Memory")
            context_parts.append(semantic_context.to_prompt_context())
            context_parts.append("")

        # Add search findings
        if findings:
            context_parts.append("## Search Results")
            for f in findings[:10]:
                context_parts.append(f"- {f.content}")
            context_parts.append("")

        if not context_parts:
            return "I couldn't find any relevant information for your question."

        context_text = "\n".join(context_parts)

        prompt = f"""Synthesize a comprehensive answer using both known information and search results.

Question: {query}

{context_text}

Provide a clear, accurate answer that combines what is known with what was found.
If there are contradictions, note them.

Answer:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip()
        except Exception as e:
            logger.warning(f"Failed to synthesize answer: {e}")
            if findings:
                return findings[0].content
            return "Unable to synthesize answer."

    async def _detect_knowledge_gaps(
        self,
        query: str,
        findings: list[SearchFinding],
        semantic_context: SemanticContext | None,
    ) -> list[str]:
        """Detect knowledge gaps that weren't answered.

        Args:
            query: Original query
            findings: Search findings
            semantic_context: Semantic context

        Returns:
            List of identified knowledge gaps
        """
        findings_text = "\n".join(f"- {f.content}" for f in findings[:10])
        context_text = semantic_context.to_prompt_context() if semantic_context else ""

        prompt = f"""Identify what aspects of this question remain unanswered or unclear.

Question: {query}

Known information:
{context_text}

Search results:
{findings_text}

List 1-3 specific knowledge gaps (things we still don't know).
If the question is fully answered, respond with: NONE

Knowledge gaps:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)

            if "NONE" in content.upper():
                return []

            gaps = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in content.strip().split("\n")
                if line.strip() and not line.strip().upper() == "NONE"
            ]
            return gaps[:3]

        except Exception as e:
            logger.warning(f"Failed to detect knowledge gaps: {e}")
            return []


# =============================================================================
# ReAct Integration: Graph-Enhanced ReAct Loop
# =============================================================================


class GraphEnhancedReActOrchestrator(SemanticSearchOrchestrator):
    """ReAct orchestrator with semantic graph exploration.

    Extends SemanticSearchOrchestrator to use graph context
    during each ReAct iteration, enabling:
    1. Graph-informed action selection
    2. Context-aware query refinement
    3. Multi-hop reasoning through graph relationships

    The key insight is that the semantic graph provides
    "what we already know" which informs "what we need to search for".

    Example:
        ```python
        orchestrator = GraphEnhancedReActOrchestrator(
            llm=llm,
            memory=memory,
            search_tool=search,
            semantic_context=semantic,
        )

        result = await orchestrator.research_with_react(
            query="What open source projects does Doug contribute to?",
            user_id="assistant",
            max_iterations=5,
        )

        # ReAct trace shows graph-informed reasoning
        for step in result.react_steps:
            print(f"{step.type}: {step.content}")
        ```
    """

    async def research_with_react(
        self,
        query: str,
        user_id: str,
        max_iterations: int = 5,
        context: dict[str, Any] | None = None,
    ) -> SemanticSearchResult:
        """Execute ReAct-style research with graph exploration.

        Each iteration:
        1. THOUGHT: Consider what we know (graph) vs need to find (search)
        2. ACTION: Either search, explore graph deeper, or answer
        3. OBSERVATION: Record findings and update context

        Args:
            query: The question to research
            user_id: User making the request
            max_iterations: Maximum ReAct iterations
            context: Additional context

        Returns:
            SemanticSearchResult with ReAct trace
        """
        start_time = datetime.now()

        # Initialize context
        semantic_ctx = None
        if self.semantic_context and self.semantic_config.enable_graph_expansion:
            try:
                semantic_ctx = await self.semantic_context.enrich(
                    query=query,
                    user_id=user_id,
                )
            except Exception as e:
                logger.warning(f"Initial semantic context failed: {e}")

        all_findings: list[SearchFinding] = []
        all_queries: list[str] = []
        related_concepts: list[str] = []
        iterations_used = 0

        # Accumulate observations across iterations
        observations: list[str] = []

        for iteration in range(max_iterations):
            iterations_used = iteration + 1

            # THOUGHT: What do we know vs what do we need?
            thought = await self._generate_thought(
                query=query,
                semantic_context=semantic_ctx,
                observations=observations,
                iteration=iteration,
            )

            # Check if we have enough to answer
            if thought.get("action") == "answer":
                break

            # ACTION: Search for more information
            search_query = thought.get("search_query", query)
            if search_query and self.search_tool:
                try:
                    search_result = await self.search_tool({"query": search_query})
                    findings = self._extract_findings(search_query, search_result)
                    all_findings.extend(findings)
                    all_queries.append(search_query)

                    # OBSERVATION: Record what we found
                    if findings:
                        obs = f"Search '{search_query}' found: {findings[0].content[:200]}"
                        observations.append(obs)
                    else:
                        observations.append(f"Search '{search_query}' found no results")

                except Exception as e:
                    observations.append(f"Search failed: {e}")

            # Optionally explore graph deeper
            if thought.get("explore_graph") and self.semantic_context:
                try:
                    deeper_ctx = await self.semantic_context.enrich(
                        query=thought.get("graph_query", query),
                        user_id=user_id,
                    )
                    if deeper_ctx and not deeper_ctx.is_empty():
                        related_concepts.extend(deeper_ctx.relevant_entities[:5])
                        observations.append(
                            f"Graph exploration found: {', '.join(deeper_ctx.relevant_entities[:3])}"
                        )
                except Exception as e:
                    logger.debug(f"Graph exploration failed: {e}")

        # Final synthesis
        answer = await self._synthesize_with_context(
            query=query,
            findings=all_findings,
            semantic_context=semantic_ctx,
        )

        # Detect knowledge gaps
        knowledge_gaps = []
        if self.semantic_config.detect_knowledge_gaps:
            knowledge_gaps = await self._detect_knowledge_gaps(
                query=query,
                findings=all_findings,
                semantic_context=semantic_ctx,
            )

        return SemanticSearchResult(
            answer=answer,
            success=True,
            findings=all_findings,
            search_queries=all_queries,
            iterations=iterations_used,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            semantic_context=semantic_ctx,
            related_concepts=list(set(related_concepts)),
            graph_context_used=bool(semantic_ctx and not semantic_ctx.is_empty()),
            knowledge_gaps=knowledge_gaps,
        )

    async def _generate_thought(
        self,
        query: str,
        semantic_context: SemanticContext | None,
        observations: list[str],
        iteration: int,
    ) -> dict[str, Any]:
        """Generate a thought about what to do next.

        Args:
            query: Original query
            semantic_context: Current semantic context
            observations: Observations from previous iterations
            iteration: Current iteration number

        Returns:
            Dict with action, search_query, explore_graph, etc.
        """
        context_text = ""
        if semantic_context and not semantic_context.is_empty():
            context_text = f"Known from memory:\n{semantic_context.to_prompt_context()}\n"

        obs_text = ""
        if observations:
            obs_text = "Previous observations:\n" + "\n".join(f"- {o}" for o in observations) + "\n"

        prompt = f"""You are researching: {query}

{context_text}
{obs_text}
Iteration: {iteration + 1}

Decide what to do next:
- If you have enough information to answer, output: ACTION: answer
- If you need to search, output: ACTION: search QUERY: <your search query>
- If you want to explore related concepts in memory, output: ACTION: explore QUERY: <concept to explore>

Output your decision:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip().upper()

            if "ACTION: ANSWER" in content:
                return {"action": "answer"}
            elif "ACTION: SEARCH" in content:
                # Extract query after QUERY:
                if "QUERY:" in content:
                    query_part = content.split("QUERY:", 1)[1].strip()
                    # Take first line
                    search_query = query_part.split("\n")[0].strip()
                    return {"action": "search", "search_query": search_query.lower()}
                return {"action": "search", "search_query": query}
            elif "ACTION: EXPLORE" in content:
                if "QUERY:" in content:
                    query_part = content.split("QUERY:", 1)[1].strip()
                    graph_query = query_part.split("\n")[0].strip()
                    return {"action": "explore", "explore_graph": True, "graph_query": graph_query.lower()}
                return {"action": "explore", "explore_graph": True, "graph_query": query}
            else:
                # Default: search with original query
                return {"action": "search", "search_query": query}

        except Exception as e:
            logger.warning(f"Failed to generate thought: {e}")
            return {"action": "search", "search_query": query}

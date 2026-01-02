"""Search orchestration with parallel agents and memory integration.

This module provides orchestration for search scenarios that enables:
- Parallel search paths (fork different queries, join results)
- Search result storage in memory
- Belief formation from search findings
- ReAct loop integration for multi-step research

Example:
    ```python
    from draagon_ai.orchestration.search_orchestration import (
        SearchOrchestrator,
        SearchConfig,
        SearchResult,
    )

    orchestrator = SearchOrchestrator(
        llm=llm_provider,
        memory=memory_provider,
        search_tool=search_handler,
    )

    result = await orchestrator.research(
        query="What is the capital of France?",
        user_id="user_123",
    )

    # Result includes answer + learned facts
    print(result.answer)
    print(result.learnings)
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable, Protocol, runtime_checkable
import asyncio
import logging
import uuid

from .multi_agent_orchestrator import (
    AgentSpec,
    AgentRole,
    AgentResult,
    TaskContext,
    MultiAgentOrchestrator,
    OrchestrationMode,
)
from .shared_memory import SharedWorkingMemory, SharedObservation

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> Any:
        """Send chat completion request."""
        ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory providers."""

    async def store(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        """Store content with metadata, return memory ID."""
        ...

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Search for relevant memories."""
        ...


# Type for search tool handler
SearchHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# =============================================================================
# Data Structures
# =============================================================================


class SearchStrategy(str, Enum):
    """Search execution strategy."""

    SINGLE = "single"  # One search, one answer
    ITERATIVE = "iterative"  # ReAct-style refinement
    PARALLEL = "parallel"  # Fork multiple search paths
    HYBRID = "hybrid"  # Start parallel, then iterative refinement


@dataclass
class SearchConfig:
    """Configuration for search orchestration.

    Attributes:
        strategy: How to execute searches
        max_parallel_searches: Max concurrent search queries
        max_iterations: Max ReAct iterations
        store_findings: Whether to store findings in memory
        min_confidence_to_store: Minimum confidence to store in memory
        timeout_seconds: Overall timeout for research
    """

    strategy: SearchStrategy = SearchStrategy.ITERATIVE
    max_parallel_searches: int = 3
    max_iterations: int = 5
    store_findings: bool = True
    min_confidence_to_store: float = 0.7
    timeout_seconds: float = 60.0


@dataclass
class SearchFinding:
    """A finding from search that may become a memory.

    Attributes:
        content: The factual content found
        source_query: Query that found this
        source_url: URL if available
        confidence: How confident we are in this finding
        finding_type: Type of knowledge (FACT, SKILL, etc.)
        stored_memory_id: Memory ID if stored
    """

    content: str
    source_query: str
    source_url: str | None = None
    confidence: float = 0.8
    finding_type: str = "FACT"
    stored_memory_id: str | None = None


@dataclass
class SearchResult:
    """Result from a search orchestration.

    Attributes:
        answer: Synthesized answer to the original query
        success: Whether research completed successfully
        findings: List of findings discovered
        learnings: Findings that were stored as memories
        search_queries: All queries executed
        iterations: Number of ReAct iterations used
        duration_ms: Total duration in milliseconds
    """

    answer: str
    success: bool = True
    error: str | None = None

    # Knowledge gathered
    findings: list[SearchFinding] = field(default_factory=list)
    learnings: list[SearchFinding] = field(default_factory=list)

    # Execution details
    search_queries: list[str] = field(default_factory=list)
    iterations: int = 0
    duration_ms: float = 0.0


# =============================================================================
# Search Orchestrator
# =============================================================================


class SearchOrchestrator:
    """Orchestrates search with parallel paths and memory integration.

    Provides multi-step research capabilities:
    1. Query analysis - Determine search strategy
    2. Parallel search - Fork multiple query variations
    3. Result synthesis - Combine findings
    4. Learning - Store valuable findings in memory
    5. Belief formation - Create beliefs from verified findings

    Example:
        ```python
        orchestrator = SearchOrchestrator(
            llm=llm,
            memory=memory,
            search_tool=search_handler,
            config=SearchConfig(strategy=SearchStrategy.PARALLEL),
        )

        result = await orchestrator.research(
            query="Compare Python and Rust for web development",
            user_id="user_123",
        )
        ```
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        search_tool: SearchHandler | None = None,
        config: SearchConfig | None = None,
    ):
        """Initialize search orchestrator.

        Args:
            llm: LLM provider for analysis and synthesis
            memory: Optional memory provider for storing findings
            search_tool: Search tool handler
            config: Search configuration
        """
        self.llm = llm
        self.memory = memory
        self.search_tool = search_tool
        self.config = config or SearchConfig()

        # Multi-agent orchestrator for parallel paths
        self._orchestrator = MultiAgentOrchestrator()

    async def research(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute research with configured strategy.

        Args:
            query: The question to research
            user_id: User making the request
            context: Additional context

        Returns:
            SearchResult with answer and findings
        """
        start_time = datetime.now()

        try:
            if self.config.strategy == SearchStrategy.SINGLE:
                result = await self._single_search(query, user_id, context)
            elif self.config.strategy == SearchStrategy.ITERATIVE:
                result = await self._iterative_search(query, user_id, context)
            elif self.config.strategy == SearchStrategy.PARALLEL:
                result = await self._parallel_search(query, user_id, context)
            else:  # HYBRID
                result = await self._hybrid_search(query, user_id, context)

            # Store findings in memory
            if self.config.store_findings and self.memory:
                await self._store_findings(result, user_id)

            result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return result

        except asyncio.TimeoutError:
            return SearchResult(
                answer="Research timed out. Please try a more specific question.",
                success=False,
                error="timeout",
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
        except Exception as e:
            logger.exception(f"Research failed: {e}")
            return SearchResult(
                answer=f"Research failed: {str(e)}",
                success=False,
                error=str(e),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def _single_search(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute single search and synthesize answer."""
        if not self.search_tool:
            return SearchResult(
                answer="No search tool available.",
                success=False,
                error="no_search_tool",
            )

        # Execute search
        search_result = await self.search_tool({"query": query})
        findings = self._extract_findings(query, search_result)

        # Synthesize answer
        answer = await self._synthesize_answer(query, findings)

        return SearchResult(
            answer=answer,
            success=True,
            findings=findings,
            search_queries=[query],
            iterations=1,
        )

    async def _iterative_search(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute iterative (ReAct-style) search."""
        if not self.search_tool:
            return SearchResult(
                answer="No search tool available.",
                success=False,
                error="no_search_tool",
            )

        all_findings: list[SearchFinding] = []
        search_queries: list[str] = []
        current_query = query

        for iteration in range(self.config.max_iterations):
            # Execute search
            search_result = await self.search_tool({"query": current_query})
            search_queries.append(current_query)
            findings = self._extract_findings(current_query, search_result)
            all_findings.extend(findings)

            # Check if we have enough information
            has_answer = await self._check_sufficient_info(query, all_findings)
            if has_answer:
                break

            # Generate refined query
            next_query = await self._generate_refined_query(
                original=query,
                previous_queries=search_queries,
                findings=all_findings,
            )

            if not next_query or next_query == current_query:
                break

            current_query = next_query

        # Synthesize final answer
        answer = await self._synthesize_answer(query, all_findings)

        return SearchResult(
            answer=answer,
            success=True,
            findings=all_findings,
            search_queries=search_queries,
            iterations=len(search_queries),
        )

    async def _parallel_search(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute parallel searches with different query variations."""
        if not self.search_tool:
            return SearchResult(
                answer="No search tool available.",
                success=False,
                error="no_search_tool",
            )

        # Generate query variations
        query_variations = await self._generate_query_variations(query)

        # Limit to max parallel
        query_variations = query_variations[: self.config.max_parallel_searches]

        # Execute searches in parallel
        search_tasks = [
            self.search_tool({"query": q}) for q in query_variations
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect findings from all searches
        all_findings: list[SearchFinding] = []
        successful_queries: list[str] = []

        for q, result in zip(query_variations, search_results):
            if isinstance(result, Exception):
                logger.warning(f"Search failed for '{q}': {result}")
                continue
            findings = self._extract_findings(q, result)
            all_findings.extend(findings)
            successful_queries.append(q)

        # Synthesize answer from all findings
        answer = await self._synthesize_answer(query, all_findings)

        return SearchResult(
            answer=answer,
            success=len(successful_queries) > 0,
            findings=all_findings,
            search_queries=successful_queries,
            iterations=1,
        )

    async def _hybrid_search(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Parallel search followed by iterative refinement."""
        # Start with parallel search
        parallel_result = await self._parallel_search(query, user_id, context)

        if not parallel_result.success:
            return parallel_result

        # Check if we need more info
        has_answer = await self._check_sufficient_info(query, parallel_result.findings)
        if has_answer:
            return parallel_result

        # Iterative refinement
        refined_query = await self._generate_refined_query(
            original=query,
            previous_queries=parallel_result.search_queries,
            findings=parallel_result.findings,
        )

        if refined_query and self.search_tool:
            search_result = await self.search_tool({"query": refined_query})
            additional_findings = self._extract_findings(refined_query, search_result)
            parallel_result.findings.extend(additional_findings)
            parallel_result.search_queries.append(refined_query)
            parallel_result.iterations += 1

            # Re-synthesize with additional findings
            parallel_result.answer = await self._synthesize_answer(
                query, parallel_result.findings
            )

        return parallel_result

    # =========================================================================
    # Query Generation
    # =========================================================================

    async def _generate_query_variations(self, query: str) -> list[str]:
        """Generate variations of the search query for parallel execution.

        Args:
            query: Original query

        Returns:
            List of query variations (including original)
        """
        prompt = f"""Generate 3 different search query variations for this question.
Each variation should approach the topic from a different angle.

Original question: {query}

Output each query on a separate line, nothing else:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)
            variations = [
                line.strip()
                for line in content.strip().split("\n")
                if line.strip() and not line.strip().startswith("-")
            ]
            # Always include original
            if query not in variations:
                variations.insert(0, query)
            return variations[:4]  # Max 4 variations
        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
            return [query]

    async def _generate_refined_query(
        self,
        original: str,
        previous_queries: list[str],
        findings: list[SearchFinding],
    ) -> str | None:
        """Generate a refined query based on what we've learned.

        Args:
            original: Original question
            previous_queries: Queries already tried
            findings: Findings so far

        Returns:
            Refined query or None if no refinement needed
        """
        findings_text = "\n".join(f"- {f.content}" for f in findings[:5])

        prompt = f"""Based on the original question and what we've found, generate ONE refined search query
to find additional information we still need.

Original question: {original}

Previous queries tried:
{chr(10).join(f"- {q}" for q in previous_queries)}

Findings so far:
{findings_text}

If we have enough information to answer, respond with: SUFFICIENT

Otherwise, output just the new search query, nothing else:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()

            if "SUFFICIENT" in content.upper():
                return None

            return content if content else None

        except Exception as e:
            logger.warning(f"Failed to generate refined query: {e}")
            return None

    async def _check_sufficient_info(
        self,
        query: str,
        findings: list[SearchFinding],
    ) -> bool:
        """Check if we have sufficient information to answer.

        Args:
            query: Original question
            findings: Findings so far

        Returns:
            True if we have enough info
        """
        if not findings:
            return False

        findings_text = "\n".join(f"- {f.content}" for f in findings[:10])

        prompt = f"""Do we have enough information to fully answer this question?

Question: {query}

Information found:
{findings_text}

Answer YES or NO:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)
            return "YES" in content.upper()
        except Exception:
            return len(findings) >= 3  # Fallback heuristic

    # =========================================================================
    # Finding Extraction and Synthesis
    # =========================================================================

    def _extract_findings(
        self,
        query: str,
        search_result: dict[str, Any],
    ) -> list[SearchFinding]:
        """Extract findings from search results.

        Args:
            query: Search query used
            search_result: Raw search result

        Returns:
            List of extracted findings
        """
        findings = []

        results = search_result.get("results", [])
        for result in results:
            snippet = result.get("snippet", result.get("content", ""))
            if snippet:
                findings.append(
                    SearchFinding(
                        content=snippet,
                        source_query=query,
                        source_url=result.get("url"),
                        confidence=0.8,
                        finding_type="FACT",
                    )
                )

        return findings

    async def _synthesize_answer(
        self,
        query: str,
        findings: list[SearchFinding],
    ) -> str:
        """Synthesize an answer from findings.

        Args:
            query: Original question
            findings: All findings collected

        Returns:
            Synthesized answer
        """
        if not findings:
            return "I couldn't find any relevant information for your question."

        findings_text = "\n".join(
            f"- {f.content}" for f in findings[:10]
        )

        prompt = f"""Based on the search results, provide a clear and accurate answer.

Question: {query}

Search Results:
{findings_text}

Synthesized Answer:"""

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip()
        except Exception as e:
            logger.warning(f"Failed to synthesize answer: {e}")
            # Fallback: return first finding
            return findings[0].content if findings else "Unable to synthesize answer."

    # =========================================================================
    # Memory Integration
    # =========================================================================

    async def _store_findings(
        self,
        result: SearchResult,
        user_id: str,
    ) -> None:
        """Store valuable findings in memory.

        Args:
            result: Search result with findings
            user_id: User who made the request
        """
        if not self.memory:
            return

        for finding in result.findings:
            if finding.confidence < self.config.min_confidence_to_store:
                continue

            try:
                memory_id = await self.memory.store(
                    content=finding.content,
                    metadata={
                        "memory_type": finding.finding_type,
                        "source": "search",
                        "source_query": finding.source_query,
                        "source_url": finding.source_url,
                        "user_id": user_id,
                        "confidence": finding.confidence,
                        "created_at": datetime.now().isoformat(),
                    },
                )
                finding.stored_memory_id = memory_id
                result.learnings.append(finding)
                logger.debug(
                    f"Stored search finding as memory {memory_id}: {finding.content[:50]}..."
                )
            except Exception as e:
                logger.warning(f"Failed to store finding: {e}")


# =============================================================================
# Parallel Search with Shared Memory
# =============================================================================


class ParallelSearchOrchestrator(SearchOrchestrator):
    """Search orchestrator with multi-agent coordination.

    Extends SearchOrchestrator to use the multi-agent orchestrator
    for parallel search paths with shared working memory.

    Each search path is an agent that:
    1. Executes its query variation
    2. Stores findings in shared memory
    3. Checks for conflicts with other agents' findings
    4. Contributes to belief formation

    Example:
        ```python
        orchestrator = ParallelSearchOrchestrator(
            llm=llm,
            memory=memory,
            search_tool=search_handler,
        )

        result = await orchestrator.research_with_agents(
            query="What are the pros and cons of Python vs Rust?",
            user_id="user_123",
        )

        # Result includes findings from all agents + conflict resolution
        print(result.answer)
        print(result.findings)
        ```
    """

    async def research_with_agents(
        self,
        query: str,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute research with parallel agent coordination.

        Uses multi-agent orchestrator to run multiple search agents
        in parallel, with shared working memory for coordination.

        Args:
            query: The question to research
            user_id: User making the request
            context: Additional context

        Returns:
            SearchResult with coordinated findings
        """
        start_time = datetime.now()
        task_id = str(uuid.uuid4())

        # Create shared working memory for this research task
        shared_memory = SharedWorkingMemory(task_id=task_id)

        # Generate query variations
        query_variations = await self._generate_query_variations(query)

        # Create agent specs for each query variation
        agents = [
            AgentSpec(
                agent_id=f"searcher_{i}",
                name=f"Search Agent {i}",
                role=AgentRole.RESEARCHER,
                description=f"Searches for: {q}",
                timeout_seconds=30.0,
            )
            for i, q in enumerate(query_variations)
        ]

        # Create task context with shared memory
        task_context = TaskContext(
            task_id=task_id,
            query=query,
            user_id=user_id,
        )
        task_context.working_memory["__shared__"] = shared_memory
        task_context.working_memory["query_variations"] = query_variations
        task_context.working_memory["search_tool"] = self.search_tool

        # Define agent executor
        async def search_agent_executor(
            agent: AgentSpec,
            context: TaskContext,
        ) -> AgentResult:
            """Execute a single search agent."""
            try:
                # Get this agent's query
                agent_index = int(agent.agent_id.split("_")[1])
                agent_query = context.working_memory["query_variations"][agent_index]
                search_tool = context.working_memory["search_tool"]

                if not search_tool:
                    return AgentResult(
                        agent_id=agent.agent_id,
                        success=False,
                        error="No search tool",
                    )

                # Execute search
                search_result = await search_tool({"query": agent_query})

                # Extract findings
                findings = self._extract_findings(agent_query, search_result)

                # Add findings to shared memory
                shared = context.working_memory.get("__shared__")
                if shared and isinstance(shared, SharedWorkingMemory):
                    for finding in findings:
                        await shared.add_observation(
                            content=finding.content,
                            source_agent_id=agent.agent_id,
                            attention_weight=finding.confidence,
                            is_belief_candidate=True,
                            belief_type=finding.finding_type,
                        )

                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={
                        "query": agent_query,
                        "findings": findings,
                    },
                )

            except Exception as e:
                logger.exception(f"Search agent {agent.agent_id} failed: {e}")
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=False,
                    error=str(e),
                )

        # Execute parallel orchestration
        orchestration_result = await self._orchestrator.orchestrate(
            agents=agents,
            context=task_context,
            executor=search_agent_executor,
            mode=OrchestrationMode.PARALLEL,
        )

        # Collect all findings from agent results
        all_findings: list[SearchFinding] = []
        all_queries: list[str] = []

        for agent_result in orchestration_result.agent_results:
            if agent_result.success and agent_result.output:
                findings = agent_result.output.get("findings", [])
                all_findings.extend(findings)
                all_queries.append(agent_result.output.get("query", ""))

        # Check for conflicts in shared memory
        conflicts = await shared_memory.get_conflicts()
        if conflicts:
            logger.info(f"Found {len(conflicts)} conflicts to resolve")
            # TODO: Implement conflict resolution
            # For now, just log and continue

        # Synthesize answer from all findings
        answer = await self._synthesize_answer(query, all_findings)

        # Store findings in long-term memory
        result = SearchResult(
            answer=answer,
            success=orchestration_result.success,
            error=orchestration_result.error,
            findings=all_findings,
            search_queries=all_queries,
            iterations=1,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )

        if self.config.store_findings and self.memory:
            await self._store_findings(result, user_id)

        return result

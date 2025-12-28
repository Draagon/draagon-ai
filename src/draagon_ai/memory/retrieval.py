"""Retrieval Augmentation for draagon-ai memory.

This module implements advanced retrieval patterns:
- Self-RAG: Assess retrieval quality and refine queries
- CRAG: Corrective RAG with chunk grading
- Contradiction Detection: Find and filter conflicting memories

These patterns improve RAG quality by validating and filtering
retrieved results before they're used for generation.

Usage:
    from draagon_ai.memory.retrieval import RetrievalAugmenter, RetrievalConfig

    augmenter = RetrievalAugmenter(llm_provider, memory_provider)

    # Self-RAG with quality assessment
    result = await augmenter.search_with_self_rag(
        query="What is Doug's birthday?",
        user_id="doug",
    )

    # CRAG with chunk grading
    result = await augmenter.search_with_crag(
        query="How do I restart the service?",
        user_id="doug",
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from draagon_ai.memory.base import SearchResult


logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MemorySearchProvider(Protocol):
    """Protocol for memory providers that support search."""

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by retrieval augmentation."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Generate a response from messages."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RetrievalConfig:
    """Configuration for retrieval augmentation.

    Attributes:
        self_rag_enabled: Enable Self-RAG quality assessment
        crag_enabled: Enable CRAG chunk grading
        contradiction_detection: Enable contradiction detection
        max_retries: Max query refinement retries for Self-RAG
        relevance_threshold: Score threshold for "relevant" grade
        ambiguous_threshold: Score threshold for "ambiguous" grade
        auto_invalidate: Auto-invalidate contradicting memories
    """

    self_rag_enabled: bool = True
    crag_enabled: bool = True
    contradiction_detection: bool = True
    max_retries: int = 1
    relevance_threshold: float = 0.7
    ambiguous_threshold: float = 0.4
    auto_invalidate: bool = True


@dataclass
class RelevanceAssessment:
    """Assessment of retrieval relevance.

    Attributes:
        score: Relevance score 0-1
        action: Recommended action (use, refine, search_web)
        reason: Human-readable reason
        suggestion: Refinement suggestion if action is "refine"
    """

    score: float
    action: str  # "use", "refine", "search_web"
    reason: str
    suggestion: str | None = None


@dataclass
class GradedChunk:
    """A search result with CRAG grading.

    Attributes:
        result: The original search result
        grade: Grade assigned (relevant, irrelevant, ambiguous)
        knowledge_strip: Extracted key fact if relevant
        confidence: Grading confidence 0-1
    """

    result: SearchResult
    grade: str  # "relevant", "irrelevant", "ambiguous"
    knowledge_strip: str | None = None
    confidence: float = 1.0


@dataclass
class Contradiction:
    """A detected contradiction between memories.

    Attributes:
        memory_id_1: First memory ID
        memory_id_2: Second memory ID
        content_1: First memory content
        content_2: Second memory content
        conflict_type: Type of conflict (direct, temporal, partial)
        to_invalidate: ID of memory to invalidate (older one)
        reason: Why these contradict
    """

    memory_id_1: str
    memory_id_2: str
    content_1: str
    content_2: str
    conflict_type: str
    to_invalidate: str | None
    reason: str


@dataclass
class SelfRAGResult:
    """Result from Self-RAG search.

    Attributes:
        results: Filtered search results
        assessment: Relevance assessment
        refined_query: Refined query if used
        retrieval_attempts: Number of retrieval attempts
        timings: Timing breakdown in milliseconds
        contradictions: Detected contradictions
        has_contradictions: Whether contradictions were found
    """

    results: list[SearchResult]
    assessment: RelevanceAssessment
    refined_query: str | None = None
    retrieval_attempts: int = 1
    timings: dict[str, int] = field(default_factory=dict)
    contradictions: list[Contradiction] = field(default_factory=list)
    has_contradictions: bool = False


@dataclass
class CRAGResult:
    """Result from CRAG search.

    Attributes:
        results: Filtered relevant chunks
        knowledge_strips: Extracted key facts
        grading: Grade counts (relevant, irrelevant, ambiguous)
        needs_web_search: Whether web search is recommended
        timings: Timing breakdown in milliseconds
    """

    results: list[GradedChunk]
    knowledge_strips: list[str] = field(default_factory=list)
    grading: dict[str, int] = field(default_factory=dict)
    needs_web_search: bool = False
    timings: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Prompts
# =============================================================================


ASSESS_RELEVANCE_PROMPT = """You are evaluating whether retrieved documents are relevant to answer a query.

Query: {query}

Retrieved Documents:
{documents}

Evaluate the relevance of these documents to the query. Consider:
1. Do the documents contain information that directly answers the query?
2. Are the documents about the same topic as the query?
3. Is the information current and trustworthy?

Respond in this exact format:
SCORE: [0.0-1.0 relevance score]
ACTION: [use|refine|search_web]
REASON: [brief explanation]
SUGGESTION: [if action is refine, suggest a better query]

Actions:
- use: Documents are highly relevant, use them
- refine: Documents are partially relevant, try a refined query
- search_web: Documents are not relevant, need external search"""


GRADE_CHUNKS_PROMPT = """You are grading retrieved document chunks for relevance to a query.

Query: {query}

For each document, grade it and extract the key fact if relevant.

Documents:
{documents}

Respond with one line per document in this format:
DOC[N]: GRADE=[relevant|irrelevant|ambiguous] STRIP="[key fact if relevant]"

Example:
DOC[1]: GRADE=relevant STRIP="Doug's birthday is March 15th"
DOC[2]: GRADE=irrelevant STRIP=""
DOC[3]: GRADE=ambiguous STRIP="Something about dates"

Grading criteria:
- relevant: Directly answers or helps answer the query
- irrelevant: Off-topic or wrong information
- ambiguous: Related but not clearly helpful"""


DETECT_CONTRADICTIONS_PROMPT = """You are detecting contradictions between memory items.

Memories:
{memories}

Find any contradictions where two memories provide conflicting information about the same topic.
For each contradiction, identify which memory is older and should be invalidated.

Respond in this format (one line per contradiction, or "NONE" if no contradictions):
CONFLICT: id1="{id1}" id2="{id2}" type=[direct|temporal|partial] invalidate="{id_to_invalidate}" reason="{why}"

Types:
- direct: Complete contradiction (e.g., "birthday is March 15" vs "birthday is March 16")
- temporal: Information changed over time (e.g., "password is X" vs "password changed to Y")
- partial: Partial conflict (e.g., "has 5 cats" vs "has 6 cats")

Always invalidate the OLDER memory (earlier created_at timestamp)."""


REFINE_QUERY_PROMPT = """The original query didn't retrieve relevant results. Suggest a better query.

Original query: {query}
Issue: {issue}

Suggest a refined query that might find better results. Just output the new query, nothing else."""


# =============================================================================
# Retrieval Augmenter
# =============================================================================


class RetrievalAugmenter:
    """Augmented retrieval with Self-RAG and CRAG patterns.

    This class wraps a memory provider and adds:
    - Self-RAG: Assess retrieval quality and refine queries
    - CRAG: Grade chunks and extract knowledge strips
    - Contradiction Detection: Find and filter conflicts

    Example:
        augmenter = RetrievalAugmenter(llm, memory)

        # Self-RAG search
        result = await augmenter.search_with_self_rag(
            "What is Doug's birthday?",
            user_id="doug",
        )
        print(f"Found {len(result.results)} relevant results")
        print(f"Assessment: {result.assessment.action}")

        # CRAG search
        result = await augmenter.search_with_crag(
            "How do I restart the service?",
            user_id="doug",
        )
        print(f"Knowledge strips: {result.knowledge_strips}")
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemorySearchProvider,
        config: RetrievalConfig | None = None,
    ):
        """Initialize the retrieval augmenter.

        Args:
            llm: LLM provider for assessment and grading
            memory: Memory provider for search
            config: Configuration options
        """
        self.llm = llm
        self.memory = memory
        self.config = config or RetrievalConfig()

    async def search_with_self_rag(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        detect_contradictions: bool | None = None,
        auto_invalidate: bool | None = None,
    ) -> SelfRAGResult:
        """Search with Self-RAG quality assessment.

        Implements the Self-RAG pattern:
        1. Retrieve initial results
        2. Assess relevance using LLM
        3. If partial relevance, refine query and retry
        4. Detect contradictions and filter stale memories
        5. Return results with assessment metadata

        Args:
            query: Search query
            user_id: User ID for scoping
            limit: Maximum results
            detect_contradictions: Override config for contradiction detection
            auto_invalidate: Override config for auto-invalidation

        Returns:
            SelfRAGResult with results and assessment
        """
        detect_contradictions = (
            detect_contradictions
            if detect_contradictions is not None
            else self.config.contradiction_detection
        )
        auto_invalidate = (
            auto_invalidate
            if auto_invalidate is not None
            else self.config.auto_invalidate
        )

        retrieval_attempts = 0
        current_query = query
        refined_query = None
        timings: dict[str, int] = {}

        while retrieval_attempts <= self.config.max_retries:
            retrieval_attempts += 1

            # Fetch results (over-fetch for contradiction detection)
            fetch_limit = limit * 2 if detect_contradictions else limit
            search_start = time.time()
            results = await self.memory.search(
                current_query,
                user_id=user_id,
                limit=fetch_limit,
            )
            timings["search_ms"] = int((time.time() - search_start) * 1000)

            # No results - recommend web search
            if not results:
                return SelfRAGResult(
                    results=[],
                    assessment=RelevanceAssessment(
                        score=0.0,
                        action="search_web",
                        reason="No matching documents found",
                    ),
                    refined_query=refined_query,
                    retrieval_attempts=retrieval_attempts,
                    timings=timings,
                )

            # Assess relevance
            assess_start = time.time()
            assessment = await self._assess_relevance(query, results)
            timings["assess_ms"] = int((time.time() - assess_start) * 1000)

            # If highly relevant or out of retries, finalize
            if assessment.action == "use" or retrieval_attempts > self.config.max_retries:
                # Detect contradictions
                contradictions: list[Contradiction] = []
                if detect_contradictions and len(results) >= 2:
                    contra_start = time.time()
                    contradictions = await self._detect_contradictions(results)
                    timings["contradiction_ms"] = int((time.time() - contra_start) * 1000)

                    # Filter out contradicting (older) memories
                    if contradictions:
                        to_invalidate = {c.to_invalidate for c in contradictions if c.to_invalidate}
                        results = [r for r in results if r.memory.id not in to_invalidate]

                        # Auto-invalidate if configured
                        if auto_invalidate:
                            await self._invalidate_memories(list(to_invalidate))

                return SelfRAGResult(
                    results=results[:limit],
                    assessment=assessment,
                    refined_query=refined_query,
                    retrieval_attempts=retrieval_attempts,
                    timings=timings,
                    contradictions=contradictions,
                    has_contradictions=len(contradictions) > 0,
                )

            # Refine query and retry
            if assessment.action == "refine" and retrieval_attempts <= self.config.max_retries:
                refine_start = time.time()
                refined_query = await self._refine_query(
                    query, assessment.suggestion or "Try a more specific query"
                )
                timings["refine_ms"] = int((time.time() - refine_start) * 1000)
                current_query = refined_query
                logger.info(f"Self-RAG: Refined query to '{refined_query}'")
                continue

            # Otherwise return what we have
            return SelfRAGResult(
                results=results[:limit],
                assessment=assessment,
                refined_query=refined_query,
                retrieval_attempts=retrieval_attempts,
                timings=timings,
            )

        # Should not reach here
        return SelfRAGResult(
            results=results[:limit],
            assessment=assessment,
            refined_query=refined_query,
            retrieval_attempts=retrieval_attempts,
            timings=timings,
        )

    async def search_with_crag(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> CRAGResult:
        """Search with Corrective RAG chunk grading.

        Implements the CRAG pattern:
        1. Retrieve initial results
        2. Grade each chunk (relevant/irrelevant/ambiguous)
        3. Extract knowledge strips from relevant chunks
        4. Recommend web search if ambiguous > relevant

        Args:
            query: Search query
            user_id: User ID for scoping
            limit: Maximum results

        Returns:
            CRAGResult with graded chunks and knowledge strips
        """
        timings: dict[str, int] = {}

        # Fetch results (over-fetch for filtering)
        search_start = time.time()
        results = await self.memory.search(
            query,
            user_id=user_id,
            limit=limit * 2,
        )
        timings["search_ms"] = int((time.time() - search_start) * 1000)

        if not results:
            return CRAGResult(
                results=[],
                knowledge_strips=[],
                grading={"relevant": 0, "irrelevant": 0, "ambiguous": 0},
                needs_web_search=True,
                timings=timings,
            )

        # Grade chunks
        grade_start = time.time()
        graded_chunks = await self._grade_chunks(query, results)
        timings["grade_ms"] = int((time.time() - grade_start) * 1000)

        # Filter and collect
        relevant_chunks = []
        knowledge_strips = []
        grading = {"relevant": 0, "irrelevant": 0, "ambiguous": 0}

        for chunk in graded_chunks:
            grading[chunk.grade] = grading.get(chunk.grade, 0) + 1

            if chunk.grade == "relevant":
                relevant_chunks.append(chunk)
                if chunk.knowledge_strip:
                    knowledge_strips.append(chunk.knowledge_strip)
            elif chunk.grade == "ambiguous":
                relevant_chunks.append(chunk)  # Include ambiguous too

        # Determine if web search is needed
        needs_web_search = (
            grading["ambiguous"] > grading["relevant"]
            or grading["relevant"] == 0
        )

        logger.info(
            f"CRAG: {len(results)} → {len(relevant_chunks)} chunks "
            f"(R:{grading['relevant']} I:{grading['irrelevant']} A:{grading['ambiguous']})"
        )

        return CRAGResult(
            results=relevant_chunks[:limit],
            knowledge_strips=knowledge_strips,
            grading=grading,
            needs_web_search=needs_web_search,
            timings=timings,
        )

    async def _assess_relevance(
        self,
        query: str,
        results: list[SearchResult],
    ) -> RelevanceAssessment:
        """Assess relevance of retrieved results.

        Args:
            query: Original query
            results: Retrieved search results

        Returns:
            RelevanceAssessment with score, action, and reason
        """
        # Format documents for prompt
        docs = "\n".join(
            f"[{i+1}] (score: {r.score:.2f}) {r.memory.content[:200]}"
            for i, r in enumerate(results[:5])
        )

        prompt = ASSESS_RELEVANCE_PROMPT.format(query=query, documents=docs)

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])

            # Parse response
            lines = response.strip().split("\n")
            score = 0.5
            action = "use"
            reason = "Assessment complete"
            suggestion = None

            for line in lines:
                line = line.strip()
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("ACTION:"):
                    action = line.split(":", 1)[1].strip().lower()
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
                elif line.startswith("SUGGESTION:"):
                    suggestion = line.split(":", 1)[1].strip()

            return RelevanceAssessment(
                score=score,
                action=action,
                reason=reason,
                suggestion=suggestion,
            )

        except Exception as e:
            logger.warning(f"Relevance assessment failed: {e}")
            # Fall back to score-based heuristic
            avg_score = sum(r.score for r in results) / len(results) if results else 0
            if avg_score >= self.config.relevance_threshold:
                return RelevanceAssessment(
                    score=avg_score,
                    action="use",
                    reason="High average similarity score",
                )
            elif avg_score >= self.config.ambiguous_threshold:
                return RelevanceAssessment(
                    score=avg_score,
                    action="refine",
                    reason="Moderate similarity, may need refinement",
                    suggestion="Try more specific terms",
                )
            else:
                return RelevanceAssessment(
                    score=avg_score,
                    action="search_web",
                    reason="Low similarity scores",
                )

    async def _grade_chunks(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[GradedChunk]:
        """Grade each chunk for relevance.

        Args:
            query: Original query
            results: Retrieved search results

        Returns:
            List of GradedChunk with grades and knowledge strips
        """
        # Format documents for prompt
        docs = "\n".join(
            f"[{i+1}] {r.memory.content[:300]}"
            for i, r in enumerate(results[:10])
        )

        prompt = GRADE_CHUNKS_PROMPT.format(query=query, documents=docs)

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])

            # Parse response
            graded = []
            for i, result in enumerate(results[:10]):
                # Default to score-based grading
                grade = "relevant" if result.score >= self.config.relevance_threshold else "ambiguous"
                strip = None

                # Try to find this doc in response
                doc_marker = f"DOC[{i+1}]:"
                for line in response.split("\n"):
                    if doc_marker in line:
                        if "GRADE=relevant" in line:
                            grade = "relevant"
                        elif "GRADE=irrelevant" in line:
                            grade = "irrelevant"
                        elif "GRADE=ambiguous" in line:
                            grade = "ambiguous"

                        # Extract knowledge strip
                        if 'STRIP="' in line:
                            start = line.index('STRIP="') + 7
                            end = line.rfind('"')
                            if end > start:
                                strip = line[start:end]
                        break

                graded.append(GradedChunk(
                    result=result,
                    grade=grade,
                    knowledge_strip=strip if strip and strip.strip() else None,
                ))

            return graded

        except Exception as e:
            logger.warning(f"Chunk grading failed: {e}")
            # Fall back to score-based grading
            return [
                GradedChunk(
                    result=r,
                    grade="relevant" if r.score >= self.config.relevance_threshold else "ambiguous",
                )
                for r in results[:10]
            ]

    async def _detect_contradictions(
        self,
        results: list[SearchResult],
    ) -> list[Contradiction]:
        """Detect contradictions between memories.

        Args:
            results: Search results to check for conflicts

        Returns:
            List of detected Contradiction objects
        """
        if len(results) < 2:
            return []

        # Format memories for prompt
        memories = "\n".join(
            f"ID: {r.memory.id}\n"
            f"Content: {r.memory.content}\n"
            f"Created: {r.memory.created_at.isoformat() if r.memory.created_at else 'unknown'}\n"
            for r in results[:10]
        )

        prompt = DETECT_CONTRADICTIONS_PROMPT.format(memories=memories)

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])

            if "NONE" in response.upper():
                return []

            # Parse contradictions
            contradictions = []
            for line in response.split("\n"):
                if "CONFLICT:" not in line:
                    continue

                try:
                    # Parse id1="..." id2="..." type=... invalidate="..." reason="..."
                    parts = line.split("CONFLICT:", 1)[1].strip()

                    id1 = self._extract_quoted("id1", parts)
                    id2 = self._extract_quoted("id2", parts)
                    conflict_type = self._extract_value("type", parts)
                    to_invalidate = self._extract_quoted("invalidate", parts)
                    reason = self._extract_quoted("reason", parts)

                    if id1 and id2:
                        # Get content for these IDs
                        content1 = next((r.memory.content for r in results if r.memory.id == id1), "")
                        content2 = next((r.memory.content for r in results if r.memory.id == id2), "")

                        contradictions.append(Contradiction(
                            memory_id_1=id1,
                            memory_id_2=id2,
                            content_1=content1,
                            content_2=content2,
                            conflict_type=conflict_type or "direct",
                            to_invalidate=to_invalidate,
                            reason=reason or "Conflicting information",
                        ))
                except Exception as e:
                    logger.warning(f"Failed to parse contradiction line: {e}")
                    continue

            return contradictions

        except Exception as e:
            logger.warning(f"Contradiction detection failed: {e}")
            return []

    async def _refine_query(self, query: str, issue: str) -> str:
        """Refine a query based on feedback.

        Args:
            query: Original query
            issue: What was wrong with results

        Returns:
            Refined query string
        """
        prompt = REFINE_QUERY_PROMPT.format(query=query, issue=issue)

        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            return response.strip()
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return query  # Return original if refinement fails

    async def _invalidate_memories(self, memory_ids: list[str]) -> None:
        """Invalidate memories (mark as superseded).

        This is a no-op in the base implementation. Override in
        subclasses to actually update memory status.

        Args:
            memory_ids: IDs of memories to invalidate
        """
        # Base implementation just logs
        for memory_id in memory_ids:
            logger.info(f"Would invalidate memory: {memory_id}")

    @staticmethod
    def _extract_quoted(key: str, text: str) -> str | None:
        """Extract a quoted value like key="value" from text."""
        import re
        pattern = rf'{key}="([^"]*)"'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_value(key: str, text: str) -> str | None:
        """Extract a value like key=value from text."""
        import re
        pattern = rf'{key}=(\S+)'
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            # Strip quotes if present
            return value.strip('"')
        return None


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Configuration
    "RetrievalConfig",
    "RelevanceAssessment",
    "GradedChunk",
    "Contradiction",
    # Results
    "SelfRAGResult",
    "CRAGResult",
    # Main class
    "RetrievalAugmenter",
    # Protocols
    "MemorySearchProvider",
    "LLMProvider",
]

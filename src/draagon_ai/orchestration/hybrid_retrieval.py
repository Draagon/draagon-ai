"""Hybrid Parallel Retrieval Orchestrator.

Runs multiple retrieval strategies (Local, Graph, Vector) in parallel,
then merges and synthesizes results for optimal coverage and accuracy.

Architecture:
    Query → Analyzer → Parallel Agents → Merge → Synthesize → Answer

Key insight: The semantic graph can SCOPE the vector search, reducing noise.
Graph finds WHICH projects are relevant, Vector finds WHAT content matches.

Usage:
    from draagon_ai.orchestration import HybridRetrievalOrchestrator

    orchestrator = HybridRetrievalOrchestrator(
        llm=my_llm,
        semantic_memory=my_neo4j,
        vector_store=my_qdrant,
        local_index=my_local,
    )

    result = await orchestrator.retrieve(
        query="What patterns do other teams use for customer authentication?",
        context={"project": "my-service"},
    )

    print(result.answer)
    print(result.sources)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProvider(Protocol):
    """LLM provider protocol."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        ...


class SemanticMemoryProvider(Protocol):
    """Semantic memory (knowledge graph) provider."""

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        ...

    async def find_entities(self, names: list[str]) -> list[Any]:
        ...

    async def get_related(
        self, entity_id: str, relationship_types: list[str] | None = None, depth: int = 1
    ) -> list[Any]:
        ...


class VectorStoreProvider(Protocol):
    """Vector store provider (Qdrant, etc.)."""

    async def search(
        self, query: str, limit: int = 10, filter: dict[str, Any] | None = None
    ) -> list[Any]:
        ...


class LocalIndexProvider(Protocol):
    """Local project index provider."""

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        ...


# =============================================================================
# Enums and Configuration
# =============================================================================


class QueryType(str, Enum):
    """Types of queries with different routing needs."""

    ENTITY_LOOKUP = "entity_lookup"  # "What is CustomerService?"
    RELATIONSHIP = "relationship"  # "How does X connect to Y?"
    SIMILARITY = "similarity"  # "Find code similar to this"
    PATTERN = "pattern"  # "How do teams handle X?"
    CROSS_PROJECT = "cross_project"  # "What other projects use X?"
    LOCAL_ONLY = "local_only"  # "What does our CLAUDE.md say?"
    MULTI_HOP = "multi_hop"  # "What calls the service that uses Customer?"
    GENERAL = "general"  # Default


class RetrievalPath(str, Enum):
    """Available retrieval paths."""

    LOCAL = "local"
    GRAPH = "graph"
    VECTOR = "vector"


@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval."""

    # Agent enablement
    enable_local: bool = True
    enable_graph: bool = True
    enable_vector: bool = True

    # Query expansion
    enable_query_expansion: bool = True  # Expand ambiguous queries
    expansion_confidence_threshold: float = 0.70  # Min confidence for expansions
    max_query_expansions: int = 4  # Max parallel expanded queries
    use_weighted_rrf: bool = True  # Use weighted RRF for merging expansion results

    # Parallelism
    parallel_execution: bool = True
    graph_scopes_vector: bool = True  # Use graph to scope vector search

    # Limits
    max_observations_per_agent: int = 10
    max_total_observations: int = 25
    max_context_tokens: int = 8000

    # Deduplication
    similarity_threshold: float = 0.85

    # Timeouts (milliseconds)
    agent_timeout_ms: int = 5000
    total_timeout_ms: int = 10000

    # Fallback
    fallback_on_timeout: bool = True
    fallback_order: list[str] = field(
        default_factory=lambda: ["local", "graph", "vector"]
    )


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class QueryClassification:
    """Result of analyzing a query."""

    query: str
    query_type: QueryType = QueryType.GENERAL
    paths: list[RetrievalPath] = field(default_factory=list)
    detected_entities: list[str] = field(default_factory=list)
    graph_scopes_vector: bool = False
    confidence: float = 0.5


@dataclass
class Observation:
    """A single observation from any retrieval source."""

    content: str
    source: str  # "local:CLAUDE.md", "graph:neo4j", "vector:qdrant"
    source_project: str | None = None
    confidence: float = 1.0
    observation_type: str = "fact"  # fact, example, relationship, instruction
    entities_mentioned: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Hash for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class Scope:
    """Scope for filtered retrieval."""

    projects: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)


@dataclass
class HybridResult:
    """Result of hybrid retrieval."""

    query: str
    answer: str = ""  # Set after synthesis
    observations: list[Observation] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    # Metrics
    total_time_ms: float = 0.0
    local_time_ms: float = 0.0
    graph_time_ms: float = 0.0
    vector_time_ms: float = 0.0
    expansion_time_ms: float = 0.0

    # Coverage
    local_observations: int = 0
    graph_observations: int = 0
    vector_observations: int = 0
    after_dedup: int = 0

    # Query expansion
    expansions_used: list[str] = field(default_factory=list)
    expansion_confidences: list[float] = field(default_factory=list)
    ambiguous_terms_resolved: list[str] = field(default_factory=list)

    # Quality
    classification: QueryClassification | None = None
    scope_used: Scope | None = None
    conflicts_detected: int = 0


# =============================================================================
# Query Analyzer
# =============================================================================


class QueryAnalyzer:
    """Analyzes queries to determine routing using LLM semantic understanding.

    Following the LLM-First Architecture principle from CLAUDE.md:
    NEVER use regex or keyword patterns for semantic understanding.
    The LLM handles ALL semantic analysis.
    """

    CLASSIFICATION_PROMPT = """Classify this query to determine how to retrieve information.

Query: {query}

Types (pick exactly ONE):
- RELATIONSHIP: Asks how things connect or relate (e.g., "How does X connect to Y?", "What calls X?")
- CROSS_PROJECT: Asks about patterns across teams/projects (e.g., "How do other teams handle...", "What patterns exist for...")
- LOCAL_ONLY: Asks specifically about this project (e.g., "What does our CLAUDE.md say...", "In this repo...")
- SIMILARITY: Asks for similar examples (e.g., "Find code like this...", "Show me examples of...")
- ENTITY_LOOKUP: Asks about a specific named thing (e.g., "What is CustomerService?", "Explain the OrderEntity")
- GENERAL: Everything else

Also extract any named entities (proper nouns, class names, service names) from the query.

Respond with ONLY this XML:
<classification>
  <type>TYPE_NAME</type>
  <entities>Entity1, Entity2</entities>
  <confidence>0.0-1.0</confidence>
</classification>"""

    # CamelCase pattern for entity extraction fallback
    ENTITY_PATTERN = r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"

    def __init__(self, llm: LLMProvider | None = None):
        """Initialize analyzer.

        Args:
            llm: LLM provider for semantic classification. If None, uses heuristic fallback.
        """
        self.llm = llm
        self._use_llm = llm is not None

    async def classify_async(self, query: str) -> QueryClassification:
        """Classify query using LLM semantic understanding."""
        if not self._use_llm:
            return self._classify_heuristic(query)

        classification = QueryClassification(query=query)

        try:
            response = await self.llm.chat(
                messages=[{
                    "role": "user",
                    "content": self.CLASSIFICATION_PROMPT.format(query=query),
                }],
                temperature=0.0,
                max_tokens=150,
            )

            # Extract content from ChatResponse if needed
            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse XML response
            type_match = re.search(r"<type>(\w+)</type>", response_text)
            entities_match = re.search(r"<entities>([^<]*)</entities>", response_text)
            confidence_match = re.search(r"<confidence>([\d.]+)</confidence>", response_text)

            if type_match:
                type_str = type_match.group(1).upper()
                try:
                    classification.query_type = QueryType(type_str.lower())
                except ValueError:
                    classification.query_type = QueryType.GENERAL

            if entities_match:
                entities_str = entities_match.group(1).strip()
                if entities_str and entities_str.lower() != "none":
                    classification.detected_entities = [
                        e.strip() for e in entities_str.split(",") if e.strip()
                    ]

            if confidence_match:
                try:
                    classification.confidence = float(confidence_match.group(1))
                except ValueError:
                    classification.confidence = 0.8

            # Set paths based on type
            classification.paths = self._get_paths_for_type(classification.query_type)

            # Enable graph scoping for cross-project queries
            if classification.query_type == QueryType.CROSS_PROJECT:
                classification.graph_scopes_vector = True

        except Exception as e:
            logger.warning(f"LLM classification failed, using heuristic: {e}")
            return self._classify_heuristic(query)

        return classification

    def classify(self, query: str) -> QueryClassification:
        """Synchronous classification (uses heuristic for backwards compatibility).

        For LLM-based classification, use classify_async().
        """
        return self._classify_heuristic(query)

    def _classify_heuristic(self, query: str) -> QueryClassification:
        """Heuristic fallback when LLM not available.

        Note: This uses pattern matching as a FALLBACK only.
        The LLM-based classify_async() should be preferred.
        """
        classification = QueryClassification(query=query)
        query_lower = query.lower()

        # Extract potential entities (CamelCase words)
        entities = re.findall(self.ENTITY_PATTERN, query)
        classification.detected_entities = list(set(entities))

        # Simple heuristic patterns (fallback only)
        if any(p in query_lower for p in ["our ", "this repo", "this project", "my project"]):
            classification.query_type = QueryType.LOCAL_ONLY
            classification.paths = [RetrievalPath.LOCAL]
            classification.confidence = 0.7

        elif any(p in query_lower for p in ["connect to", "relates to", "calls", "uses"]):
            classification.query_type = QueryType.RELATIONSHIP
            classification.paths = [RetrievalPath.GRAPH, RetrievalPath.LOCAL]
            classification.confidence = 0.7

        elif any(p in query_lower for p in ["other teams", "other projects", "how do", "patterns for"]):
            classification.query_type = QueryType.CROSS_PROJECT
            classification.paths = [RetrievalPath.LOCAL, RetrievalPath.GRAPH, RetrievalPath.VECTOR]
            classification.graph_scopes_vector = True
            classification.confidence = 0.7

        elif any(p in query_lower for p in ["similar to", "like this", "examples of"]):
            classification.query_type = QueryType.SIMILARITY
            classification.paths = [RetrievalPath.VECTOR]
            classification.confidence = 0.7

        elif entities:
            classification.query_type = QueryType.ENTITY_LOOKUP
            classification.paths = [RetrievalPath.GRAPH, RetrievalPath.LOCAL]
            classification.confidence = 0.6

        else:
            classification.query_type = QueryType.GENERAL
            classification.paths = [RetrievalPath.LOCAL, RetrievalPath.GRAPH, RetrievalPath.VECTOR]
            classification.confidence = 0.5

        return classification

    def _get_paths_for_type(self, query_type: QueryType) -> list[RetrievalPath]:
        """Get retrieval paths for a query type."""
        path_map = {
            QueryType.LOCAL_ONLY: [RetrievalPath.LOCAL],
            QueryType.RELATIONSHIP: [RetrievalPath.GRAPH, RetrievalPath.LOCAL],
            QueryType.CROSS_PROJECT: [RetrievalPath.LOCAL, RetrievalPath.GRAPH, RetrievalPath.VECTOR],
            QueryType.SIMILARITY: [RetrievalPath.VECTOR],
            QueryType.ENTITY_LOOKUP: [RetrievalPath.GRAPH, RetrievalPath.LOCAL],
            QueryType.GENERAL: [RetrievalPath.LOCAL, RetrievalPath.GRAPH, RetrievalPath.VECTOR],
        }
        return path_map.get(query_type, [RetrievalPath.LOCAL, RetrievalPath.GRAPH, RetrievalPath.VECTOR])


# =============================================================================
# Query Expander
# =============================================================================


@dataclass
class QueryExpansion:
    """A single expanded query with confidence and provenance."""

    query: str
    confidence: float
    reasoning: str
    target_entities: list[str] = field(default_factory=list)
    source_ambiguity: str = ""  # The ambiguous term that was resolved


@dataclass
class ExpansionResult:
    """Result of query expansion."""

    original_query: str
    expansions: list[QueryExpansion]
    graph_context: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    ambiguous_terms: list[str] = field(default_factory=list)


class QueryExpander:
    """Expands ambiguous queries using knowledge graph context and LLM.

    This implements the Hybrid approach (Approach C from research):
    1. Extract ambiguous terms from query
    2. Look up semantic context from knowledge graph
    3. Use LLM to generate prioritized expansions
    4. Filter by confidence threshold

    Example:
        Query: "How do other teams handle authentication?"
        User Context: {team: "Engineering"}
        Graph Context: Teams = [Platform, Data, Mobile, QA]

        Expansions:
        1. "How does Platform team handle authentication?" (0.85)
        2. "How does Data team handle authentication?" (0.82)
        3. "How does Mobile team handle authentication?" (0.80)
        4. "How does QA team handle authentication?" (0.78)
    """

    EXPANSION_PROMPT = """You are expanding an ambiguous query using context from a knowledge graph.

Query: "{query}"

{user_context_section}

{graph_context_section}

Task: Generate {max_expansions} specific query variations that resolve the ambiguous terms.
Each variation should target specific entities from the graph context.

Rules:
1. Replace ambiguous terms with concrete entity names
2. Assign confidence scores (0.0-1.0) based on relevance
3. Higher confidence = more likely the user's intent
4. If user context excludes certain entities (like their own team), don't include those

Respond with ONLY this XML:
<expansions>
  <expansion>
    <query>The expanded query text with specific entity names</query>
    <confidence>0.85</confidence>
    <reasoning>Brief explanation of why this expansion</reasoning>
    <target_entities>Entity1, Entity2</target_entities>
    <resolved_term>The ambiguous term that was resolved</resolved_term>
  </expansion>
</expansions>"""

    # Patterns that indicate ambiguous references needing expansion
    AMBIGUOUS_PATTERNS = [
        (r"\bother teams?\b", "team_reference"),
        (r"\bthe api\b", "api_reference"),
        (r"\bthe service\b", "service_reference"),
        (r"\bthe database\b", "database_reference"),
        (r"\beveryone\b", "all_users"),
        (r"\ball projects?\b", "project_reference"),
    ]

    def __init__(
        self,
        llm: LLMProvider | None = None,
        semantic_memory: SemanticMemoryProvider | None = None,
    ):
        """Initialize the expander.

        Args:
            llm: LLM provider for semantic expansion
            semantic_memory: Knowledge graph for entity resolution
        """
        self.llm = llm
        self.semantic_memory = semantic_memory

    async def expand(
        self,
        query: str,
        user_context: dict[str, Any] | None = None,
        confidence_threshold: float = 0.70,
        max_expansions: int = 5,
    ) -> ExpansionResult:
        """Expand ambiguous query using graph context.

        Args:
            query: The original query with potential ambiguities
            user_context: User info (team, recent work, etc.)
            confidence_threshold: Minimum confidence to include
            max_expansions: Maximum number of expansions to generate

        Returns:
            ExpansionResult with prioritized expansions
        """
        start_time = time.perf_counter()

        # Step 1: Detect ambiguous terms
        ambiguous_terms = self._detect_ambiguities(query)

        if not ambiguous_terms:
            # No ambiguity - return original query
            return ExpansionResult(
                original_query=query,
                expansions=[
                    QueryExpansion(
                        query=query,
                        confidence=1.0,
                        reasoning="Query is already specific, no expansion needed",
                    )
                ],
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Step 2: Gather graph context for ambiguous terms
        graph_context = await self._gather_graph_context(ambiguous_terms, user_context)

        # Step 3: Generate expansions with LLM
        if self.llm and graph_context:
            expansions = await self._llm_expand(
                query=query,
                ambiguous_terms=ambiguous_terms,
                graph_context=graph_context,
                user_context=user_context,
                max_expansions=max_expansions,
            )
        else:
            # Fallback: heuristic expansion
            expansions = self._heuristic_expand(
                query=query,
                ambiguous_terms=ambiguous_terms,
                graph_context=graph_context,
                user_context=user_context,
            )

        # Step 4: Filter by confidence
        filtered = [e for e in expansions if e.confidence >= confidence_threshold]
        if not filtered and expansions:
            # Keep at least the highest confidence
            filtered = [max(expansions, key=lambda x: x.confidence)]

        # Limit to max
        filtered = filtered[:max_expansions]

        return ExpansionResult(
            original_query=query,
            expansions=filtered,
            graph_context=graph_context,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            ambiguous_terms=ambiguous_terms,
        )

    def _detect_ambiguities(self, query: str) -> list[str]:
        """Detect ambiguous terms in the query."""
        ambiguities = []
        query_lower = query.lower()

        for pattern, term_type in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, query_lower):
                match = re.search(pattern, query_lower)
                if match:
                    ambiguities.append(match.group(0))

        return ambiguities

    async def _gather_graph_context(
        self,
        ambiguous_terms: list[str],
        user_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Gather relevant context from the knowledge graph."""
        context: dict[str, Any] = {}

        if not self.semantic_memory:
            return context

        for term in ambiguous_terms:
            term_lower = term.lower()

            # Handle "other teams" pattern
            if "team" in term_lower:
                try:
                    # Search for team-related content using multiple strategies
                    teams = set()

                    # Strategy 1: Search for "team" mentions directly
                    results = await self.semantic_memory.search("team", limit=20)

                    for result in results:
                        # Extract content
                        content = (
                            result.content if hasattr(result, "content")
                            else str(result.get("content", ""))
                        )

                        # Extract entities
                        entities = []
                        if hasattr(result, "entities"):
                            entities = result.entities
                        elif isinstance(result, dict) and "entities" in result:
                            entities = result["entities"]

                        # Look for team-related entities from the entity list
                        for entity in entities:
                            entity_name = entity if isinstance(entity, str) else getattr(entity, "name", "")
                            if entity_name:
                                # Filter out generic terms like "authentication"
                                if entity_name.lower() not in ("authentication", "team", "database", "api"):
                                    teams.add(entity_name)

                        # Also try to extract team names from content
                        # Pattern: "X team uses..." or "X is a member of the Y team"
                        team_pattern = r'\b([A-Z][a-zA-Z]+)\s+team\b'
                        content_matches = re.findall(team_pattern, content)
                        for match in content_matches:
                            if match.lower() not in ("the", "other", "our", "each", "every"):
                                teams.add(match)

                    # Strategy 2: Try direct entity search for "team"
                    try:
                        team_entities = await self.semantic_memory.find_entities(["team", "Team"])
                        for entity in team_entities:
                            if hasattr(entity, "name"):
                                name = entity.name
                            elif isinstance(entity, dict) and "name" in entity:
                                name = entity["name"]
                            else:
                                continue
                            # Filter out generic terms
                            if name.lower() not in ("authentication", "team", "database", "api"):
                                teams.add(name)
                    except Exception:
                        pass

                    all_teams = list(teams)
                    context["all_teams"] = all_teams

                    # Exclude user's team if known
                    if user_context and user_context.get("team"):
                        user_team = user_context["team"]
                        context["user_team"] = user_team
                        context["other_teams"] = [t for t in all_teams if t.lower() != user_team.lower()]
                    else:
                        context["other_teams"] = all_teams

                except Exception as e:
                    logger.warning(f"Failed to gather team context: {e}")

            # Handle "the API" pattern
            elif "api" in term_lower:
                try:
                    results = await self.semantic_memory.search("API service endpoint", limit=10)
                    apis = set()

                    for result in results:
                        entities = []
                        if hasattr(result, "entities"):
                            entities = result.entities
                        elif isinstance(result, dict) and "entities" in result:
                            entities = result["entities"]

                        for entity in entities:
                            entity_name = entity if isinstance(entity, str) else getattr(entity, "name", "")
                            if entity_name and "api" in entity_name.lower():
                                apis.add(entity_name)

                    context["apis"] = list(apis)

                    # Check user's recent context
                    if user_context and user_context.get("last_worked_on"):
                        context["user_recent_api"] = user_context["last_worked_on"]

                except Exception as e:
                    logger.warning(f"Failed to gather API context: {e}")

            # Handle "the database" pattern
            elif "database" in term_lower:
                try:
                    results = await self.semantic_memory.search("database storage", limit=10)
                    databases = set()

                    for result in results:
                        entities = []
                        if hasattr(result, "entities"):
                            entities = result.entities
                        elif isinstance(result, dict) and "entities" in result:
                            entities = result["entities"]

                        for entity in entities:
                            entity_name = entity if isinstance(entity, str) else getattr(entity, "name", "")
                            if entity_name:
                                databases.add(entity_name)

                    context["databases"] = list(databases)

                except Exception as e:
                    logger.warning(f"Failed to gather database context: {e}")

        return context

    async def _llm_expand(
        self,
        query: str,
        ambiguous_terms: list[str],
        graph_context: dict[str, Any],
        user_context: dict[str, Any] | None,
        max_expansions: int,
    ) -> list[QueryExpansion]:
        """Generate expansions using LLM with graph context."""
        # Build user context section
        user_context_section = "User Context: Unknown"
        if user_context:
            parts = []
            if user_context.get("user_name"):
                parts.append(f"User: {user_context['user_name']}")
            if user_context.get("team"):
                parts.append(f"User's Team: {user_context['team']} (EXCLUDE from 'other teams')")
            if user_context.get("last_worked_on"):
                parts.append(f"Recently worked on: {user_context['last_worked_on']}")
            if parts:
                user_context_section = "User Context:\n- " + "\n- ".join(parts)

        # Build graph context section
        graph_parts = []
        if graph_context.get("other_teams"):
            graph_parts.append(f"Available teams (excluding user's): {', '.join(graph_context['other_teams'])}")
        elif graph_context.get("all_teams"):
            graph_parts.append(f"All teams: {', '.join(graph_context['all_teams'])}")
        if graph_context.get("apis"):
            graph_parts.append(f"Known APIs: {', '.join(graph_context['apis'])}")
        if graph_context.get("databases"):
            graph_parts.append(f"Known databases: {', '.join(graph_context['databases'])}")

        graph_context_section = "Graph Context:\n- " + "\n- ".join(graph_parts) if graph_parts else "Graph Context: No relevant entities found"

        prompt = self.EXPANSION_PROMPT.format(
            query=query,
            user_context_section=user_context_section,
            graph_context_section=graph_context_section,
            max_expansions=max_expansions,
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )

            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse XML response
            expansions = []
            expansion_matches = re.findall(
                r"<expansion>(.*?)</expansion>",
                response_text,
                re.DOTALL,
            )

            for match in expansion_matches:
                query_match = re.search(r"<query>(.*?)</query>", match, re.DOTALL)
                conf_match = re.search(r"<confidence>([\d.]+)</confidence>", match)
                reason_match = re.search(r"<reasoning>(.*?)</reasoning>", match, re.DOTALL)
                entities_match = re.search(r"<target_entities>(.*?)</target_entities>", match)
                resolved_match = re.search(r"<resolved_term>(.*?)</resolved_term>", match)

                if query_match:
                    exp_query = query_match.group(1).strip()
                    confidence = float(conf_match.group(1)) if conf_match else 0.7
                    reasoning = reason_match.group(1).strip() if reason_match else ""
                    entities_str = entities_match.group(1).strip() if entities_match else ""
                    resolved = resolved_match.group(1).strip() if resolved_match else ""

                    target_entities = [e.strip() for e in entities_str.split(",") if e.strip()]

                    expansions.append(QueryExpansion(
                        query=exp_query,
                        confidence=confidence,
                        reasoning=reasoning,
                        target_entities=target_entities,
                        source_ambiguity=resolved,
                    ))

            return expansions

        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return self._heuristic_expand(query, ambiguous_terms, graph_context, user_context)

    def _heuristic_expand(
        self,
        query: str,
        ambiguous_terms: list[str],
        graph_context: dict[str, Any],
        user_context: dict[str, Any] | None,
    ) -> list[QueryExpansion]:
        """Heuristic expansion when LLM not available."""
        expansions = []

        for term in ambiguous_terms:
            term_lower = term.lower()

            if "team" in term_lower:
                # Expand "other teams" to specific teams
                teams = graph_context.get("other_teams", graph_context.get("all_teams", []))
                base_conf = 0.85

                for i, team in enumerate(teams[:5]):
                    expanded = query.lower().replace(term_lower, f"{team} team")
                    expanded = expanded[0].upper() + expanded[1:]

                    expansions.append(QueryExpansion(
                        query=expanded,
                        confidence=base_conf - (i * 0.03),
                        reasoning=f"Replacing '{term}' with specific team: {team}",
                        target_entities=[team],
                        source_ambiguity=term,
                    ))

            elif "api" in term_lower:
                apis = graph_context.get("apis", [])
                user_recent = graph_context.get("user_recent_api")

                # Put user's recent API first if known
                if user_recent and user_recent in apis:
                    apis = [user_recent] + [a for a in apis if a != user_recent]

                for i, api in enumerate(apis[:4]):
                    expanded = query.lower().replace(term_lower, api.lower())
                    expanded = expanded[0].upper() + expanded[1:]

                    expansions.append(QueryExpansion(
                        query=expanded,
                        confidence=0.90 - (i * 0.10),
                        reasoning=f"Disambiguating '{term}' to: {api}",
                        target_entities=[api],
                        source_ambiguity=term,
                    ))

            elif "database" in term_lower:
                databases = graph_context.get("databases", [])

                for i, db in enumerate(databases[:4]):
                    expanded = query.lower().replace(term_lower, db.lower())
                    expanded = expanded[0].upper() + expanded[1:]

                    expansions.append(QueryExpansion(
                        query=expanded,
                        confidence=0.85 - (i * 0.05),
                        reasoning=f"Expanding '{term}' to: {db}",
                        target_entities=[db],
                        source_ambiguity=term,
                    ))

        if not expansions:
            # No expansion possible - return original
            expansions.append(QueryExpansion(
                query=query,
                confidence=0.7,
                reasoning="Could not resolve ambiguous terms from graph context",
            ))

        return expansions


# =============================================================================
# Weighted Reciprocal Rank Fusion
# =============================================================================


@dataclass
class RRFResult:
    """Result after RRF merging."""

    observation: Observation
    rrf_score: float
    contributing_queries: list[str] = field(default_factory=list)
    query_confidences: list[float] = field(default_factory=list)


class WeightedRRFMerger:
    """Merges results from multiple query expansions using weighted RRF.

    RRF (Reciprocal Rank Fusion) combines ranked results from multiple
    retrieval queries. We weight by expansion confidence so higher-confidence
    expansions contribute more to the final ranking.

    Formula: RRF_score = Σ (1 / (k + rank_i)) × confidence_i

    Reference: RAG-Fusion paper (arXiv:2402.03367)
    """

    def __init__(self, k: int = 60, similarity_threshold: float = 0.85):
        """Initialize the merger.

        Args:
            k: RRF smoothing constant (standard is 60)
            similarity_threshold: Threshold for near-duplicate detection
        """
        self.k = k
        self.similarity_threshold = similarity_threshold

    def merge(
        self,
        results_per_query: list[list[Observation]],
        expansions: list[QueryExpansion],
    ) -> list[RRFResult]:
        """Merge results from multiple expansions using weighted RRF.

        Args:
            results_per_query: List of observation lists, one per expansion
            expansions: The query expansions (for confidence weights)

        Returns:
            Merged and ranked list of RRFResults
        """
        if len(results_per_query) != len(expansions):
            raise ValueError("results_per_query and expansions must have same length")

        # Group by content (for deduplication and score accumulation)
        content_scores: dict[str, dict] = {}

        for exp_idx, observations in enumerate(results_per_query):
            confidence = expansions[exp_idx].confidence
            exp_query = expansions[exp_idx].query

            for rank, obs in enumerate(observations, start=1):
                content_key = obs.content_hash

                # Check for near-duplicates with existing content
                matched_key = self._find_near_duplicate(obs.content, content_scores)
                if matched_key:
                    content_key = matched_key

                if content_key not in content_scores:
                    content_scores[content_key] = {
                        "observation": obs,
                        "rrf_score": 0.0,
                        "contributing_queries": [],
                        "query_confidences": [],
                    }

                # Weighted RRF contribution
                rrf_contribution = (1 / (self.k + rank)) * confidence
                content_scores[content_key]["rrf_score"] += rrf_contribution

                if exp_query not in content_scores[content_key]["contributing_queries"]:
                    content_scores[content_key]["contributing_queries"].append(exp_query)
                    content_scores[content_key]["query_confidences"].append(confidence)

        # Convert to RRFResult list
        merged = [
            RRFResult(
                observation=data["observation"],
                rrf_score=data["rrf_score"],
                contributing_queries=data["contributing_queries"],
                query_confidences=data["query_confidences"],
            )
            for data in content_scores.values()
        ]

        # Sort by RRF score descending
        merged.sort(key=lambda x: x.rrf_score, reverse=True)

        return merged

    def _find_near_duplicate(
        self, content: str, existing: dict[str, dict]
    ) -> str | None:
        """Find near-duplicate content key."""
        content_words = set(content.lower().split())

        for key, data in existing.items():
            existing_words = set(data["observation"].content.lower().split())

            if not content_words or not existing_words:
                continue

            intersection = len(content_words & existing_words)
            union = len(content_words | existing_words)
            overlap = intersection / union if union > 0 else 0.0

            if overlap >= self.similarity_threshold:
                return key

        return None


# =============================================================================
# Embedding Strategies (FR-011)
# =============================================================================


class EmbeddingStrategy(str, Enum):
    """Available embedding strategies for query processing."""

    RAW = "raw"  # Direct query embedding
    HYDE = "hyde"  # Hypothetical Document Embedding
    QUERY2DOC = "query2doc"  # Original + LLM expansion
    GROUNDED = "grounded"  # Phase0/1 + Graph context


@dataclass
class EmbeddingResult:
    """Result of applying an embedding strategy."""

    strategy: EmbeddingStrategy
    embedding: list[float]
    expanded_text: str  # The text that was embedded
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result from a single strategy execution including retrieval."""

    strategy: EmbeddingStrategy
    observations: list[Observation]
    embedding_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float
    expanded_text: str


@dataclass
class MultiStrategyResult:
    """Combined result from multiple strategies."""

    query: str
    strategy_results: list[StrategyResult]
    merged_observations: list[RRFResult]
    total_latency_ms: float
    strategies_executed: list[EmbeddingStrategy]


class EmbeddingStrategyExecutor:
    """Executes multiple embedding strategies for query processing.

    Implements the four strategies from FR-011:
    - Raw: Direct query embedding (fastest, no hallucination risk)
    - HyDE: Hypothetical Document Embedding (bridges query-document gap)
    - Query2Doc: Original + expansion (balances precision and recall)
    - Grounded: Phase0/1 + graph context (most accurate, slowest)

    Each strategy produces an embedding, which is used for vector retrieval.
    Results are merged using weighted RRF based on strategy confidence.

    Usage:
        executor = EmbeddingStrategyExecutor(llm=llm, embedder=embedder)
        result = await executor.execute(
            query="How do teams handle auth?",
            strategies=[EmbeddingStrategy.RAW, EmbeddingStrategy.HYDE],
        )
    """

    # HyDE prompt - generates hypothetical answer document
    HYDE_PROMPT = """Write a detailed paragraph that answers this question as if you are an expert:

Question: {query}

Write a comprehensive answer with specific details, examples, and technical terms that someone searching for this information would find helpful. Do not say "I don't know" - write what a knowledgeable answer WOULD contain."""

    # Query2Doc prompt - expands query with context
    QUERY2DOC_PROMPT = """Expand this search query with relevant context, terminology, and related concepts:

Query: {query}

Write 2-3 sentences that elaborate on what the user is asking. Include:
- Related technical terms
- Synonyms and alternative phrasings
- Contextual information that would help find relevant documents

Output ONLY the expanded text, no explanations."""

    # Grounded expansion prompt - uses graph context
    GROUNDED_PROMPT = """Expand this query using ONLY the provided context. Do not invent information.

Query: {query}

Known entities from knowledge graph:
{entities}

Related context:
{graph_context}

User context:
{user_context}

Write an expanded version of the query that:
1. Replaces ambiguous terms with specific entity names from the graph
2. Adds relevant context from the provided information
3. Does NOT invent entities or facts not in the context

Output ONLY the expanded query text."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        embedder: Any = None,  # EmbeddingProvider
        semantic_memory: SemanticMemoryProvider | None = None,
        decomposer: Any = None,  # DecompositionService for Phase0/1
    ):
        """Initialize the strategy executor.

        Args:
            llm: LLM for generating expansions (HyDE, Query2Doc, Grounded)
            embedder: Embedding provider for vector generation
            semantic_memory: Knowledge graph for grounded expansion
            decomposer: Decomposition service for Phase0/1 analysis
        """
        self.llm = llm
        self.embedder = embedder
        self.semantic_memory = semantic_memory
        self.decomposer = decomposer

        # Strategy confidence weights (can be learned via bandit)
        self.strategy_weights = {
            EmbeddingStrategy.RAW: 0.7,
            EmbeddingStrategy.HYDE: 0.85,
            EmbeddingStrategy.QUERY2DOC: 0.80,
            EmbeddingStrategy.GROUNDED: 0.90,
        }

    async def execute_embedding(
        self,
        query: str,
        strategy: EmbeddingStrategy,
        user_context: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Execute a single embedding strategy.

        Args:
            query: The original query
            strategy: Which strategy to use
            user_context: Optional user context for grounded expansion

        Returns:
            EmbeddingResult with the generated embedding
        """
        start_time = time.perf_counter()

        if strategy == EmbeddingStrategy.RAW:
            expanded_text = query
        elif strategy == EmbeddingStrategy.HYDE:
            expanded_text = await self._hyde_expand(query)
        elif strategy == EmbeddingStrategy.QUERY2DOC:
            expanded_text = await self._query2doc_expand(query)
        elif strategy == EmbeddingStrategy.GROUNDED:
            expanded_text = await self._grounded_expand(query, user_context)
        else:
            expanded_text = query

        # Generate embedding
        if self.embedder:
            embedding = await self.embedder.embed(expanded_text)
        else:
            embedding = []

        latency_ms = (time.perf_counter() - start_time) * 1000

        return EmbeddingResult(
            strategy=strategy,
            embedding=embedding,
            expanded_text=expanded_text,
            latency_ms=latency_ms,
            metadata={"original_query": query},
        )

    async def execute_all(
        self,
        query: str,
        strategies: list[EmbeddingStrategy] | None = None,
        user_context: dict[str, Any] | None = None,
        parallel: bool = True,
    ) -> list[EmbeddingResult]:
        """Execute multiple embedding strategies.

        Args:
            query: The original query
            strategies: Which strategies to use (default: all)
            user_context: Optional user context
            parallel: Run strategies in parallel (default: True)

        Returns:
            List of EmbeddingResults
        """
        if strategies is None:
            strategies = list(EmbeddingStrategy)

        if parallel:
            tasks = [
                self.execute_embedding(query, s, user_context)
                for s in strategies
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for s in strategies:
                result = await self.execute_embedding(query, s, user_context)
                results.append(result)
            return results

    async def _hyde_expand(self, query: str) -> str:
        """Generate hypothetical document for HyDE strategy."""
        if not self.llm:
            return query

        prompt = self.HYDE_PROMPT.format(query=query)
        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            # Handle ChatResponse object or string
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip()
        except Exception as e:
            logger.warning(f"HyDE expansion failed: {e}")
            return query

    async def _query2doc_expand(self, query: str) -> str:
        """Generate Query2Doc expansion (original + expansion)."""
        if not self.llm:
            return query

        prompt = self.QUERY2DOC_PROMPT.format(query=query)
        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
            )
            # Handle ChatResponse object or string
            content = response.content if hasattr(response, "content") else str(response)
            # Concatenate original query with expansion
            return f"{query} {content.strip()}"
        except Exception as e:
            logger.warning(f"Query2Doc expansion failed: {e}")
            return query

    async def _grounded_expand(
        self,
        query: str,
        user_context: dict[str, Any] | None = None,
    ) -> str:
        """Generate grounded expansion using Phase0/1 + graph context."""
        if not self.llm:
            return query

        # Extract entities via decomposition or simple extraction
        entities = []
        if self.decomposer:
            try:
                decomposition = await self.decomposer.decompose(query)
                entities = [e.name for e in decomposition.entities]
            except Exception:
                pass

        # Gather graph context
        graph_context_str = ""
        if self.semantic_memory and entities:
            try:
                for entity in entities[:3]:  # Limit to avoid token explosion
                    results = await self.semantic_memory.search(entity, limit=3)
                    for r in results:
                        content = (
                            r.content if hasattr(r, "content")
                            else r.get("content", "")
                        )
                        if content:
                            graph_context_str += f"- {content}\n"
            except Exception:
                pass

        # Format user context
        user_context_str = ""
        if user_context:
            user_context_str = ", ".join(f"{k}: {v}" for k, v in user_context.items())

        # Generate grounded expansion
        prompt = self.GROUNDED_PROMPT.format(
            query=query,
            entities=", ".join(entities) if entities else "None detected",
            graph_context=graph_context_str if graph_context_str else "No additional context",
            user_context=user_context_str if user_context_str else "None",
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temp for factual grounding
                max_tokens=200,
            )
            # Handle ChatResponse object or string
            content = response.content if hasattr(response, "content") else str(response)
            # Concatenate for Query2Doc-style benefit
            return f"{query} {content.strip()}"
        except Exception as e:
            logger.warning(f"Grounded expansion failed: {e}")
            return query

    def get_strategy_weight(self, strategy: EmbeddingStrategy) -> float:
        """Get the confidence weight for a strategy."""
        return self.strategy_weights.get(strategy, 0.5)

    def update_strategy_weight(
        self,
        strategy: EmbeddingStrategy,
        reward: float,
        learning_rate: float = 0.1,
    ):
        """Update strategy weight based on observed reward.

        This enables learning which strategies work best over time.

        Args:
            strategy: The strategy to update
            reward: Observed reward (0.0 - 1.0)
            learning_rate: How quickly to adapt
        """
        current = self.strategy_weights.get(strategy, 0.5)
        # Simple exponential moving average update
        self.strategy_weights[strategy] = current + learning_rate * (reward - current)


# =============================================================================
# Retrieval Agents
# =============================================================================


class RetrievalAgent(ABC):
    """Base class for retrieval agents."""

    @property
    @abstractmethod
    def path(self) -> RetrievalPath:
        """Return the retrieval path this agent handles."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        scope: Scope | None = None,
    ) -> list[Observation]:
        """Retrieve observations for query."""
        ...


class LocalRetrievalAgent(RetrievalAgent):
    """Retrieves from local project context."""

    def __init__(self, local_index: LocalIndexProvider | None = None):
        self.local_index = local_index

    @property
    def path(self) -> RetrievalPath:
        return RetrievalPath.LOCAL

    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        scope: Scope | None = None,
    ) -> list[Observation]:
        """Search local index."""
        if not self.local_index:
            return []

        try:
            results = await self.local_index.search(query, limit=max_results)
            observations = []

            for result in results:
                content = (
                    result.content
                    if hasattr(result, "content")
                    else str(result.get("content", result))
                )
                source_file = (
                    result.source_path
                    if hasattr(result, "source_path")
                    else result.get("source", "local")
                )

                observations.append(
                    Observation(
                        content=content,
                        source=f"local:{source_file}",
                        confidence=0.9,  # Local is trusted
                        observation_type="instruction",
                        metadata={"source_file": source_file},
                    )
                )

            return observations

        except Exception as e:
            logger.warning(f"Local retrieval failed: {e}")
            return []


class GraphRetrievalAgent(RetrievalAgent):
    """Retrieves from semantic knowledge graph."""

    def __init__(self, semantic_memory: SemanticMemoryProvider | None = None):
        self.semantic = semantic_memory

    @property
    def path(self) -> RetrievalPath:
        return RetrievalPath.GRAPH

    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        scope: Scope | None = None,
    ) -> list[Observation]:
        """Search knowledge graph."""
        if not self.semantic:
            return []

        try:
            results = await self.semantic.search(query, limit=max_results)
            observations = []

            for result in results:
                content = (
                    result.content
                    if hasattr(result, "content")
                    else str(result.get("content", result))
                )

                # Extract entities if available
                entities = []
                if hasattr(result, "entities"):
                    entities = [e.name for e in result.entities]
                elif isinstance(result, dict) and "entities" in result:
                    entities = result["entities"]

                # Extract source project
                source_project = None
                if hasattr(result, "project"):
                    source_project = result.project
                elif isinstance(result, dict):
                    source_project = result.get("project")

                observations.append(
                    Observation(
                        content=content,
                        source="graph:neo4j",
                        source_project=source_project,
                        confidence=0.85,
                        observation_type="relationship",
                        entities_mentioned=entities,
                        metadata={"from_graph": True},
                    )
                )

            return observations

        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")
            return []

    async def find_scope(self, entities: list[str]) -> Scope:
        """Find scope (projects) that contain these entities."""
        if not self.semantic:
            return Scope()

        try:
            found = await self.semantic.find_entities(entities)
            projects = set()

            for entity in found:
                if hasattr(entity, "project"):
                    projects.add(entity.project)
                elif isinstance(entity, dict) and "project" in entity:
                    projects.add(entity["project"])

            return Scope(projects=list(projects), entities=entities)

        except Exception as e:
            logger.warning(f"Scope finding failed: {e}")
            return Scope()


class VectorRetrievalAgent(RetrievalAgent):
    """Retrieves from vector store (Qdrant, etc.)."""

    def __init__(self, vector_store: VectorStoreProvider | None = None):
        self.vector_store = vector_store

    @property
    def path(self) -> RetrievalPath:
        return RetrievalPath.VECTOR

    async def retrieve(
        self,
        query: str,
        max_results: int = 10,
        scope: Scope | None = None,
    ) -> list[Observation]:
        """Search vector store, optionally scoped."""
        if not self.vector_store:
            return []

        try:
            # Build filter from scope
            filter_dict = None
            if scope and scope.projects:
                filter_dict = {"project": {"$in": scope.projects}}

            results = await self.vector_store.search(
                query, limit=max_results, filter=filter_dict
            )
            observations = []

            for result in results:
                content = (
                    result.content
                    if hasattr(result, "content")
                    else str(result.get("content", result))
                )

                # Extract similarity score
                score = 0.7
                if hasattr(result, "score"):
                    score = result.score
                elif isinstance(result, dict) and "score" in result:
                    score = result["score"]

                source_project = None
                if hasattr(result, "project"):
                    source_project = result.project
                elif isinstance(result, dict):
                    source_project = result.get("project")

                observations.append(
                    Observation(
                        content=content,
                        source="vector:qdrant",
                        source_project=source_project,
                        confidence=min(0.9, score),  # Cap at 0.9
                        observation_type="example",
                        metadata={
                            "similarity_score": score,
                            "scoped": scope is not None,
                        },
                    )
                )

            return observations

        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return []


# =============================================================================
# Result Merger
# =============================================================================


class ResultMerger:
    """Merges and ranks observations from multiple sources."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def merge(self, observations: list[Observation]) -> list[Observation]:
        """Merge observations: deduplicate and rank."""
        if not observations:
            return []

        # Sort by confidence first
        sorted_obs = sorted(observations, key=lambda o: -o.confidence)

        # Deduplicate
        unique = self._deduplicate(sorted_obs)

        # Rank
        ranked = self._rank(unique)

        return ranked

    def _deduplicate(self, observations: list[Observation]) -> list[Observation]:
        """Remove duplicate/very similar observations."""
        unique = []
        seen_hashes = set()

        for obs in observations:
            # Exact duplicate check
            if obs.content_hash in seen_hashes:
                continue

            # Check for near-duplicates (simple overlap check)
            is_near_dup = False
            for existing in unique:
                overlap = self._content_overlap(obs.content, existing.content)
                if overlap > self.similarity_threshold:
                    # Boost confidence of existing if corroborated
                    existing.confidence = min(1.0, existing.confidence + 0.05)
                    existing.metadata["corroborated_by"] = existing.metadata.get(
                        "corroborated_by", []
                    )
                    existing.metadata["corroborated_by"].append(obs.source)
                    is_near_dup = True
                    break

            if not is_near_dup:
                unique.append(obs)
                seen_hashes.add(obs.content_hash)

        return unique

    def _content_overlap(self, a: str, b: str) -> float:
        """Calculate word overlap between two strings."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _rank(self, observations: list[Observation]) -> list[Observation]:
        """Rank observations by quality signals."""

        def score(obs: Observation) -> float:
            base = obs.confidence

            # Source agreement boost
            corroboration = len(obs.metadata.get("corroborated_by", []))
            agreement_boost = 0.05 * corroboration

            # Local context boost (team-curated)
            local_boost = 0.1 if obs.source.startswith("local:") else 0

            # Graph entity boost
            entity_boost = 0.1 if obs.entities_mentioned else 0

            # Length penalty (very long = less focused)
            length_penalty = max(0, (len(obs.content) - 500) / 5000) * 0.1

            return base + agreement_boost + local_boost + entity_boost - length_penalty

        return sorted(observations, key=score, reverse=True)


# =============================================================================
# Synthesis
# =============================================================================


class SynthesisAgent:
    """Synthesizes final answer from observations."""

    SYNTHESIS_PROMPT = """You are synthesizing information from multiple retrieval sources to answer a query.

QUERY: {query}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Synthesize a coherent answer from the retrieved information
2. Cite sources using brackets: [local:file], [graph:Entity], [vector:project/file]
3. Note if sources agree or conflict on any points
4. If information is incomplete, mention what's missing
5. Prioritize: local project context > graph relationships > similar examples

Answer:"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def synthesize(
        self,
        query: str,
        observations: list[Observation],
        max_context_chars: int = 10000,
    ) -> str:
        """Generate answer from observations."""
        context = self._build_context(observations, max_context_chars)

        prompt = self.SYNTHESIS_PROMPT.format(query=query, context=context)

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            # Handle both string returns (mock) and ChatResponse objects (real LLM)
            if hasattr(response, "content"):
                return response.content
            return str(response)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Error synthesizing answer: {e}"

    def _build_context(
        self, observations: list[Observation], max_chars: int
    ) -> str:
        """Build context string from observations."""
        parts = []
        current_chars = 0

        # Group by source type
        by_source: dict[str, list[Observation]] = {}
        for obs in observations:
            source_type = obs.source.split(":")[0]
            if source_type not in by_source:
                by_source[source_type] = []
            by_source[source_type].append(obs)

        # Headers for each source type
        headers = {
            "local": "FROM YOUR PROJECT:",
            "graph": "FROM ENTERPRISE KNOWLEDGE:",
            "vector": "FROM SIMILAR DOCUMENTS:",
        }

        for source_type in ["local", "graph", "vector"]:
            obs_list = by_source.get(source_type, [])
            if not obs_list:
                continue

            header = headers.get(source_type, f"FROM {source_type.upper()}:")
            parts.append(f"\n{header}")
            current_chars += len(header) + 1

            for obs in obs_list:
                entry = f"\n[{obs.source}] {obs.content}"
                if current_chars + len(entry) > max_chars:
                    parts.append("\n[... truncated ...]")
                    break
                parts.append(entry)
                current_chars += len(entry)

        return "\n".join(parts)


# =============================================================================
# Main Orchestrator
# =============================================================================


class HybridRetrievalOrchestrator:
    """Orchestrates parallel hybrid retrieval across Local, Graph, and Vector sources.

    With query expansion enabled, the orchestrator:
    1. Detects ambiguous terms (e.g., "other teams")
    2. Looks up context from knowledge graph
    3. Generates prioritized query expansions with confidence scores
    4. Retrieves in parallel for each expansion
    5. Merges results using weighted Reciprocal Rank Fusion
    6. Synthesizes final answer

    Example:
        Query: "How do other teams handle authentication?"
        User context: {team: "Engineering"}

        Expansions:
        1. "How does Platform team handle authentication?" (0.85)
        2. "How does Data team handle authentication?" (0.82)
        ...

        Results merged with RRF, weighted by expansion confidence.
    """

    def __init__(
        self,
        llm: LLMProvider,
        semantic_memory: SemanticMemoryProvider | None = None,
        vector_store: VectorStoreProvider | None = None,
        local_index: LocalIndexProvider | None = None,
        config: HybridRetrievalConfig | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            llm: LLM provider for synthesis
            semantic_memory: Knowledge graph (Neo4j)
            vector_store: Vector store (Qdrant)
            local_index: Local project index
            config: Configuration options
        """
        self.llm = llm
        self.config = config or HybridRetrievalConfig()
        self.semantic_memory = semantic_memory

        # Initialize agents
        self.local_agent = LocalRetrievalAgent(local_index)
        self.graph_agent = GraphRetrievalAgent(semantic_memory)
        self.vector_agent = VectorRetrievalAgent(vector_store)

        # Initialize components
        # Pass LLM to analyzer for semantic classification (LLM-First Architecture)
        self.analyzer = QueryAnalyzer(llm=llm)
        self.merger = ResultMerger(self.config.similarity_threshold)
        self.synthesizer = SynthesisAgent(llm)

        # Query expansion components
        self.expander = QueryExpander(llm=llm, semantic_memory=semantic_memory)
        self.rrf_merger = WeightedRRFMerger(similarity_threshold=self.config.similarity_threshold)

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> HybridResult:
        """Perform hybrid retrieval and synthesis.

        Args:
            query: The user query
            context: Additional context (e.g., current project)
            user_context: User info for query expansion (e.g., {team: "Engineering"})

        Returns:
            HybridResult with answer and observations
        """
        start_time = time.perf_counter()
        result = HybridResult(query=query)

        # Step 1: Classify query using LLM semantic understanding
        classification = await self.analyzer.classify_async(query)
        result.classification = classification
        logger.info(
            f"Query classified as {classification.query_type.value}, "
            f"paths: {[p.value for p in classification.paths]}"
        )

        # Step 2: Determine which agents to run
        agents_to_run = self._get_agents_for_paths(classification.paths)

        # Step 3: Optionally expand ambiguous queries
        queries_to_run: list[tuple[str, float]] = [(query, 1.0)]  # (query, confidence)

        if self.config.enable_query_expansion:
            expansion_start = time.perf_counter()
            expansion_result = await self.expander.expand(
                query=query,
                user_context=user_context,
                confidence_threshold=self.config.expansion_confidence_threshold,
                max_expansions=self.config.max_query_expansions,
            )
            result.expansion_time_ms = (time.perf_counter() - expansion_start) * 1000

            if expansion_result.ambiguous_terms and expansion_result.expansions:
                # Query was expanded
                result.ambiguous_terms_resolved = expansion_result.ambiguous_terms
                queries_to_run = [
                    (exp.query, exp.confidence)
                    for exp in expansion_result.expansions
                ]
                result.expansions_used = [exp.query for exp in expansion_result.expansions]
                result.expansion_confidences = [exp.confidence for exp in expansion_result.expansions]
                logger.info(
                    f"Query expanded into {len(queries_to_run)} variations: "
                    f"{result.ambiguous_terms_resolved}"
                )

        # Ensure we always have at least the original query
        if not queries_to_run:
            queries_to_run = [(query, 1.0)]

        # Step 4: Run retrieval for each query (parallel if multiple expansions)
        all_observations: list[list[Observation]] = []
        all_timings: dict[str, float] = {}

        if len(queries_to_run) > 1 and self.config.use_weighted_rrf:
            # Multiple expansions - retrieve in parallel, merge with RRF
            retrieval_tasks = []
            for exp_query, _ in queries_to_run:
                if self.config.parallel_execution:
                    task = self._parallel_retrieve(exp_query, agents_to_run, classification)
                else:
                    task = self._sequential_retrieve(exp_query, agents_to_run, classification)
                retrieval_tasks.append(task)

            retrieval_results = await asyncio.gather(*retrieval_tasks)

            for obs_list, timings in retrieval_results:
                all_observations.append(obs_list)
                # Accumulate timings
                for key, val in timings.items():
                    all_timings[key] = all_timings.get(key, 0) + val

            # Merge with weighted RRF
            rrf_results = self.rrf_merger.merge(
                all_observations,
                [
                    QueryExpansion(query=q, confidence=c, reasoning="")
                    for q, c in queries_to_run
                ],
            )
            observations = [r.observation for r in rrf_results]
            timings = {k: v / len(queries_to_run) for k, v in all_timings.items()}  # Average

        else:
            # Single query or RRF disabled
            if self.config.parallel_execution:
                observations, timings = await self._parallel_retrieve(
                    queries_to_run[0][0], agents_to_run, classification
                )
            else:
                observations, timings = await self._sequential_retrieve(
                    queries_to_run[0][0], agents_to_run, classification
                )

        result.local_time_ms = timings.get("local", 0)
        result.graph_time_ms = timings.get("graph", 0)
        result.vector_time_ms = timings.get("vector", 0)

        # Count observations by source
        for obs in observations:
            source_type = obs.source.split(":")[0]
            if source_type == "local":
                result.local_observations += 1
            elif source_type == "graph":
                result.graph_observations += 1
            elif source_type == "vector":
                result.vector_observations += 1

        # Step 4: Merge results
        merged = self.merger.merge(observations)
        result.after_dedup = len(merged)
        result.observations = merged[: self.config.max_total_observations]

        # Step 5: Synthesize answer
        result.answer = await self.synthesizer.synthesize(query, result.observations)

        # Extract sources
        result.sources = list(set(obs.source for obs in result.observations))

        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Hybrid retrieval complete: {len(result.observations)} observations, "
            f"{result.total_time_ms:.0f}ms"
        )

        return result

    def _get_agents_for_paths(
        self, paths: list[RetrievalPath]
    ) -> list[RetrievalAgent]:
        """Get agents for the specified paths."""
        agents = []

        if RetrievalPath.LOCAL in paths and self.config.enable_local:
            agents.append(self.local_agent)

        if RetrievalPath.GRAPH in paths and self.config.enable_graph:
            agents.append(self.graph_agent)

        if RetrievalPath.VECTOR in paths and self.config.enable_vector:
            agents.append(self.vector_agent)

        return agents

    async def _parallel_retrieve(
        self,
        query: str,
        agents: list[RetrievalAgent],
        classification: QueryClassification,
    ) -> tuple[list[Observation], dict[str, float]]:
        """Run agents in parallel."""
        observations: list[Observation] = []
        timings: dict[str, float] = {}

        # If graph scopes vector, run in two phases
        if classification.graph_scopes_vector and self.graph_agent in agents:
            # Phase 1: Run local and graph in parallel
            phase1_agents = [a for a in agents if a.path != RetrievalPath.VECTOR]
            phase1_tasks = []

            for agent in phase1_agents:
                task = self._timed_retrieve(agent, query)
                phase1_tasks.append(task)

            phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

            scope = Scope()
            for agent, result in zip(phase1_agents, phase1_results):
                if isinstance(result, Exception):
                    logger.warning(f"{agent.path.value} agent failed: {result}")
                    continue

                obs_list, elapsed = result
                observations.extend(obs_list)
                timings[agent.path.value] = elapsed

                # Extract scope from graph results
                if agent.path == RetrievalPath.GRAPH:
                    projects = set()
                    for obs in obs_list:
                        if obs.source_project:
                            projects.add(obs.source_project)
                    scope.projects = list(projects)

            # Phase 2: Run vector with scope (if applicable)
            if self.vector_agent in agents:
                if scope.projects:
                    logger.info(f"Scoping vector search to projects: {scope.projects}")

                obs_list, elapsed = await self._timed_retrieve(
                    self.vector_agent, query, scope if scope.projects else None
                )
                observations.extend(obs_list)
                timings["vector"] = elapsed

        else:
            # Run all agents in parallel
            tasks = [self._timed_retrieve(agent, query) for agent in agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent, result in zip(agents, results):
                if isinstance(result, Exception):
                    logger.warning(f"{agent.path.value} agent failed: {result}")
                    continue

                obs_list, elapsed = result
                observations.extend(obs_list)
                timings[agent.path.value] = elapsed

        return observations, timings

    async def _sequential_retrieve(
        self,
        query: str,
        agents: list[RetrievalAgent],
        classification: QueryClassification,
    ) -> tuple[list[Observation], dict[str, float]]:
        """Run agents sequentially."""
        observations: list[Observation] = []
        timings: dict[str, float] = {}
        scope = None

        for agent in agents:
            try:
                # Use scope from graph for vector
                if (
                    agent.path == RetrievalPath.VECTOR
                    and classification.graph_scopes_vector
                    and scope
                ):
                    obs_list, elapsed = await self._timed_retrieve(
                        agent, query, scope
                    )
                else:
                    obs_list, elapsed = await self._timed_retrieve(agent, query)

                observations.extend(obs_list)
                timings[agent.path.value] = elapsed

                # Extract scope from graph
                if agent.path == RetrievalPath.GRAPH:
                    projects = set()
                    for obs in obs_list:
                        if obs.source_project:
                            projects.add(obs.source_project)
                    scope = Scope(projects=list(projects))

            except Exception as e:
                logger.warning(f"{agent.path.value} agent failed: {e}")

        return observations, timings

    async def _timed_retrieve(
        self,
        agent: RetrievalAgent,
        query: str,
        scope: Scope | None = None,
    ) -> tuple[list[Observation], float]:
        """Run retrieval with timing."""
        start = time.perf_counter()

        try:
            observations = await asyncio.wait_for(
                agent.retrieve(
                    query,
                    max_results=self.config.max_observations_per_agent,
                    scope=scope,
                ),
                timeout=self.config.agent_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning(f"{agent.path.value} agent timed out")
            observations = []

        elapsed = (time.perf_counter() - start) * 1000
        return observations, elapsed


# =============================================================================
# Convenience Functions
# =============================================================================


async def hybrid_retrieve(
    query: str,
    llm: LLMProvider,
    semantic_memory: SemanticMemoryProvider | None = None,
    vector_store: VectorStoreProvider | None = None,
    local_index: LocalIndexProvider | None = None,
) -> HybridResult:
    """Convenience function for one-off hybrid retrieval.

    Args:
        query: The query to answer
        llm: LLM provider
        semantic_memory: Optional knowledge graph
        vector_store: Optional vector store
        local_index: Optional local index

    Returns:
        HybridResult with answer
    """
    orchestrator = HybridRetrievalOrchestrator(
        llm=llm,
        semantic_memory=semantic_memory,
        vector_store=vector_store,
        local_index=local_index,
    )

    return await orchestrator.retrieve(query)

"""Two-Pass Semantic Orchestrator with Memory Integration.

This module implements the critical two-pass retrieval architecture:

PASS 1 (Pre-Expansion):
    Before expanding a statement, query memory to get context:
    - Who is "he"? → Recent mentions from working memory
    - What do we know about "tea"? → Facts from semantic memory
    - What happened before? → Episodes from episodic memory

EXPAND:
    Use LLM with retrieved context to produce richer semantic frames

PASS 2 (Post-Expansion):
    After expansion, query again with resolved entities:
    - Search for "Doug" not "he"
    - Search by synset IDs for precise matching
    - Find supporting and contradicting evidence

RE-SCORE:
    Update variant scores based on evidence

GENERATE:
    Convert winning semantic frame back to natural language

Usage:
    >>> orchestrator = TwoPassSemanticOrchestrator(memory, llm)
    >>> result = await orchestrator.process("He prefers tea in the morning")
    >>> print(result.response_text)
    "Doug prefers tea in the morning. Note: this may differ from his general coffee preference."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

# Import core types from draagon-ai (stable base types)
from draagon_ai.memory.base import Memory, MemoryType, SearchResult

# Relative imports for prototype code
from expansion import (
    ExpansionInput,
    EntityInfo,
    SemanticExpansionService,
)
from semantic_types import (
    CrossLayerEdge,
    CrossLayerRelation,
    ExpansionVariant,
    SemanticFrame,
)
from wsd import WordSenseDisambiguator

if TYPE_CHECKING:
    pass


# =============================================================================
# Protocols
# =============================================================================


class MemoryProvider(Protocol):
    """Protocol for memory providers."""

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories by query."""
        ...

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories by entity overlap."""
        ...


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Any:
        """Send chat messages and get response.

        Returns:
            Either a string directly, or a ChatResponse object with .content attribute.
        """
        ...


def _extract_llm_content(response: Any) -> str:
    """Extract string content from LLM response.

    Handles both direct string returns (from simple/mock providers)
    and ChatResponse objects (from real providers like GroqLLM).
    """
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    return str(response)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PreExpansionContext:
    """Context retrieved before semantic expansion.

    This is the output of Pass 1 - everything we know that might
    help disambiguate and expand the statement.
    """

    # Resolved entity references
    resolved_entities: dict[str, EntityInfo] = field(default_factory=dict)

    # Retrieved from working memory (recent, task-relevant)
    working_observations: list[dict[str, Any]] = field(default_factory=list)

    # Retrieved from episodic memory (summaries of past episodes)
    episodic_summaries: list[str] = field(default_factory=list)

    # Retrieved from semantic memory (facts, preferences, skills)
    semantic_facts: list[str] = field(default_factory=list)

    # Retrieved from metacognitive memory (patterns, calibration)
    metacognitive_patterns: list[str] = field(default_factory=list)

    # Relevant beliefs
    relevant_beliefs: list[str] = field(default_factory=list)

    # Raw search results for reference
    raw_results: list[SearchResult] = field(default_factory=list)

    def to_expansion_input(self, immediate_context: list[str] | None = None) -> ExpansionInput:
        """Convert to ExpansionInput for the expansion service."""
        return ExpansionInput(
            immediate_context=immediate_context or [],
            session_entities=self.resolved_entities,
            working_observations=self.working_observations,
            episodic_summaries=self.episodic_summaries,
            semantic_facts=self.semantic_facts,
            metacognitive_patterns=self.metacognitive_patterns,
            relevant_beliefs=self.relevant_beliefs,
        )


@dataclass
class VariantEvidence:
    """Evidence collected for a specific variant in Pass 2."""

    variant_id: str

    # Supporting memories
    supporting: list[SearchResult] = field(default_factory=list)

    # Contradicting memories
    contradicting: list[SearchResult] = field(default_factory=list)

    # Neutral/related memories
    related: list[SearchResult] = field(default_factory=list)

    # Cross-layer associations detected
    associations: list[CrossLayerEdge] = field(default_factory=list)

    @property
    def support_score(self) -> float:
        """Compute support score from evidence."""
        if not self.supporting and not self.contradicting:
            return 0.5  # Neutral

        total = len(self.supporting) + len(self.contradicting)
        return len(self.supporting) / total if total > 0 else 0.5

    @property
    def conflict_score(self) -> float:
        """Compute conflict severity (0 = no conflicts, 1 = major conflicts)."""
        if not self.contradicting:
            return 0.0

        # Weight by relevance scores
        total_conflict = sum(r.score for r in self.contradicting)
        return min(1.0, total_conflict / 2.0)  # Normalize


@dataclass
class DetectedConflict:
    """A detected conflict between new information and existing memory."""

    new_content: str  # What was just said
    existing_content: str  # What we already knew
    existing_memory_id: str
    conflict_type: str  # "preference_mismatch", "fact_contradiction", etc.
    severity: float  # 0-1
    needs_clarification: bool = False
    suggested_question: str | None = None


@dataclass
class StorageDecision:
    """Decision about how to store expanded information."""

    variant_id: str
    should_store: bool
    storage_layer: str  # "working", "semantic", "episodic"
    memory_type: MemoryType
    confidence: float
    synset_ids: list[str] = field(default_factory=list)
    cross_layer_links: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ProcessingResult:
    """Complete result from two-pass semantic processing."""

    # Input
    statement: str
    immediate_context: list[str] = field(default_factory=list)

    # Pass 1 output
    pre_expansion_context: PreExpansionContext | None = None

    # Expansion output
    variants: list[ExpansionVariant] = field(default_factory=list)

    # Pass 2 output
    variant_evidence: dict[str, VariantEvidence] = field(default_factory=dict)

    # Analysis
    detected_conflicts: list[DetectedConflict] = field(default_factory=list)
    storage_decisions: list[StorageDecision] = field(default_factory=list)

    # Generated response
    response_text: str = ""
    response_confidence: float = 0.0

    # Memory tracking for reinforcement learning
    # Maps memory_id -> role ("supporting", "contradicting", "context")
    used_memories: dict[str, str] = field(default_factory=dict)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def primary_variant(self) -> ExpansionVariant | None:
        """Get the highest-scored variant."""
        return self.variants[0] if self.variants else None

    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.detected_conflicts) > 0

    @property
    def processing_time_ms(self) -> float:
        """Get processing time in milliseconds."""
        if not self.completed_at:
            return 0.0
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    async def record_outcome(
        self,
        memory_provider: Any,  # LayeredMemoryProvider
        outcome: str,
        confidence: float = 1.0,
    ) -> None:
        """Record usage outcome for all memories used in this result.

        Call this after verifying the response was correct/incorrect to
        trigger memory reinforcement learning.

        Args:
            memory_provider: LayeredMemoryProvider with record_usage method
            outcome: "success" | "failure" | "neutral"
            confidence: How confident we are in the outcome (0-1)

        Example:
            result = await orchestrator.process("Doug has 3 cats")

            # After user confirms response was helpful
            await result.record_outcome(memory_provider, "success")

            # After user indicates response was wrong
            await result.record_outcome(memory_provider, "failure")
        """
        if not hasattr(memory_provider, "record_usage"):
            return

        for memory_id, role in self.used_memories.items():
            # Supporting memories get boost on success, demote on failure
            # Contradicting memories get opposite treatment
            if role == "contradicting":
                # If we succeeded despite contradiction, the contradiction was wrong
                # If we failed, the contradiction was right (we should have listened)
                actual_outcome = "failure" if outcome == "success" else "success"
            else:
                actual_outcome = outcome

            await memory_provider.record_usage(
                memory_id,
                actual_outcome,
                confidence=confidence,
            )


# =============================================================================
# Pre-Expansion Retriever (Pass 1)
# =============================================================================


class PreExpansionRetriever:
    """Retrieves context from memory BEFORE semantic expansion.

    This is Pass 1 of the two-pass architecture. The goal is to
    gather enough context to help the LLM produce a better expansion.

    Key queries:
    1. Entity resolution: Who is "he"? → Check recent working memory
    2. Topic context: What do we know about "tea"? → Semantic memory
    3. Recent history: What happened before? → Episodic memory
    """

    def __init__(
        self,
        memory: MemoryProvider,
        max_working_results: int = 5,
        max_episodic_results: int = 3,
        max_semantic_results: int = 5,
    ):
        self.memory = memory
        self.max_working_results = max_working_results
        self.max_episodic_results = max_episodic_results
        self.max_semantic_results = max_semantic_results

    async def retrieve(
        self,
        statement: str,
        immediate_context: list[str] | None = None,
    ) -> PreExpansionContext:
        """Retrieve pre-expansion context from memory.

        Args:
            statement: The statement to be expanded
            immediate_context: Recent utterances (if any)

        Returns:
            PreExpansionContext with all retrieved information
        """
        context = PreExpansionContext()

        # Extract potential entities from statement
        potential_entities = self._extract_potential_entities(statement)

        # Query 1: Working memory for recent observations
        working_results = await self._query_working_memory(
            statement, potential_entities
        )
        context.working_observations = [
            {"content": r.memory.content, "source": r.memory.source}
            for r in working_results
        ]
        context.raw_results.extend(working_results)

        # Try to resolve pronouns from working memory
        context.resolved_entities = self._resolve_entities_from_results(
            potential_entities, working_results, immediate_context or []
        )

        # Query 2: Semantic memory for facts
        semantic_results = await self._query_semantic_memory(
            statement, list(context.resolved_entities.keys())
        )
        context.semantic_facts = [r.memory.content for r in semantic_results]
        context.raw_results.extend(semantic_results)

        # Query 3: Episodic memory for past episodes
        episodic_results = await self._query_episodic_memory(
            statement, list(context.resolved_entities.keys())
        )
        context.episodic_summaries = [r.memory.content for r in episodic_results]
        context.raw_results.extend(episodic_results)

        # Query 4: Beliefs (from semantic memory with belief type)
        belief_results = await self._query_beliefs(
            list(context.resolved_entities.keys())
        )
        context.relevant_beliefs = [r.memory.content for r in belief_results]
        context.raw_results.extend(belief_results)

        return context

    def _extract_potential_entities(self, statement: str) -> list[str]:
        """Extract potential entity mentions from statement.

        This is a simple extraction - just capitalized words and pronouns.
        Real WSD will happen during expansion.
        """
        # Pronouns that need resolution
        pronouns = {"he", "she", "they", "it", "we", "his", "her", "their", "its"}

        # Find capitalized words (potential proper nouns)
        words = statement.split()
        entities = []

        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)

            # Check for pronouns
            if clean_word.lower() in pronouns:
                entities.append(clean_word.lower())

            # Check for capitalized words (skip first word of sentence)
            elif i > 0 and clean_word and clean_word[0].isupper():
                entities.append(clean_word)

        # Also extract key nouns (simple heuristic)
        keywords = re.findall(r'\b[a-z]{4,}\b', statement.lower())
        entities.extend(keywords[:5])  # Top 5 keywords

        return list(set(entities))

    async def _query_working_memory(
        self,
        statement: str,
        entities: list[str],
    ) -> list[SearchResult]:
        """Query working memory for recent observations."""
        # Search by statement
        results = await self.memory.search(
            statement,
            memory_types=[MemoryType.OBSERVATION],
            limit=self.max_working_results,
        )

        # Also search by entities
        if entities:
            entity_results = await self.memory.search_by_entities(
                entities,
                limit=self.max_working_results,
            )
            # Merge, avoiding duplicates
            seen_ids = {r.memory.id for r in results}
            for r in entity_results:
                if r.memory.id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.memory.id)

        return results[:self.max_working_results]

    async def _query_semantic_memory(
        self,
        statement: str,
        resolved_entities: list[str],
    ) -> list[SearchResult]:
        """Query semantic memory for facts and preferences."""
        results = await self.memory.search(
            statement,
            memory_types=[MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.SKILL],
            limit=self.max_semantic_results,
        )

        # Also search by resolved entities
        if resolved_entities:
            entity_results = await self.memory.search_by_entities(
                resolved_entities,
                limit=self.max_semantic_results,
            )
            seen_ids = {r.memory.id for r in results}
            for r in entity_results:
                if r.memory.id not in seen_ids and r.memory.memory_type in [
                    MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.SKILL
                ]:
                    results.append(r)
                    seen_ids.add(r.memory.id)

        return results[:self.max_semantic_results]

    async def _query_episodic_memory(
        self,
        statement: str,
        resolved_entities: list[str],
    ) -> list[SearchResult]:
        """Query episodic memory for past episodes."""
        results = await self.memory.search(
            statement,
            memory_types=[MemoryType.EPISODIC],
            limit=self.max_episodic_results,
        )

        return results

    async def _query_beliefs(
        self,
        entities: list[str],
    ) -> list[SearchResult]:
        """Query for beliefs about entities."""
        if not entities:
            return []

        results = await self.memory.search_by_entities(
            entities,
            limit=3,
        )

        # Filter to belief type
        return [r for r in results if r.memory.memory_type == MemoryType.BELIEF]

    def _resolve_entities_from_results(
        self,
        potential_entities: list[str],
        results: list[SearchResult],
        immediate_context: list[str],
    ) -> dict[str, EntityInfo]:
        """Try to resolve pronouns to named entities.

        Strategy:
        1. Look for recent mentions in immediate context
        2. Look for entities in retrieved memories
        3. Associate pronouns with most recently mentioned names
        """
        resolved: dict[str, EntityInfo] = {}

        # Find named entities from results
        named_entities: list[tuple[str, str]] = []  # (name, type)
        for result in results:
            for entity in result.memory.entities:
                if entity and entity[0].isupper():
                    # Guess type based on context (simple heuristic)
                    entity_type = "PERSON" if any(
                        p in entity.lower() for p in ["person", "user", "doug", "john", "mary"]
                    ) else "ENTITY"
                    named_entities.append((entity, entity_type))

        # Find names in immediate context
        context_text = " ".join(immediate_context)
        for match in re.finditer(r'\b([A-Z][a-z]+)\b', context_text):
            name = match.group(1)
            named_entities.append((name, "PERSON"))

        # Resolve pronouns to most recent name
        for entity in potential_entities:
            if entity.lower() in {"he", "his"}:
                # Find most recent male name
                for name, etype in named_entities:
                    if etype == "PERSON":
                        resolved[name] = EntityInfo(entity_type="PERSON")
                        break
            elif entity.lower() in {"she", "her"}:
                # Find most recent female name (would need NER, use first name)
                for name, etype in named_entities:
                    if etype == "PERSON":
                        resolved[name] = EntityInfo(entity_type="PERSON")
                        break
            elif entity[0].isupper():
                # Already a proper noun
                resolved[entity] = EntityInfo(entity_type="ENTITY")

        return resolved


# =============================================================================
# Post-Expansion Retriever (Pass 2)
# =============================================================================


class PostExpansionRetriever:
    """Retrieves evidence from memory AFTER semantic expansion.

    This is Pass 2 of the two-pass architecture. Now we have:
    - Resolved entities (Doug, not "he")
    - Synset IDs (tea.n.01, not just "tea")
    - Semantic triples

    We can do more precise queries to find:
    - Supporting evidence
    - Contradicting evidence
    - Cross-layer associations
    """

    # LLM prompt for semantic conflict detection (follows XML output convention)
    CONFLICT_DETECTION_PROMPT = """Analyze whether these two statements are semantically compatible.

Statement A (new): "{statement_a}"
Statement B (existing memory): "{statement_b}"

Classify the relationship:
- SUPPORTING: Statement B provides evidence that supports Statement A
- CONTRADICTING: Statement B conflicts with or contradicts Statement A (different numbers, opposite preferences, etc.)
- RELATED: Statement B is related but neither clearly supports nor contradicts Statement A

Output in XML format:
<classification>
<relationship>SUPPORTING|CONTRADICTING|RELATED</relationship>
<confidence>0.0-1.0</confidence>
<reason>Brief explanation</reason>
</classification>"""

    def __init__(
        self,
        memory: MemoryProvider,
        llm: LLMProvider | None = None,
        similarity_threshold: float = 0.6,
        contradiction_threshold: float = 0.7,
    ):
        self.memory = memory
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold

    async def retrieve_evidence(
        self,
        variant: ExpansionVariant,
    ) -> VariantEvidence:
        """Retrieve evidence for a specific variant.

        Args:
            variant: The expansion variant to evaluate

        Returns:
            VariantEvidence with supporting/contradicting memories
        """
        evidence = VariantEvidence(variant_id=variant.variant_id)

        # Extract entities from variant
        entities = variant.frame.get_entities()

        # Search by resolved entities
        if entities:
            entity_results = await self.memory.search_by_entities(
                entities,
                limit=10,
            )

            # Classify results as supporting or contradicting
            for result in entity_results:
                classification = await self._classify_evidence(variant, result)

                if classification == "supporting":
                    evidence.supporting.append(result)
                elif classification == "contradicting":
                    evidence.contradicting.append(result)
                    # Create cross-layer association
                    evidence.associations.append(CrossLayerEdge(
                        source_node_id=variant.variant_id,
                        source_layer="working",
                        target_node_id=result.memory.id,
                        target_layer=self._get_memory_layer(result.memory),
                        relation=CrossLayerRelation.CONTRADICTS,
                        confidence=result.score,
                        context={"conflict_type": "semantic_contradiction"},
                    ))
                else:
                    evidence.related.append(result)

        # Search by each triple
        for triple in variant.frame.triples:
            triple_text = triple.to_text()
            triple_results = await self.memory.search(
                triple_text,
                limit=5,
            )

            for result in triple_results:
                if result.memory.id not in {r.memory.id for r in evidence.supporting + evidence.contradicting}:
                    classification = await self._classify_evidence(variant, result)
                    if classification == "supporting":
                        evidence.supporting.append(result)
                        evidence.associations.append(CrossLayerEdge(
                            source_node_id=variant.variant_id,
                            source_layer="working",
                            target_node_id=result.memory.id,
                            target_layer=self._get_memory_layer(result.memory),
                            relation=CrossLayerRelation.SUPPORTS,
                            confidence=result.score,
                        ))

        return evidence

    async def _classify_evidence_llm(
        self,
        variant: ExpansionVariant,
        result: SearchResult,
    ) -> tuple[str, float, str]:
        """Use LLM to classify evidence relationship.

        Returns:
            Tuple of (classification, confidence, reason)
        """
        # Build statement from variant
        statement_a = variant.frame.original_text
        if variant.frame.triples:
            statement_a = "; ".join(t.to_text() for t in variant.frame.triples)

        statement_b = result.memory.content

        prompt = self.CONFLICT_DETECTION_PROMPT.format(
            statement_a=statement_a,
            statement_b=statement_b,
        )

        raw_response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for classification
            max_tokens=200,
        )
        response = _extract_llm_content(raw_response)

        # Parse XML response
        relationship = "related"
        confidence = 0.5
        reason = ""

        if "<relationship>" in response:
            match = re.search(r"<relationship>(\w+)</relationship>", response)
            if match:
                rel = match.group(1).lower()
                if rel == "supporting":
                    relationship = "supporting"
                elif rel == "contradicting":
                    relationship = "contradicting"
                else:
                    relationship = "related"

        if "<confidence>" in response:
            match = re.search(r"<confidence>([\d.]+)</confidence>", response)
            if match:
                try:
                    confidence = float(match.group(1))
                except ValueError:
                    pass

        if "<reason>" in response:
            match = re.search(r"<reason>(.+?)</reason>", response, re.DOTALL)
            if match:
                reason = match.group(1).strip()

        return (relationship, confidence, reason)

    def _classify_evidence_heuristic(
        self,
        variant: ExpansionVariant,
        result: SearchResult,
    ) -> str:
        """Fallback heuristic classification when no LLM available.

        Note: This is a basic heuristic. Use LLM for accurate detection.
        """
        memory_content = result.memory.content.lower()

        # Check for explicit negation patterns
        negation_patterns = ["not", "doesn't", "don't", "never", "no longer", "stopped"]
        has_negation = any(p in memory_content for p in negation_patterns)

        # Check if entities match
        variant_entities = {e.lower() for e in variant.frame.get_entities()}
        memory_entities = {e.lower() for e in result.memory.entities}
        entity_overlap = variant_entities & memory_entities

        if not entity_overlap:
            return "related"

        # Check for predicate conflicts
        for triple in variant.frame.triples:
            # Look for same subject with opposite predicates
            if triple.subject.lower() in memory_entities:
                # Check for preference/like conflicts
                if triple.predicate in ["PREFERS", "LIKES", "WANTS"]:
                    # If memory says they like something ELSE, potential conflict
                    if triple.object.lower() not in memory_content:
                        if any(w in memory_content for w in ["prefer", "like", "want", "love"]):
                            if has_negation:
                                return "contradicting"
                            # Different preference mentioned
                            return "related"  # Could be complementary
                    else:
                        # Same object mentioned
                        if has_negation:
                            return "contradicting"
                        return "supporting"

        # Default based on score
        if result.score >= self.similarity_threshold:
            return "supporting" if not has_negation else "contradicting"

        return "related"

    async def _classify_evidence(
        self,
        variant: ExpansionVariant,
        result: SearchResult,
    ) -> str:
        """Classify a search result as supporting, contradicting, or related.

        Uses LLM if available for accurate semantic comparison,
        falls back to heuristic otherwise.
        """
        if self.llm:
            classification, confidence, reason = await self._classify_evidence_llm(variant, result)
            # Store reason for debugging (could be added to evidence)
            return classification
        else:
            return self._classify_evidence_heuristic(variant, result)

    def _get_memory_layer(self, memory: Memory) -> str:
        """Determine which memory layer a memory belongs to."""
        if memory.memory_type in [MemoryType.OBSERVATION]:
            return "working"
        elif memory.memory_type in [MemoryType.EPISODIC]:
            return "episodic"
        elif memory.memory_type in [MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.SKILL]:
            return "semantic"
        elif memory.memory_type in [MemoryType.INSIGHT, MemoryType.SELF_KNOWLEDGE]:
            return "metacognitive"
        return "unknown"


# =============================================================================
# Natural Language Generator
# =============================================================================


class NaturalLanguageGenerator:
    """Converts semantic frames back to natural language.

    Takes a semantic frame and generates a human-readable response.
    Can also generate clarification questions for conflicts.
    """

    NLG_PROMPT = '''Convert this semantic understanding into a natural, conversational response.

Semantic Frame:
- Statement: "{original_text}"
- Understood as: {triples_text}
- Presuppositions: {presuppositions_text}
- Implications: {implications_text}
{conflicts_section}

Context:
{context_text}

Generate a response that:
1. Confirms the understanding naturally (don't be robotic)
2. Notes any important conflicts or ambiguities
3. Asks for clarification if needed

Respond with just the natural language text, no XML or formatting.'''

    CONFLICT_SECTION = '''
Conflicts detected:
{conflicts_text}
These should be acknowledged diplomatically.'''

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    async def generate(
        self,
        variant: ExpansionVariant,
        evidence: VariantEvidence | None = None,
        conflicts: list[DetectedConflict] | None = None,
        context: PreExpansionContext | None = None,
    ) -> str:
        """Generate natural language from a semantic frame.

        Args:
            variant: The expansion variant to verbalize
            evidence: Evidence collected in Pass 2
            conflicts: Detected conflicts
            context: Pre-expansion context

        Returns:
            Natural language text
        """
        if not self.llm:
            # Fallback: Simple template-based generation
            return self._generate_simple(variant, conflicts)

        # Build prompt
        frame = variant.frame

        # Format triples
        triples_text = "; ".join(t.to_text() for t in frame.triples) or "No specific relations extracted"

        # Format presuppositions
        presuppositions_text = "; ".join(
            p.content for p in frame.presuppositions[:3]
        ) or "None identified"

        # Format implications
        implications_text = "; ".join(
            f"{i.content} (confidence: {i.confidence:.0%})"
            for i in frame.implications[:3]
        ) or "None identified"

        # Build conflicts section
        conflicts_section = ""
        if conflicts:
            conflicts_text = "\n".join(
                f"- {c.new_content} vs existing: {c.existing_content}"
                for c in conflicts
            )
            conflicts_section = self.CONFLICT_SECTION.format(conflicts_text=conflicts_text)

        # Build context section
        context_text = ""
        if context:
            if context.semantic_facts:
                context_text += "Known facts:\n" + "\n".join(f"- {f}" for f in context.semantic_facts[:3])
            if context.relevant_beliefs:
                context_text += "\nBeliefs:\n" + "\n".join(f"- {b}" for b in context.relevant_beliefs[:2])
        context_text = context_text or "No additional context"

        prompt = self.NLG_PROMPT.format(
            original_text=frame.original_text,
            triples_text=triples_text,
            presuppositions_text=presuppositions_text,
            implications_text=implications_text,
            conflicts_section=conflicts_section,
            context_text=context_text,
        )

        raw_response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return _extract_llm_content(raw_response).strip()

    def _generate_simple(
        self,
        variant: ExpansionVariant,
        conflicts: list[DetectedConflict] | None = None,
    ) -> str:
        """Simple template-based NLG when no LLM available."""
        frame = variant.frame
        parts = []

        # Generate main understanding
        if frame.triples:
            triple_texts = [t.to_text() for t in frame.triples]
            parts.append(f"Understood: {'; '.join(triple_texts)}")
        else:
            parts.append(f"Understood: {frame.original_text}")

        # Add confidence
        parts.append(f"Confidence: {variant.combined_score:.0%}")

        # Note conflicts
        if conflicts:
            conflict_notes = [
                f"Note: {c.new_content} may conflict with: {c.existing_content}"
                for c in conflicts[:2]
            ]
            parts.extend(conflict_notes)

        # Add open questions
        if frame.open_questions:
            parts.append(f"Questions: {frame.open_questions[0]}")

        return " | ".join(parts)

    async def generate_clarification_question(
        self,
        conflict: DetectedConflict,
    ) -> str:
        """Generate a natural clarification question for a conflict."""
        if conflict.suggested_question:
            return conflict.suggested_question

        if self.llm:
            prompt = f'''Generate a natural, conversational question to clarify this conflict:

New information: "{conflict.new_content}"
Existing knowledge: "{conflict.existing_content}"
Conflict type: {conflict.conflict_type}

The question should be:
1. Polite and not accusatory
2. Get clarification without being pedantic
3. Sound natural in conversation

Respond with just the question.'''

            raw_response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return _extract_llm_content(raw_response).strip()

        # Simple fallback
        return f"I want to make sure I understand - {conflict.new_content}, right?"


# =============================================================================
# Two-Pass Semantic Orchestrator (Main Class)
# =============================================================================


class TwoPassSemanticOrchestrator:
    """Orchestrates two-pass semantic expansion with memory integration.

    This is the main entry point for semantically processing statements.
    It coordinates:
    1. Pre-expansion memory retrieval (Pass 1)
    2. Semantic expansion with LLM
    3. Post-expansion evidence retrieval (Pass 2)
    4. Variant re-scoring based on evidence
    5. Natural language response generation

    Usage:
        >>> orchestrator = TwoPassSemanticOrchestrator(memory, llm)
        >>> result = await orchestrator.process("He prefers tea in the morning")
        >>> print(result.response_text)
    """

    def __init__(
        self,
        memory: MemoryProvider,
        llm: LLMProvider | None = None,
        wsd: WordSenseDisambiguator | None = None,
        support_boost: float = 0.1,
        contradiction_penalty: float = 0.15,
    ):
        """Initialize the orchestrator.

        Args:
            memory: Memory provider for retrieval
            llm: LLM provider for expansion and NLG
            wsd: Word sense disambiguator
            support_boost: Score boost for supporting evidence
            contradiction_penalty: Score penalty for contradictions
        """
        self.memory = memory
        self.llm = llm
        self.support_boost = support_boost
        self.contradiction_penalty = contradiction_penalty

        # Initialize components
        self.pre_retriever = PreExpansionRetriever(memory)
        self.post_retriever = PostExpansionRetriever(memory, llm=llm)  # LLM for conflict detection
        self.expansion_service = SemanticExpansionService(llm=llm, wsd=wsd)
        self.nlg = NaturalLanguageGenerator(llm)

    async def process(
        self,
        statement: str,
        immediate_context: list[str] | None = None,
        max_variants: int = 3,
    ) -> ProcessingResult:
        """Process a statement through the two-pass pipeline.

        Args:
            statement: The statement to process
            immediate_context: Recent utterances for context
            max_variants: Maximum variants to generate

        Returns:
            ProcessingResult with all outputs
        """
        result = ProcessingResult(
            statement=statement,
            immediate_context=immediate_context or [],
        )

        # === PASS 1: Pre-Expansion Retrieval ===
        result.pre_expansion_context = await self.pre_retriever.retrieve(
            statement, immediate_context
        )

        # Track pre-expansion memories for reinforcement
        for search_result in result.pre_expansion_context.raw_results:
            result.used_memories[search_result.memory.id] = "context"

        # === DETECT MEMORY-TO-MEMORY CONFLICTS ===
        # Check if retrieved memories conflict with each other
        await self._detect_memory_conflicts(result)

        # === EXPANSION ===
        expansion_input = result.pre_expansion_context.to_expansion_input(
            immediate_context
        )
        result.variants = await self.expansion_service.expand(
            statement, expansion_input, max_variants
        )

        if not result.variants:
            result.completed_at = datetime.now()
            result.response_text = f"Unable to process: {statement}"
            return result

        # === PASS 2: Post-Expansion Retrieval ===
        for variant in result.variants:
            evidence = await self.post_retriever.retrieve_evidence(variant)
            result.variant_evidence[variant.variant_id] = evidence

            # Re-score variant based on evidence
            self._rescore_variant(variant, evidence)

            # Track used memories for reinforcement learning
            for supporting in evidence.supporting:
                result.used_memories[supporting.memory.id] = "supporting"
            for contradicting in evidence.contradicting:
                result.used_memories[contradicting.memory.id] = "contradicting"
            for related in evidence.related:
                result.used_memories[related.memory.id] = "context"

            # Detect conflicts
            for contradiction in evidence.contradicting:
                result.detected_conflicts.append(DetectedConflict(
                    new_content=statement,
                    existing_content=contradiction.memory.content,
                    existing_memory_id=contradiction.memory.id,
                    conflict_type="semantic_contradiction",
                    severity=contradiction.score,
                    needs_clarification=contradiction.score > 0.7,
                ))

        # Re-sort variants by updated scores
        result.variants.sort(key=lambda v: v.combined_score, reverse=True)

        # === GENERATE RESPONSE ===
        primary = result.variants[0]
        primary_evidence = result.variant_evidence.get(primary.variant_id)

        result.response_text = await self.nlg.generate(
            primary,
            evidence=primary_evidence,
            conflicts=result.detected_conflicts[:2],  # Top 2 conflicts
            context=result.pre_expansion_context,
        )
        result.response_confidence = primary.combined_score

        # === STORAGE DECISIONS ===
        result.storage_decisions = self._make_storage_decisions(result)

        result.completed_at = datetime.now()
        return result

    def _rescore_variant(
        self,
        variant: ExpansionVariant,
        evidence: VariantEvidence,
    ) -> None:
        """Update variant scores based on evidence.

        Boost score for supporting evidence, penalize for contradictions.
        """
        # Boost for supporting evidence
        support_count = len(evidence.supporting)
        if support_count > 0:
            boost = min(0.3, self.support_boost * support_count)
            variant.semantic_memory_weight = min(1.0, variant.semantic_memory_weight + boost)

        # Penalty for contradictions
        contradiction_count = len(evidence.contradicting)
        if contradiction_count > 0:
            penalty = min(0.4, self.contradiction_penalty * contradiction_count)
            variant.belief_weight = max(0.1, variant.belief_weight - penalty)
            # Also reduce base confidence
            variant.base_confidence = max(0.3, variant.base_confidence - penalty / 2)

    # LLM prompt for memory-to-memory conflict detection
    MEMORY_CONFLICT_PROMPT = """Do these two memories about the same topic conflict?

Memory A: "{memory_a}"
Memory B: "{memory_b}"

Classify:
- CONFLICTING: They say contradictory things (different numbers, opposite preferences, incompatible facts)
- CONSISTENT: They are compatible or complementary
- UNRELATED: They are about different topics

Output in XML format:
<classification>
<relationship>CONFLICTING|CONSISTENT|UNRELATED</relationship>
<confidence>0.0-1.0</confidence>
<reason>Brief explanation</reason>
</classification>"""

    async def _detect_memory_conflicts(self, result: ProcessingResult) -> None:
        """Detect conflicts between retrieved memories.

        This checks if any of the pre-expansion memories conflict with each other.
        For example, if memory says "Doug has 3 cats" and another says "Doug has 6 cats",
        we should detect this as a conflict.
        """
        memories = result.pre_expansion_context.raw_results
        if len(memories) < 2:
            return

        # Compare pairs of memories with overlapping entities
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i + 1:]:
                # Check if they share entities (likely about the same thing)
                entities_a = set(e.lower() for e in mem_a.memory.entities)
                entities_b = set(e.lower() for e in mem_b.memory.entities)

                if not entities_a or not entities_b:
                    continue

                # Only compare if they share at least one entity
                shared_entities = entities_a & entities_b
                if not shared_entities:
                    continue

                # Use LLM to check for conflict if available
                is_conflict = False
                severity = 0.5

                if self.llm:
                    prompt = self.MEMORY_CONFLICT_PROMPT.format(
                        memory_a=mem_a.memory.content,
                        memory_b=mem_b.memory.content,
                    )

                    raw_response = await self.llm.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=200,
                    )
                    response = _extract_llm_content(raw_response)

                    # Parse response
                    if "<relationship>" in response:
                        match = re.search(r"<relationship>(\w+)</relationship>", response)
                        if match and match.group(1).upper() == "CONFLICTING":
                            is_conflict = True

                            # Extract confidence
                            conf_match = re.search(r"<confidence>([\d.]+)</confidence>", response)
                            if conf_match:
                                try:
                                    severity = float(conf_match.group(1))
                                except ValueError:
                                    severity = 0.7
                else:
                    # Heuristic: Check for numbers or opposite words
                    import re as regex_module
                    numbers_a = set(regex_module.findall(r'\d+', mem_a.memory.content))
                    numbers_b = set(regex_module.findall(r'\d+', mem_b.memory.content))

                    # If both have numbers and they're different
                    if numbers_a and numbers_b and numbers_a != numbers_b:
                        is_conflict = True
                        severity = 0.8

                    # Check for opposite words
                    opposites = [
                        ("loves", "hates"), ("likes", "dislikes"),
                        ("prefers", "avoids"), ("wants", "rejects"),
                        ("yes", "no"), ("true", "false"),
                    ]
                    content_a = mem_a.memory.content.lower()
                    content_b = mem_b.memory.content.lower()
                    for word_a, word_b in opposites:
                        if (word_a in content_a and word_b in content_b) or \
                           (word_b in content_a and word_a in content_b):
                            is_conflict = True
                            severity = 0.9
                            break

                if is_conflict:
                    result.detected_conflicts.append(DetectedConflict(
                        new_content=mem_a.memory.content,
                        existing_content=mem_b.memory.content,
                        existing_memory_id=mem_b.memory.id,
                        conflict_type="memory_contradiction",
                        severity=severity,
                        needs_clarification=severity > 0.7,
                    ))
                    # Update memory roles
                    result.used_memories[mem_a.memory.id] = "contradicting"
                    result.used_memories[mem_b.memory.id] = "contradicting"

    def _make_storage_decisions(
        self,
        result: ProcessingResult,
    ) -> list[StorageDecision]:
        """Decide how to store the processed information."""
        decisions = []

        primary = result.primary_variant
        if not primary:
            return decisions

        # High confidence without conflicts -> store as fact/preference
        if primary.combined_score >= 0.8 and not result.has_conflicts:
            # Determine memory type from frame
            memory_type = MemoryType.FACT
            if any(t.predicate in ["PREFERS", "LIKES", "WANTS"] for t in primary.frame.triples):
                memory_type = MemoryType.PREFERENCE

            decisions.append(StorageDecision(
                variant_id=primary.variant_id,
                should_store=True,
                storage_layer="semantic",
                memory_type=memory_type,
                confidence=primary.combined_score,
                synset_ids=[
                    ws.synset_id for ws in primary.frame.word_senses.values()
                ],
                reason="High confidence, no conflicts",
            ))

        # Has conflicts -> store as observation pending reconciliation
        elif result.has_conflicts:
            decisions.append(StorageDecision(
                variant_id=primary.variant_id,
                should_store=True,
                storage_layer="working",
                memory_type=MemoryType.OBSERVATION,
                confidence=primary.combined_score,
                reason=f"Conflicts detected: {len(result.detected_conflicts)}",
            ))

        # Medium confidence -> store as observation
        elif primary.combined_score >= 0.5:
            decisions.append(StorageDecision(
                variant_id=primary.variant_id,
                should_store=True,
                storage_layer="working",
                memory_type=MemoryType.OBSERVATION,
                confidence=primary.combined_score,
                reason="Medium confidence, stored as observation",
            ))

        # Low confidence -> don't store
        else:
            decisions.append(StorageDecision(
                variant_id=primary.variant_id,
                should_store=False,
                storage_layer="",
                memory_type=MemoryType.OBSERVATION,
                confidence=primary.combined_score,
                reason="Confidence too low to store",
            ))

        return decisions


# =============================================================================
# Convenience Function
# =============================================================================


async def process_with_memory(
    statement: str,
    memory: MemoryProvider,
    llm: LLMProvider | None = None,
    context: list[str] | None = None,
) -> ProcessingResult:
    """Convenience function for two-pass semantic processing.

    Args:
        statement: The statement to process
        memory: Memory provider
        llm: LLM provider (optional)
        context: Recent utterances (optional)

    Returns:
        ProcessingResult with all outputs
    """
    orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)
    return await orchestrator.process(statement, context)

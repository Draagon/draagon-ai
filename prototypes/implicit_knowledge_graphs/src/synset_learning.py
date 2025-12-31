"""Runtime Synset Learning Service.

Learns new synset definitions at runtime through:
1. Detection of unknown terms during WSD processing
2. Extraction of definitions from agent outputs
3. User explanations in conversation
4. Background research agent queries

Key features:
- Confidence-based threshold system (not just found/not-found)
- Reinforcement loop that increases/decreases confidence based on usage
- LLM-driven definition extraction (no regex for semantic understanding)
- Hybrid storage: in-memory + Qdrant for persistence

Example:
    >>> from synset_learning import SynsetLearningService, SynsetLearningConfig
    >>>
    >>> service = SynsetLearningService(
    ...     config=SynsetLearningConfig(confidence_threshold=0.8),
    ...     evolving_db=evolving_db,
    ...     llm=llm_provider,
    ... )
    >>>
    >>> # Record unknown term during WSD
    >>> await service.record_unknown_term("ArgoCD", context="GitOps deployment")
    >>>
    >>> # Later, extract definition from agent output
    >>> learned = await service.extract_definitions_from_output(
    ...     output="ArgoCD is a declarative GitOps tool...",
    ...     unknown_terms=["ArgoCD"],
    ... )
    >>>
    >>> # Reinforce based on usage outcome
    >>> await service.reinforce("argocd.tech.01", success=True)
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from identifiers import LearnedSynset, SynsetInfo, SynsetSource

if TYPE_CHECKING:
    from evolving_synsets import EvolvingSynsetDatabase

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
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str: ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory storage."""

    async def store(
        self,
        content: str,
        memory_type: str,
        scope: str,
        **kwargs: Any,
    ) -> Any: ...

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# =============================================================================
# Configuration
# =============================================================================


class ResolutionSource(str, Enum):
    """How an unknown term was resolved."""
    AGENT_OUTPUT = "agent_output"  # Extracted from agent's response
    USER_EXPLANATION = "user_explanation"  # User explained it
    RESEARCH = "research"  # Background research agent
    SIMILAR_MATCH = "similar_match"  # Found similar definition in DB
    UNRESOLVED = "unresolved"  # Still unknown


@dataclass
class SynsetLearningConfig:
    """Configuration for synset learning service.

    Attributes:
        confidence_threshold: Minimum confidence to consider a synset "known" (0.0-1.0)
            Below this threshold, the term is considered uncertain and may need
            reinforcement or re-research. Default: 0.8 (80%)

        min_confidence_for_use: Minimum confidence to use a learned synset (0.0-1.0)
            Synsets below this won't be returned by queries. Default: 0.5 (50%)

        reinforcement_boost: How much to increase confidence on success
            Default: 0.05 (5% per success)

        reinforcement_penalty: How much to decrease confidence on failure
            Default: 0.08 (8% per failure - penalize more than reward)

        max_confidence: Maximum confidence cap
            Default: 0.99 (never 100% certain)

        min_confidence: Minimum confidence floor
            Default: 0.1 (10% - allow recovery)

        unknown_term_ttl_seconds: How long to keep unknown terms in buffer
            Default: 300 (5 minutes)

        max_extraction_attempts: Max times to try extracting definition
            Default: 3

        definition_extraction_temperature: LLM temperature for extraction
            Default: 0.3 (low for consistency)

        max_extraction_chars: Maximum characters to send to LLM for extraction
            Default: 4000 (fits most context windows)

        chunk_overlap_chars: Overlap between chunks when processing large texts
            Default: 200 (preserve context continuity)
    """
    confidence_threshold: float = 0.8
    min_confidence_for_use: float = 0.5
    reinforcement_boost: float = 0.05
    reinforcement_penalty: float = 0.08
    max_confidence: float = 0.99
    min_confidence: float = 0.1
    unknown_term_ttl_seconds: int = 300
    max_extraction_attempts: int = 3
    definition_extraction_temperature: float = 0.3
    max_extraction_chars: int = 4000
    chunk_overlap_chars: int = 200

    # LLM prompts
    definition_extraction_prompt: str = """You are extracting term definitions from text.

Given the following unknown terms and text that may contain their definitions,
extract any definitions you can find.

Unknown terms: {terms}

Text to analyze:
{text}

For each term that is defined or explained in the text, provide:
1. The term (exactly as given)
2. A concise definition (1-2 sentences)
3. Part of speech (n=noun, v=verb, adj=adjective, adv=adverb)
4. Example usage (from the text if available)
5. Domain category (TECHNOLOGY, AI_ML, CLOUD_INFRASTRUCTURE, DEVOPS_CICD, etc.)
6. Your confidence that this definition is correct (0.0-1.0)

Output format (XML):
<definitions>
  <definition>
    <term>the term</term>
    <explanation>concise definition</explanation>
    <pos>n</pos>
    <example>example sentence</example>
    <domain>TECHNOLOGY</domain>
    <confidence>0.85</confidence>
  </definition>
</definitions>

If a term is mentioned but not actually defined, do not include it.
Only include terms where you found a clear definition or explanation.
If no terms are defined, output: <definitions></definitions>
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class UnknownTermRecord:
    """A term that couldn't be resolved during WSD."""
    term: str
    context: str
    pos: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    attempts: int = 1
    resolved: bool = False
    resolution_source: ResolutionSource = ResolutionSource.UNRESOLVED
    resolution_synset_id: str | None = None

    @property
    def is_expired(self) -> bool:
        """Check if this record has expired (5 minute default)."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > 300  # 5 minutes


@dataclass
class DefinitionExtraction:
    """A definition extracted from text."""
    term: str
    explanation: str
    pos: str
    example: str
    domain: str
    confidence: float
    source_text: str = ""

    def to_learned_synset(self, synset_id: str | None = None) -> LearnedSynset:
        """Convert to LearnedSynset."""
        if synset_id is None:
            # Generate ID: term.domain_code.01
            domain_code = self._domain_to_code(self.domain)
            synset_id = f"{self.term.lower().replace(' ', '_')}.{domain_code}.01"

        return LearnedSynset(
            synset_id=synset_id,
            word=self.term.lower(),
            pos=self.pos or "n",
            definition=self.explanation,
            examples=[self.example] if self.example else [],
            hypernyms=[],
            hyponyms=[],
            aliases=[],
            domain=self.domain,
            source=SynsetSource.LLM,
            confidence=self.confidence,
        )

    def _domain_to_code(self, domain: str) -> str:
        """Convert domain name to short code."""
        codes = {
            "TECHNOLOGY": "tech",
            "AI_ML": "ai",
            "CLOUD_INFRASTRUCTURE": "cloud",
            "DEVOPS_CICD": "devops",
            "DATABASES": "db",
            "WEB_APIS": "web",
            "OBSERVABILITY": "obs",
            "PROGRAMMING_FRAMEWORKS": "prog",
            "GENERAL_TECH": "tech",
        }
        return codes.get(domain, "tech")


@dataclass
class ReinforcementResult:
    """Result of reinforcing a synset."""
    synset_id: str
    old_confidence: float
    new_confidence: float
    success: bool
    usage_count: int
    success_rate: float
    needs_research: bool = False  # True if confidence dropped too low


# =============================================================================
# Synset Learning Service
# =============================================================================


class SynsetLearningService:
    """Service for learning synsets at runtime.

    This service manages the lifecycle of learned synsets:
    1. Recording unknown terms encountered during WSD
    2. Extracting definitions from agent outputs using LLM
    3. Persisting learned synsets with confidence tracking
    4. Reinforcement loop to increase/decrease confidence based on usage

    The key insight is that we track CONFIDENCE, not just presence:
    - A synset with 95% confidence is "known"
    - A synset with 60% confidence needs more reinforcement
    - A synset with 30% confidence should trigger re-research

    This allows the system to learn gradually and recover from mistakes.
    """

    def __init__(
        self,
        config: SynsetLearningConfig | None = None,
        evolving_db: "EvolvingSynsetDatabase | None" = None,
        llm: LLMProvider | None = None,
        memory: MemoryProvider | None = None,
        qdrant_store: "QdrantSynsetStore | None" = None,
    ):
        """Initialize the synset learning service.

        Args:
            config: Learning configuration
            evolving_db: Evolving synset database for in-memory storage
            llm: LLM provider for definition extraction
            memory: Memory provider for persistence
            qdrant_store: Qdrant store for vector-based synset storage
        """
        self.config = config or SynsetLearningConfig()
        self._evolving_db = evolving_db
        self._llm = llm
        self._memory = memory
        self._qdrant = qdrant_store

        # In-memory buffer for unknown terms
        self._unknown_terms: dict[str, UnknownTermRecord] = {}

        # Metrics
        self.metrics = {
            "terms_recorded": 0,
            "terms_resolved": 0,
            "definitions_extracted": 0,
            "reinforcements_positive": 0,
            "reinforcements_negative": 0,
            "research_triggered": 0,
        }

    # -------------------------------------------------------------------------
    # Unknown Term Recording
    # -------------------------------------------------------------------------

    async def record_unknown_term(
        self,
        term: str,
        context: str,
        pos: str | None = None,
        session_id: str = "",
    ) -> UnknownTermRecord:
        """Record a term that couldn't be resolved during WSD.

        This is called when the WSD pipeline encounters a word with no synset.
        The term is buffered for later resolution (from agent output, user
        explanation, or research).

        Args:
            term: The unknown term
            context: Surrounding context (sentence or paragraph)
            pos: Part of speech if known
            session_id: Session identifier for grouping

        Returns:
            The unknown term record (new or existing)
        """
        key = term.lower()

        # Check if already recorded
        if key in self._unknown_terms:
            existing = self._unknown_terms[key]
            existing.attempts += 1
            # Update context if new one is longer/better
            if len(context) > len(existing.context):
                existing.context = context
            return existing

        # Create new record
        record = UnknownTermRecord(
            term=term,
            context=context,
            pos=pos,
            session_id=session_id,
        )
        self._unknown_terms[key] = record
        self.metrics["terms_recorded"] += 1

        logger.debug(f"Recorded unknown term: {term}")

        # Also store in memory if available (for cross-agent visibility)
        if self._memory:
            await self._memory.store(
                content=f"Unknown term encountered: {term}",
                memory_type="observation",
                scope="session",
                metadata={
                    "term": term,
                    "context": context[:200],  # Truncate
                    "type": "unknown_synset",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        return record

    def get_unknown_terms(
        self,
        session_id: str | None = None,
        include_resolved: bool = False,
    ) -> list[UnknownTermRecord]:
        """Get all unknown terms, optionally filtered.

        Args:
            session_id: Filter by session
            include_resolved: Include already-resolved terms

        Returns:
            List of unknown term records
        """
        results = []
        for record in self._unknown_terms.values():
            if record.is_expired:
                continue
            if session_id and record.session_id != session_id:
                continue
            if not include_resolved and record.resolved:
                continue
            results.append(record)
        return results

    def get_pending_term_words(self) -> list[str]:
        """Get list of unresolved term words."""
        return [
            r.term for r in self._unknown_terms.values()
            if not r.resolved and not r.is_expired
        ]

    def clear_expired_terms(self) -> int:
        """Remove expired unknown term records.

        Returns:
            Number of records removed
        """
        expired = [
            k for k, v in self._unknown_terms.items()
            if v.is_expired
        ]
        for key in expired:
            del self._unknown_terms[key]
        return len(expired)

    # -------------------------------------------------------------------------
    # Definition Extraction (LLM-Driven)
    # -------------------------------------------------------------------------

    def _get_chunker(self):
        """Lazy initialization of text chunker."""
        if not hasattr(self, '_chunker') or self._chunker is None:
            from text_chunking import TextChunker, ChunkingConfig
            self._chunker = TextChunker(ChunkingConfig(
                max_chunk_tokens=int(self.config.max_extraction_chars / 4),
                overlap_tokens=int(self.config.chunk_overlap_chars / 4),
            ))
        return self._chunker

    async def extract_definitions_from_output(
        self,
        output: str,
        unknown_terms: list[str] | None = None,
        tool_results: list[dict] | None = None,
    ) -> list[DefinitionExtraction]:
        """Extract definitions from agent output using LLM.

        This is the core LLM-driven extraction. We do NOT use regex to find
        definitions - the LLM understands semantically whether a term was
        actually defined or just mentioned.

        For large texts, the system:
        1. Chunks the text intelligently at sentence boundaries
        2. Prioritizes chunks containing the target terms
        3. Processes the most relevant chunks first
        4. Merges results across chunks

        Args:
            output: Agent's text output (can be very long)
            unknown_terms: Terms to look for (defaults to pending unknowns)
            tool_results: Optional tool results to also search

        Returns:
            List of extracted definitions
        """
        if not self._llm:
            logger.warning("No LLM provider - cannot extract definitions")
            return []

        if unknown_terms is None:
            unknown_terms = self.get_pending_term_words()

        if not unknown_terms:
            return []

        # Combine output and tool results
        full_text = output
        if tool_results:
            for result in tool_results:
                if isinstance(result, dict):
                    full_text += "\n" + str(result.get("output", ""))

        # Use smart chunking for large texts
        text_to_process = self._prepare_text_for_extraction(full_text, unknown_terms)

        # Build prompt
        prompt = self.config.definition_extraction_prompt.format(
            terms=", ".join(unknown_terms),
            text=text_to_process,
        )

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.definition_extraction_temperature,
            )

            extractions = self._parse_extraction_response(response, full_text)
            self.metrics["definitions_extracted"] += len(extractions)

            # Mark terms as resolved
            for ext in extractions:
                key = ext.term.lower()
                if key in self._unknown_terms:
                    self._unknown_terms[key].resolved = True
                    self._unknown_terms[key].resolution_source = ResolutionSource.AGENT_OUTPUT
                    self.metrics["terms_resolved"] += 1

            return extractions

        except Exception as e:
            logger.error(f"Failed to extract definitions: {e}")
            return []

    def _prepare_text_for_extraction(
        self,
        text: str,
        target_terms: list[str],
    ) -> str:
        """Prepare text for definition extraction, chunking if necessary.

        For short texts, returns the text unchanged.
        For long texts, extracts and combines the most relevant chunks.

        Args:
            text: The full text
            target_terms: Terms we're looking for

        Returns:
            Text ready for LLM processing (within size limits)
        """
        max_chars = self.config.max_extraction_chars

        if len(text) <= max_chars:
            return text

        # Use smart chunking
        chunker = self._get_chunker()
        chunks = chunker.chunk_for_definition_extraction(
            text,
            target_terms,
            max_chars=max_chars,
        )

        if not chunks:
            return text[:max_chars]

        # Combine chunks up to max size, prioritizing those with target terms
        combined = []
        current_length = 0

        for chunk in chunks:
            if current_length + len(chunk.text) + 10 > max_chars:
                break
            combined.append(chunk.text)
            current_length += len(chunk.text) + 10  # +10 for separator

        if not combined:
            # At least include first chunk truncated
            return chunks[0].text[:max_chars]

        return "\n\n---\n\n".join(combined)

    def _parse_extraction_response(
        self,
        response: str,
        source_text: str,
    ) -> list[DefinitionExtraction]:
        """Parse LLM extraction response (XML format)."""
        extractions = []

        # Find all <definition> blocks
        definition_pattern = re.compile(
            r"<definition>(.*?)</definition>",
            re.DOTALL | re.IGNORECASE,
        )

        for match in definition_pattern.finditer(response):
            block = match.group(1)

            # Extract fields
            term = self._extract_xml_field(block, "term")
            explanation = self._extract_xml_field(block, "explanation")
            pos = self._extract_xml_field(block, "pos") or "n"
            example = self._extract_xml_field(block, "example") or ""
            domain = self._extract_xml_field(block, "domain") or "TECHNOLOGY"
            confidence_str = self._extract_xml_field(block, "confidence")

            try:
                confidence = float(confidence_str) if confidence_str else 0.7
            except ValueError:
                confidence = 0.7

            if term and explanation:
                extractions.append(DefinitionExtraction(
                    term=term,
                    explanation=explanation,
                    pos=pos,
                    example=example,
                    domain=domain.upper(),
                    confidence=confidence,
                    source_text=source_text[:200],
                ))

        return extractions

    def _extract_xml_field(self, text: str, field: str) -> str | None:
        """Extract a field from XML-like text."""
        pattern = re.compile(
            rf"<{field}>(.*?)</{field}>",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(text)
        return match.group(1).strip() if match else None

    # -------------------------------------------------------------------------
    # User Explanation Learning
    # -------------------------------------------------------------------------

    async def learn_from_user_explanation(
        self,
        term: str,
        explanation: str,
        user_id: str = "",
        pos: str | None = None,
        domain: str = "TECHNOLOGY",
    ) -> LearnedSynset | None:
        """Learn a synset from user's explanation.

        User-provided definitions get high confidence (0.95) since users
        typically know what they're talking about when explaining terms.

        Args:
            term: The term being defined
            explanation: User's explanation
            user_id: Who provided the definition
            pos: Part of speech
            domain: Domain category

        Returns:
            The learned synset, or None if failed
        """
        synset = LearnedSynset(
            synset_id=f"{term.lower().replace(' ', '_')}.user.01",
            word=term.lower(),
            pos=pos or "n",
            definition=explanation,
            examples=[],
            hypernyms=[],
            hyponyms=[],
            aliases=[],
            domain=domain,
            source=SynsetSource.USER,
            confidence=0.95,  # High confidence for user-provided
        )

        await self.persist_learned_synset(synset)

        # Mark as resolved if it was unknown
        key = term.lower()
        if key in self._unknown_terms:
            self._unknown_terms[key].resolved = True
            self._unknown_terms[key].resolution_source = ResolutionSource.USER_EXPLANATION
            self._unknown_terms[key].resolution_synset_id = synset.synset_id
            self.metrics["terms_resolved"] += 1

        logger.info(f"Learned from user: {term} = {explanation[:50]}...")
        return synset

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    async def persist_learned_synset(
        self,
        synset: LearnedSynset,
        to_qdrant: bool = True,
    ) -> str:
        """Persist a learned synset to storage.

        Synsets are stored in:
        1. In-memory EvolvingSynsetDatabase (for immediate use)
        2. Qdrant (for persistence and semantic search)
        3. Memory provider (as a learning record)

        Args:
            synset: The synset to persist
            to_qdrant: Whether to also store in Qdrant

        Returns:
            The synset ID
        """
        # Add to in-memory database
        if self._evolving_db:
            self._evolving_db.add_synset(synset)

        # Store in Qdrant
        if to_qdrant and self._qdrant:
            await self._qdrant.store_synset(synset)

        # Store as memory record
        if self._memory:
            await self._memory.store(
                content=f"Learned definition: {synset.word} = {synset.definition}",
                memory_type="knowledge",
                scope="agent",
                confidence=synset.confidence,
                metadata={
                    "synset_id": synset.synset_id,
                    "type": "learned_synset",
                    "source": synset.source.value,
                    "domain": synset.domain,
                },
            )

        logger.info(
            f"Persisted synset: {synset.synset_id} "
            f"(confidence={synset.confidence:.2f})"
        )
        return synset.synset_id

    # -------------------------------------------------------------------------
    # Reinforcement Loop
    # -------------------------------------------------------------------------

    async def reinforce(
        self,
        synset_id: str,
        success: bool,
    ) -> ReinforcementResult | None:
        """Reinforce a synset based on usage outcome.

        This is the key to the learning system. When a synset is used:
        - Success: Increase confidence (capped at max_confidence)
        - Failure: Decrease confidence (floored at min_confidence)

        If confidence drops too low, the synset is flagged for re-research.

        Args:
            synset_id: The synset that was used
            success: Whether the usage was successful

        Returns:
            ReinforcementResult with old/new confidence, or None if not found
        """
        # Try evolving DB first
        synset = None
        if self._evolving_db:
            synset = self._evolving_db.get_synset_by_id(synset_id)

        if synset is None:
            logger.warning(f"Cannot reinforce unknown synset: {synset_id}")
            return None

        old_confidence = synset.confidence

        # Apply reinforcement
        if success:
            new_confidence = min(
                self.config.max_confidence,
                old_confidence + self.config.reinforcement_boost,
            )
            synset.record_usage(success=True)
            self.metrics["reinforcements_positive"] += 1
        else:
            new_confidence = max(
                self.config.min_confidence,
                old_confidence - self.config.reinforcement_penalty,
            )
            synset.record_usage(success=False)
            self.metrics["reinforcements_negative"] += 1

        synset.confidence = new_confidence

        # Check if needs research
        needs_research = new_confidence < self.config.min_confidence_for_use
        if needs_research:
            self.metrics["research_triggered"] += 1
            logger.warning(
                f"Synset {synset_id} confidence dropped to {new_confidence:.2f} "
                "- flagging for re-research"
            )

        # Update in Qdrant if available
        if self._qdrant:
            await self._qdrant.update_synset_confidence(synset_id, new_confidence)

        result = ReinforcementResult(
            synset_id=synset_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            success=success,
            usage_count=synset.usage_count,
            success_rate=synset.success_rate,
            needs_research=needs_research,
        )

        logger.debug(
            f"Reinforced {synset_id}: {old_confidence:.2f} → {new_confidence:.2f} "
            f"({'success' if success else 'failure'})"
        )

        return result

    async def get_synset_with_confidence_check(
        self,
        word: str,
        pos: str | None = None,
        domain: str | None = None,
    ) -> tuple[LearnedSynset | None, bool]:
        """Get a synset and check if it meets confidence threshold.

        This is the main query method that respects confidence thresholds.

        Args:
            word: Word to look up
            pos: Part of speech filter
            domain: Domain filter

        Returns:
            Tuple of (synset, is_confident) where is_confident indicates
            whether the synset meets the confidence threshold.
        """
        if not self._evolving_db:
            return None, False

        synsets = self._evolving_db.get_synsets(word, pos, domain)

        if not synsets:
            return None, False

        # Return best match
        best = synsets[0]

        # Check confidence
        is_confident = best.confidence >= self.config.confidence_threshold

        return best, is_confident

    # -------------------------------------------------------------------------
    # Querying Uncertain Terms
    # -------------------------------------------------------------------------

    def get_uncertain_synsets(
        self,
        threshold: float | None = None,
    ) -> list[LearnedSynset]:
        """Get synsets that are below confidence threshold.

        These are candidates for re-research or user confirmation.

        Args:
            threshold: Custom threshold (defaults to config.confidence_threshold)

        Returns:
            List of uncertain synsets
        """
        if not self._evolving_db:
            return []

        threshold = threshold or self.config.confidence_threshold
        uncertain = []

        for synset in self._evolving_db.get_all_synsets():
            if synset.confidence < threshold:
                uncertain.append(synset)

        # Sort by confidence ascending (most uncertain first)
        uncertain.sort(key=lambda s: s.confidence)
        return uncertain

    def get_synsets_needing_reinforcement(self) -> list[LearnedSynset]:
        """Get synsets that have been used but need more reinforcement.

        These are synsets with:
        - Confidence below threshold
        - Some usage history
        - Not from bootstrap (those are assumed correct)

        Returns:
            List of synsets needing reinforcement
        """
        if not self._evolving_db:
            return []

        needing = []
        for synset in self._evolving_db.get_all_synsets():
            if (
                synset.confidence < self.config.confidence_threshold
                and synset.usage_count > 0
                and synset.source != SynsetSource.BOOTSTRAP
            ):
                needing.append(synset)

        return needing

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get learning statistics."""
        stats = dict(self.metrics)

        if self._evolving_db:
            all_synsets = self._evolving_db.get_all_synsets()
            stats["total_learned_synsets"] = len(all_synsets)
            stats["synsets_above_threshold"] = len([
                s for s in all_synsets
                if s.confidence >= self.config.confidence_threshold
            ])
            stats["synsets_below_threshold"] = len([
                s for s in all_synsets
                if s.confidence < self.config.confidence_threshold
            ])
            stats["average_confidence"] = (
                sum(s.confidence for s in all_synsets) / len(all_synsets)
                if all_synsets else 0.0
            )

        stats["pending_unknown_terms"] = len(self.get_pending_term_words())
        stats["config"] = {
            "confidence_threshold": self.config.confidence_threshold,
            "min_confidence_for_use": self.config.min_confidence_for_use,
        }

        return stats


# =============================================================================
# Qdrant Synset Store (Stub - Full implementation in separate file)
# =============================================================================


class QdrantSynsetStore:
    """Qdrant-backed storage for learned synsets.

    This is a stub class. The full implementation will be in qdrant_synset_store.py
    and will provide:
    - Vector storage with definition embeddings
    - Semantic similarity search
    - Exact match by word/alias
    - Confidence updates
    """

    COLLECTION_NAME = "learned_synsets"

    def __init__(
        self,
        client: Any,  # QdrantClient
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize Qdrant synset store."""
        self._client = client
        self._embedder = embedding_provider
        self._synsets: dict[str, LearnedSynset] = {}  # In-memory fallback

    async def store_synset(self, synset: LearnedSynset) -> str:
        """Store a synset with its definition embedding."""
        # For now, just store in memory
        self._synsets[synset.synset_id] = synset
        return synset.synset_id

    async def search_by_word(self, word: str) -> list[LearnedSynset]:
        """Exact match search by word or alias."""
        word_lower = word.lower()
        results = []
        for synset in self._synsets.values():
            if synset.word == word_lower or word_lower in synset.aliases:
                results.append(synset)
        return results

    async def search_similar(
        self,
        definition: str,
        limit: int = 5,
    ) -> list[tuple[LearnedSynset, float]]:
        """Semantic search for similar definitions."""
        # Stub: returns empty for now
        return []

    async def update_synset_confidence(
        self,
        synset_id: str,
        confidence: float,
    ) -> bool:
        """Update confidence for a synset."""
        if synset_id in self._synsets:
            self._synsets[synset_id].confidence = confidence
            return True
        return False

    async def load_all(self) -> list[LearnedSynset]:
        """Load all synsets from Qdrant."""
        return list(self._synsets.values())

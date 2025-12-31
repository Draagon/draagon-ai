"""Unified Phase 0 + Phase 1 Pipeline.

This module provides a fully integrated pipeline that chains:
1. Phase 0: Content Analysis → WSD → Entity Classification
2. Phase 1: Decomposition → Presuppositions → Commonsense → Temporal → etc.

The pipeline handles:
- Content-aware processing (PROSE, CODE, DATA, CONFIG, MIXED)
- Automatic chunking for large documents
- Full WSD with synset resolution
- Entity type classification
- Multi-type semantic decomposition

Example:
    >>> from .integrated_pipeline import IntegratedPipeline
    >>>
    >>> pipeline = IntegratedPipeline(llm=my_llm)
    >>>
    >>> # Process any content type
    >>> result = await pipeline.process("Doug forgot the meeting again")
    >>>
    >>> # Result includes Phase 0 output (WSD, entities) + Phase 1 (decomposition)
    >>> print(result.wsd_results)  # {"forgot": DisambiguationResult(...)}
    >>> print(result.decomposition.presuppositions)  # [Presupposition(...)]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from enum import Enum

from ..content_analyzer import ContentType, ContentAnalysis, ProcessingStrategy
from content_aware_wsd import ContentAwareWSD, ContentAwareWSDResult
from ..entity_classifier import EntityClassifier, ClassificationResult
from ..identifiers import EntityType, UniversalSemanticIdentifier
from text_chunking import TextChunker, ChunkingConfig, TextChunk
from ..wsd import WSDConfig, DisambiguationResult

from .models import (
    DecomposedKnowledge,
    Presupposition,
    CommonsenseInference,
    SemanticRole,
    TemporalInfo,
    ModalityInfo,
    NegationInfo,
    WeightedBranch,
    CrossReference,
)
from .config import DecompositionConfig
from .pipeline import DecompositionPipeline


logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Any:
        """Send chat messages and get response."""
        ...


class MemoryProvider(Protocol):
    """Protocol for memory providers."""

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memory for related items."""
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class IntegratedPipelineConfig:
    """Configuration for the integrated Phase 0 + Phase 1 pipeline.

    Attributes:
        wsd_config: Configuration for Word Sense Disambiguation
        decomposition_config: Configuration for decomposition stages
        chunking_config: Configuration for document chunking
        min_nl_length: Minimum natural language text length to process
        max_content_length: Maximum content length before chunking
        require_wordnet: Raise error if WordNet unavailable
        parallel_chunk_processing: Process chunks in parallel
        max_parallel_chunks: Maximum chunks to process in parallel
    """

    wsd_config: WSDConfig = field(default_factory=WSDConfig)
    decomposition_config: DecompositionConfig = field(default_factory=DecompositionConfig)
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)

    min_nl_length: int = 20
    max_content_length: int = 8000
    require_wordnet: bool = True
    parallel_chunk_processing: bool = True
    max_parallel_chunks: int = 5


# =============================================================================
# Phase 0 Output
# =============================================================================


@dataclass
class Phase0Result:
    """Complete output from Phase 0 processing.

    Captures all identification, WSD, and entity classification results.

    Attributes:
        content_analysis: Content type analysis result
        disambiguation_results: WSD results for disambiguated words
        entity_identifiers: Universal semantic identifiers for entities
        entity_classifications: Entity type classifications
        structural_knowledge: Extracted structural knowledge (for CODE/DATA)
        processed_text: The natural language text that was processed
        skipped_wsd: Whether WSD was skipped
        skip_reason: Reason WSD was skipped (if applicable)
    """

    content_analysis: ContentAnalysis
    disambiguation_results: dict[str, DisambiguationResult] = field(default_factory=dict)
    entity_identifiers: dict[str, UniversalSemanticIdentifier] = field(default_factory=dict)
    entity_classifications: dict[str, ClassificationResult] = field(default_factory=dict)
    structural_knowledge: list[dict[str, Any]] = field(default_factory=list)
    processed_text: str = ""
    skipped_wsd: bool = False
    skip_reason: str = ""

    def get_wsd_results_for_decomposition(self) -> dict[str, str]:
        """Get WSD results in format expected by decomposition pipeline.

        Returns:
            Dict mapping word -> synset_id
        """
        return {
            word: result.synset_id
            for word, result in self.disambiguation_results.items()
        }

    def get_wsd_alternatives(self) -> dict[str, list[tuple[str, float]]]:
        """Get WSD alternatives in format expected by decomposition pipeline.

        For each ambiguous word, returns all possible senses (including primary)
        with their confidence scores. Alternatives from WSD don't have explicit
        confidence, so we distribute the remaining confidence after the primary.

        Returns:
            Dict mapping word -> list of (synset_id, confidence) tuples
        """
        alternatives = {}
        for word, result in self.disambiguation_results.items():
            if result.alternatives:
                # Build list of all senses with confidence scores
                all_senses: list[tuple[str, float]] = []

                # Primary sense gets its actual confidence
                all_senses.append((result.synset_id, result.confidence))

                # Distribute remaining confidence among alternatives
                remaining_confidence = max(0.0, 1.0 - result.confidence)
                num_alternatives = len(result.alternatives)
                per_alt_confidence = remaining_confidence / num_alternatives if num_alternatives > 0 else 0.0

                for alt_synset_id in result.alternatives:
                    # Skip if alternative is same as primary (shouldn't happen but be safe)
                    if alt_synset_id != result.synset_id:
                        all_senses.append((alt_synset_id, per_alt_confidence))

                alternatives[word] = all_senses
        return alternatives

    def get_entity_ids(self) -> list[str]:
        """Get entity IDs for decomposition pipeline."""
        return list(self.entity_identifiers.keys())

    def get_entity_types(self) -> dict[str, str]:
        """Get entity types in format expected by decomposition pipeline.

        Returns:
            Dict mapping entity text -> entity type value
        """
        return {
            text: classification.entity_type.value
            for text, classification in self.entity_classifications.items()
        }


# =============================================================================
# Integrated Result
# =============================================================================


@dataclass
class IntegratedResult:
    """Complete output from the integrated pipeline.

    Contains both Phase 0 (identification) and Phase 1 (decomposition) results.

    Attributes:
        source_text: Original input text
        content_type: Detected content type
        phase0: Phase 0 results (WSD, entities)
        decomposition: Phase 1 decomposition results
        chunks_processed: Number of chunks processed (for large docs)
        total_duration_ms: Total processing time
        processing_timestamp: When processing occurred
    """

    source_text: str
    content_type: ContentType
    phase0: Phase0Result
    decomposition: DecomposedKnowledge
    chunks_processed: int = 1
    total_duration_ms: float = 0.0
    processing_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def wsd_results(self) -> dict[str, DisambiguationResult]:
        """Convenience accessor for WSD results."""
        return self.phase0.disambiguation_results

    @property
    def entities(self) -> dict[str, UniversalSemanticIdentifier]:
        """Convenience accessor for entity identifiers."""
        return self.phase0.entity_identifiers

    @property
    def presuppositions(self) -> list[Presupposition]:
        """Convenience accessor for presuppositions."""
        return self.decomposition.presuppositions

    @property
    def inferences(self) -> list[CommonsenseInference]:
        """Convenience accessor for commonsense inferences."""
        return self.decomposition.commonsense_inferences

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_text": self.source_text,
            "content_type": self.content_type.value,
            "phase0": {
                "wsd_results": {
                    word: {
                        "synset_id": r.synset_id,
                        "confidence": r.confidence,
                        "method": r.method,
                    }
                    for word, r in self.phase0.disambiguation_results.items()
                },
                "entity_identifiers": {
                    k: v.to_dict() for k, v in self.phase0.entity_identifiers.items()
                },
                "structural_knowledge": self.phase0.structural_knowledge,
                "processed_text": self.phase0.processed_text,
            },
            "decomposition": self.decomposition.to_dict(),
            "chunks_processed": self.chunks_processed,
            "total_duration_ms": self.total_duration_ms,
            "processing_timestamp": self.processing_timestamp.isoformat(),
        }


# =============================================================================
# Integrated Pipeline
# =============================================================================


class IntegratedPipeline:
    """Unified Phase 0 + Phase 1 Pipeline.

    This is the main entry point for processing text through the complete
    semantic decomposition pipeline. It handles:

    1. Content Analysis - Determine what type of content we're processing
    2. WSD - Disambiguate word senses in natural language
    3. Entity Classification - Classify entities by type
    4. Chunking - Split large documents for processing
    5. Decomposition - Extract presuppositions, commonsense, temporal, etc.

    Example:
        >>> pipeline = IntegratedPipeline(llm=my_llm)
        >>>
        >>> # Simple text
        >>> result = await pipeline.process("Doug forgot the meeting again")
        >>> print(result.presuppositions)
        >>>
        >>> # Code with docstrings
        >>> result = await pipeline.process('''
        ...     def deposit(amount):
        ...         \"\"\"Add funds to the bank account.\"\"\"
        ...         pass
        ... ''')
        >>> # Only docstring is processed for WSD
        >>> print(result.wsd_results)  # "bank" → bank.n.01 (financial)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: IntegratedPipelineConfig | None = None,
        llm: LLMProvider | None = None,
        memory: MemoryProvider | None = None,
    ):
        """Initialize the integrated pipeline.

        Args:
            config: Pipeline configuration
            llm: LLM provider for WSD and decomposition
            memory: Memory provider for cross-referencing
        """
        self.config = config or IntegratedPipelineConfig()
        self.llm = llm
        self.memory = memory

        # Initialize Phase 0 components
        self._content_wsd = ContentAwareWSD(
            llm=llm,
            wsd_config=self.config.wsd_config,
            require_wordnet=self.config.require_wordnet,
            min_nl_length=self.config.min_nl_length,
        )
        self._entity_classifier = EntityClassifier(llm=llm)
        self._chunker = TextChunker(self.config.chunking_config)

        # Initialize Phase 1 pipeline
        self._decomposition = DecompositionPipeline(
            config=self.config.decomposition_config,
            llm=llm,
            memory=memory,
        )

        # Metrics
        self.metrics = {
            "total_processed": 0,
            "prose_processed": 0,
            "code_processed": 0,
            "data_skipped": 0,
            "config_skipped": 0,
            "chunks_processed": 0,
            "phase0_duration_ms": 0.0,
            "phase1_duration_ms": 0.0,
        }

    async def process(self, content: str) -> IntegratedResult:
        """Process content through the full integrated pipeline.

        This is the main entry point. It:
        1. Analyzes content type
        2. Routes to appropriate processing (PROSE vs CODE vs DATA)
        3. Runs Phase 0 (WSD, entities)
        4. Runs Phase 1 (decomposition)
        5. Returns integrated result

        Args:
            content: Any content (prose, code, data, etc.)

        Returns:
            IntegratedResult with Phase 0 + Phase 1 output
        """
        start_time = time.time()
        self.metrics["total_processed"] += 1

        # Check if chunking needed
        if len(content) > self.config.max_content_length:
            result = await self._process_chunked(content)
        else:
            result = await self._process_single(content)

        result.total_duration_ms = (time.time() - start_time) * 1000
        return result

    async def _process_single(self, content: str) -> IntegratedResult:
        """Process a single piece of content (no chunking)."""
        # Phase 0: Content-aware WSD
        phase0_start = time.time()
        phase0_result = await self._run_phase0(content)
        self.metrics["phase0_duration_ms"] += (time.time() - phase0_start) * 1000

        # Phase 1: Decomposition
        phase1_start = time.time()
        decomposition = await self._run_phase1(
            text=phase0_result.processed_text or content,
            phase0=phase0_result,
        )
        self.metrics["phase1_duration_ms"] += (time.time() - phase1_start) * 1000

        return IntegratedResult(
            source_text=content,
            content_type=phase0_result.content_analysis.content_type,
            phase0=phase0_result,
            decomposition=decomposition,
            chunks_processed=1,
        )

    async def _process_chunked(self, content: str) -> IntegratedResult:
        """Process large content by chunking."""
        # First, analyze the content type
        wsd_result = await self._content_wsd.process(content)

        # Get text to chunk (NL portion for prose/code, original for data/config)
        if wsd_result.content_analysis.content_type in (ContentType.DATA, ContentType.CONFIG):
            # Skip WSD/decomposition for data/config
            self.metrics["data_skipped" if wsd_result.content_analysis.content_type == ContentType.DATA else "config_skipped"] += 1
            return IntegratedResult(
                source_text=content,
                content_type=wsd_result.content_analysis.content_type,
                phase0=Phase0Result(
                    content_analysis=wsd_result.content_analysis,
                    structural_knowledge=wsd_result.structural_knowledge,
                    skipped_wsd=True,
                    skip_reason=wsd_result.skip_reason,
                ),
                decomposition=DecomposedKnowledge(
                    source_text=content,
                    pipeline_version=self.VERSION,
                ),
                chunks_processed=0,
            )

        # Chunk the processable text
        text_to_chunk = wsd_result.processed_text or content
        chunks = self._chunker.chunk_for_processing(text_to_chunk)
        self.metrics["chunks_processed"] += len(chunks)

        # Process chunks
        if self.config.parallel_chunk_processing:
            chunk_results = await self._process_chunks_parallel(chunks)
        else:
            chunk_results = await self._process_chunks_sequential(chunks)

        # Merge results
        return self._merge_chunk_results(
            content=content,
            content_analysis=wsd_result.content_analysis,
            chunk_results=chunk_results,
        )

    async def _process_chunks_parallel(
        self,
        chunks: list[TextChunk],
    ) -> list[tuple[Phase0Result, DecomposedKnowledge]]:
        """Process chunks in parallel."""
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_parallel_chunks)

        async def process_with_semaphore(chunk: TextChunk):
            async with semaphore:
                return await self._process_chunk(chunk)

        results = await asyncio.gather(
            *[process_with_semaphore(c) for c in chunks],
            return_exceptions=True,
        )

        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Chunk processing failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def _process_chunks_sequential(
        self,
        chunks: list[TextChunk],
    ) -> list[tuple[Phase0Result, DecomposedKnowledge]]:
        """Process chunks sequentially."""
        results = []
        for chunk in chunks:
            try:
                result = await self._process_chunk(chunk)
                results.append(result)
            except Exception as e:
                logger.warning(f"Chunk processing failed: {e}")
        return results

    async def _process_chunk(
        self,
        chunk: TextChunk,
    ) -> tuple[Phase0Result, DecomposedKnowledge]:
        """Process a single chunk."""
        phase0 = await self._run_phase0(chunk.text)
        decomposition = await self._run_phase1(
            text=phase0.processed_text or chunk.text,
            phase0=phase0,
        )
        return (phase0, decomposition)

    def _merge_chunk_results(
        self,
        content: str,
        content_analysis: ContentAnalysis,
        chunk_results: list[tuple[Phase0Result, DecomposedKnowledge]],
    ) -> IntegratedResult:
        """Merge results from multiple chunks."""
        if not chunk_results:
            return IntegratedResult(
                source_text=content,
                content_type=content_analysis.content_type,
                phase0=Phase0Result(content_analysis=content_analysis),
                decomposition=DecomposedKnowledge(source_text=content),
                chunks_processed=0,
            )

        # Merge Phase 0 results
        merged_wsd: dict[str, DisambiguationResult] = {}
        merged_entities: dict[str, UniversalSemanticIdentifier] = {}
        merged_classifications: dict[str, ClassificationResult] = {}
        merged_structural: list[dict[str, Any]] = []

        # Merge Phase 1 results
        merged_presuppositions: list[Presupposition] = []
        merged_inferences: list[CommonsenseInference] = []
        merged_roles: list[SemanticRole] = []
        all_processed_text: list[str] = []

        for phase0, decomp in chunk_results:
            # Merge WSD (prefer higher confidence)
            for word, result in phase0.disambiguation_results.items():
                if word not in merged_wsd or result.confidence > merged_wsd[word].confidence:
                    merged_wsd[word] = result

            # Merge entities (prefer higher confidence)
            for text, identifier in phase0.entity_identifiers.items():
                if text not in merged_entities or identifier.confidence > merged_entities[text].confidence:
                    merged_entities[text] = identifier

            # Merge classifications
            for text, classification in phase0.entity_classifications.items():
                if text not in merged_classifications or classification.confidence > merged_classifications[text].confidence:
                    merged_classifications[text] = classification

            merged_structural.extend(phase0.structural_knowledge)
            all_processed_text.append(phase0.processed_text)

            # Merge decomposition
            merged_presuppositions.extend(decomp.presuppositions)
            merged_inferences.extend(decomp.commonsense_inferences)
            merged_roles.extend(decomp.semantic_roles)

        # Deduplicate presuppositions and inferences
        merged_presuppositions = self._deduplicate_presuppositions(merged_presuppositions)
        merged_inferences = self._deduplicate_inferences(merged_inferences)

        # Create merged Phase 0 result
        merged_phase0 = Phase0Result(
            content_analysis=content_analysis,
            disambiguation_results=merged_wsd,
            entity_identifiers=merged_entities,
            entity_classifications=merged_classifications,
            structural_knowledge=merged_structural,
            processed_text=" ".join(all_processed_text),
        )

        # Create merged decomposition
        merged_decomposition = DecomposedKnowledge(
            source_text=content,
            entity_ids=list(merged_entities.keys()),
            semantic_roles=merged_roles,
            presuppositions=merged_presuppositions,
            commonsense_inferences=merged_inferences,
            pipeline_version=self.VERSION,
        )

        # Use first chunk's temporal/modality/negation as representative
        if chunk_results:
            first_decomp = chunk_results[0][1]
            merged_decomposition.temporal = first_decomp.temporal
            merged_decomposition.modality = first_decomp.modality
            merged_decomposition.negation = first_decomp.negation

        return IntegratedResult(
            source_text=content,
            content_type=content_analysis.content_type,
            phase0=merged_phase0,
            decomposition=merged_decomposition,
            chunks_processed=len(chunk_results),
        )

    def _deduplicate_presuppositions(
        self,
        presuppositions: list[Presupposition],
    ) -> list[Presupposition]:
        """Remove duplicate presuppositions, keeping highest confidence."""
        seen: dict[str, Presupposition] = {}
        for p in presuppositions:
            key = (p.content, p.trigger_type.value)
            if key not in seen or p.confidence > seen[key].confidence:
                seen[key] = p
        return list(seen.values())

    def _deduplicate_inferences(
        self,
        inferences: list[CommonsenseInference],
    ) -> list[CommonsenseInference]:
        """Remove duplicate inferences, keeping highest confidence."""
        seen: dict[str, CommonsenseInference] = {}
        for i in inferences:
            key = (i.relation.value, i.head, i.tail)
            if key not in seen or i.confidence > seen[key].confidence:
                seen[key] = i
        return list(seen.values())

    async def _run_phase0(self, content: str) -> Phase0Result:
        """Run Phase 0: Content Analysis + WSD + Entity Classification.

        Args:
            content: Content to process

        Returns:
            Phase0Result with WSD and entity results
        """
        # Step 1: Content-aware WSD
        wsd_result = await self._content_wsd.process(content, extract_all_words=False)

        # Update metrics
        if wsd_result.content_analysis.content_type == ContentType.PROSE:
            self.metrics["prose_processed"] += 1
        elif wsd_result.content_analysis.content_type == ContentType.CODE:
            self.metrics["code_processed"] += 1

        # Step 2: Entity Classification (on disambiguated words and capitalized terms)
        entity_identifiers: dict[str, UniversalSemanticIdentifier] = {}
        entity_classifications: dict[str, ClassificationResult] = {}

        nl_text = wsd_result.processed_text or content

        # Extract potential entities (capitalized words, proper nouns)
        potential_entities = self._extract_potential_entities(nl_text)

        for entity_text in potential_entities:
            # Classify entity type
            classification = await self._entity_classifier.classify(
                text=entity_text,
                context=nl_text,
            )
            entity_classifications[entity_text] = classification

            # Create universal identifier
            identifier = self._create_identifier(
                text=entity_text,
                classification=classification,
                wsd_result=wsd_result.disambiguation_results.get(entity_text.lower()),
            )
            entity_identifiers[entity_text] = identifier

        return Phase0Result(
            content_analysis=wsd_result.content_analysis,
            disambiguation_results=wsd_result.disambiguation_results,
            entity_identifiers=entity_identifiers,
            entity_classifications=entity_classifications,
            structural_knowledge=wsd_result.structural_knowledge,
            processed_text=wsd_result.processed_text,
            skipped_wsd=wsd_result.skipped_processing,
            skip_reason=wsd_result.skip_reason,
        )

    def _extract_potential_entities(self, text: str) -> list[str]:
        """Extract potential entities from text.

        Looks for:
        - Capitalized words not at sentence start
        - Multi-word proper nouns
        - Pronouns (for anaphora tracking)

        Args:
            text: Natural language text

        Returns:
            List of potential entity strings
        """
        import re

        entities = set()

        # Pattern for capitalized words
        # Match capitalized words that are not at the start of a sentence
        words = text.split()
        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = word.strip('.,!?;:"\'()[]{}')
            if not clean_word:
                continue

            # Check if capitalized
            if clean_word[0].isupper():
                # Check if at sentence start
                if i > 0:
                    prev_word = words[i - 1].strip()
                    if not prev_word.endswith(('.', '!', '?')):
                        entities.add(clean_word)
                elif len(clean_word) > 1 and clean_word[1:].islower():
                    # Single capital at start - might be proper noun
                    entities.add(clean_word)

        # Also extract multi-word proper nouns (consecutive capitalized words)
        proper_noun_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in proper_noun_pattern.finditer(text):
            entities.add(match.group(1))

        return list(entities)

    def _create_identifier(
        self,
        text: str,
        classification: ClassificationResult,
        wsd_result: DisambiguationResult | None,
    ) -> UniversalSemanticIdentifier:
        """Create a universal semantic identifier for an entity.

        Args:
            text: The entity text
            classification: Entity type classification
            wsd_result: WSD result if available

        Returns:
            UniversalSemanticIdentifier
        """
        wordnet_synset = wsd_result.synset_id if wsd_result else None
        confidence = (classification.confidence + (wsd_result.confidence if wsd_result else 0.5)) / 2

        return UniversalSemanticIdentifier(
            entity_type=classification.entity_type,
            wordnet_synset=wordnet_synset,
            canonical_name=text if classification.entity_type == EntityType.INSTANCE else None,
            confidence=confidence,
        )

    async def _run_phase1(
        self,
        text: str,
        phase0: Phase0Result,
    ) -> DecomposedKnowledge:
        """Run Phase 1: Decomposition pipeline.

        Args:
            text: Text to decompose
            phase0: Phase 0 results

        Returns:
            DecomposedKnowledge
        """
        if phase0.skipped_wsd:
            # For DATA/CONFIG, return minimal decomposition
            return DecomposedKnowledge(
                source_text=text,
                entity_ids=phase0.get_entity_ids(),
                pipeline_version=self.VERSION,
            )

        return await self._decomposition.decompose(
            text=text,
            entity_ids=phase0.get_entity_ids(),
            wsd_results=phase0.get_wsd_results_for_decomposition(),
            wsd_alternatives=phase0.get_wsd_alternatives(),
            entity_types=phase0.get_entity_types(),
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics."""
        return {
            **self.metrics,
            "content_wsd": self._content_wsd.get_metrics(),
            "entity_classifier": self._entity_classifier.get_metrics(),
            "decomposition": self._decomposition.get_last_metrics().to_dict() if self._decomposition.get_last_metrics() else None,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def process_text(
    text: str,
    llm: LLMProvider | None = None,
) -> IntegratedResult:
    """Process text through the integrated pipeline.

    Convenience function that creates a pipeline and processes text.

    Args:
        text: Text to process
        llm: Optional LLM provider

    Returns:
        IntegratedResult
    """
    pipeline = IntegratedPipeline(llm=llm)
    return await pipeline.process(text)


async def decompose_with_wsd(
    text: str,
    llm: LLMProvider | None = None,
) -> DecomposedKnowledge:
    """Decompose text with full WSD preprocessing.

    Convenience function that returns just the decomposition result.

    Args:
        text: Text to process
        llm: Optional LLM provider

    Returns:
        DecomposedKnowledge with Phase 0 data embedded
    """
    result = await process_text(text, llm)
    return result.decomposition

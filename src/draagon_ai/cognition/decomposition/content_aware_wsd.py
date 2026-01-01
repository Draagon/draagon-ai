"""Content-Aware Word Sense Disambiguation.

Integrates content type analysis with WSD to properly handle different
content types (prose, code, data, config, etc.).

Key Principle: WSD is for natural language understanding, not all semantic
understanding. Code syntax, CSV data, and JSON structures need different
processing strategies.

Processing Strategy by Content Type:
- PROSE: Full WSD on entire content
- CODE: Extract NL (docstrings, comments) → WSD on those
- DATA: Schema extraction (no WSD)
- CONFIG: Pattern matching (no WSD)
- MIXED: Split into components, apply appropriate strategy to each

Example:
    >>> from draagon_ai.cognition.decomposition import ContentAwareWSD
    >>>
    >>> # This handles code intelligently
    >>> wsd = ContentAwareWSD(llm=llm_provider)
    >>> result = await wsd.process('''
    ...     def process_bank_transaction(account):
    ...         \"\"\"Transfer funds to a financial institution.\"\"\"
    ...         pass
    ... ''')
    >>>
    >>> # Only the docstring is processed for WSD
    >>> print(result.disambiguation_results)  # "bank" → bank.n.01 (financial)
    >>>
    >>> # Prose is processed fully
    >>> result = await wsd.process("I walked to the river bank")
    >>> print(result.disambiguation_results)  # "bank" → bank.n.02 (river bank)

Migrated from prototypes/implicit_knowledge_graphs/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .content_analyzer import (
    ContentAnalyzer,
    ContentAnalysis,
    ContentType,
    ProcessingStrategy,
)
from .wsd import (
    WordSenseDisambiguator,
    WSDConfig,
    DisambiguationResult,
)

if TYPE_CHECKING:
    from .synset_learning import SynsetLearningService

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
    ) -> Any: ...


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class ContentAwareWSDResult:
    """Result from content-aware WSD processing.

    Attributes:
        content_analysis: The content type analysis
        disambiguation_results: Word sense disambiguation results for NL portions
        structural_knowledge: Any structural knowledge extracted (types, contracts)
        processed_text: The text that was actually processed for WSD
        skipped_processing: True if content was not suitable for WSD
        skip_reason: Reason why WSD was skipped (if applicable)
    """

    content_analysis: ContentAnalysis
    disambiguation_results: dict[str, DisambiguationResult] = field(default_factory=dict)
    structural_knowledge: list[dict[str, Any]] = field(default_factory=list)
    processed_text: str = ""
    skipped_processing: bool = False
    skip_reason: str = ""

    def has_disambiguations(self) -> bool:
        """Check if any disambiguations were performed."""
        return len(self.disambiguation_results) > 0

    def get_synset_ids(self) -> list[str]:
        """Get all resolved synset IDs."""
        return [r.synset_id for r in self.disambiguation_results.values()]

    def get_high_confidence_disambiguations(
        self,
        threshold: float = 0.7,
    ) -> dict[str, DisambiguationResult]:
        """Get only high-confidence disambiguation results."""
        return {
            word: result
            for word, result in self.disambiguation_results.items()
            if result.confidence >= threshold
        }


# =============================================================================
# Content-Aware WSD
# =============================================================================


class ContentAwareWSD:
    """Content-aware word sense disambiguation.

    Intelligently handles different content types by:
    1. Analyzing what type of content was provided
    2. Extracting natural language portions from code/mixed content
    3. Applying WSD only to natural language
    4. Preserving structural knowledge from code/data

    This prevents nonsensical WSD attempts on code syntax (e.g., trying
    to disambiguate "bank" in "bank.deposit()") while still extracting
    semantic value from comments and docstrings.
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        wsd_config: WSDConfig | None = None,
        require_wordnet: bool = True,
        synset_learner: "SynsetLearningService | None" = None,
        min_nl_length: int = 20,
    ):
        """Initialize content-aware WSD.

        Args:
            llm: LLM provider for content analysis and WSD
            wsd_config: Configuration for WSD (uses defaults if not provided)
            require_wordnet: If True, raise error if WordNet unavailable
            synset_learner: Optional synset learning service
            min_nl_length: Minimum NL text length to attempt WSD
        """
        self._llm = llm
        self._content_analyzer = ContentAnalyzer(llm=llm)
        self._wsd = WordSenseDisambiguator(
            config=wsd_config,
            llm=llm,
            require_wordnet=require_wordnet,
            synset_learner=synset_learner,
        )
        self._min_nl_length = min_nl_length

        # Metrics
        self.metrics = {
            "total_processed": 0,
            "prose_processed": 0,
            "code_nl_extracted": 0,
            "data_skipped": 0,
            "config_skipped": 0,
            "mixed_processed": 0,
            "wsd_performed": 0,
            "wsd_skipped_no_nl": 0,
        }

    async def process(
        self,
        content: str,
        extract_all_words: bool = False,
    ) -> ContentAwareWSDResult:
        """Process content with content-aware WSD.

        Args:
            content: The content to process (can be any type)
            extract_all_words: If True, disambiguate all content words.
                              If False, only disambiguate ambiguous words.

        Returns:
            ContentAwareWSDResult with analysis and disambiguation results
        """
        self.metrics["total_processed"] += 1

        # Step 1: Analyze content type
        analysis = await self._content_analyzer.analyze(content)

        # Step 2: Determine processing strategy
        result = ContentAwareWSDResult(content_analysis=analysis)

        if analysis.content_type == ContentType.DATA:
            self.metrics["data_skipped"] += 1
            result.skipped_processing = True
            result.skip_reason = "Data content - WSD not applicable"
            result.structural_knowledge = self._extract_data_schema(analysis)
            return result

        if analysis.content_type == ContentType.CONFIG:
            self.metrics["config_skipped"] += 1
            result.skipped_processing = True
            result.skip_reason = "Configuration content - WSD not applicable"
            result.structural_knowledge = self._extract_config_patterns(analysis)
            return result

        # Step 3: Extract natural language for processing
        if analysis.content_type == ContentType.PROSE:
            self.metrics["prose_processed"] += 1
            nl_text = analysis.get_natural_language_text() or content
        elif analysis.content_type in (ContentType.CODE, ContentType.MIXED):
            self.metrics["code_nl_extracted"] += 1
            nl_text = analysis.get_natural_language_text()
            # Also capture structural knowledge from code
            if analysis.structural_knowledge:
                result.structural_knowledge = [
                    {
                        "type": sk.relationship_type,
                        "subject": sk.subject,
                        "object": sk.object,
                    }
                    for sk in analysis.structural_knowledge
                ]
        else:
            # UNKNOWN or other types - try as prose
            nl_text = content

        result.processed_text = nl_text

        # Step 4: Check if there's enough NL to process
        if not nl_text or len(nl_text.strip()) < self._min_nl_length:
            self.metrics["wsd_skipped_no_nl"] += 1
            result.skipped_processing = True
            result.skip_reason = (
                f"Insufficient natural language text ({len(nl_text.strip())} chars)"
            )
            return result

        # Step 5: Perform WSD on natural language
        self.metrics["wsd_performed"] += 1

        if extract_all_words:
            # Disambiguate all content words
            result.disambiguation_results = await self._wsd.disambiguate_all(nl_text)
        else:
            # Disambiguate only ambiguous words (words with multiple synsets)
            result.disambiguation_results = await self._disambiguate_ambiguous_only(
                nl_text
            )

        return result

    async def disambiguate_word(
        self,
        word: str,
        content: str,
    ) -> DisambiguationResult | None:
        """Disambiguate a specific word in content.

        Content-aware version that extracts appropriate context.

        Args:
            word: The word to disambiguate
            content: The full content (any type)

        Returns:
            DisambiguationResult or None if cannot disambiguate
        """
        # Analyze content to get NL portions
        analysis = await self._content_analyzer.analyze(content)

        # Get the appropriate text for context
        if analysis.content_type == ContentType.PROSE:
            context_text = content
        elif analysis.has_natural_language():
            context_text = analysis.get_natural_language_text()
        else:
            # No NL context available, use original but may fail
            context_text = content

        # Check if word is even in the NL text
        if word.lower() not in context_text.lower():
            # Word might be in code syntax, not NL - skip WSD
            logger.debug(
                f"Word '{word}' not found in NL portion, skipping WSD"
            )
            return None

        return await self._wsd.disambiguate(word, context_text)

    async def _disambiguate_ambiguous_only(
        self,
        text: str,
    ) -> dict[str, DisambiguationResult]:
        """Disambiguate only words that have multiple synsets.

        More efficient than disambiguating all words since unambiguous
        words don't need the full pipeline.

        Args:
            text: Natural language text

        Returns:
            Dict of word -> DisambiguationResult for ambiguous words
        """
        results = {}

        # Extract content words
        words = self._wsd._extract_content_words(text)

        for word in words:
            # Check if word has multiple synsets
            synsets = self._wsd.get_synsets(word)
            if len(synsets) > 1:
                # Ambiguous - needs disambiguation
                result = await self._wsd.disambiguate(word, text)
                if result:
                    results[word] = result

        return results

    def _extract_data_schema(
        self,
        analysis: ContentAnalysis,
    ) -> list[dict[str, Any]]:
        """Extract schema information from data content.

        For CSV/JSON/etc., we extract column names, types, relationships
        rather than doing WSD.
        """
        schema = []

        # If we have structural knowledge from LLM analysis, use it
        if analysis.structural_knowledge:
            for sk in analysis.structural_knowledge:
                schema.append({
                    "type": sk.relationship_type,
                    "subject": sk.subject,
                    "object": sk.object,
                    "confidence": sk.confidence,
                })

        # Add format info
        if analysis.detected_format:
            schema.append({
                "type": "format",
                "subject": "content",
                "object": analysis.detected_format,
            })

        return schema

    def _extract_config_patterns(
        self,
        analysis: ContentAnalysis,
    ) -> list[dict[str, Any]]:
        """Extract pattern information from config content.

        For YAML/TOML/etc., we extract key hierarchies and common patterns
        rather than doing WSD.
        """
        patterns = []

        if analysis.structural_knowledge:
            for sk in analysis.structural_knowledge:
                patterns.append({
                    "type": sk.relationship_type,
                    "subject": sk.subject,
                    "object": sk.object,
                })

        if analysis.detected_format:
            patterns.append({
                "type": "format",
                "subject": "config",
                "object": analysis.detected_format,
            })

        return patterns

    def get_metrics(self) -> dict[str, int]:
        """Get processing metrics."""
        return dict(self.metrics)

    def get_combined_metrics(self) -> dict[str, Any]:
        """Get metrics from both content analyzer and WSD."""
        return {
            "content_aware": dict(self.metrics),
            "content_analyzer": self._content_analyzer.get_metrics(),
            "wsd": self._wsd.get_metrics(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def process_content_for_wsd(
    content: str,
    llm: LLMProvider | None = None,
) -> ContentAwareWSDResult:
    """Process content with content-aware WSD.

    Convenience function that handles any content type appropriately.

    Args:
        content: Any content (prose, code, data, etc.)
        llm: Optional LLM provider

    Returns:
        ContentAwareWSDResult with analysis and disambiguation
    """
    processor = ContentAwareWSD(llm=llm, require_wordnet=False)
    return await processor.process(content)


async def disambiguate_in_context(
    word: str,
    content: str,
    llm: LLMProvider | None = None,
) -> DisambiguationResult | None:
    """Disambiguate a word with content-aware context extraction.

    Args:
        word: Word to disambiguate
        content: Content containing the word (any type)
        llm: Optional LLM provider

    Returns:
        DisambiguationResult or None
    """
    processor = ContentAwareWSD(llm=llm, require_wordnet=False)
    return await processor.disambiguate_word(word, content)

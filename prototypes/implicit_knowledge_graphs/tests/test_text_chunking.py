"""Tests for text chunking and sentence segmentation.

Tests the text_chunking module's ability to:
- Segment sentences correctly (handling abbreviations, edge cases)
- Extract context around target words
- Chunk large texts for processing
- Handle extremely long documents efficiently
"""

import pytest

import sys
from pathlib import Path

# Add prototype src to path
prototype_root = Path(__file__).parent.parent
src_path = prototype_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from text_chunking import (
    TextChunker,
    ChunkingConfig,
    TextChunk,
    segment_sentences,
    get_context_for_wsd,
    chunk_large_text,
)


# =============================================================================
# Test: Sentence Segmentation
# =============================================================================


class TestSentenceSegmentation:
    """Test sentence boundary detection."""

    def test_simple_sentences(self):
        """Basic sentence splitting."""
        text = "Hello world. This is a test. It works."
        sentences = segment_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "This is a test."
        assert sentences[2] == "It works."

    def test_question_and_exclamation(self):
        """Handles ? and ! correctly."""
        text = "What is this? It's amazing! Yes it is."
        sentences = segment_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "What is this?"
        assert sentences[1] == "It's amazing!"
        assert sentences[2] == "Yes it is."

    def test_abbreviations_not_split(self):
        """Abbreviations don't cause false splits."""
        text = "Dr. Smith went to the store. He bought milk."
        sentences = segment_sentences(text)

        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]

    def test_multiple_abbreviations(self):
        """Multiple abbreviations in one sentence."""
        text = "Mr. and Mrs. Smith visited Dr. Jones. They stayed for tea."
        sentences = segment_sentences(text)

        assert len(sentences) == 2
        assert "Mr. and Mrs. Smith" in sentences[0]

    def test_decimal_numbers(self):
        """Decimal numbers don't cause splits."""
        text = "The price is $3.50 per unit. That's reasonable."
        sentences = segment_sentences(text)

        assert len(sentences) == 2
        assert "3.50" in sentences[0]

    def test_ellipsis(self):
        """Ellipsis handling."""
        text = "He said... well... I don't know. Then he left."
        sentences = segment_sentences(text)

        # Ellipsis should be kept together, not split
        assert len(sentences) >= 1
        assert "left" in sentences[-1]

    def test_empty_input(self):
        """Empty string returns empty list."""
        assert segment_sentences("") == []
        assert segment_sentences("   ") == []

    def test_no_punctuation(self):
        """Text without ending punctuation."""
        text = "This sentence has no ending"
        sentences = segment_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "This sentence has no ending"

    def test_initials(self):
        """Initials like J. K. Rowling."""
        text = "J. K. Rowling wrote Harry Potter. It was popular."
        sentences = segment_sentences(text)

        # This is tricky - may split or not depending on implementation
        # The key is not crashing
        assert len(sentences) >= 1

    def test_quoted_speech(self):
        """Quoted speech handling."""
        text = '"Hello," she said. "How are you?"'
        sentences = segment_sentences(text)

        assert len(sentences) >= 1


# =============================================================================
# Test: Context Extraction
# =============================================================================


class TestContextExtraction:
    """Test extracting context around target words."""

    def test_basic_context(self):
        """Extract context around a word."""
        text = "The bank is near the river. People fish there."
        chunker = TextChunker()

        context = chunker.extract_context_around_word(text, "bank")

        assert "bank" in context.lower()
        assert "river" in context.lower()

    def test_context_window(self):
        """Context respects word count limits."""
        # Create a very long text (>1000 chars to trigger chunking)
        words = ["word" + str(i) for i in range(300)]
        words[150] = "TARGET"
        text = " ".join(words)

        # Ensure it's over the threshold
        assert len(text) > 1000

        config = ChunkingConfig(context_window_words=10, max_chunk_tokens=100)
        chunker = TextChunker(config)

        context = chunker.extract_context_around_word(
            text, "TARGET", context_words=10, include_full_sentences=False
        )

        # Should have target and some surrounding words, but not all 300
        assert "TARGET" in context
        assert len(context.split()) < 100

    def test_word_not_found(self):
        """Word not in text returns truncated text."""
        text = "This is a short sentence."
        chunker = TextChunker()

        context = chunker.extract_context_around_word(text, "nonexistent")

        # Should return the original text since it's short
        assert context == text or "short sentence" in context

    def test_include_full_sentences(self):
        """Expand to sentence boundaries when requested."""
        text = "First sentence here. The bank processes deposits daily. Last sentence."
        chunker = TextChunker()

        context = chunker.extract_context_around_word(
            text, "bank", include_full_sentences=True
        )

        # Should include the full sentence with "bank"
        assert "processes deposits" in context

    def test_case_insensitive_match(self):
        """Word matching is case-insensitive."""
        text = "The BANK is open. Visit it today."
        chunker = TextChunker()

        context = chunker.extract_context_around_word(text, "bank")
        assert "BANK" in context

    def test_very_long_text(self):
        """Large text is handled efficiently."""
        # Create a 10000 word document
        filler = "This is filler text to pad the document. " * 500
        text = filler + "The bank handles financial transactions." + filler

        chunker = TextChunker()
        context = chunker.extract_context_around_word(text, "bank")

        # Context should be extracted, not the full 10000 words
        assert "bank" in context.lower()
        assert len(context) < len(text)


# =============================================================================
# Test: Text Chunking
# =============================================================================


class TestTextChunking:
    """Test chunking text for batch processing."""

    def test_short_text_single_chunk(self):
        """Short text returns single chunk."""
        text = "This is a short text."
        chunker = TextChunker()

        chunks = chunker.chunk_for_processing(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self):
        """Long text is split into multiple chunks."""
        text = "This is sentence number one. " * 100
        config = ChunkingConfig(max_chunk_tokens=50)  # ~200 chars
        chunker = TextChunker(config)

        chunks = chunker.chunk_for_processing(text)

        assert len(chunks) > 1
        # All chunks should be within size limit
        for chunk in chunks:
            assert len(chunk.text) <= config.max_chunk_chars + 100  # Some tolerance

    def test_chunk_metadata(self):
        """Chunks include position metadata."""
        text = "First sentence. Second sentence. Third sentence."
        config = ChunkingConfig(max_chunk_tokens=20)
        chunker = TextChunker(config)

        chunks = chunker.chunk_for_processing(text)

        # First chunk starts at 0
        assert chunks[0].start_char == 0

    def test_sentence_preservation(self):
        """Chunks don't break mid-sentence when possible."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        config = ChunkingConfig(max_chunk_tokens=30)
        chunker = TextChunker(config)

        chunks = chunker.chunk_for_processing(text, preserve_sentences=True)

        # Each chunk should end with a sentence-ending punctuation
        for chunk in chunks:
            text_stripped = chunk.text.strip()
            if text_stripped:
                # Should end with period (unless it's the last incomplete chunk)
                assert text_stripped.endswith('.') or chunk == chunks[-1]

    def test_overlap_between_chunks(self):
        """Chunks have overlap for context continuity."""
        text = "One two three four five. " * 50
        config = ChunkingConfig(max_chunk_tokens=50, overlap_tokens=10)
        chunker = TextChunker(config)

        chunks = chunker.chunk_for_processing(text)

        if len(chunks) > 1:
            # Check that there's some overlap (words appear in both chunks)
            chunk1_words = set(chunks[0].text.lower().split())
            chunk2_words = set(chunks[1].text.lower().split())
            overlap = chunk1_words & chunk2_words
            # Some common words should appear in both
            assert len(overlap) > 0

    def test_contains_word_method(self):
        """TextChunk.contains_word works correctly."""
        chunk = TextChunk(
            text="The bank handles deposits.",
            start_char=0,
            end_char=26,
        )

        assert chunk.contains_word("bank") is True
        assert chunk.contains_word("Bank") is True  # Case insensitive
        assert chunk.contains_word("river") is False
        assert chunk.contains_word("ban") is False  # Partial match shouldn't work


# =============================================================================
# Test: Definition Extraction Chunking
# =============================================================================


class TestDefinitionExtractionChunking:
    """Test specialized chunking for definition extraction."""

    def test_prioritize_chunks_with_terms(self):
        """Chunks containing target terms are prioritized."""
        # Create text where the term is in the middle
        text = "Filler text here. " * 20
        text += "Kubernetes is a container orchestration platform. "
        text += "More filler text. " * 20

        chunker = TextChunker(ChunkingConfig(max_chunk_tokens=100))
        chunks = chunker.chunk_for_definition_extraction(
            text,
            target_terms=["Kubernetes"],
            max_chars=500,
        )

        # First chunk should contain Kubernetes
        assert len(chunks) > 0
        assert chunks[0].contains_word("Kubernetes")

    def test_multiple_terms(self):
        """Works with multiple target terms."""
        # Create a longer text to ensure multiple chunks
        text = (
            "Part 1: Docker is a containerization tool for packaging applications. " * 10 +
            "Part 2: Kubernetes orchestrates containers across clusters. " * 10 +
            "Part 3: Unrelated content here about other things. " * 20
        )

        chunker = TextChunker(ChunkingConfig(max_chunk_tokens=50))
        chunks = chunker.chunk_for_definition_extraction(
            text,
            target_terms=["Docker", "Kubernetes"],
            max_chars=2000,
        )

        # Chunks containing terms should be prioritized
        # Check that at least one of the first chunks has a target term
        has_docker = any(c.contains_word("Docker") for c in chunks[:4])
        has_kubernetes = any(c.contains_word("Kubernetes") for c in chunks[:4])
        assert has_docker or has_kubernetes


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_context_for_wsd(self):
        """get_context_for_wsd function works."""
        text = "The bass played a deep note. The fish swam away."
        context = get_context_for_wsd(text, "bass")

        assert "bass" in context.lower()

    def test_chunk_large_text(self):
        """chunk_large_text function works."""
        text = "This is a test sentence. " * 100
        chunks = chunk_large_text(text, max_tokens=50)

        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)


# =============================================================================
# Test: Token Estimation
# =============================================================================


class TestTokenEstimation:
    """Test token count estimation."""

    def test_estimate_tokens(self):
        """Token estimation is roughly correct."""
        text = "This is a test sentence with about ten words here."
        chunker = TextChunker()

        estimate = chunker.estimate_tokens(text)

        # Should be roughly chars/4
        assert 10 <= estimate <= 20


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self):
        """Empty text handling."""
        chunker = TextChunker()

        assert chunker.extract_context_around_word("", "word") == ""
        assert chunker.chunk_for_processing("") == []

    def test_single_word(self):
        """Single word text."""
        chunker = TextChunker()

        context = chunker.extract_context_around_word("bank", "bank")
        assert context == "bank"

    def test_unicode_text(self):
        """Unicode characters handled correctly."""
        text = "Le café est délicieux. Je l'adore."
        sentences = segment_sentences(text)

        assert len(sentences) == 2
        assert "café" in sentences[0]

    def test_very_long_sentence(self):
        """Single very long sentence is handled."""
        # 1000 word sentence with no periods
        text = " ".join(["word"] * 1000)
        chunker = TextChunker(ChunkingConfig(max_chunk_tokens=100))

        chunks = chunker.chunk_for_processing(text)

        # Should still produce chunks even without sentence boundaries
        assert len(chunks) >= 1

    def test_all_punctuation(self):
        """Text that's mostly punctuation."""
        text = "... !!! ??? ... Hello. ... !!!"
        sentences = segment_sentences(text)

        # Should handle gracefully
        assert isinstance(sentences, list)


# =============================================================================
# Test: Integration with WSD
# =============================================================================


class TestWSDIntegration:
    """Test that chunking integrates properly with WSD use case."""

    def test_bank_disambiguation_context(self):
        """Context extraction preserves disambiguating words."""
        text = (
            "The financial district has many buildings. "
            "I went to the bank to deposit my paycheck. "
            "The teller was very helpful with the transaction. "
            "Then I walked to the river bank to relax."
        )

        # For the first "bank" (financial)
        config = ChunkingConfig(context_window_words=15)
        chunker = TextChunker(config)

        # Find first occurrence context
        context = chunker.extract_context_around_word(
            text, "bank", include_full_sentences=True
        )

        # Should include disambiguating words like "deposit" or "paycheck"
        assert "bank" in context.lower()
        # Context should help disambiguate

    def test_large_document_wsd(self):
        """WSD on a large document extracts relevant context."""
        # Simulate a large document with the target word buried in the middle
        preamble = "This document discusses various topics. " * 100
        key_sentence = "The bass guitar produces low frequency sounds. "
        epilogue = "Thank you for reading this document. " * 100

        document = preamble + key_sentence + epilogue

        chunker = TextChunker()
        context = chunker.extract_context_around_word(document, "bass")

        # Context should include the key disambiguating words
        assert "bass" in context.lower()
        # Should be much shorter than the full document
        assert len(context) < len(document) / 2

"""Text Chunking and Sentence Segmentation.

Provides utilities for handling extremely large sentences and documents
in the WSD and synset learning pipelines.

Key features:
- Sentence boundary detection (handles abbreviations, edge cases)
- Smart chunking that preserves context around target words
- Token budget management for LLM calls
- Sliding window context extraction

Example:
    >>> from text_chunking import TextChunker, ChunkingConfig
    >>>
    >>> chunker = TextChunker(ChunkingConfig(max_chunk_tokens=500))
    >>>
    >>> # Get context around a target word
    >>> context = chunker.extract_context_around_word(
    ...     "The bank handles deposits...",
    ...     target_word="bank",
    ...     context_words=20,
    ... )
    >>>
    >>> # Chunk a large document for processing
    >>> chunks = chunker.chunk_for_processing(large_document, max_tokens=2000)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ChunkingConfig:
    """Configuration for text chunking.

    Attributes:
        max_chunk_tokens: Maximum tokens per chunk (approximate)
        overlap_tokens: Overlap between chunks to preserve context
        context_window_words: Words to include around target for WSD
        min_chunk_length: Minimum chunk length (chars) to avoid tiny chunks
        sentence_boundary_buffer: Extra chars to check for sentence boundaries
        chars_per_token: Approximate chars per token for estimation
    """
    max_chunk_tokens: int = 500
    overlap_tokens: int = 50
    context_window_words: int = 25
    min_chunk_length: int = 100
    sentence_boundary_buffer: int = 50
    chars_per_token: float = 4.0  # Approximate for English

    @property
    def max_chunk_chars(self) -> int:
        """Max chunk size in characters."""
        return int(self.max_chunk_tokens * self.chars_per_token)

    @property
    def overlap_chars(self) -> int:
        """Overlap size in characters."""
        return int(self.overlap_tokens * self.chars_per_token)


# =============================================================================
# Sentence Segmentation
# =============================================================================


# Abbreviations that shouldn't end sentences
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd",
    "corp", "co", "eg", "ie", "al", "fig", "vol", "no", "pp", "ed", "est",
    "approx", "dept", "div", "govt", "max", "min", "avg", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "st",
    "ave", "blvd", "rd", "ft", "lb", "oz", "pt", "qt", "gal", "ml", "kg",
    "km", "cm", "mm", "sec", "min", "hr", "wk", "mo", "yr", "gen", "col",
    "lt", "sgt", "capt", "maj", "adm", "gov", "sen", "rep", "hon", "rev",
}


def segment_sentences(text: str) -> list[str]:
    """Segment text into sentences.

    Handles:
    - Standard sentence endings (. ! ?)
    - Abbreviations (Dr., Mr., etc.)
    - Decimal numbers (3.14)
    - URLs and emails
    - Quoted speech
    - Ellipsis (...)

    Args:
        text: The text to segment

    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    sentences = []
    current = []
    i = 0

    while i < len(text):
        char = text[i]
        current.append(char)

        # Check for sentence-ending punctuation
        if char in '.!?':
            # Look ahead to determine if this is actually a sentence end
            is_sentence_end = _is_sentence_boundary(text, i)

            if is_sentence_end:
                # Include any closing quotes or parentheses
                while i + 1 < len(text) and text[i + 1] in '"\')':
                    i += 1
                    current.append(text[i])

                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        i += 1

    # Don't forget the last sentence if it doesn't end with punctuation
    if current:
        sentence = ''.join(current).strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def _is_sentence_boundary(text: str, pos: int) -> bool:
    """Determine if position is a true sentence boundary.

    Args:
        text: The full text
        pos: Position of the punctuation mark

    Returns:
        True if this is a sentence boundary
    """
    if pos >= len(text) - 1:
        return True  # End of text

    char = text[pos]
    next_char = text[pos + 1] if pos + 1 < len(text) else ''

    # ! and ? are almost always sentence endings
    if char in '!?':
        # Unless followed by lowercase (e.g., "really?!" or quoted speech)
        if next_char and next_char.isalpha() and next_char.islower():
            return False
        return True

    # For periods, we need more checks
    if char == '.':
        # Ellipsis - not a sentence end unless followed by capital
        if pos + 2 < len(text) and text[pos + 1:pos + 3] == '..':
            # Skip to end of ellipsis
            return False

        # Check for abbreviation
        word_before = _get_word_before(text, pos)
        if word_before.lower().rstrip('.') in ABBREVIATIONS:
            return False

        # Check for decimal number (e.g., 3.14)
        if word_before and word_before[-1].isdigit() and next_char.isdigit():
            return False

        # Check for initials (e.g., "J. K. Rowling")
        if len(word_before) == 1 and word_before.isupper():
            if next_char == ' ' and pos + 2 < len(text):
                after_space = text[pos + 2]
                if after_space.isupper() and (pos + 3 >= len(text) or text[pos + 3] in '. '):
                    return False

        # Check if next non-space character is uppercase (new sentence)
        j = pos + 1
        while j < len(text) and text[j] in ' \t\n"\')':
            j += 1

        if j < len(text):
            next_meaningful = text[j]
            # New sentence typically starts with uppercase
            if next_meaningful.isupper():
                return True
            # If lowercase, probably not a sentence boundary
            if next_meaningful.islower():
                return False

        return True

    return False


def _get_word_before(text: str, pos: int) -> str:
    """Get the word immediately before a position."""
    if pos == 0:
        return ''

    end = pos
    start = pos - 1

    # Skip back to start of word
    while start > 0 and text[start - 1] not in ' \t\n':
        start -= 1

    return text[start:end]


# =============================================================================
# Text Chunker
# =============================================================================


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    text: str
    start_char: int
    end_char: int
    sentence_indices: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.text)

    def contains_word(self, word: str) -> bool:
        """Check if chunk contains a word (case-insensitive)."""
        return bool(re.search(rf'\b{re.escape(word)}\b', self.text, re.IGNORECASE))


class TextChunker:
    """Chunks text for efficient processing.

    Provides methods for:
    - Extracting context around target words (for WSD)
    - Chunking documents for LLM processing
    - Managing token budgets
    """

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize the chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()

    def extract_context_around_word(
        self,
        text: str,
        target_word: str,
        context_words: int | None = None,
        include_full_sentences: bool = True,
    ) -> str:
        """Extract context around a target word.

        This is the primary method for WSD - it extracts just enough
        context to disambiguate the word without processing the entire text.

        Args:
            text: The full text
            target_word: The word to find context for
            context_words: Number of words on each side (default from config)
            include_full_sentences: If True, expand to full sentence boundaries

        Returns:
            Context string centered on the target word
        """
        if not text or not target_word:
            return ""

        context_words = context_words or self.config.context_window_words

        # Find target word position(s)
        pattern = rf'\b{re.escape(target_word)}\b'
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if not matches:
            # Word not found, return truncated text
            return text[:self.config.max_chunk_chars]

        # Use first match
        match = matches[0]
        word_start = match.start()
        word_end = match.end()

        if include_full_sentences:
            # Find sentence boundaries
            return self._extract_sentence_context(text, word_start, word_end, context_words)
        else:
            # Just extract word window
            return self._extract_word_window(text, word_start, word_end, context_words)

    def _extract_word_window(
        self,
        text: str,
        word_start: int,
        word_end: int,
        context_words: int,
    ) -> str:
        """Extract a window of words around the target."""
        # Tokenize into words with positions
        words = list(re.finditer(r'\b\w+\b', text))

        # Find which word token contains our target
        target_idx = None
        for i, m in enumerate(words):
            if m.start() <= word_start < m.end():
                target_idx = i
                break

        if target_idx is None:
            # Fallback: character-based window
            start = max(0, word_start - context_words * 6)
            end = min(len(text), word_end + context_words * 6)
            return text[start:end].strip()

        # Calculate window
        start_idx = max(0, target_idx - context_words)
        end_idx = min(len(words), target_idx + context_words + 1)

        # Get character positions
        start_char = words[start_idx].start()
        end_char = words[end_idx - 1].end()

        return text[start_char:end_char].strip()

    def _extract_sentence_context(
        self,
        text: str,
        word_start: int,
        word_end: int,
        context_words: int,
    ) -> str:
        """Extract context expanded to sentence boundaries."""
        # First get word window
        window = self._extract_word_window(text, word_start, word_end, context_words)

        # Find where this window is in the original text
        window_start = text.find(window)
        if window_start == -1:
            return window

        window_end = window_start + len(window)

        # Expand to sentence boundaries
        # Look backwards for sentence start
        sent_start = window_start
        while sent_start > 0:
            if text[sent_start - 1] in '.!?' and sent_start > 1:
                # Check if this is actually a sentence boundary
                if _is_sentence_boundary(text, sent_start - 1):
                    break
            sent_start -= 1

        # Look forwards for sentence end
        sent_end = window_end
        while sent_end < len(text):
            if text[sent_end] in '.!?':
                if _is_sentence_boundary(text, sent_end):
                    sent_end += 1  # Include the punctuation
                    break
            sent_end += 1

        result = text[sent_start:sent_end].strip()

        # If result is too long, fall back to word window
        if len(result) > self.config.max_chunk_chars:
            return window

        return result

    def chunk_for_processing(
        self,
        text: str,
        max_chars: int | None = None,
        preserve_sentences: bool = True,
    ) -> list[TextChunk]:
        """Chunk text for batch processing.

        Splits text into chunks that respect:
        - Maximum size limits
        - Sentence boundaries (if preserve_sentences=True)
        - Overlap for context continuity

        Args:
            text: The text to chunk
            max_chars: Maximum characters per chunk
            preserve_sentences: Try to break at sentence boundaries

        Returns:
            List of TextChunk objects
        """
        if not text:
            return []

        max_chars = max_chars or self.config.max_chunk_chars

        if len(text) <= max_chars:
            return [TextChunk(text=text, start_char=0, end_char=len(text))]

        if preserve_sentences:
            return self._chunk_by_sentences(text, max_chars)
        else:
            return self._chunk_by_chars(text, max_chars)

    def _chunk_by_sentences(self, text: str, max_chars: int) -> list[TextChunk]:
        """Chunk text respecting sentence boundaries."""
        sentences = segment_sentences(text)

        if not sentences:
            return self._chunk_by_chars(text, max_chars)

        chunks = []
        current_sentences = []
        current_length = 0
        current_start = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence) + 1  # +1 for space

            if current_length + sentence_len > max_chars and current_sentences:
                # Save current chunk
                chunk_text = ' '.join(current_sentences)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    sentence_indices=list(range(i - len(current_sentences), i)),
                ))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences,
                    self.config.overlap_chars,
                )
                current_sentences = overlap_sentences
                current_length = sum(len(s) + 1 for s in current_sentences)
                current_start = current_start + len(chunk_text) - current_length

            current_sentences.append(sentence)
            current_length += sentence_len

        # Don't forget last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
            ))

        return chunks

    def _get_overlap_sentences(
        self,
        sentences: list[str],
        target_chars: int,
    ) -> list[str]:
        """Get sentences from end that fit within target characters."""
        overlap = []
        total = 0

        for sentence in reversed(sentences):
            if total + len(sentence) + 1 > target_chars:
                break
            overlap.insert(0, sentence)
            total += len(sentence) + 1

        return overlap

    def _chunk_by_chars(self, text: str, max_chars: int) -> list[TextChunk]:
        """Simple character-based chunking with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look back for space
                search_start = max(start, end - 50)
                last_space = text.rfind(' ', search_start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                ))

            # Move start with overlap
            start = end - self.config.overlap_chars
            if start >= end:
                start = end

        return chunks

    def chunk_for_definition_extraction(
        self,
        text: str,
        target_terms: list[str],
        max_chars: int = 4000,
    ) -> list[TextChunk]:
        """Chunk text for definition extraction, prioritizing chunks with target terms.

        This is specialized for the synset learning use case - we want chunks
        that are likely to contain definitions of the target terms.

        Args:
            text: The text to process
            target_terms: Terms we're looking for definitions of
            max_chars: Maximum chars per chunk

        Returns:
            List of chunks, prioritized by term presence
        """
        chunks = self.chunk_for_processing(text, max_chars)

        # Score and sort by term presence
        scored_chunks = []
        for chunk in chunks:
            score = sum(1 for term in target_terms if chunk.contains_word(term))
            scored_chunks.append((score, chunk))

        # Sort by score descending, but keep original order for ties
        scored_chunks.sort(key=lambda x: -x[0])

        return [chunk for score, chunk in scored_chunks]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is a rough estimate - actual tokenization varies by model.

        Args:
            text: The text to estimate

        Returns:
            Estimated token count
        """
        return int(len(text) / self.config.chars_per_token)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_context_for_wsd(
    text: str,
    word: str,
    context_words: int = 25,
) -> str:
    """Get context around a word for WSD.

    Convenience function that creates a chunker and extracts context.

    Args:
        text: Full text
        word: Target word
        context_words: Words on each side

    Returns:
        Context string
    """
    chunker = TextChunker()
    return chunker.extract_context_around_word(text, word, context_words)


def chunk_large_text(
    text: str,
    max_tokens: int = 500,
) -> list[str]:
    """Chunk large text into processable pieces.

    Convenience function for simple chunking.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    config = ChunkingConfig(max_chunk_tokens=max_tokens)
    chunker = TextChunker(config)
    chunks = chunker.chunk_for_processing(text)
    return [c.text for c in chunks]

"""Negation detection with scope analysis.

This module detects negation in text and determines its scope - which parts
of the sentence are affected by the negation. This is crucial for knowledge
extraction because:

- "Doug forgot the meeting" → Doug didn't attend
- "Doug didn't forget the meeting" → Doug attended (negation flips meaning)

We handle several types of negation:
1. Explicit negation: "not", "n't", "never", "no", etc.
2. Morphological negation: "unhappy", "impossible", "disagree"
3. Implicit negation: "fail", "refuse", "prevent", "lack"
4. Negative polarity items: "any", "ever" (in negative contexts)

Scope detection uses syntactic heuristics:
- Negation typically scopes over its VP (verb phrase)
- Raised negation can scope over subordinate clauses
- Focus particles can narrow scope

Example:
    >>> from .negation import NegationExtractor
    >>> extractor = NegationExtractor()
    >>> result = extractor.extract_sync("Doug didn't forget the meeting")
    >>> print(result.is_negated)  # True
    >>> print(result.negated_spans)  # [(5, 17)] - "forget the meeting"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from .models import NegationInfo, Polarity
from .config import NegationConfig


# =============================================================================
# Data Types
# =============================================================================


class NegationType(str, Enum):
    """Types of negation."""

    EXPLICIT = "explicit"  # not, n't, never
    MORPHOLOGICAL = "morphological"  # un-, in-, dis-
    IMPLICIT = "implicit"  # fail, refuse, prevent
    INHERENT = "inherent"  # absent, lack, miss
    DOUBLE = "double"  # negative concord


@dataclass
class NegationCue:
    """A detected negation marker."""

    text: str
    """The negation word/morpheme."""

    negation_type: NegationType
    """Type of negation."""

    span: tuple[int, int]
    """Character span in source text."""

    scope_start: int | None = None
    """Start of negation scope."""

    scope_end: int | None = None
    """End of negation scope."""

    confidence: float = 0.9
    """Confidence in this detection."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "negation_type": self.negation_type.value,
            "span": self.span,
            "scope_start": self.scope_start,
            "scope_end": self.scope_end,
            "confidence": self.confidence,
        }


@dataclass
class NegationAnalysis:
    """Complete negation analysis for a sentence."""

    cues: list[NegationCue] = field(default_factory=list)
    """All detected negation cues."""

    is_negated: bool = False
    """Whether the main proposition is negated."""

    polarity: Polarity = Polarity.POSITIVE
    """Overall polarity of the sentence."""

    scope_text: str | None = None
    """Text under negation scope."""

    has_double_negation: bool = False
    """Whether double negation was detected."""

    affirmative_meaning: str | None = None
    """What the sentence would mean without negation."""

    confidence: float = 0.9
    """Overall confidence in analysis."""

    def to_negation_info(self) -> NegationInfo:
        """Convert to NegationInfo model."""
        negated_spans = []
        for cue in self.cues:
            if cue.scope_start is not None and cue.scope_end is not None:
                negated_spans.append((cue.scope_start, cue.scope_end))

        return NegationInfo(
            is_negated=self.polarity != Polarity.POSITIVE,
            polarity=self.polarity,
            negation_cue=self.cues[0].text if self.cues else None,
            negation_scope=self.scope_text,
            negation_scope_span=negated_spans[0] if negated_spans else None,
        )


# =============================================================================
# Negation Detector
# =============================================================================


class NegationDetector:
    """Detects negation cues in text.

    Uses pattern matching to identify explicit, morphological, and implicit
    negation markers.
    """

    # Explicit negation patterns
    EXPLICIT_PATTERNS = {
        # Contracted forms
        r"\b\w+n't\b": ("n't", 0.95),
        # Full forms
        r"\bnot\b": ("not", 0.95),
        r"\bnever\b": ("never", 0.95),
        r"\bno\b(?!\s+(?:one|body|thing|where))": ("no", 0.90),
        r"\bnone\b": ("none", 0.95),
        r"\bnobody\b": ("nobody", 0.95),
        r"\bnothing\b": ("nothing", 0.95),
        r"\bnowhere\b": ("nowhere", 0.95),
        r"\bno\s+one\b": ("no one", 0.95),
        r"\bneither\b": ("neither", 0.90),
        r"\bnor\b": ("nor", 0.85),
        r"\bwithout\b": ("without", 0.85),
        r"\bhardly\b": ("hardly", 0.85),
        r"\bbarely\b": ("barely", 0.85),
        r"\bscarcely\b": ("scarcely", 0.85),
        r"\bseldom\b": ("seldom", 0.80),
        r"\brarely\b": ("rarely", 0.80),
    }

    # Morphological negation - prefixes
    NEGATIVE_PREFIXES = {
        "un": 0.85,
        "in": 0.80,
        "im": 0.85,
        "il": 0.85,
        "ir": 0.85,
        "non": 0.90,
        "dis": 0.85,
        "mis": 0.70,  # Lower - sometimes means "wrongly" not "not"
        "anti": 0.75,
        "a": 0.60,  # Only for Greek roots: amoral, asymmetric
    }

    # Words that look prefixed but aren't
    FALSE_PREFIXES = {
        "under", "understand", "universe", "union", "unique", "unit",
        "uncle", "until", "undo",  # undo IS negative, handled separately
        "in", "into", "inside", "ink", "inch", "install", "instead",
        "important", "impact", "improve", "include",
        "impress", "image", "imagine",
        "discuss", "discover", "display", "dispose", "district",
        "miss", "mist", "mistake",  # mistake is different
        "anti",  # as a word, not prefix
    }

    # True morphologically negated words (for validation)
    KNOWN_NEGATED_WORDS = {
        "unhappy", "unlikely", "unable", "uncertain", "unclear",
        "impossible", "improbable", "impatient", "immature",
        "illegal", "illiterate", "illogical",
        "irregular", "irrelevant", "irresponsible",
        "disagree", "disappear", "disconnect", "discourage",
        "nonfiction", "nonprofit", "nonsense", "nonexistent",
        "asymmetric", "atypical", "amoral",
    }

    # Implicit negation verbs
    IMPLICIT_NEGATORS = {
        # Failure verbs
        "fail": ("failure to do X", 0.90),
        "failed": ("failure to do X", 0.90),
        "fails": ("failure to do X", 0.90),
        "failing": ("failure to do X", 0.85),
        # Refusal verbs
        "refuse": ("not doing X", 0.90),
        "refused": ("not doing X", 0.90),
        "refuses": ("not doing X", 0.90),
        "refusing": ("not doing X", 0.85),
        "decline": ("not doing X", 0.85),
        "declined": ("not doing X", 0.85),
        "reject": ("not accepting X", 0.85),
        "rejected": ("not accepting X", 0.85),
        # Prevention verbs
        "prevent": ("X not happening", 0.90),
        "prevented": ("X not happening", 0.90),
        "prevents": ("X not happening", 0.90),
        "preventing": ("X not happening", 0.85),
        "stop": ("X not continuing", 0.80),
        "stopped": ("X not continuing", 0.80),
        "block": ("X not happening", 0.80),
        "blocked": ("X not happening", 0.80),
        "prohibit": ("X not allowed", 0.90),
        "prohibited": ("X not allowed", 0.90),
        "forbid": ("X not allowed", 0.90),
        "forbade": ("X not allowed", 0.90),
        "forbidden": ("X not allowed", 0.90),
        # Absence verbs
        "lack": ("absence of X", 0.85),
        "lacked": ("absence of X", 0.85),
        "lacks": ("absence of X", 0.85),
        "lacking": ("absence of X", 0.85),
        "miss": ("absence of X", 0.75),  # Ambiguous
        "missed": ("absence of X", 0.75),
        "absent": ("not present", 0.85),
        # Denial verbs
        "deny": ("X not true", 0.90),
        "denied": ("X not true", 0.90),
        "denies": ("X not true", 0.90),
        "doubt": ("X possibly not true", 0.70),
        "doubted": ("X possibly not true", 0.70),
    }

    def __init__(self, config: NegationConfig):
        """Initialize detector with configuration."""
        self.config = config

        # Compile patterns
        self._explicit_patterns = [
            (re.compile(pattern, re.IGNORECASE), label, conf)
            for pattern, (label, conf) in self.EXPLICIT_PATTERNS.items()
        ]

    def detect(self, text: str) -> list[NegationCue]:
        """Detect all negation cues in text.

        Args:
            text: Input text

        Returns:
            List of NegationCue objects
        """
        cues = []

        # Detect explicit negation
        cues.extend(self._detect_explicit(text))

        # Detect morphological negation
        if self.config.detect_prefixes:
            cues.extend(self._detect_morphological(text))

        # Detect implicit negation
        cues.extend(self._detect_implicit(text))

        # Sort by position
        cues.sort(key=lambda c: c.span[0])

        return cues

    def _detect_explicit(self, text: str) -> list[NegationCue]:
        """Detect explicit negation markers."""
        cues = []

        for pattern, label, confidence in self._explicit_patterns:
            for match in pattern.finditer(text):
                cues.append(NegationCue(
                    text=match.group(),
                    negation_type=NegationType.EXPLICIT,
                    span=(match.start(), match.end()),
                    confidence=confidence,
                ))

        return cues

    def _detect_morphological(self, text: str) -> list[NegationCue]:
        """Detect morphologically negated words."""
        cues = []
        words = re.finditer(r'\b\w+\b', text)

        for match in words:
            word = match.group().lower()

            # Skip if in false prefix list
            if word in self.FALSE_PREFIXES:
                continue

            # Check known negated words first
            if word in self.KNOWN_NEGATED_WORDS:
                cues.append(NegationCue(
                    text=match.group(),
                    negation_type=NegationType.MORPHOLOGICAL,
                    span=(match.start(), match.end()),
                    confidence=0.95,
                ))
                continue

            # Check for negative prefixes
            for prefix, conf in self.NEGATIVE_PREFIXES.items():
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    # Verify the stem is a real word (simple heuristic)
                    stem = word[len(prefix):]

                    # Some validation - stem should start with consonant cluster
                    # that makes sense after removing prefix
                    if self._is_likely_negated_word(word, prefix, stem):
                        cues.append(NegationCue(
                            text=match.group(),
                            negation_type=NegationType.MORPHOLOGICAL,
                            span=(match.start(), match.end()),
                            confidence=conf,
                        ))
                        break

        return cues

    def _is_likely_negated_word(self, word: str, prefix: str, stem: str) -> bool:
        """Check if a word is likely morphologically negated.

        This is a heuristic check. For production, would use a morphological
        analyzer or dictionary lookup.
        """
        # Very short stems unlikely
        if len(stem) < 3:
            return False

        # Double letter at boundary often indicates true prefix
        # e.g., "il-legal", "im-mature", "ir-regular"
        if prefix == "il" and stem.startswith("l"):
            return True
        if prefix == "im" and stem.startswith("m"):
            return True
        if prefix == "ir" and stem.startswith("r"):
            return True

        # "un-" before common endings
        if prefix == "un" and any(stem.endswith(s) for s in ["able", "ful", "ly", "ed", "ing"]):
            return True

        # "dis-" before common stems
        if prefix == "dis" and any(stem.startswith(s) for s in ["agree", "appear", "connect", "like", "honor"]):
            return True

        # "non-" is usually productive
        if prefix == "non":
            return True

        return False

    def _detect_implicit(self, text: str) -> list[NegationCue]:
        """Detect implicit negation from verbs."""
        cues = []
        text_lower = text.lower()

        for word, (meaning, confidence) in self.IMPLICIT_NEGATORS.items():
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                cues.append(NegationCue(
                    text=match.group(),
                    negation_type=NegationType.IMPLICIT,
                    span=(match.start(), match.end()),
                    confidence=confidence,
                ))

        return cues


# =============================================================================
# Scope Analyzer
# =============================================================================


class ScopeAnalyzer:
    """Analyzes the scope of negation.

    Determines which part of the sentence is under the scope of negation.
    Uses syntactic heuristics without requiring a full parser.
    """

    # Scope boundary markers - negation typically doesn't cross these
    SCOPE_BOUNDARIES = {
        r'\bbut\b',
        r'\balthough\b',
        r'\bthough\b',
        r'\beven though\b',
        r'\bhowever\b',
        r'\bnevertheless\b',
        r'\byet\b',
        r'[.!?;]',  # Sentence boundaries
    }

    # Focus particles that can restrict scope
    FOCUS_PARTICLES = {
        "only": "restricts to focused element",
        "just": "restricts to focused element",
        "even": "adds to alternatives",
        "also": "adds to alternatives",
    }

    def __init__(self, config: NegationConfig):
        """Initialize scope analyzer."""
        self.config = config
        self._boundary_pattern = re.compile(
            '|'.join(self.SCOPE_BOUNDARIES),
            re.IGNORECASE
        )

    def analyze_scope(
        self,
        text: str,
        cues: list[NegationCue],
    ) -> list[NegationCue]:
        """Analyze scope for each negation cue.

        Args:
            text: Source text
            cues: Detected negation cues

        Returns:
            Cues with scope_start and scope_end filled in
        """
        for cue in cues:
            if cue.negation_type == NegationType.EXPLICIT:
                self._analyze_explicit_scope(text, cue)
            elif cue.negation_type == NegationType.MORPHOLOGICAL:
                self._analyze_morphological_scope(text, cue)
            elif cue.negation_type == NegationType.IMPLICIT:
                self._analyze_implicit_scope(text, cue)

        return cues

    def _analyze_explicit_scope(self, text: str, cue: NegationCue) -> None:
        """Analyze scope for explicit negation.

        Heuristic: Scope extends from negation to end of clause/VP.
        """
        neg_end = cue.span[1]

        # Find next boundary
        boundary_match = self._boundary_pattern.search(text, neg_end)
        if boundary_match:
            scope_end = boundary_match.start()
        else:
            scope_end = len(text)

        # Scope starts right after negation
        cue.scope_start = neg_end
        cue.scope_end = scope_end

    def _analyze_morphological_scope(self, text: str, cue: NegationCue) -> None:
        """Analyze scope for morphological negation.

        Scope is typically just the negated word itself and its modifiees.
        """
        # For morphological negation, scope is narrow - just the word
        cue.scope_start = cue.span[0]
        cue.scope_end = cue.span[1]

    def _analyze_implicit_scope(self, text: str, cue: NegationCue) -> None:
        """Analyze scope for implicit negation.

        For verbs like "fail to X", scope is the complement X.
        """
        verb_end = cue.span[1]

        # Look for "to + VP" pattern
        to_match = re.search(r'\s+to\s+', text[verb_end:])
        if to_match:
            scope_start = verb_end + to_match.end()

            # Find next boundary
            boundary_match = self._boundary_pattern.search(text, scope_start)
            if boundary_match:
                scope_end = boundary_match.start()
            else:
                scope_end = len(text)

            cue.scope_start = scope_start
            cue.scope_end = scope_end
        else:
            # Fallback: scope to end of clause
            boundary_match = self._boundary_pattern.search(text, verb_end)
            if boundary_match:
                scope_end = boundary_match.start()
            else:
                scope_end = len(text)

            cue.scope_start = verb_end
            cue.scope_end = scope_end


# =============================================================================
# Negation Extractor
# =============================================================================


class NegationExtractor:
    """Main extractor for negation analysis.

    Combines cue detection and scope analysis to produce complete
    negation information.

    Example:
        >>> extractor = NegationExtractor()
        >>> result = extractor.extract_sync("Doug didn't forget")
        >>> print(result.is_negated)  # True
        >>> print(result.scope_text)  # "forget"
    """

    def __init__(
        self,
        config: NegationConfig | None = None,
        llm: Any | None = None,
    ):
        """Initialize the negation extractor.

        Args:
            config: Extraction configuration
            llm: Optional LLM provider for scope analysis
        """
        self.config = config or NegationConfig()
        self.llm = llm

        self.detector = NegationDetector(self.config)
        self.scope_analyzer = ScopeAnalyzer(self.config)

    def extract_sync(self, text: str) -> NegationInfo:
        """Synchronously extract negation information.

        Args:
            text: Input text

        Returns:
            NegationInfo with polarity and scope information
        """
        analysis = self._analyze(text)
        return analysis.to_negation_info()

    async def extract(self, text: str) -> NegationInfo:
        """Asynchronously extract negation information.

        If LLM is available and config allows, uses LLM for scope analysis.

        Args:
            text: Input text

        Returns:
            NegationInfo with polarity and scope information
        """
        if self.llm and self.config.scope_method == "llm":
            return await self._extract_with_llm(text)
        else:
            return self.extract_sync(text)

    def _analyze(self, text: str) -> NegationAnalysis:
        """Perform complete negation analysis.

        Args:
            text: Input text

        Returns:
            Complete NegationAnalysis
        """
        # Detect cues
        cues = self.detector.detect(text)

        if not cues:
            return NegationAnalysis(
                cues=[],
                is_negated=False,
                polarity=Polarity.POSITIVE,
                confidence=0.95,
            )

        # Analyze scope
        cues = self.scope_analyzer.analyze_scope(text, cues)

        # Determine overall polarity
        explicit_count = sum(
            1 for c in cues if c.negation_type == NegationType.EXPLICIT
        )
        has_double_negation = explicit_count >= 2

        # Double negation often = positive (but with emphasis)
        if has_double_negation:
            polarity = Polarity.DOUBLE_NEGATIVE
        else:
            polarity = Polarity.NEGATIVE

        # Get scope text for primary negation
        primary_cue = cues[0]
        scope_text = None
        if primary_cue.scope_start is not None and primary_cue.scope_end is not None:
            scope_text = text[primary_cue.scope_start:primary_cue.scope_end].strip()

        # Compute affirmative meaning
        affirmative = self._compute_affirmative(text, cues)

        # Confidence based on cue types
        avg_confidence = sum(c.confidence for c in cues) / len(cues)

        return NegationAnalysis(
            cues=cues,
            is_negated=True,
            polarity=polarity,
            scope_text=scope_text,
            has_double_negation=has_double_negation,
            affirmative_meaning=affirmative,
            confidence=avg_confidence,
        )

    def _compute_affirmative(
        self,
        text: str,
        cues: list[NegationCue],
    ) -> str | None:
        """Compute what the sentence would mean without negation.

        This is useful for knowledge extraction - we want to know both
        what IS and what ISN'T the case.
        """
        if not cues:
            return None

        result = text

        # Remove explicit negation markers
        for cue in sorted(cues, key=lambda c: c.span[0], reverse=True):
            if cue.negation_type == NegationType.EXPLICIT:
                start, end = cue.span

                # Handle contracted forms specially
                if cue.text.endswith("n't"):
                    # "didn't" -> "did", "couldn't" -> "could"
                    base = text[start:end - 3]
                    result = result[:start] + base + result[end:]
                else:
                    # Remove the negation word
                    result = result[:start] + result[end:]

        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result if result != text else None

    async def _extract_with_llm(self, text: str) -> NegationInfo:
        """Use LLM for more accurate scope analysis.

        Falls back to heuristic if LLM call fails.
        """
        prompt = f"""Analyze the negation in this sentence:

Sentence: "{text}"

Identify:
1. Is there negation? (yes/no)
2. What word/phrase indicates negation?
3. What is under the scope of negation (what is being negated)?
4. What would the affirmative version mean?

Respond in XML:
<negation>
  <is_negated>yes or no</is_negated>
  <negation_cue>the negating word</negation_cue>
  <scope>the negated content</scope>
  <affirmative>what the sentence would mean without negation</affirmative>
  <confidence>0.0-1.0</confidence>
</negation>"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )

            # Parse XML response
            return self._parse_llm_response(response, text)
        except Exception:
            # Fallback to heuristic
            return self.extract_sync(text)

    def _parse_llm_response(self, response: str, text: str) -> NegationInfo:
        """Parse LLM XML response into NegationInfo."""
        # Simple XML parsing
        is_negated_match = re.search(r'<is_negated>(.*?)</is_negated>', response, re.IGNORECASE)
        cue_match = re.search(r'<negation_cue>(.*?)</negation_cue>', response)
        scope_match = re.search(r'<scope>(.*?)</scope>', response)
        conf_match = re.search(r'<confidence>(.*?)</confidence>', response)

        is_negated = is_negated_match and "yes" in is_negated_match.group(1).lower()

        if not is_negated:
            return NegationInfo(polarity=Polarity.POSITIVE, confidence=0.9)

        negation_cue = cue_match.group(1).strip() if cue_match else None
        scope_text = scope_match.group(1).strip() if scope_match else None

        try:
            confidence = float(conf_match.group(1)) if conf_match else 0.8
        except ValueError:
            confidence = 0.8

        # Find span of scope in original text
        negated_span = None
        if scope_text and scope_text in text:
            start = text.find(scope_text)
            negated_span = (start, start + len(scope_text))

        return NegationInfo(
            polarity=Polarity.NEGATIVE,
            negation_cue=negation_cue,
            scope_text=scope_text,
            negated_span=negated_span,
            confidence=confidence,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def detect_negation(text: str) -> bool:
    """Quick check if text contains negation.

    Args:
        text: Input text

    Returns:
        True if negation detected
    """
    extractor = NegationExtractor()
    result = extractor.extract_sync(text)
    return result.polarity != Polarity.POSITIVE


def get_polarity(text: str) -> Polarity:
    """Get the polarity of a sentence.

    Args:
        text: Input text

    Returns:
        Polarity enum value
    """
    extractor = NegationExtractor()
    result = extractor.extract_sync(text)
    return result.polarity

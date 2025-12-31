"""Temporal and aspectual information extraction.

This module extracts temporal information from text:

1. **Tense**: Past, present, future, and their perfect/progressive forms
2. **Aspect** (Vendler categories):
   - State: "know", "love", "own" (no internal structure)
   - Activity: "run", "swim", "write" (unbounded, homogeneous)
   - Accomplishment: "build a house", "write a letter" (bounded, durative)
   - Achievement: "arrive", "find", "win" (bounded, instantaneous)
   - Semelfactive: "knock", "cough" (instantaneous, can repeat)

3. **Temporal expressions**: "yesterday", "last week", "in 2024"
4. **Temporal relations**: Before, after, during, overlaps

This is crucial for knowledge extraction because temporal information
affects when knowledge is valid:
- "Doug owned a cat" → past (may not be true now)
- "Doug owns a cat" → present (currently true)
- "Doug will own a cat" → future (not yet true)

Example:
    >>> from decomposition.temporal import TemporalExtractor
    >>> extractor = TemporalExtractor()
    >>> result = extractor.extract_sync("Doug forgot the meeting yesterday")
    >>> print(result.tense)  # Tense.PAST
    >>> print(result.temporal_expression)  # "yesterday"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from enum import Enum

from decomposition.models import TemporalInfo, Tense, Aspect
from decomposition.config import TemporalConfig


# =============================================================================
# Additional Types
# =============================================================================


class TemporalReference(str, Enum):
    """Types of temporal reference."""

    ABSOLUTE = "absolute"  # "January 1, 2024"
    RELATIVE = "relative"  # "yesterday", "next week"
    DURATIONAL = "durational"  # "for 3 hours"
    FREQUENTATIVE = "frequentative"  # "every day", "sometimes"
    GENERIC = "generic"  # "usually", "always" (timeless truths)


@dataclass
class TemporalExpression:
    """A detected temporal expression."""

    text: str
    """The temporal expression text."""

    reference_type: TemporalReference
    """Type of temporal reference."""

    span: tuple[int, int]
    """Character span in source text."""

    normalized: str | None = None
    """Normalized form (e.g., ISO date)."""

    confidence: float = 0.9
    """Detection confidence."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "reference_type": self.reference_type.value,
            "span": self.span,
            "normalized": self.normalized,
            "confidence": self.confidence,
        }


# =============================================================================
# Tense Detector
# =============================================================================


class TenseDetector:
    """Detects grammatical tense from verb forms.

    Uses morphological patterns and auxiliary verbs to determine tense.
    More sophisticated analysis would require POS tagging.
    """

    # Auxiliary patterns
    PAST_AUXILIARIES = ["was", "were", "had", "did", "could", "would", "might"]
    PRESENT_AUXILIARIES = ["is", "are", "am", "has", "have", "do", "does", "can", "may"]
    FUTURE_MARKERS = ["will", "shall", "going to", "'ll", "gonna"]

    # Progressive markers
    PROGRESSIVE_PATTERN = re.compile(
        r'\b(is|are|am|was|were|been)\s+\w+ing\b',
        re.IGNORECASE
    )

    # Perfect markers
    PERFECT_PATTERN = re.compile(
        r'\b(has|have|had)\s+(\w+ed|\w+en|\w+)\b',
        re.IGNORECASE
    )

    # Past tense regular verbs
    PAST_PATTERN = re.compile(r'\b\w+ed\b')

    # Common irregular past tense verbs
    IRREGULAR_PAST = {
        "was", "were", "had", "did", "went", "came", "saw", "took", "made",
        "got", "said", "knew", "thought", "found", "gave", "told", "felt",
        "became", "left", "put", "kept", "began", "brought", "wrote",
        "stood", "heard", "let", "meant", "set", "met", "ran", "paid",
        "sat", "spoke", "lay", "led", "read", "grew", "lost", "fell",
        "sent", "built", "spent", "cut", "hit", "forgot", "forgot",
    }

    def detect(self, text: str) -> tuple[Tense, float]:
        """Detect the primary tense of a sentence.

        Args:
            text: Input text

        Returns:
            Tuple of (Tense, confidence)
        """
        text_lower = text.lower()

        # Check for future markers first
        for marker in self.FUTURE_MARKERS:
            if marker in text_lower:
                return Tense.FUTURE, 0.90

        # Check for progressive aspect (ongoing)
        if self.PROGRESSIVE_PATTERN.search(text_lower):
            # Determine past vs present progressive
            if re.search(r'\b(was|were)\s+\w+ing\b', text_lower):
                return Tense.PAST, 0.85
            else:
                return Tense.PRESENT, 0.85

        # Check for perfect aspect
        perfect_match = self.PERFECT_PATTERN.search(text_lower)
        if perfect_match:
            aux = perfect_match.group(1).lower()
            if aux == "had":
                return Tense.PAST, 0.90  # Past perfect
            else:
                return Tense.PRESENT, 0.85  # Present perfect

        # Check for past auxiliaries
        for aux in self.PAST_AUXILIARIES:
            if re.search(rf'\b{aux}\b', text_lower):
                return Tense.PAST, 0.80

        # Check for irregular past tense verbs
        words = set(re.findall(r'\b\w+\b', text_lower))
        if words & self.IRREGULAR_PAST:
            return Tense.PAST, 0.85

        # Check for regular past tense (-ed)
        if self.PAST_PATTERN.search(text_lower):
            # Filter out adjectives like "interested", "excited"
            ed_words = self.PAST_PATTERN.findall(text_lower)
            likely_verbs = [w for w in ed_words if not self._is_likely_adjective(w)]
            if likely_verbs:
                return Tense.PAST, 0.75

        # Check for present auxiliaries
        for aux in self.PRESENT_AUXILIARIES:
            if re.search(rf'\b{aux}\b', text_lower):
                return Tense.PRESENT, 0.75

        # Default to present with lower confidence
        return Tense.PRESENT, 0.50

    def _is_likely_adjective(self, word: str) -> bool:
        """Check if an -ed word is likely an adjective rather than verb."""
        adjective_ed = {
            "interested", "excited", "bored", "tired", "worried", "concerned",
            "surprised", "amazed", "pleased", "satisfied", "disappointed",
            "frustrated", "confused", "complicated", "detailed", "supposed",
        }
        return word in adjective_ed


# =============================================================================
# Aspect Classifier
# =============================================================================


class AspectClassifier:
    """Classifies Vendler aspectual categories.

    Uses verb properties and linguistic tests:
    - States: incompatible with progressive (*"I am knowing")
    - Activities: compatible with "for X time"
    - Accomplishments: compatible with "in X time"
    - Achievements: "at X moment", instantaneous
    - Semelfactives: inherently iterative
    """

    # Stative verbs (no internal temporal structure)
    STATIVE_VERBS = {
        # Mental states
        "know", "believe", "think", "understand", "remember", "forget",
        "want", "need", "like", "love", "hate", "prefer", "wish", "hope",
        # Perception
        "see", "hear", "smell", "taste", "feel",
        # Possession
        "have", "own", "possess", "belong", "contain", "include",
        # Relational
        "be", "seem", "appear", "look", "sound", "mean",
        "cost", "weigh", "measure", "equal", "resemble", "differ",
    }

    # Activity verbs (unbounded processes)
    ACTIVITY_VERBS = {
        "run", "walk", "swim", "drive", "fly", "dance", "sing", "play",
        "write", "read", "work", "study", "search", "look", "watch",
        "talk", "speak", "listen", "laugh", "cry", "sleep", "wait",
        "push", "pull", "carry", "hold", "move", "breathe",
    }

    # Achievement verbs (instantaneous, telic, punctual)
    ACHIEVEMENT_VERBS = {
        # Arrival/departure
        "arrive", "leave", "depart", "return", "land", "dock",
        # Start/stop
        "start", "begin", "stop", "end", "finish", "conclude", "terminate",
        # Finding/losing
        "find", "lose", "win", "die", "born", "recognize", "realize",
        "notice", "reach", "enter", "exit", "spot", "discover", "identify",
        # Physical events (punctual)
        "explode", "break", "crash", "collapse", "shatter", "burst",
        "launch", "ignite", "detonate", "erupt", "blast",
        # Appearance/disappearance
        "appear", "disappear", "emerge", "vanish", "surface", "pop",
        # State changes (instantaneous)
        "wake", "fall", "slip", "trip", "drop", "catch", "grab",
        "hit", "strike", "touch", "open", "close", "shut",
        # Achievement of goal
        "score", "accomplish", "achieve", "attain", "succeed", "fail",
    }

    # Semelfactive verbs (single bounded events that can repeat)
    SEMELFACTIVE_VERBS = {
        "knock", "tap", "kick", "hit", "punch", "slap", "blink",
        "cough", "sneeze", "hiccup", "flash", "beep", "click", "snap",
    }

    # Accomplishment verbs (bounded durative events with endpoint)
    ACCOMPLISHMENT_VERBS = {
        "build", "make", "create", "write", "compose", "paint", "draw",
        "construct", "assemble", "cook", "bake", "prepare",
        "repair", "fix", "mend", "destroy", "demolish",
        "cross", "traverse", "climb", "descend",
        "learn", "memorize", "solve", "prove", "recover", "heal",
    }

    # Irregular past tenses for aspect checking
    IRREGULAR_PAST_TO_BASE = {
        # Accomplishment irregulars
        "built": "build", "made": "make", "wrote": "write",
        "drew": "draw", "cooked": "cook", "solved": "solve",
        "created": "create", "destroyed": "destroy",
        # Achievement irregulars - common
        "arrived": "arrive", "left": "leave", "began": "begin",
        "found": "find", "lost": "lose", "won": "win",
        "broke": "break", "burst": "burst", "fell": "fall",
        "woke": "wake", "caught": "catch", "hit": "hit",
        "shut": "shut", "struck": "strike",
        # Achievement irregulars - launches/explosions
        "launched": "launch", "exploded": "explode", "crashed": "crash",
        "collapsed": "collapse", "erupted": "erupt",
        # Activity irregulars
        "ran": "run", "swam": "swim", "drove": "drive",
        "flew": "fly", "sang": "sing", "spoke": "speak",
        "read": "read", "held": "hold", "slept": "sleep",
        # Stative irregulars
        "knew": "know", "felt": "feel", "thought": "think",
        "believed": "believe", "understood": "understand",
    }

    # Accomplishment indicators (bounded telic events)
    ACCOMPLISHMENT_PATTERNS = [
        r'\b(build|make|create|write|draw|paint|compose)\s+\w+',
        r'\b(finish|complete|accomplish)\s+\w+',
        r'\b(go|walk|run|drive)\s+to\s+',
        r'\b(read|watch)\s+(a|the|this)\s+\w+',
    ]

    def __init__(self, config: TemporalConfig):
        """Initialize aspect classifier."""
        self.config = config
        self._accomplishment_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.ACCOMPLISHMENT_PATTERNS
        ]

    def classify(self, text: str, verb: str | None = None) -> tuple[Aspect, float]:
        """Classify the aspectual category.

        Args:
            text: Full sentence
            verb: Main verb (if known)

        Returns:
            Tuple of (Aspect, confidence)
        """
        text_lower = text.lower()

        # If verb provided, check lexical aspect
        if verb:
            verb_lower = verb.lower()

            # Check base form (remove inflections)
            base = self._get_base_form(verb_lower)

            if base in self.STATIVE_VERBS:
                return Aspect.STATE, 0.90

            if base in self.ACHIEVEMENT_VERBS:
                return Aspect.ACHIEVEMENT, 0.85

            if base in self.SEMELFACTIVE_VERBS:
                return Aspect.SEMELFACTIVE, 0.85

            if base in self.ACTIVITY_VERBS:
                # Could be promoted to accomplishment with bounded object
                if self._has_bounded_object(text_lower, base):
                    return Aspect.ACCOMPLISHMENT, 0.75
                return Aspect.ACTIVITY, 0.80

        # Check for irregular past tense forms first
        words = text_lower.split()
        for word in words:
            word = word.rstrip('.,!?;:')
            if word in self.IRREGULAR_PAST_TO_BASE:
                base = self.IRREGULAR_PAST_TO_BASE[word]
                if base in self.ACCOMPLISHMENT_VERBS:
                    return Aspect.ACCOMPLISHMENT, 0.80
                if base in self.ACHIEVEMENT_VERBS:
                    return Aspect.ACHIEVEMENT, 0.80
                if base in self.ACTIVITY_VERBS:
                    return Aspect.ACTIVITY, 0.80
                if base in self.STATIVE_VERBS:
                    return Aspect.STATE, 0.80

        # Fall back to pattern matching
        for pattern in self._accomplishment_patterns:
            if pattern.search(text_lower):
                return Aspect.ACCOMPLISHMENT, 0.70

        # Check for stative patterns
        for verb in self.STATIVE_VERBS:
            if re.search(rf'\b{verb}s?\b', text_lower):
                return Aspect.STATE, 0.75

        # Check for achievements - need to handle various inflections
        for verb in self.ACHIEVEMENT_VERBS:
            # Handle verbs ending in 'e' (arrive -> arrived, not arriveed)
            if verb.endswith('e'):
                patterns = [
                    rf'\b{verb}s?\b',  # arrive, arrives
                    rf'\b{verb}d\b',  # arrived
                    rf'\b{verb[:-1]}ing\b',  # arriving
                ]
            else:
                patterns = [
                    rf'\b{verb}s?\b',  # find, finds
                    rf'\b{verb}ed\b',  # started
                    rf'\b{verb}ing\b',  # finding
                ]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return Aspect.ACHIEVEMENT, 0.75

        # Check for accomplishment verbs
        for verb in self.ACCOMPLISHMENT_VERBS:
            if verb.endswith('e'):
                patterns = [rf'\b{verb}s?\b', rf'\b{verb}d\b', rf'\b{verb[:-1]}ing\b']
            else:
                patterns = [rf'\b{verb}s?\b', rf'\b{verb}ed\b', rf'\b{verb}ing\b']
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return Aspect.ACCOMPLISHMENT, 0.75

        # Default to activity with low confidence
        return Aspect.ACTIVITY, 0.40

    def _get_base_form(self, verb: str) -> str:
        """Get approximate base form of verb.

        For production, would use a proper lemmatizer.
        """
        # Remove common suffixes
        if verb.endswith("ing"):
            base = verb[:-3]
            if base.endswith(("runn", "swimm", "sitt")):
                base = base[:-1]  # running -> run
            return base

        if verb.endswith("ed"):
            return verb[:-2] if not verb.endswith("ied") else verb[:-3] + "y"

        if verb.endswith("s") and not verb.endswith("ss"):
            return verb[:-1]

        return verb

    def _has_bounded_object(self, text: str, verb: str) -> bool:
        """Check if activity verb has a bounded object (making it accomplishment).

        E.g., "run" is activity, but "run a marathon" is accomplishment.
        """
        # Look for determiner + noun after verb
        pattern = rf'\b{verb}\w*\s+(a|an|the|this|that)\s+\w+'
        return bool(re.search(pattern, text))


# =============================================================================
# Temporal Expression Extractor
# =============================================================================


class TemporalExpressionExtractor:
    """Extracts temporal expressions from text.

    Identifies time references like "yesterday", "next week", "in 2024",
    "for 3 hours", etc.
    """

    # Relative temporal expressions
    RELATIVE_EXPRESSIONS = {
        # Past
        "yesterday": (-1, "day"),
        "last night": (-1, "day"),
        "last week": (-1, "week"),
        "last month": (-1, "month"),
        "last year": (-1, "year"),
        "the other day": (-2, "day"),
        "recently": (-7, "day"),
        "earlier": (-1, "unspecified"),
        "before": (-1, "unspecified"),
        "previously": (-1, "unspecified"),
        "ago": (-1, "unspecified"),
        # Present
        "today": (0, "day"),
        "now": (0, "moment"),
        "currently": (0, "moment"),
        "at the moment": (0, "moment"),
        "right now": (0, "moment"),
        "this week": (0, "week"),
        "this month": (0, "month"),
        "this year": (0, "year"),
        # Future
        "tomorrow": (1, "day"),
        "next week": (1, "week"),
        "next month": (1, "month"),
        "next year": (1, "year"),
        "soon": (1, "unspecified"),
        "later": (1, "unspecified"),
        "afterwards": (1, "unspecified"),
        "in the future": (1, "unspecified"),
    }

    # Duration patterns
    DURATION_PATTERNS = [
        (r'for\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?', "for_duration"),
        (r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', "ago_duration"),
        (r'in\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?', "in_duration"),
        (r'(all|the whole)\s+(day|night|week|month|year)', "all_duration"),
        (r'since\s+(\w+)', "since"),
        (r'until\s+(\w+)', "until"),
    ]

    # Frequency patterns
    FREQUENCY_PATTERNS = [
        (r'\b(always)\b', "always"),
        (r'\b(usually|normally|typically)\b', "usually"),
        (r'\b(often|frequently)\b', "often"),
        (r'\b(sometimes|occasionally)\b', "sometimes"),
        (r'\b(rarely|seldom)\b', "rarely"),
        (r'\b(never)\b', "never"),
        (r'every\s+(day|week|month|year)', "every"),
        (r'once\s+a\s+(day|week|month|year)', "once_per"),
        (r'(\d+)\s+times?\s+(a|per)\s+(day|week|month|year)', "times_per"),
    ]

    # Date patterns
    DATE_PATTERNS = [
        # ISO format
        (r'\b(\d{4}-\d{2}-\d{2})\b', "iso"),
        # Written dates
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b', "written"),
        (r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})?\b', "written_alt"),
        # Year only
        (r'\bin\s+(\d{4})\b', "year"),
    ]

    def __init__(self, config: TemporalConfig):
        """Initialize extractor."""
        self.config = config

        # Compile patterns
        self._duration_patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in self.DURATION_PATTERNS]
        self._frequency_patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in self.FREQUENCY_PATTERNS]
        self._date_patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in self.DATE_PATTERNS]

    def extract(self, text: str) -> list[TemporalExpression]:
        """Extract all temporal expressions from text.

        Args:
            text: Input text

        Returns:
            List of TemporalExpression objects
        """
        expressions = []

        # Extract relative expressions
        expressions.extend(self._extract_relative(text))

        # Extract durations
        expressions.extend(self._extract_durations(text))

        # Extract frequencies
        expressions.extend(self._extract_frequencies(text))

        # Extract dates
        expressions.extend(self._extract_dates(text))

        # Sort by position and remove overlaps
        expressions.sort(key=lambda e: e.span[0])
        expressions = self._remove_overlaps(expressions)

        return expressions

    def _extract_relative(self, text: str) -> list[TemporalExpression]:
        """Extract relative temporal expressions."""
        expressions = []
        text_lower = text.lower()

        for expr, (offset, unit) in self.RELATIVE_EXPRESSIONS.items():
            pattern = re.compile(rf'\b{re.escape(expr)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                normalized = self._normalize_relative(offset, unit)
                expressions.append(TemporalExpression(
                    text=match.group(),
                    reference_type=TemporalReference.RELATIVE,
                    span=(match.start(), match.end()),
                    normalized=normalized,
                    confidence=0.90,
                ))

        return expressions

    def _extract_durations(self, text: str) -> list[TemporalExpression]:
        """Extract duration expressions."""
        expressions = []

        for pattern, pattern_type in self._duration_patterns:
            for match in pattern.finditer(text):
                normalized = f"duration:{match.group()}"
                expressions.append(TemporalExpression(
                    text=match.group(),
                    reference_type=TemporalReference.DURATIONAL,
                    span=(match.start(), match.end()),
                    normalized=normalized,
                    confidence=0.85,
                ))

        return expressions

    def _extract_frequencies(self, text: str) -> list[TemporalExpression]:
        """Extract frequency expressions."""
        expressions = []

        for pattern, pattern_type in self._frequency_patterns:
            for match in pattern.finditer(text):
                expressions.append(TemporalExpression(
                    text=match.group(),
                    reference_type=TemporalReference.FREQUENTATIVE,
                    span=(match.start(), match.end()),
                    normalized=pattern_type,
                    confidence=0.85,
                ))

        return expressions

    def _extract_dates(self, text: str) -> list[TemporalExpression]:
        """Extract absolute date expressions."""
        expressions = []

        for pattern, pattern_type in self._date_patterns:
            for match in pattern.finditer(text):
                normalized = self._normalize_date(match, pattern_type)
                expressions.append(TemporalExpression(
                    text=match.group(),
                    reference_type=TemporalReference.ABSOLUTE,
                    span=(match.start(), match.end()),
                    normalized=normalized,
                    confidence=0.90,
                ))

        return expressions

    def _normalize_relative(self, offset: int, unit: str) -> str:
        """Normalize relative expression to approximate ISO date."""
        if unit == "unspecified":
            return f"relative:{'+' if offset > 0 else ''}{offset}"

        today = datetime.now()

        if unit == "day":
            target = today + timedelta(days=offset)
        elif unit == "week":
            target = today + timedelta(weeks=offset)
        elif unit == "month":
            # Approximate
            target = today + timedelta(days=offset * 30)
        elif unit == "year":
            target = today + timedelta(days=offset * 365)
        else:
            return f"relative:{offset}_{unit}"

        return target.strftime("%Y-%m-%d")

    def _normalize_date(self, match: re.Match, pattern_type: str) -> str:
        """Normalize date expression to ISO format."""
        if pattern_type == "iso":
            return match.group(1)

        if pattern_type == "year":
            return f"{match.group(1)}-01-01"

        # For written dates, would need more sophisticated parsing
        return match.group()

    def _remove_overlaps(self, expressions: list[TemporalExpression]) -> list[TemporalExpression]:
        """Remove overlapping expressions, keeping the longer one."""
        if not expressions:
            return expressions

        result = [expressions[0]]
        for expr in expressions[1:]:
            last = result[-1]
            if expr.span[0] >= last.span[1]:
                # No overlap
                result.append(expr)
            elif (expr.span[1] - expr.span[0]) > (last.span[1] - last.span[0]):
                # Current is longer, replace
                result[-1] = expr

        return result


# =============================================================================
# Temporal Extractor
# =============================================================================


class TemporalExtractor:
    """Main extractor for temporal information.

    Combines tense detection, aspect classification, and temporal expression
    extraction.

    Example:
        >>> extractor = TemporalExtractor()
        >>> result = extractor.extract_sync("Doug forgot the meeting yesterday")
        >>> print(result.tense)  # Tense.PAST
        >>> print(result.aspect)  # Aspect.ACHIEVEMENT (forgot is punctual)
    """

    def __init__(
        self,
        config: TemporalConfig | None = None,
        llm: Any | None = None,
    ):
        """Initialize the temporal extractor.

        Args:
            config: Extraction configuration
            llm: Optional LLM provider
        """
        self.config = config or TemporalConfig()
        self.llm = llm

        self.tense_detector = TenseDetector()
        self.aspect_classifier = AspectClassifier(self.config)
        self.expression_extractor = TemporalExpressionExtractor(self.config)

    def extract_sync(self, text: str) -> TemporalInfo:
        """Synchronously extract temporal information.

        Args:
            text: Input text

        Returns:
            TemporalInfo with tense, aspect, and expressions
        """
        # Detect tense
        tense = Tense.PRESENT
        tense_confidence = 0.5
        if self.config.extract_tense:
            tense, tense_confidence = self.tense_detector.detect(text)

        # Classify aspect
        aspect = Aspect.ACTIVITY
        aspect_confidence = 0.5
        if self.config.extract_aspect:
            aspect, aspect_confidence = self.aspect_classifier.classify(text)

        # Extract temporal expressions
        expressions = []
        temporal_expression = None
        if self.config.extract_references:
            expressions = self.expression_extractor.extract(text)
            if expressions:
                # Use first (most prominent) expression
                temporal_expression = expressions[0].text

        # Compute overall confidence
        confidence = (tense_confidence + aspect_confidence) / 2
        if expressions:
            confidence = (confidence + expressions[0].confidence) / 2

        # Store expressions for retrieval
        self._last_expressions = expressions

        return TemporalInfo(
            tense=tense,
            aspect=aspect,
            reference_value=temporal_expression,
            confidence=confidence,
        )

    async def extract(self, text: str) -> TemporalInfo:
        """Asynchronously extract temporal information.

        If LLM is available, uses it for more accurate analysis.

        Args:
            text: Input text

        Returns:
            TemporalInfo
        """
        if self.llm:
            return await self._extract_with_llm(text)
        return self.extract_sync(text)

    async def _extract_with_llm(self, text: str) -> TemporalInfo:
        """Use LLM for temporal analysis."""
        prompt = f"""Analyze the temporal properties of this sentence:

Sentence: "{text}"

Determine:
1. Tense: past, present, or future
2. Aspect (Vendler category):
   - state: "know", "love" (no internal structure)
   - activity: "run", "swim" (unbounded process)
   - accomplishment: "build a house" (bounded, durative)
   - achievement: "arrive", "find" (instantaneous)
   - semelfactive: "knock", "cough" (repeatable instant)
3. Any temporal expressions (dates, times, durations)

Respond in XML:
<temporal>
  <tense>past|present|future</tense>
  <aspect>state|activity|accomplishment|achievement|semelfactive</aspect>
  <expressions>
    <expr type="relative|absolute|duration">the expression</expr>
  </expressions>
  <confidence>0.0-1.0</confidence>
</temporal>"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )
            return self._parse_llm_response(response)
        except Exception:
            return self.extract_sync(text)

    def _parse_llm_response(self, response: str) -> TemporalInfo:
        """Parse LLM response to TemporalInfo."""
        tense_match = re.search(r'<tense>(.*?)</tense>', response, re.IGNORECASE)
        aspect_match = re.search(r'<aspect>(.*?)</aspect>', response, re.IGNORECASE)
        expr_match = re.search(r'<expr[^>]*>(.*?)</expr>', response)
        conf_match = re.search(r'<confidence>(.*?)</confidence>', response)

        tense = Tense.PRESENT
        if tense_match:
            tense_str = tense_match.group(1).strip().upper()
            try:
                tense = Tense[tense_str]
            except KeyError:
                pass

        aspect = Aspect.ACTIVITY
        if aspect_match:
            aspect_str = aspect_match.group(1).strip().upper()
            try:
                aspect = Aspect[aspect_str]
            except KeyError:
                pass

        temporal_expression = expr_match.group(1).strip() if expr_match else None

        try:
            confidence = float(conf_match.group(1)) if conf_match else 0.8
        except ValueError:
            confidence = 0.8

        return TemporalInfo(
            tense=tense,
            aspect=aspect,
            temporal_expression=temporal_expression,
            confidence=confidence,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_tense(text: str) -> Tense:
    """Quick tense detection.

    Args:
        text: Input text

    Returns:
        Detected tense
    """
    detector = TenseDetector()
    tense, _ = detector.detect(text)
    return tense


def get_aspect(text: str) -> Aspect:
    """Quick aspect classification.

    Args:
        text: Input text

    Returns:
        Classified aspect
    """
    config = TemporalConfig()
    classifier = AspectClassifier(config)
    aspect, _ = classifier.classify(text)
    return aspect

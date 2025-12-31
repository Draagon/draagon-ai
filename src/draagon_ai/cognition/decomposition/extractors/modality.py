"""Modality extraction for epistemic, deontic, and evidential markers.

Modality expresses the speaker's attitude toward the proposition:

1. **Epistemic modality** (certainty/uncertainty):
   - "Doug definitely forgot" (high certainty)
   - "Doug might have forgotten" (low certainty)
   - "Doug probably forgot" (medium certainty)

2. **Deontic modality** (obligation/permission):
   - "Doug must attend the meeting" (obligation)
   - "Doug can leave early" (permission)
   - "Doug should call back" (recommendation)

3. **Evidential modality** (information source):
   - "Apparently Doug forgot" (reported/hearsay)
   - "Doug seems to have forgotten" (inferred)
   - "I saw Doug forget" (direct evidence)

This is crucial for knowledge extraction because modality affects
the factual status of the proposition:
- "Doug forgot" → factual claim
- "Doug might have forgotten" → uncertain claim
- "Doug must have forgotten" → inference
- "Doug should forget" → recommendation, not fact

Example:
    >>> from .modality import ModalityExtractor
    >>> extractor = ModalityExtractor()
    >>> result = extractor.extract_sync("Doug probably forgot the meeting")
    >>> print(result.modal_type)  # ModalType.EPISTEMIC
    >>> print(result.certainty_score)  # 0.75
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from .models import ModalityInfo, ModalType
from .config import ModalityConfig


# =============================================================================
# Additional Types
# =============================================================================


class EvidentialSource(str, Enum):
    """Source of information for evidential modality."""

    DIRECT = "direct"  # Speaker directly observed
    REPORTED = "reported"  # Someone told the speaker
    INFERRED = "inferred"  # Speaker inferred from evidence
    ASSUMED = "assumed"  # Speaker assumes based on reasoning
    UNKNOWN = "unknown"


class DeonticForce(str, Enum):
    """Strength of deontic modality."""

    OBLIGATION = "obligation"  # must, have to
    RECOMMENDATION = "recommendation"  # should, ought to
    PERMISSION = "permission"  # may, can
    PROHIBITION = "prohibition"  # must not, cannot
    ABILITY = "ability"  # can, able to


@dataclass
class ModalMarker:
    """A detected modal marker."""

    text: str
    """The modal word/phrase."""

    modal_type: ModalType
    """Type of modality."""

    span: tuple[int, int]
    """Character span in source text."""

    certainty: float | None = None
    """Certainty score for epistemic (0-1)."""

    deontic_force: DeonticForce | None = None
    """Force for deontic modality."""

    evidential_source: EvidentialSource | None = None
    """Source for evidential modality."""

    confidence: float = 0.9
    """Detection confidence."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "modal_type": self.modal_type.value,
            "span": self.span,
            "certainty": self.certainty,
            "deontic_force": self.deontic_force.value if self.deontic_force else None,
            "evidential_source": self.evidential_source.value if self.evidential_source else None,
            "confidence": self.confidence,
        }


# =============================================================================
# Epistemic Detector
# =============================================================================


class EpistemicDetector:
    """Detects epistemic modality (certainty/uncertainty).

    Maps modal expressions to certainty scores.
    Uses ModalityDisambiguator for ambiguous modals.
    """

    # Unambiguously epistemic modal patterns (no disambiguation needed)
    # These are modal + have/be patterns that are always epistemic
    EPISTEMIC_PATTERNS = [
        (r'\bmust\s+have\s+\w+', 0.90),  # "must have left"
        (r'\bmust\s+be\s+\w+', 0.90),    # "must be tired"
        (r'\bcould\s+have\s+\w+', 0.45), # "could have gone"
        (r'\bcould\s+be\s+\w+', 0.45),   # "could be sleeping"
        (r'\bmight\s+have\s+\w+', 0.40), # "might have seen"
        (r'\bmight\s+be\s+\w+', 0.40),   # "might be working"
        (r'\bwould\s+have\s+\w+', 0.65), # "would have known"
        (r'\bshould\s+have\s+\w+', 0.70), # "should have called"
    ]

    # Ambiguous modal verbs that need disambiguation
    # Note: "might", "will", "would" are classified as PRIMARILY_EPISTEMIC
    # in the disambiguator, so they'll be detected as epistemic
    AMBIGUOUS_MODALS = {
        "must": 0.90,
        "should": 0.70,
        "may": 0.50,
        "might": 0.40,
        "could": 0.45,
        "can": 0.50,
        "will": 0.85,
        "would": 0.65,
    }

    # Adverbs with certainty levels - always epistemic
    CERTAINTY_ADVERBS = {
        # Very high
        "definitely": 0.95,
        "certainly": 0.95,
        "absolutely": 0.98,
        "undoubtedly": 0.95,
        "surely": 0.90,
        "clearly": 0.90,
        "obviously": 0.90,
        "evidently": 0.85,
        # High
        "probably": 0.75,
        "likely": 0.75,
        "presumably": 0.70,
        "apparently": 0.70,
        # Medium
        "possibly": 0.45,
        "perhaps": 0.45,
        "maybe": 0.45,
        "conceivably": 0.40,
        # Low
        "unlikely": 0.25,
        "doubtfully": 0.20,
        "hardly": 0.15,
        "barely": 0.15,
    }

    # Phrases that indicate certainty - always epistemic
    CERTAINTY_PHRASES = {
        "I'm sure": 0.90,
        "I'm certain": 0.95,
        "I believe": 0.70,
        "I think": 0.65,
        "I suppose": 0.55,
        "I guess": 0.50,
        "I doubt": 0.30,
        "there's no doubt": 0.95,
        "without a doubt": 0.95,
        "it's certain": 0.95,
        "it's likely": 0.75,
        "it's possible": 0.50,
        "it's unlikely": 0.25,
        "it's impossible": 0.05,
    }

    def __init__(self, config: ModalityConfig):
        """Initialize detector."""
        self.config = config
        self._certainty_markers = {
            **self.config.epistemic_markers,
            **self.CERTAINTY_ADVERBS,
        }
        # Compile epistemic patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), certainty)
            for pattern, certainty in self.EPISTEMIC_PATTERNS
        ]
        # Lazy-load disambiguator
        self._disambiguator = None

    @property
    def disambiguator(self) -> "ModalityDisambiguator":
        """Lazy-load disambiguator to avoid circular initialization."""
        if self._disambiguator is None:
            self._disambiguator = ModalityDisambiguator()
        return self._disambiguator

    def detect(self, text: str) -> list[ModalMarker]:
        """Detect epistemic markers in text.

        Args:
            text: Input text

        Returns:
            List of ModalMarker objects
        """
        markers = []

        # Check for certainty adverbs - always epistemic
        for adverb, certainty in self._certainty_markers.items():
            pattern = re.compile(rf'\b{re.escape(adverb)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                markers.append(ModalMarker(
                    text=match.group(),
                    modal_type=ModalType.EPISTEMIC,
                    span=(match.start(), match.end()),
                    certainty=certainty,
                    confidence=0.90,
                ))

        # Check for certainty phrases - always epistemic
        for phrase, certainty in self.CERTAINTY_PHRASES.items():
            pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                markers.append(ModalMarker(
                    text=match.group(),
                    modal_type=ModalType.EPISTEMIC,
                    span=(match.start(), match.end()),
                    certainty=certainty,
                    confidence=0.85,
                ))

        # Check for unambiguous epistemic patterns (modal + have/be)
        for compiled_pattern, certainty in self._compiled_patterns:
            for match in compiled_pattern.finditer(text):
                markers.append(ModalMarker(
                    text=match.group(),
                    modal_type=ModalType.EPISTEMIC,
                    span=(match.start(), match.end()),
                    certainty=certainty,
                    confidence=0.85,  # High - pattern-matched
                ))

        # Check ambiguous modals with disambiguation
        for modal, certainty in self.AMBIGUOUS_MODALS.items():
            pattern = re.compile(rf'\b{modal}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                span = (match.start(), match.end())

                # Skip if already matched by a more specific pattern above
                already_matched = any(
                    m.span[0] <= span[0] and m.span[1] >= span[1]
                    for m in markers
                )
                if already_matched:
                    continue

                # Use disambiguator
                modal_type = self.disambiguator.disambiguate(text, modal, span)

                if modal_type == ModalType.EPISTEMIC:
                    markers.append(ModalMarker(
                        text=match.group(),
                        modal_type=ModalType.EPISTEMIC,
                        span=span,
                        certainty=certainty,
                        confidence=0.75,  # Lower - required disambiguation
                    ))

        return markers


# =============================================================================
# Deontic Detector
# =============================================================================


class ModalityDisambiguator:
    """Disambiguates between epistemic and deontic modality.

    Based on linguistic research:
    - Scope: What does the modal operate on?
    - Source: Who/what imposes the modality?
    - Context: What follows the modal verb?

    Key patterns:
    - modal + have/be + past_participle → EPISTEMIC (inference)
    - modal + stative_verb → Usually EPISTEMIC (state inference)
    - modal + action_verb + now/deadline → DEONTIC (obligation)
    - modal + action_verb (generic) → DEONTIC (obligation)

    References:
    - Kratzer (1981) - Modality semantics
    - Palmer (2001) - Mood and Modality
    """

    # Stative verbs - indicate state, not action
    STATIVE_VERBS = {
        "be", "is", "are", "am", "was", "were", "been", "being",
        "know", "believe", "think", "understand", "feel", "love", "hate",
        "want", "need", "like", "prefer", "seem", "appear", "look",
        "have", "has", "own", "possess", "belong", "contain", "include",
        "mean", "matter", "cost", "weigh", "measure", "equal",
        "tired", "happy", "sad", "angry", "hungry", "sick", "cold", "hot",
    }

    # Action/dynamic verbs - typically deontic in obligation context
    ACTION_VERBS = {
        # Movement
        "go", "come", "leave", "arrive", "start", "stop", "finish",
        "move", "walk", "run", "drive", "fly", "travel", "enter", "exit",
        # Actions
        "do", "make", "take", "give", "get", "put", "set",
        "work", "write", "read", "speak", "call", "send", "submit",
        "pay", "buy", "sell", "clean", "fix", "repair", "build",
        "attend", "complete", "report", "return", "follow", "obey",
        # Common obligation verbs
        "try", "help", "study", "learn", "practice", "prepare",
        "check", "verify", "confirm", "approve", "sign", "file",
        "wait", "stay", "remain", "keep", "maintain",
        "eat", "drink", "sleep", "rest", "exercise",
        "answer", "respond", "reply", "explain", "describe",
        "ask", "tell", "inform", "notify", "remind",
        "apologize", "forgive", "promise", "agree",
    }

    # Deontic context markers - temporal/obligation signals
    DEONTIC_CONTEXT_MARKERS = {
        "now", "immediately", "today", "tomorrow", "by", "before",
        "deadline", "required", "mandatory", "asap", "urgent",
    }

    # Epistemic context markers - inference signals
    EPISTEMIC_CONTEXT_MARKERS = {
        "probably", "likely", "perhaps", "maybe", "certainly",
        "already", "still", "yet", "obviously", "clearly",
    }

    # Modals that are primarily epistemic (rarely/never deontic)
    PRIMARILY_EPISTEMIC = {"might", "will", "would"}

    # Modals that are primarily deontic (rarely epistemic without context)
    PRIMARILY_DEONTIC = {"have to", "need to", "ought to"}

    def disambiguate(self, text: str, modal: str, span: tuple[int, int]) -> ModalType:
        """Determine if modal is epistemic or deontic.

        Args:
            text: Full sentence
            modal: The modal verb found
            span: Character span of the modal

        Returns:
            ModalType.EPISTEMIC or ModalType.DEONTIC
        """
        text_lower = text.lower()
        after_modal = text_lower[span[1]:].strip()

        # Pattern 0: Some modals are primarily epistemic
        # "might" always expresses epistemic possibility, not obligation
        # "will" expresses epistemic prediction about future
        if modal.lower() in self.PRIMARILY_EPISTEMIC:
            return ModalType.EPISTEMIC

        # Pattern 1: modal + have/be + past participle = EPISTEMIC
        # "must have left", "could be sleeping", "might have been"
        if re.match(r'\s*(?:have|be|been)\s+\w+', after_modal):
            return ModalType.EPISTEMIC

        # Pattern 2: Stative complement = EPISTEMIC
        # "must be tired", "could know", "might feel"
        first_word_match = re.match(r'\s*(\w+)', after_modal)
        if first_word_match:
            following_verb = first_word_match.group(1)
            if following_verb in self.STATIVE_VERBS:
                return ModalType.EPISTEMIC

        # Pattern 3: Action verb + deontic context = DEONTIC
        # "must leave now", "should submit by Friday"
        if first_word_match:
            following_verb = first_word_match.group(1)
            if following_verb in self.ACTION_VERBS:
                # Check for deontic context markers in rest of sentence
                for marker in self.DEONTIC_CONTEXT_MARKERS:
                    if marker in text_lower:
                        return ModalType.DEONTIC

        # Pattern 4: Epistemic context markers = EPISTEMIC
        # "must certainly know", "could probably come"
        for marker in self.EPISTEMIC_CONTEXT_MARKERS:
            if marker in text_lower:
                return ModalType.EPISTEMIC

        # Pattern 5: Second person subject + action = likely DEONTIC
        # "You must leave" = obligation
        before_modal = text_lower[:span[0]].strip()
        if before_modal.endswith("you") or text_lower.startswith("you "):
            if first_word_match and first_word_match.group(1) in self.ACTION_VERBS:
                return ModalType.DEONTIC

        # Pattern 6: Third person + action = could be either, default DEONTIC
        # "He must finish" = obligation (more common interpretation)
        if first_word_match:
            following_verb = first_word_match.group(1)
            if following_verb in self.ACTION_VERBS:
                return ModalType.DEONTIC

        # Default to EPISTEMIC for ambiguous cases (conservative)
        # Better to be uncertain than to assert obligation incorrectly
        return ModalType.EPISTEMIC


class DeonticDetector:
    """Detects deontic modality (obligation/permission).

    Identifies expressions of what should, must, or can be done.
    Uses ModalityDisambiguator for ambiguous modals like "must", "can", "may".
    """

    # Unambiguously deontic markers (no disambiguation needed)
    UNAMBIGUOUS_DEONTIC = {
        "have to": DeonticForce.OBLIGATION,
        "need to": DeonticForce.OBLIGATION,
        "has to": DeonticForce.OBLIGATION,
        "needs to": DeonticForce.OBLIGATION,
        "required to": DeonticForce.OBLIGATION,
        "obligated to": DeonticForce.OBLIGATION,
        "ought to": DeonticForce.RECOMMENDATION,
        "had better": DeonticForce.RECOMMENDATION,
        "supposed to": DeonticForce.RECOMMENDATION,
        "allowed to": DeonticForce.PERMISSION,
        "permitted to": DeonticForce.PERMISSION,
        "free to": DeonticForce.PERMISSION,
        "must not": DeonticForce.PROHIBITION,
        "mustn't": DeonticForce.PROHIBITION,
        "cannot": DeonticForce.PROHIBITION,
        "can't": DeonticForce.PROHIBITION,
        "forbidden to": DeonticForce.PROHIBITION,
        "prohibited from": DeonticForce.PROHIBITION,
        "not allowed to": DeonticForce.PROHIBITION,
        "able to": DeonticForce.ABILITY,
        "capable of": DeonticForce.ABILITY,
    }

    # Ambiguous modals that need disambiguation
    AMBIGUOUS_MODALS = {
        "must": DeonticForce.OBLIGATION,
        "should": DeonticForce.RECOMMENDATION,
        "may": DeonticForce.PERMISSION,
        "can": DeonticForce.ABILITY,
    }

    def __init__(self, config: ModalityConfig):
        """Initialize detector."""
        self.config = config
        self.disambiguator = ModalityDisambiguator()

    def detect(self, text: str) -> list[ModalMarker]:
        """Detect deontic markers in text.

        Args:
            text: Input text

        Returns:
            List of ModalMarker objects
        """
        markers = []

        # First check unambiguous deontic markers
        for marker, force in self.UNAMBIGUOUS_DEONTIC.items():
            pattern = re.compile(rf'\b{re.escape(marker)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                markers.append(ModalMarker(
                    text=match.group(),
                    modal_type=ModalType.DEONTIC,
                    span=(match.start(), match.end()),
                    deontic_force=force,
                    confidence=0.90,  # High confidence - unambiguous
                ))

        # Then check ambiguous modals with disambiguation
        for modal, force in self.AMBIGUOUS_MODALS.items():
            pattern = re.compile(rf'\b{re.escape(modal)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                span = (match.start(), match.end())

                # Use disambiguator to determine if this is deontic
                modal_type = self.disambiguator.disambiguate(text, modal, span)

                if modal_type == ModalType.DEONTIC:
                    markers.append(ModalMarker(
                        text=match.group(),
                        modal_type=ModalType.DEONTIC,
                        span=span,
                        deontic_force=force,
                        confidence=0.75,  # Lower - required disambiguation
                    ))

        return markers


# =============================================================================
# Evidential Detector
# =============================================================================


class EvidentialDetector:
    """Detects evidential modality (information source).

    Identifies how the speaker knows the information.
    """

    # Reported evidence (hearsay)
    REPORTED_MARKERS = {
        "apparently": EvidentialSource.REPORTED,
        "reportedly": EvidentialSource.REPORTED,
        "allegedly": EvidentialSource.REPORTED,
        "supposedly": EvidentialSource.REPORTED,
        "they say": EvidentialSource.REPORTED,
        "people say": EvidentialSource.REPORTED,
        "I heard": EvidentialSource.REPORTED,
        "I was told": EvidentialSource.REPORTED,
        "according to": EvidentialSource.REPORTED,
        "it's said": EvidentialSource.REPORTED,
    }

    # Inferred evidence
    INFERRED_MARKERS = {
        "seems": EvidentialSource.INFERRED,
        "appears": EvidentialSource.INFERRED,
        "looks like": EvidentialSource.INFERRED,
        "sounds like": EvidentialSource.INFERRED,
        "seems like": EvidentialSource.INFERRED,
        "appears to": EvidentialSource.INFERRED,
        "must have": EvidentialSource.INFERRED,  # Inferential "must"
        "evidently": EvidentialSource.INFERRED,
    }

    # Direct evidence
    DIRECT_MARKERS = {
        "I saw": EvidentialSource.DIRECT,
        "I watched": EvidentialSource.DIRECT,
        "I witnessed": EvidentialSource.DIRECT,
        "I heard": EvidentialSource.DIRECT,  # Ambiguous - context needed
        "I noticed": EvidentialSource.DIRECT,
        "I observed": EvidentialSource.DIRECT,
    }

    # Assumed (reasoning-based)
    ASSUMED_MARKERS = {
        "I assume": EvidentialSource.ASSUMED,
        "I presume": EvidentialSource.ASSUMED,
        "presumably": EvidentialSource.ASSUMED,
        "I imagine": EvidentialSource.ASSUMED,
        "I expect": EvidentialSource.ASSUMED,
        "I suspect": EvidentialSource.ASSUMED,
    }

    def __init__(self, config: ModalityConfig):
        """Initialize detector."""
        self.config = config
        self._all_markers = {
            **self.REPORTED_MARKERS,
            **self.INFERRED_MARKERS,
            **self.DIRECT_MARKERS,
            **self.ASSUMED_MARKERS,
            **{k: self._source_from_string(v)
               for k, v in self.config.evidential_markers.items()},
        }

    def _source_from_string(self, s: str) -> EvidentialSource:
        """Convert string to EvidentialSource."""
        try:
            return EvidentialSource[s.upper()]
        except KeyError:
            return EvidentialSource.UNKNOWN

    def detect(self, text: str) -> list[ModalMarker]:
        """Detect evidential markers in text.

        Args:
            text: Input text

        Returns:
            List of ModalMarker objects
        """
        markers = []

        for marker, source in self._all_markers.items():
            pattern = re.compile(rf'\b{re.escape(marker)}\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                markers.append(ModalMarker(
                    text=match.group(),
                    modal_type=ModalType.EVIDENTIAL,
                    span=(match.start(), match.end()),
                    evidential_source=source,
                    confidence=0.85,
                ))

        return markers


# =============================================================================
# Modality Extractor
# =============================================================================


class ModalityExtractor:
    """Main extractor for modality information.

    Combines epistemic, deontic, and evidential detection.

    Example:
        >>> extractor = ModalityExtractor()
        >>> result = extractor.extract_sync("Doug probably forgot")
        >>> print(result.modal_type)  # ModalType.EPISTEMIC
        >>> print(result.certainty_score)  # 0.75
    """

    def __init__(
        self,
        config: ModalityConfig | None = None,
        llm: Any | None = None,
    ):
        """Initialize the modality extractor.

        Args:
            config: Extraction configuration
            llm: Optional LLM provider
        """
        self.config = config or ModalityConfig()
        self.llm = llm

        self.epistemic_detector = EpistemicDetector(self.config)
        self.deontic_detector = DeonticDetector(self.config)
        self.evidential_detector = EvidentialDetector(self.config)

    def extract_sync(self, text: str) -> ModalityInfo:
        """Synchronously extract modality information.

        Args:
            text: Input text

        Returns:
            ModalityInfo with modal type and parameters
        """
        markers = []

        # Collect all markers
        if self.config.extract_epistemic:
            markers.extend(self.epistemic_detector.detect(text))

        if self.config.extract_deontic:
            markers.extend(self.deontic_detector.detect(text))

        if self.config.extract_evidential:
            markers.extend(self.evidential_detector.detect(text))

        if not markers:
            return ModalityInfo(
                modal_type=ModalType.NONE,
                confidence=0.90,
            )

        # Sort by position
        markers.sort(key=lambda m: m.span[0])

        # Primary modality is typically the first/most prominent
        primary = markers[0]

        # Compute certainty score for epistemic
        certainty_score = None
        if primary.modal_type == ModalType.EPISTEMIC and primary.certainty:
            certainty_score = primary.certainty
        elif self.config.compute_certainty:
            # Aggregate certainty from all epistemic markers
            epistemic_markers = [m for m in markers if m.modal_type == ModalType.EPISTEMIC]
            if epistemic_markers:
                certainties = [m.certainty for m in epistemic_markers if m.certainty]
                if certainties:
                    certainty_score = sum(certainties) / len(certainties)

        # Confidence is average of marker confidences
        confidence = sum(m.confidence for m in markers) / len(markers)

        # Extract deontic force and evidential source from markers
        deontic_force = None
        evidential_source = None
        for m in markers:
            if hasattr(m, 'deontic_force') and m.deontic_force:
                deontic_force = m.deontic_force
            if hasattr(m, 'evidential_source') and m.evidential_source:
                evidential_source = m.evidential_source

        # Store markers for extended access
        self._last_markers = markers
        self._last_deontic_force = deontic_force
        self._last_evidential_source = evidential_source

        return ModalityInfo(
            modal_type=primary.modal_type,
            modal_marker=primary.text if primary.modal_type in (ModalType.EPISTEMIC, ModalType.DEONTIC) else None,
            certainty=certainty_score,
            evidence_source=evidential_source.value if evidential_source else None,
            confidence=confidence,
        )

    async def extract(self, text: str) -> ModalityInfo:
        """Asynchronously extract modality information.

        If LLM is available, uses it for more nuanced analysis.

        Args:
            text: Input text

        Returns:
            ModalityInfo
        """
        if self.llm:
            return await self._extract_with_llm(text)
        return self.extract_sync(text)

    async def _extract_with_llm(self, text: str) -> ModalityInfo:
        """Use LLM for modality analysis."""
        prompt = f"""Analyze the modality in this sentence:

Sentence: "{text}"

Identify:
1. Modal type:
   - epistemic: certainty/uncertainty (probably, might, definitely)
   - deontic: obligation/permission (must, should, can)
   - evidential: information source (apparently, I heard, seems)
   - none: no modality markers

2. For epistemic: certainty score (0.0 = uncertain, 1.0 = certain)
3. For deontic: force (obligation, recommendation, permission, prohibition)
4. For evidential: source (direct, reported, inferred, assumed)

Respond in XML:
<modality>
  <type>epistemic|deontic|evidential|none</type>
  <marker>the modal word/phrase</marker>
  <certainty>0.0-1.0</certainty>
  <deontic_force>obligation|recommendation|permission|prohibition</deontic_force>
  <evidential_source>direct|reported|inferred|assumed</evidential_source>
  <confidence>0.0-1.0</confidence>
</modality>"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )
            return self._parse_llm_response(response)
        except Exception:
            return self.extract_sync(text)

    def _parse_llm_response(self, response: str) -> ModalityInfo:
        """Parse LLM response to ModalityInfo."""
        type_match = re.search(r'<type>(.*?)</type>', response, re.IGNORECASE)
        marker_match = re.search(r'<marker>(.*?)</marker>', response)
        cert_match = re.search(r'<certainty>(.*?)</certainty>', response)
        conf_match = re.search(r'<confidence>(.*?)</confidence>', response)

        modal_type = ModalType.NONE
        if type_match:
            type_str = type_match.group(1).strip().upper()
            try:
                modal_type = ModalType[type_str]
            except KeyError:
                pass

        modal_verb = marker_match.group(1).strip() if marker_match else None

        certainty_score = None
        if cert_match:
            try:
                certainty_score = float(cert_match.group(1))
            except ValueError:
                pass

        try:
            confidence = float(conf_match.group(1)) if conf_match else 0.8
        except ValueError:
            confidence = 0.8

        return ModalityInfo(
            modal_type=modal_type,
            modal_verb=modal_verb,
            certainty_score=certainty_score,
            confidence=confidence,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_certainty(text: str) -> float:
    """Quick certainty score extraction.

    Args:
        text: Input text

    Returns:
        Certainty score (0.0-1.0), defaults to 0.8 if no epistemic markers
    """
    extractor = ModalityExtractor()
    result = extractor.extract_sync(text)
    if result.certainty_score is not None:
        return result.certainty_score
    return 0.8  # Default for factual statements


def has_modal(text: str) -> bool:
    """Quick check if text contains modal markers.

    Args:
        text: Input text

    Returns:
        True if any modality detected
    """
    extractor = ModalityExtractor()
    result = extractor.extract_sync(text)
    return result.modal_type != ModalType.NONE

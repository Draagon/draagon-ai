"""Semantic Role Labeling (SRL) extraction.

This module extracts predicate-argument structures from sentences,
identifying WHO did WHAT to WHOM, WHERE, WHEN, and HOW.

Semantic Roles (PropBank/VerbNet style):
- ARG0: Agent (doer of action)
- ARG1: Patient/Theme (affected entity)
- ARG2: Instrument/Beneficiary/Attribute
- ARG3: Starting point/Beneficiary
- ARG4: Ending point
- ARGM-LOC: Location
- ARGM-TMP: Temporal
- ARGM-MNR: Manner
- ARGM-DIR: Direction
- ARGM-CAU: Cause
- ARGM-PRP: Purpose
- ARGM-NEG: Negation marker

This is crucial for knowledge extraction because it provides:
1. Structured representation of events
2. Clear identification of participants
3. Disambiguation of semantic relationships

Example:
    >>> from .semantic_roles import SemanticRoleExtractor
    >>> extractor = SemanticRoleExtractor()
    >>> roles = extractor.extract_sync("Doug gave the book to Mary yesterday")
    >>> # Returns roles like:
    >>> # - ARG0: "Doug" (agent - giver)
    >>> # - ARG1: "the book" (theme - thing given)
    >>> # - ARG2: "Mary" (recipient)
    >>> # - ARGM-TMP: "yesterday" (time)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from .models import SemanticRole
from .config import SemanticRoleConfig


# =============================================================================
# Role Types
# =============================================================================


class RoleType(str, Enum):
    """Semantic role types following PropBank conventions."""

    # Core arguments
    ARG0 = "ARG0"  # Agent, experiencer, causer
    ARG1 = "ARG1"  # Patient, theme, experiencer
    ARG2 = "ARG2"  # Instrument, benefactive, attribute, end state
    ARG3 = "ARG3"  # Start point, benefactive, attribute
    ARG4 = "ARG4"  # End point

    # Modifier arguments
    ARGM_LOC = "ARGM-LOC"  # Location
    ARGM_TMP = "ARGM-TMP"  # Temporal
    ARGM_MNR = "ARGM-MNR"  # Manner
    ARGM_DIR = "ARGM-DIR"  # Direction
    ARGM_CAU = "ARGM-CAU"  # Cause
    ARGM_PRP = "ARGM-PRP"  # Purpose
    ARGM_NEG = "ARGM-NEG"  # Negation
    ARGM_EXT = "ARGM-EXT"  # Extent
    ARGM_DIS = "ARGM-DIS"  # Discourse connective
    ARGM_ADV = "ARGM-ADV"  # Adverbial

    # Special
    V = "V"  # Verb/predicate itself


@dataclass
class Predicate:
    """A detected predicate (verb)."""

    text: str
    """The verb/predicate text."""

    lemma: str
    """Base form of the verb."""

    span: tuple[int, int]
    """Character span."""

    sense: str | None = None
    """WordNet or PropBank sense ID."""

    frame: str | None = None
    """PropBank frame ID."""


@dataclass
class Argument:
    """A detected argument."""

    text: str
    """The argument text."""

    role: RoleType
    """Semantic role type."""

    span: tuple[int, int]
    """Character span."""

    head: str | None = None
    """Head word of the argument."""

    confidence: float = 0.8
    """Detection confidence."""


@dataclass
class PredicateArgumentStructure:
    """Complete predicate-argument structure."""

    predicate: Predicate
    """The main predicate."""

    arguments: list[Argument] = field(default_factory=list)
    """List of arguments."""

    def to_semantic_roles(self) -> list[SemanticRole]:
        """Convert to list of SemanticRole objects."""
        roles = []
        for arg in self.arguments:
            roles.append(SemanticRole(
                role=arg.role.value,
                filler=arg.text,
                predicate=self.predicate.text,
                predicate_sense=self.predicate.sense,
                span=arg.span,
                confidence=arg.confidence,
            ))
        return roles


# =============================================================================
# Verb Frame Knowledge
# =============================================================================


# Simplified verb frames - in production would use PropBank/VerbNet
VERB_FRAMES = {
    # Transfer verbs: ARG0=giver, ARG1=thing, ARG2=recipient
    "give": {"ARG0": "giver", "ARG1": "thing_given", "ARG2": "recipient"},
    "gave": {"ARG0": "giver", "ARG1": "thing_given", "ARG2": "recipient"},
    "send": {"ARG0": "sender", "ARG1": "thing_sent", "ARG2": "recipient"},
    "sent": {"ARG0": "sender", "ARG1": "thing_sent", "ARG2": "recipient"},
    "tell": {"ARG0": "speaker", "ARG1": "message", "ARG2": "addressee"},
    "told": {"ARG0": "speaker", "ARG1": "message", "ARG2": "addressee"},
    "show": {"ARG0": "shower", "ARG1": "thing_shown", "ARG2": "viewer"},
    "showed": {"ARG0": "shower", "ARG1": "thing_shown", "ARG2": "viewer"},

    # Transitive verbs: ARG0=agent, ARG1=patient
    "eat": {"ARG0": "eater", "ARG1": "food"},
    "ate": {"ARG0": "eater", "ARG1": "food"},
    "read": {"ARG0": "reader", "ARG1": "material"},
    "write": {"ARG0": "writer", "ARG1": "text"},
    "wrote": {"ARG0": "writer", "ARG1": "text"},
    "break": {"ARG0": "breaker", "ARG1": "thing_broken"},
    "broke": {"ARG0": "breaker", "ARG1": "thing_broken"},
    "fix": {"ARG0": "fixer", "ARG1": "thing_fixed"},
    "fixed": {"ARG0": "fixer", "ARG1": "thing_fixed"},
    "forget": {"ARG0": "forgetter", "ARG1": "thing_forgotten"},
    "forgot": {"ARG0": "forgetter", "ARG1": "thing_forgotten"},
    "remember": {"ARG0": "rememberer", "ARG1": "thing_remembered"},
    "remembered": {"ARG0": "rememberer", "ARG1": "thing_remembered"},
    "know": {"ARG0": "knower", "ARG1": "thing_known"},
    "knew": {"ARG0": "knower", "ARG1": "thing_known"},
    "see": {"ARG0": "seer", "ARG1": "thing_seen"},
    "saw": {"ARG0": "seer", "ARG1": "thing_seen"},
    "hear": {"ARG0": "hearer", "ARG1": "thing_heard"},
    "heard": {"ARG0": "hearer", "ARG1": "thing_heard"},
    "examine": {"ARG0": "examiner", "ARG1": "thing_examined"},
    "examined": {"ARG0": "examiner", "ARG1": "thing_examined"},
    "chase": {"ARG0": "chaser", "ARG1": "thing_chased"},
    "chased": {"ARG0": "chaser", "ARG1": "thing_chased"},
    "want": {"ARG0": "wanter", "ARG1": "thing_wanted"},
    "wanted": {"ARG0": "wanter", "ARG1": "thing_wanted"},
    "leave": {"ARG0": "leaver", "ARG1": "place_left"},
    "left": {"ARG0": "leaver", "ARG1": "place_left"},
    "persuade": {"ARG0": "persuader", "ARG1": "action", "ARG2": "persuadee"},
    "persuaded": {"ARG0": "persuader", "ARG1": "action", "ARG2": "persuadee"},
    "arrive": {"ARG0": "arriver", "ARG1": "destination"},
    "arrived": {"ARG0": "arriver", "ARG1": "destination"},
    "cancel": {"ARG0": "canceller", "ARG1": "thing_cancelled"},
    "cancelled": {"ARG0": "canceller", "ARG1": "thing_cancelled"},

    # Motion verbs: ARG0=mover, ARG1=path/destination
    "go": {"ARG0": "goer", "ARG1": "destination"},
    "went": {"ARG0": "goer", "ARG1": "destination"},
    "come": {"ARG0": "comer", "ARG1": "destination"},
    "came": {"ARG0": "comer", "ARG1": "destination"},
    "run": {"ARG0": "runner", "ARG1": "path"},
    "ran": {"ARG0": "runner", "ARG1": "path"},
    "walk": {"ARG0": "walker", "ARG1": "path"},
    "walked": {"ARG0": "walker", "ARG1": "path"},

    # Creation verbs: ARG0=creator, ARG1=creation
    "make": {"ARG0": "maker", "ARG1": "thing_made"},
    "made": {"ARG0": "maker", "ARG1": "thing_made"},
    "build": {"ARG0": "builder", "ARG1": "structure"},
    "built": {"ARG0": "builder", "ARG1": "structure"},
    "create": {"ARG0": "creator", "ARG1": "creation"},
    "created": {"ARG0": "creator", "ARG1": "creation"},

    # Communication verbs: ARG0=speaker, ARG1=content
    "say": {"ARG0": "speaker", "ARG1": "utterance"},
    "said": {"ARG0": "speaker", "ARG1": "utterance"},
    "speak": {"ARG0": "speaker", "ARG1": "topic"},
    "spoke": {"ARG0": "speaker", "ARG1": "topic"},
    "talk": {"ARG0": "talker", "ARG1": "topic"},
    "talked": {"ARG0": "talker", "ARG1": "topic"},

    # Stative verbs: ARG0/ARG1 varies
    "be": {"ARG1": "entity", "ARG2": "attribute"},
    "is": {"ARG1": "entity", "ARG2": "attribute"},
    "was": {"ARG1": "entity", "ARG2": "attribute"},
    "are": {"ARG1": "entity", "ARG2": "attribute"},
    "were": {"ARG1": "entity", "ARG2": "attribute"},
    "have": {"ARG0": "possessor", "ARG1": "possession"},
    "has": {"ARG0": "possessor", "ARG1": "possession"},
    "had": {"ARG0": "possessor", "ARG1": "possession"},
    "own": {"ARG0": "owner", "ARG1": "possession"},
    "owned": {"ARG0": "owner", "ARG1": "possession"},

    # Change of state: ARG0=causer, ARG1=theme
    "stop": {"ARG0": "stopper", "ARG1": "stopped_activity"},
    "stopped": {"ARG0": "stopper", "ARG1": "stopped_activity"},
    "start": {"ARG0": "starter", "ARG1": "started_activity"},
    "started": {"ARG0": "starter", "ARG1": "started_activity"},
    "begin": {"ARG0": "beginner", "ARG1": "begun_activity"},
    "began": {"ARG0": "beginner", "ARG1": "begun_activity"},
    "continue": {"ARG0": "continuer", "ARG1": "continued_activity"},
    "continued": {"ARG0": "continuer", "ARG1": "continued_activity"},
}


# =============================================================================
# Argument Detector
# =============================================================================


class PassiveDetector:
    """Detects passive voice constructions.

    Passive voice pattern: [subject] + [be form] + [past participle] + [by phrase]
    Example: "The cat was chased by the dog"

    In passive:
    - Surface subject is actually ARG1 (patient)
    - By-phrase contains ARG0 (agent)
    - Main predicate is the past participle, not the auxiliary
    """

    # Forms of "to be"
    BE_FORMS = {"be", "is", "are", "was", "were", "been", "being", "am"}

    # Past participle patterns (simplified)
    # In practice would use morphological analysis
    PAST_PARTICIPLE_ENDINGS = ("ed", "en", "t", "n")

    # Irregular past participles (comprehensive list for passive detection)
    IRREGULAR_PP = {
        # Common irregulars
        "gone", "done", "seen", "been", "taken", "given", "broken",
        "chosen", "spoken", "written", "driven", "eaten", "fallen",
        "forgotten", "hidden", "frozen", "stolen", "worn", "torn",
        "born", "sworn", "drawn", "grown", "known", "shown", "thrown",
        "blown", "flown", "woken", "sung", "rung", "begun", "swum",
        "drunk", "shrunk", "sunk", "struck", "stuck", "hung",
        # -made verbs
        "made", "remade", "unmade",
        # -ed that don't follow regular patterns well
        "chased", "cancelled", "examined", "used", "abused",
        # -t past participles
        "built", "burnt", "dealt", "dreamt", "dwelt", "felt",
        "kept", "knelt", "leapt", "learnt", "left", "lent",
        "lost", "meant", "sent", "slept", "smelt", "spelt",
        "spent", "spilt", "swept", "wept",
        # Other irregular forms
        "bought", "brought", "caught", "fought", "sought", "taught", "thought",
        "fed", "fled", "held", "led", "met", "paid", "read", "said",
        "sold", "told", "understood", "stood", "sat", "set", "put", "cut", "hit",
        "hurt", "let", "shut", "spread", "split", "quit", "rid", "run",
        "become", "come", "overcome",
        # Launched, and other -ed that are also past tense
        "launched", "finished", "started", "killed", "opened", "closed",
        "created", "destroyed", "discovered", "found", "lost", "won",
    }

    def is_passive(self, text: str) -> tuple[bool, dict]:
        """Check if sentence is passive.

        Args:
            text: Input sentence

        Returns:
            Tuple of (is_passive, info_dict)
            info_dict contains: auxiliary, main_verb, by_phrase
        """
        words = text.lower().split()

        # Look for BE + PAST_PARTICIPLE pattern
        for i, word in enumerate(words):
            # Remove punctuation
            clean_word = word.rstrip('.,!?;:')

            if clean_word in self.BE_FORMS:
                # Check next word for past participle
                if i + 1 < len(words):
                    next_word = words[i + 1].rstrip('.,!?;:')

                    if self._is_past_participle(next_word):
                        # Found passive pattern
                        # Look for by-phrase
                        by_phrase = None
                        for j in range(i + 2, len(words)):
                            if words[j].lower() == "by":
                                # Extract agent from by-phrase
                                by_phrase_words = words[j + 1:]
                                by_phrase = " ".join(w.rstrip('.,!?;:') for w in by_phrase_words)
                                break

                        return True, {
                            "auxiliary": clean_word,
                            "main_verb": next_word,
                            "by_phrase": by_phrase,
                            "auxiliary_index": i,
                        }

        return False, {}

    def _is_past_participle(self, word: str) -> bool:
        """Check if word is a past participle."""
        word = word.lower()

        # Check irregular
        if word in self.IRREGULAR_PP:
            return True

        # Check regular endings
        for ending in self.PAST_PARTICIPLE_ENDINGS:
            if word.endswith(ending):
                return True

        return False


class ArgumentDetector:
    """Detects arguments for predicates.

    Uses positional heuristics and patterns without full parsing.
    Now includes passive voice handling.
    """

    # Preposition to role mapping
    PREP_TO_ROLE = {
        # Location
        "in": RoleType.ARGM_LOC,
        "at": RoleType.ARGM_LOC,
        "on": RoleType.ARGM_LOC,
        "near": RoleType.ARGM_LOC,
        "inside": RoleType.ARGM_LOC,
        "outside": RoleType.ARGM_LOC,
        "under": RoleType.ARGM_LOC,
        "above": RoleType.ARGM_LOC,
        # Direction
        "to": RoleType.ARG2,  # Often recipient or destination
        "from": RoleType.ARG3,  # Often source
        "into": RoleType.ARGM_DIR,
        "onto": RoleType.ARGM_DIR,
        "toward": RoleType.ARGM_DIR,
        "towards": RoleType.ARGM_DIR,
        # Manner
        "with": RoleType.ARG2,  # Often instrument
        "by": RoleType.ARGM_MNR,  # Can also be agent in passive
        "without": RoleType.ARGM_MNR,
        # Cause/Purpose
        "because": RoleType.ARGM_CAU,
        "for": RoleType.ARGM_PRP,
        # Temporal (handled separately)
    }

    # Temporal markers
    TEMPORAL_MARKERS = {
        "yesterday", "today", "tomorrow", "now", "then",
        "later", "earlier", "before", "after", "during",
        "always", "never", "sometimes", "often", "usually",
        "last", "next", "this", "every",
    }

    def detect(
        self,
        text: str,
        predicate: Predicate,
    ) -> list[Argument]:
        """Detect arguments for a predicate.

        Args:
            text: Full sentence
            predicate: The predicate to find arguments for

        Returns:
            List of detected arguments
        """
        arguments = []

        # Split text around predicate
        before = text[:predicate.span[0]].strip()
        after = text[predicate.span[1]:].strip()

        # ARG0: Usually subject (before verb)
        if before:
            arg0 = self._extract_noun_phrase(before, end=True)
            if arg0:
                span_start = text.find(arg0)
                arguments.append(Argument(
                    text=arg0,
                    role=RoleType.ARG0,
                    span=(span_start, span_start + len(arg0)),
                    head=self._get_head(arg0),
                    confidence=0.80,
                ))

        # ARG1 and other arguments: After verb
        if after:
            # Check for direct object (first NP after verb)
            parts = self._split_by_prepositions(after)

            if parts:
                # First part is likely ARG1 (direct object)
                first = parts[0]
                if not self._is_prepositional(first):
                    arg1 = self._extract_noun_phrase(first, end=False)
                    if arg1:
                        span_start = text.find(arg1)
                        arguments.append(Argument(
                            text=arg1,
                            role=RoleType.ARG1,
                            span=(span_start, span_start + len(arg1)),
                            head=self._get_head(arg1),
                            confidence=0.75,
                        ))

                # Check remaining parts for PP arguments
                for part in parts[1:]:
                    arg = self._analyze_pp(part, text)
                    if arg:
                        arguments.append(arg)

            # Check for temporal expressions
            temporal = self._find_temporal(after, text)
            if temporal:
                arguments.append(temporal)

        return arguments

    def _extract_noun_phrase(self, text: str, end: bool = False) -> str | None:
        """Extract a noun phrase from text.

        Args:
            text: Text to extract from
            end: If True, extract from end; otherwise from start

        Returns:
            Noun phrase or None
        """
        text = text.strip()
        if not text:
            return None

        # Remove punctuation
        text = text.rstrip('.,!?;:')

        if end:
            # Take last words that form NP
            words = text.split()
            if not words:
                return None

            # Simple heuristic: last N words where N <= 4
            np_words = words[-min(4, len(words)):]

            # Remove articles/determiners from beginning if standalone
            while np_words and np_words[0].lower() in {"a", "an", "the", "this", "that"}:
                if len(np_words) > 1:
                    break  # Keep determiner with noun
                np_words = np_words[1:]

            return " ".join(np_words) if np_words else None
        else:
            # Take first words that form NP
            words = text.split()
            if not words:
                return None

            # Include determiner and up to 3 more words
            np_words = []
            for i, word in enumerate(words[:5]):
                np_words.append(word)
                # Stop at preposition or punctuation
                if word.lower() in self.PREP_TO_ROLE or word in ".,;:!?":
                    np_words = np_words[:-1]  # Don't include the prep
                    break

            return " ".join(np_words) if np_words else None

    def _split_by_prepositions(self, text: str) -> list[str]:
        """Split text by prepositional boundaries."""
        preps = "|".join(re.escape(p) for p in self.PREP_TO_ROLE.keys())
        pattern = rf'\b({preps})\s+'

        parts = re.split(pattern, text, flags=re.IGNORECASE)
        result = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i].lower() in self.PREP_TO_ROLE:
                # Combine prep with following text
                result.append(parts[i] + " " + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    result.append(parts[i])
                i += 1

        return result

    def _is_prepositional(self, text: str) -> bool:
        """Check if text starts with a preposition."""
        first_word = text.split()[0].lower() if text.split() else ""
        return first_word in self.PREP_TO_ROLE

    def _analyze_pp(self, pp: str, full_text: str) -> Argument | None:
        """Analyze a prepositional phrase."""
        words = pp.split()
        if not words:
            return None

        prep = words[0].lower()
        if prep not in self.PREP_TO_ROLE:
            return None

        role = self.PREP_TO_ROLE[prep]
        content = " ".join(words[1:]).rstrip('.,!?;:')

        if not content:
            return None

        span_start = full_text.find(pp)
        if span_start < 0:
            span_start = full_text.find(content)

        return Argument(
            text=content,
            role=role,
            span=(span_start, span_start + len(content)),
            head=self._get_head(content),
            confidence=0.70,
        )

    def _find_temporal(self, text: str, full_text: str) -> Argument | None:
        """Find temporal expressions."""
        text_lower = text.lower()

        for marker in self.TEMPORAL_MARKERS:
            if marker in text_lower:
                # Find the temporal phrase
                pattern = rf'\b{re.escape(marker)}(?:\s+\w+){{0,2}}\b'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    span_start = full_text.find(match.group())
                    return Argument(
                        text=match.group(),
                        role=RoleType.ARGM_TMP,
                        span=(span_start, span_start + len(match.group())),
                        head=marker,
                        confidence=0.85,
                    )

        return None

    def _get_head(self, phrase: str) -> str | None:
        """Get the head word of a phrase (simplified)."""
        words = phrase.split()
        if not words:
            return None
        # Last word is often the head (simplified)
        return words[-1].rstrip('.,!?;:')


# =============================================================================
# Predicate Detector
# =============================================================================


class PredicateDetector:
    """Detects predicates (verbs) in text."""

    # Common verbs for quick matching
    COMMON_VERBS = set(VERB_FRAMES.keys())

    def detect(self, text: str) -> list[Predicate]:
        """Detect predicates in text.

        Args:
            text: Input text

        Returns:
            List of detected predicates
        """
        predicates = []
        words = list(re.finditer(r'\b\w+\b', text))

        for match in words:
            word = match.group()
            word_lower = word.lower()

            if word_lower in self.COMMON_VERBS:
                predicates.append(Predicate(
                    text=word,
                    lemma=self._get_lemma(word_lower),
                    span=(match.start(), match.end()),
                    frame=VERB_FRAMES.get(word_lower),
                ))

        return predicates

    def _get_lemma(self, verb: str) -> str:
        """Get base form of verb (simplified)."""
        # Very simplified lemmatization
        lemma_map = {
            "gave": "give", "sent": "send", "told": "tell", "showed": "show",
            "ate": "eat", "wrote": "write", "broke": "break", "fixed": "fix",
            "forgot": "forget", "remembered": "remember", "knew": "know",
            "saw": "see", "heard": "hear", "went": "go", "came": "come",
            "ran": "run", "walked": "walk", "made": "make", "built": "build",
            "created": "create", "said": "say", "spoke": "speak", "talked": "talk",
            "was": "be", "were": "be", "is": "be", "are": "be",
            "has": "have", "had": "have", "owned": "own",
            "stopped": "stop", "started": "start", "began": "begin",
            "continued": "continue",
        }
        return lemma_map.get(verb, verb)


# =============================================================================
# Semantic Role Extractor
# =============================================================================


class SemanticRoleExtractor:
    """Main extractor for semantic roles.

    Identifies predicates and their arguments.
    Handles both active and passive voice constructions.

    Example:
        >>> extractor = SemanticRoleExtractor()
        >>> roles = extractor.extract_sync("Doug gave the book to Mary")
        >>> for role in roles:
        ...     print(f"{role.role}: {role.filler}")
    """

    def __init__(
        self,
        config: SemanticRoleConfig | None = None,
        llm: Any | None = None,
    ):
        """Initialize the semantic role extractor.

        Args:
            config: Extraction configuration
            llm: Optional LLM provider
        """
        self.config = config or SemanticRoleConfig()
        self.llm = llm

        self.predicate_detector = PredicateDetector()
        self.argument_detector = ArgumentDetector()
        self.passive_detector = PassiveDetector()

    def extract_sync(
        self,
        text: str,
        wsd_results: dict[str, str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> list[SemanticRole]:
        """Synchronously extract semantic roles.

        Handles both active and passive voice.

        Args:
            text: Input text
            wsd_results: Optional WSD results for verbs
            entity_ids: Optional entity identifiers

        Returns:
            List of SemanticRole objects
        """
        all_roles = []

        # Check for passive voice first
        is_passive, passive_info = self.passive_detector.is_passive(text)

        if is_passive:
            # Handle passive construction specially
            roles = self._extract_passive_roles(text, passive_info, wsd_results)
            all_roles.extend(roles)
        else:
            # Active voice - use standard extraction
            predicates = self.predicate_detector.detect(text)

            for predicate in predicates:
                # Add WSD sense if available
                if wsd_results and predicate.text.lower() in wsd_results:
                    predicate.sense = wsd_results[predicate.text.lower()]

                # Detect arguments
                arguments = self.argument_detector.detect(text, predicate)

                # Create structure
                structure = PredicateArgumentStructure(
                    predicate=predicate,
                    arguments=arguments,
                )

                # Convert to SemanticRole objects
                roles = structure.to_semantic_roles()

                # Filter by confidence
                roles = [
                    r for r in roles
                    if r.confidence >= self.config.confidence_threshold
                ]

                # Limit per predicate
                roles = roles[:self.config.max_roles_per_predicate]

                # Filter modifiers if configured
                if not self.config.include_modifiers:
                    roles = [
                        r for r in roles
                        if not r.role.startswith("ARGM")
                    ]

                all_roles.extend(roles)

        return all_roles

    def _extract_passive_roles(
        self,
        text: str,
        passive_info: dict,
        wsd_results: dict[str, str] | None = None,
    ) -> list[SemanticRole]:
        """Extract semantic roles from passive construction.

        In passive voice:
        - Surface subject -> ARG1 (patient)
        - By-phrase -> ARG0 (agent)
        - Main verb is past participle, not auxiliary

        Args:
            text: Full sentence
            passive_info: Info from passive detector
            wsd_results: Optional WSD results

        Returns:
            List of semantic roles
        """
        roles = []

        main_verb = passive_info.get("main_verb", "")
        by_phrase = passive_info.get("by_phrase")
        aux_index = passive_info.get("auxiliary_index", 0)

        # Find main verb span in original text
        main_verb_match = re.search(rf'\b{re.escape(main_verb)}\b', text, re.IGNORECASE)
        if main_verb_match:
            verb_span = (main_verb_match.start(), main_verb_match.end())
            actual_verb = main_verb_match.group()
        else:
            verb_span = (0, len(main_verb))
            actual_verb = main_verb

        # Get sense if available
        sense = None
        if wsd_results and main_verb.lower() in wsd_results:
            sense = wsd_results[main_verb.lower()]

        # Extract surface subject (before auxiliary) -> ARG1 (patient)
        words = text.split()
        if aux_index > 0:
            subject_words = words[:aux_index]
            subject = " ".join(subject_words).strip()
            if subject:
                span_start = 0
                roles.append(SemanticRole(
                    role="ARG1",  # Patient in passive
                    filler=subject,
                    predicate=actual_verb,
                    predicate_sense=sense,
                    span=(span_start, span_start + len(subject)),
                    confidence=0.85,
                ))

        # Extract by-phrase -> ARG0 (agent)
        if by_phrase:
            by_match = re.search(r'\bby\s+(.+?)(?:\s+(?:in|at|on|to|from|with)|[.,!?;:]|$)', text, re.IGNORECASE)
            if by_match:
                agent = by_match.group(1).strip().rstrip('.,!?;:')
                span_start = by_match.start(1)
                roles.append(SemanticRole(
                    role="ARG0",  # Agent from by-phrase
                    filler=agent,
                    predicate=actual_verb,
                    predicate_sense=sense,
                    span=(span_start, span_start + len(agent)),
                    confidence=0.80,
                ))

        return roles

    async def extract(
        self,
        text: str,
        wsd_results: dict[str, str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> list[SemanticRole]:
        """Asynchronously extract semantic roles.

        Uses LLM if available for better accuracy.

        Args:
            text: Input text
            wsd_results: Optional WSD results
            entity_ids: Optional entity identifiers

        Returns:
            List of SemanticRole objects
        """
        if self.llm and self.config.use_llm:
            return await self._extract_with_llm(text, wsd_results)
        return self.extract_sync(text, wsd_results, entity_ids)

    async def _extract_with_llm(
        self,
        text: str,
        wsd_results: dict[str, str] | None = None,
    ) -> list[SemanticRole]:
        """Use LLM for semantic role extraction."""
        prompt = self.config.prompt_template.format(text=text)

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )
            return self._parse_llm_response(response, text)
        except Exception:
            return self.extract_sync(text, wsd_results)

    def _parse_llm_response(
        self,
        response: str,
        text: str,
    ) -> list[SemanticRole]:
        """Parse LLM response to SemanticRole objects."""
        roles = []

        # Parse predicate blocks
        pred_pattern = r'<predicate>(.*?)</predicate>'
        for pred_match in re.finditer(pred_pattern, response, re.DOTALL):
            pred_block = pred_match.group(1)

            # Get verb
            verb_match = re.search(r'<verb>(.*?)</verb>', pred_block)
            verb = verb_match.group(1).strip() if verb_match else "unknown"

            # Get arguments
            arg_pattern = r'<arg\s+role="([^"]+)"(?:\s+confidence="([\d.]+)")?\s*>(.*?)</arg>'
            for arg_match in re.finditer(arg_pattern, pred_block, re.DOTALL):
                role_str = arg_match.group(1)
                conf_str = arg_match.group(2)
                filler = arg_match.group(3).strip()

                try:
                    confidence = float(conf_str) if conf_str else 0.8
                except ValueError:
                    confidence = 0.8

                # Find span in original text
                span = None
                if filler in text:
                    start = text.find(filler)
                    span = (start, start + len(filler))

                roles.append(SemanticRole(
                    role=role_str,
                    filler=filler,
                    predicate=verb,
                    span=span,
                    confidence=confidence,
                ))

        return roles


# =============================================================================
# Convenience Functions
# =============================================================================


def get_semantic_roles(text: str) -> list[SemanticRole]:
    """Quick semantic role extraction.

    Args:
        text: Input text

    Returns:
        List of semantic roles
    """
    extractor = SemanticRoleExtractor()
    return extractor.extract_sync(text)


def get_agent(text: str) -> str | None:
    """Get the agent (ARG0) of the main event.

    Args:
        text: Input text

    Returns:
        Agent string or None
    """
    extractor = SemanticRoleExtractor()
    roles = extractor.extract_sync(text)

    for role in roles:
        if role.role == "ARG0":
            return role.filler

    return None


def get_patient(text: str) -> str | None:
    """Get the patient/theme (ARG1) of the main event.

    Args:
        text: Input text

    Returns:
        Patient string or None
    """
    extractor = SemanticRoleExtractor()
    roles = extractor.extract_sync(text)

    for role in roles:
        if role.role == "ARG1":
            return role.filler

    return None

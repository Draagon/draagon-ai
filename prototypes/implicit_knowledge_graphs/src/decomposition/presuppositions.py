"""Presupposition extraction from text.

Presuppositions are what must be true for a statement to make sense.
This module detects presupposition triggers and extracts the presupposed content.

Supported trigger types:
- Definite descriptions ("the X" -> X exists)
- Factive verbs ("realize that X" -> X is true)
- Change-of-state ("stop X" -> was doing X before)
- Iteratives ("again" -> happened before)
- Temporal clauses ("before X" -> X occurred)
- Possessives ("X's Y" -> X has Y)
- Implicatives ("manage to X" -> X was difficult)
- Comparatives ("more than X" -> X has the property)
- Cleft sentences ("It was X who..." -> someone did it)
- Counterfactuals ("if X had..." -> X didn't happen)

Example:
    >>> from decomposition.presuppositions import PresuppositionExtractor
    >>> from decomposition.config import PresuppositionConfig
    >>>
    >>> extractor = PresuppositionExtractor(PresuppositionConfig())
    >>> presups = await extractor.extract("Doug forgot the meeting again")
    >>> for p in presups:
    ...     print(f"{p.trigger_type}: {p.content}")
    ... # iterative: Doug forgot before
    ... # definite_description: A specific meeting exists
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from decomposition.models import Presupposition, PresuppositionTrigger
from decomposition.config import PresuppositionConfig


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


def _extract_llm_content(response: Any) -> str:
    """Extract string content from LLM response."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    return str(response)


# =============================================================================
# Trigger Detection
# =============================================================================


@dataclass
class DetectedTrigger:
    """A detected presupposition trigger in text."""

    trigger_type: PresuppositionTrigger
    trigger_text: str
    span: tuple[int, int]
    context: str  # Surrounding text for content generation
    confidence: float = 1.0


class TriggerDetector:
    """Detects presupposition triggers using pattern matching.

    This is the pattern-based component that identifies WHERE triggers
    occur. Content generation (WHAT is presupposed) is handled separately.
    """

    # Patterns for each trigger type
    # Format: (pattern, trigger_type, confidence, context_extraction_groups)

    DEFINITE_DESC_PATTERN = re.compile(
        r'\b(the)\s+(\w+(?:\s+\w+)?)\b',
        re.IGNORECASE
    )

    FACTIVE_VERBS = {
        "realize", "realized", "realizes", "realizing",
        "know", "knew", "knows", "knowing",
        "regret", "regretted", "regrets", "regretting",
        "discover", "discovered", "discovers", "discovering",
        "notice", "noticed", "notices", "noticing",
        "remember", "remembered", "remembers", "remembering",
        "forget", "forgot", "forgets", "forgetting",  # implicative but also factive
        "learn", "learned", "learns", "learning",
        "understand", "understood", "understands", "understanding",
        "see", "saw", "sees", "seeing",  # perception verbs
        "hear", "heard", "hears", "hearing",
    }

    CHANGE_OF_STATE_VERBS = {
        "stop", "stopped", "stops", "stopping",
        "start", "started", "starts", "starting",
        "begin", "began", "begins", "beginning",
        "continue", "continued", "continues", "continuing",
        "resume", "resumed", "resumes", "resuming",
        "quit", "quitted", "quits", "quitting",
        "finish", "finished", "finishes", "finishing",
        "end", "ended", "ends", "ending",
    }

    ITERATIVE_MARKERS = {
        "again": 0.95,
        "another": 0.90,
        "still": 0.85,
        "anymore": 0.80,
        "once more": 0.95,
        "yet again": 0.95,
    }

    TEMPORAL_MARKERS = {
        "before": 0.90,
        "after": 0.90,
        "when": 0.85,
        "while": 0.80,
        "since": 0.85,
        "until": 0.85,
    }

    IMPLICATIVE_VERBS = {
        "manage", "managed", "manages", "managing",
        "forget", "forgot", "forgets",  # "forget to X" -> was supposed to X
        "remember", "remembered", "remembers",  # "remember to X" -> was supposed to X
        "bother", "bothered", "bothers",
        "dare", "dared", "dares",
        "happen", "happened", "happens",
        "fail", "failed", "fails", "failing",
    }

    POSSESSIVE_PATTERN = re.compile(
        r"(\w+)'s\s+(\w+(?:\s+\w+)?)",
        re.IGNORECASE
    )

    CLEFT_PATTERNS = [
        re.compile(r"\bIt\s+was\s+(\w+)\s+who\b", re.IGNORECASE),
        re.compile(r"\bIt\s+is\s+(\w+)\s+who\b", re.IGNORECASE),
        re.compile(r"\bIt's\s+(\w+)\s+that\b", re.IGNORECASE),
        re.compile(r"\bWhat\s+(\w+)\s+(\w+)\s+was\b", re.IGNORECASE),
    ]

    COMPARATIVE_PATTERN = re.compile(
        r"\b(\w+er)\s+than\s+(\w+(?:\s+\w+)?)",
        re.IGNORECASE
    )

    COUNTERFACTUAL_PATTERNS = [
        re.compile(r"\bif\s+(\w+)\s+had\b", re.IGNORECASE),
        re.compile(r"\bwish\s+(?:\w+\s+)?had\b", re.IGNORECASE),
        re.compile(r"\bwould\s+have\s+(\w+)\s+if\b", re.IGNORECASE),
    ]

    def __init__(self, config: PresuppositionConfig):
        """Initialize the trigger detector.

        Args:
            config: Configuration for presupposition extraction
        """
        self.config = config

    def detect_triggers(self, text: str) -> list[DetectedTrigger]:
        """Detect all presupposition triggers in text.

        Args:
            text: The text to analyze

        Returns:
            List of detected triggers with their positions
        """
        triggers = []
        enabled = set(self.config.triggers_enabled)

        # Definite descriptions
        if "definite_description" in enabled:
            triggers.extend(self._detect_definite_descriptions(text))

        # Factive verbs
        if "factive_verb" in enabled:
            triggers.extend(self._detect_factive_verbs(text))

        # Change of state
        if "change_of_state" in enabled:
            triggers.extend(self._detect_change_of_state(text))

        # Iteratives
        if "iterative" in enabled:
            triggers.extend(self._detect_iteratives(text))

        # Temporal clauses
        if "temporal_clause" in enabled:
            triggers.extend(self._detect_temporal_clauses(text))

        # Possessives
        if "possessive" in enabled:
            triggers.extend(self._detect_possessives(text))

        # Implicatives
        if "implicative" in enabled:
            triggers.extend(self._detect_implicatives(text))

        # Comparatives
        if "comparative" in enabled:
            triggers.extend(self._detect_comparatives(text))

        # Clefts
        if "cleft" in enabled:
            triggers.extend(self._detect_clefts(text))

        # Counterfactuals
        if "counterfactual" in enabled:
            triggers.extend(self._detect_counterfactuals(text))

        # Filter by confidence if not including weak triggers
        if not self.config.include_weak_triggers:
            triggers = [t for t in triggers if t.confidence >= 0.7]

        return triggers

    def _detect_definite_descriptions(self, text: str) -> list[DetectedTrigger]:
        """Detect 'the X' patterns."""
        triggers = []
        for match in self.DEFINITE_DESC_PATTERN.finditer(text):
            # Skip common phrases that aren't presuppositional
            noun_phrase = match.group(2).lower()
            skip_phrases = {"way", "same", "fact", "thing", "one", "other"}
            if noun_phrase in skip_phrases:
                continue

            triggers.append(DetectedTrigger(
                trigger_type=PresuppositionTrigger.DEFINITE_DESC,
                trigger_text=match.group(0),
                span=(match.start(), match.end()),
                context=text,
                confidence=0.85,
            ))
        return triggers

    def _detect_factive_verbs(self, text: str) -> list[DetectedTrigger]:
        """Detect factive verbs like 'realize', 'know'."""
        triggers = []
        text_lower = text.lower()
        words = text_lower.split()

        for i, word in enumerate(words):
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.FACTIVE_VERBS:
                # Find position in original text
                pattern = re.compile(r'\b' + re.escape(clean_word) + r'\b', re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    # Get context (rest of sentence after verb)
                    context_start = match.end()
                    context = text[context_start:].strip()

                    triggers.append(DetectedTrigger(
                        trigger_type=PresuppositionTrigger.FACTIVE_VERB,
                        trigger_text=match.group(0),
                        span=(match.start(), match.end()),
                        context=context,
                        confidence=0.90,
                    ))
        return triggers

    def _detect_change_of_state(self, text: str) -> list[DetectedTrigger]:
        """Detect change-of-state verbs like 'stop', 'start'."""
        triggers = []
        text_lower = text.lower()

        for verb in self.CHANGE_OF_STATE_VERBS:
            pattern = re.compile(r'\b' + re.escape(verb) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                # Get what follows the verb
                context_start = match.end()
                context = text[context_start:].strip()

                triggers.append(DetectedTrigger(
                    trigger_type=PresuppositionTrigger.CHANGE_OF_STATE,
                    trigger_text=match.group(0),
                    span=(match.start(), match.end()),
                    context=context,
                    confidence=0.90,
                ))
        return triggers

    def _detect_iteratives(self, text: str) -> list[DetectedTrigger]:
        """Detect iterative markers like 'again', 'another'."""
        triggers = []
        text_lower = text.lower()

        for marker, confidence in self.ITERATIVE_MARKERS.items():
            pattern = re.compile(r'\b' + re.escape(marker) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                triggers.append(DetectedTrigger(
                    trigger_type=PresuppositionTrigger.ITERATIVE,
                    trigger_text=match.group(0),
                    span=(match.start(), match.end()),
                    context=text,
                    confidence=confidence,
                ))
        return triggers

    def _detect_temporal_clauses(self, text: str) -> list[DetectedTrigger]:
        """Detect temporal clause markers like 'before', 'after'."""
        triggers = []

        for marker, confidence in self.TEMPORAL_MARKERS.items():
            pattern = re.compile(r'\b' + re.escape(marker) + r'\s+(\w+(?:\s+\w+)*)', re.IGNORECASE)
            for match in pattern.finditer(text):
                triggers.append(DetectedTrigger(
                    trigger_type=PresuppositionTrigger.TEMPORAL_CLAUSE,
                    trigger_text=match.group(0),
                    span=(match.start(), match.end()),
                    context=match.group(1) if match.lastindex else text,
                    confidence=confidence,
                ))
        return triggers

    def _detect_possessives(self, text: str) -> list[DetectedTrigger]:
        """Detect possessive constructions like 'X's Y'."""
        triggers = []
        for match in self.POSSESSIVE_PATTERN.finditer(text):
            possessor = match.group(1)
            possessed = match.group(2)

            # Skip common non-presuppositional uses
            if possessor.lower() in {"it", "that", "this"}:
                continue

            triggers.append(DetectedTrigger(
                trigger_type=PresuppositionTrigger.POSSESSIVE,
                trigger_text=match.group(0),
                span=(match.start(), match.end()),
                context=f"{possessor} has {possessed}",
                confidence=0.90,
            ))
        return triggers

    def _detect_implicatives(self, text: str) -> list[DetectedTrigger]:
        """Detect implicative verbs like 'manage', 'forget to'."""
        triggers = []

        for verb in self.IMPLICATIVE_VERBS:
            # Pattern for "verb to X" constructions
            pattern = re.compile(
                r'\b' + re.escape(verb) + r'\s+to\s+(\w+(?:\s+\w+)*)',
                re.IGNORECASE
            )
            for match in pattern.finditer(text):
                triggers.append(DetectedTrigger(
                    trigger_type=PresuppositionTrigger.IMPLICATIVE,
                    trigger_text=match.group(0),
                    span=(match.start(), match.end()),
                    context=match.group(1) if match.lastindex else text,
                    confidence=0.85,
                ))
        return triggers

    def _detect_comparatives(self, text: str) -> list[DetectedTrigger]:
        """Detect comparative constructions like 'taller than X'."""
        triggers = []
        for match in self.COMPARATIVE_PATTERN.finditer(text):
            triggers.append(DetectedTrigger(
                trigger_type=PresuppositionTrigger.COMPARATIVE,
                trigger_text=match.group(0),
                span=(match.start(), match.end()),
                context=match.group(2) if match.lastindex >= 2 else text,
                confidence=0.85,
            ))

        # Also detect "more X than Y" and "as X as Y"
        more_pattern = re.compile(
            r'\bmore\s+(\w+)\s+than\s+(\w+(?:\s+\w+)?)',
            re.IGNORECASE
        )
        for match in more_pattern.finditer(text):
            triggers.append(DetectedTrigger(
                trigger_type=PresuppositionTrigger.COMPARATIVE,
                trigger_text=match.group(0),
                span=(match.start(), match.end()),
                context=f"{match.group(2)} has {match.group(1)}",
                confidence=0.85,
            ))

        as_pattern = re.compile(
            r'\bas\s+(\w+)\s+as\s+(\w+(?:\s+\w+)?)',
            re.IGNORECASE
        )
        for match in as_pattern.finditer(text):
            triggers.append(DetectedTrigger(
                trigger_type=PresuppositionTrigger.COMPARATIVE,
                trigger_text=match.group(0),
                span=(match.start(), match.end()),
                context=f"{match.group(2)} has {match.group(1)}",
                confidence=0.80,
            ))

        return triggers

    def _detect_clefts(self, text: str) -> list[DetectedTrigger]:
        """Detect cleft sentences like 'It was X who...'."""
        triggers = []
        for pattern in self.CLEFT_PATTERNS:
            for match in pattern.finditer(text):
                triggers.append(DetectedTrigger(
                    trigger_type=PresuppositionTrigger.CLEFT,
                    trigger_text=match.group(0),
                    span=(match.start(), match.end()),
                    context=text,
                    confidence=0.85,
                ))
        return triggers

    def _detect_counterfactuals(self, text: str) -> list[DetectedTrigger]:
        """Detect counterfactual constructions like 'if X had...'."""
        triggers = []
        for pattern in self.COUNTERFACTUAL_PATTERNS:
            for match in pattern.finditer(text):
                triggers.append(DetectedTrigger(
                    trigger_type=PresuppositionTrigger.COUNTERFACTUAL,
                    trigger_text=match.group(0),
                    span=(match.start(), match.end()),
                    context=text,
                    confidence=0.90,
                ))
        return triggers


# =============================================================================
# Content Generation
# =============================================================================


class ContentGenerator:
    """Generates presupposition content from triggers.

    Can use templates for simple cases or LLM for complex cases.
    """

    # Templates for generating presupposition content
    TEMPLATES = {
        PresuppositionTrigger.DEFINITE_DESC: "A specific {noun_phrase} exists that is contextually identifiable",
        PresuppositionTrigger.FACTIVE_VERB: "{complement} is true",
        PresuppositionTrigger.CHANGE_OF_STATE: "The subject was previously {prior_state}",
        PresuppositionTrigger.ITERATIVE: "This happened before",
        PresuppositionTrigger.TEMPORAL_CLAUSE: "{event} occurred",
        PresuppositionTrigger.POSSESSIVE: "{possessor} has {possessed}",
        PresuppositionTrigger.IMPLICATIVE: "The subject was supposed to {action}",
        PresuppositionTrigger.COMPARATIVE: "{compared_to} has the property being compared",
        PresuppositionTrigger.CLEFT: "Someone performed the action",
        PresuppositionTrigger.COUNTERFACTUAL: "The condition did not actually occur",
    }

    def __init__(self, config: PresuppositionConfig, llm: LLMProvider | None = None):
        """Initialize the content generator.

        Args:
            config: Configuration for presupposition extraction
            llm: Optional LLM provider for complex cases
        """
        self.config = config
        self.llm = llm

    async def generate_content(
        self,
        trigger: DetectedTrigger,
        full_text: str,
    ) -> str:
        """Generate presupposition content for a trigger.

        Args:
            trigger: The detected trigger
            full_text: The full source text

        Returns:
            The presupposed content as text
        """
        if self.config.use_llm_for_content and self.llm:
            return await self._generate_with_llm(trigger, full_text)
        else:
            return self._generate_with_template(trigger, full_text)

    def _generate_with_template(
        self,
        trigger: DetectedTrigger,
        full_text: str,
    ) -> str:
        """Generate content using templates."""
        template = self.TEMPLATES.get(
            trigger.trigger_type,
            "Something is presupposed"
        )

        # Extract variables for template
        variables = self._extract_template_variables(trigger, full_text)
        try:
            return template.format(**variables)
        except KeyError:
            return template

    def _extract_template_variables(
        self,
        trigger: DetectedTrigger,
        full_text: str,
    ) -> dict[str, str]:
        """Extract variables needed for template filling."""
        variables: dict[str, str] = {}

        if trigger.trigger_type == PresuppositionTrigger.DEFINITE_DESC:
            # Extract noun phrase from "the X"
            match = re.search(r'\bthe\s+(\w+(?:\s+\w+)?)', trigger.trigger_text, re.IGNORECASE)
            if match:
                variables["noun_phrase"] = match.group(1)

        elif trigger.trigger_type == PresuppositionTrigger.FACTIVE_VERB:
            # The complement is the context
            variables["complement"] = trigger.context

        elif trigger.trigger_type == PresuppositionTrigger.CHANGE_OF_STATE:
            # Infer prior state from verb
            verb = trigger.trigger_text.lower()
            if "stop" in verb:
                variables["prior_state"] = f"doing {trigger.context}"
            elif "start" in verb or "begin" in verb:
                variables["prior_state"] = f"not doing {trigger.context}"
            elif "continue" in verb:
                variables["prior_state"] = f"doing {trigger.context}"
            else:
                variables["prior_state"] = trigger.context

        elif trigger.trigger_type == PresuppositionTrigger.POSSESSIVE:
            match = re.search(r"(\w+)'s\s+(\w+)", trigger.trigger_text)
            if match:
                variables["possessor"] = match.group(1)
                variables["possessed"] = match.group(2)

        elif trigger.trigger_type == PresuppositionTrigger.TEMPORAL_CLAUSE:
            variables["event"] = trigger.context

        elif trigger.trigger_type == PresuppositionTrigger.IMPLICATIVE:
            variables["action"] = trigger.context

        elif trigger.trigger_type == PresuppositionTrigger.COMPARATIVE:
            variables["compared_to"] = trigger.context

        return variables

    async def _generate_with_llm(
        self,
        trigger: DetectedTrigger,
        full_text: str,
    ) -> str:
        """Generate content using LLM for more accurate results."""
        if not self.llm:
            return self._generate_with_template(trigger, full_text)

        prompt = f"""Generate the presupposition content for this trigger.

Sentence: "{full_text}"
Trigger type: {trigger.trigger_type.value}
Trigger text: "{trigger.trigger_text}"

A presupposition is what must be true for the sentence to make sense.

For example:
- "Doug stopped smoking" presupposes "Doug used to smoke"
- "The meeting was canceled" presupposes "A specific meeting exists"
- "She won another award" presupposes "She won an award before"

Generate ONLY the presupposition content (what is presupposed), nothing else.
Be specific to this sentence, not generic.

Presupposition:"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
                max_tokens=100,
            )
            content = _extract_llm_content(response).strip()
            # Clean up response
            content = content.replace("Presupposition:", "").strip()
            content = content.strip('"\'')
            return content
        except Exception:
            # Fall back to template
            return self._generate_with_template(trigger, full_text)


# =============================================================================
# Main Extractor
# =============================================================================


class PresuppositionExtractor:
    """Main presupposition extraction class.

    Combines trigger detection and content generation to extract
    complete presuppositions from text.

    Example:
        >>> config = PresuppositionConfig()
        >>> extractor = PresuppositionExtractor(config, llm=my_llm)
        >>> presups = await extractor.extract("Doug forgot the meeting again")
    """

    def __init__(
        self,
        config: PresuppositionConfig | None = None,
        llm: LLMProvider | None = None,
    ):
        """Initialize the presupposition extractor.

        Args:
            config: Configuration (uses defaults if not provided)
            llm: Optional LLM provider for content generation
        """
        self.config = config or PresuppositionConfig()
        self.llm = llm
        self.detector = TriggerDetector(self.config)
        self.generator = ContentGenerator(self.config, llm)

        # Metrics
        self.metrics = {
            "total_extractions": 0,
            "triggers_detected": 0,
            "llm_calls": 0,
        }

    async def extract(
        self,
        text: str,
        entity_ids: list[str] | None = None,
    ) -> list[Presupposition]:
        """Extract presuppositions from text.

        Args:
            text: The text to analyze
            entity_ids: Optional list of entity IDs in the text

        Returns:
            List of extracted presuppositions
        """
        self.metrics["total_extractions"] += 1

        # Detect triggers
        triggers = self.detector.detect_triggers(text)
        self.metrics["triggers_detected"] += len(triggers)

        if not triggers:
            return []

        # Generate content for each trigger
        presuppositions = []
        for trigger in triggers:
            content = await self.generator.generate_content(trigger, text)

            presup = Presupposition(
                content=content,
                trigger_type=trigger.trigger_type,
                trigger_text=trigger.trigger_text,
                trigger_span=trigger.span,
                confidence=trigger.confidence,
                cancellable=self._is_cancellable(trigger.trigger_type),
                entity_ids=entity_ids or [],
            )
            presuppositions.append(presup)

        # Apply confidence threshold
        presuppositions = [
            p for p in presuppositions
            if p.confidence >= self.config.confidence_threshold
        ]

        # Limit count
        presuppositions = presuppositions[:self.config.max_presuppositions]

        return presuppositions

    def extract_sync(
        self,
        text: str,
        entity_ids: list[str] | None = None,
    ) -> list[Presupposition]:
        """Synchronous extraction using only templates (no LLM).

        Args:
            text: The text to analyze
            entity_ids: Optional list of entity IDs

        Returns:
            List of extracted presuppositions
        """
        self.metrics["total_extractions"] += 1

        # Detect triggers
        triggers = self.detector.detect_triggers(text)
        self.metrics["triggers_detected"] += len(triggers)

        if not triggers:
            return []

        # Generate content using templates only
        presuppositions = []
        for trigger in triggers:
            content = self.generator._generate_with_template(trigger, text)

            presup = Presupposition(
                content=content,
                trigger_type=trigger.trigger_type,
                trigger_text=trigger.trigger_text,
                trigger_span=trigger.span,
                confidence=trigger.confidence,
                cancellable=self._is_cancellable(trigger.trigger_type),
                entity_ids=entity_ids or [],
            )
            presuppositions.append(presup)

        # Apply confidence threshold
        presuppositions = [
            p for p in presuppositions
            if p.confidence >= self.config.confidence_threshold
        ]

        # Limit count
        presuppositions = presuppositions[:self.config.max_presuppositions]

        return presuppositions

    def _is_cancellable(self, trigger_type: PresuppositionTrigger) -> bool:
        """Determine if a presupposition type is typically cancellable.

        Some presuppositions can be cancelled in context:
        "Doug stopped smoking, or rather he never started."

        Returns:
            True if the presupposition type is typically cancellable
        """
        # Most presuppositions are cancellable except for logical/semantic ones
        non_cancellable = {
            PresuppositionTrigger.COUNTERFACTUAL,  # Counterfactuals are definitional
        }
        return trigger_type not in non_cancellable

    def get_metrics(self) -> dict[str, int]:
        """Get extraction metrics."""
        return dict(self.metrics)

    def reset_metrics(self) -> None:
        """Reset metrics to zero."""
        for key in self.metrics:
            self.metrics[key] = 0


# =============================================================================
# Convenience Functions
# =============================================================================


def detect_triggers(text: str) -> list[tuple[str, PresuppositionTrigger, tuple[int, int]]]:
    """Convenience function to detect triggers without creating extractor.

    Args:
        text: Text to analyze

    Returns:
        List of (trigger_text, trigger_type, span) tuples
    """
    config = PresuppositionConfig()
    detector = TriggerDetector(config)
    triggers = detector.detect_triggers(text)
    return [(t.trigger_text, t.trigger_type, t.span) for t in triggers]

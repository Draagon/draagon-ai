"""Commonsense inference generation using ATOMIC-style relations.

This module generates commonsense inferences about events described in text,
based on the ATOMIC (and ATOMIC 2020) commonsense knowledge framework.

ATOMIC Relations:
- **xIntent**: Why might X do this? (motivation)
- **xNeed**: What does X need before this? (precondition)
- **xAttr**: How would X be described? (attributes)
- **xEffect**: What happens to X after? (effect on actor)
- **xWant**: What does X want after? (desire)
- **xReact**: How does X feel after? (emotional reaction)
- **oEffect**: What happens to others? (effect on recipients)
- **oReact**: How do others feel? (others' reactions)
- **oWant**: What do others want after?

This is crucial for knowledge extraction because it captures implicit
knowledge that humans understand but isn't explicitly stated:
- "Doug forgot the meeting" implies:
  - xEffect: Doug may miss important information
  - xReact: Doug feels embarrassed or guilty
  - oReact: Others feel frustrated or concerned

Example:
    >>> from decomposition.commonsense import CommonsenseExtractor
    >>> extractor = CommonsenseExtractor()
    >>> inferences = await extractor.extract("Doug forgot the meeting")
    >>> for inf in inferences:
    ...     print(f"{inf.relation}: {inf.content}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from decomposition.models import CommonsenseInference, CommonsenseRelation
from decomposition.config import CommonsenseConfig


# =============================================================================
# Relation Definitions
# =============================================================================


@dataclass
class RelationDefinition:
    """Definition of an ATOMIC relation."""

    relation: CommonsenseRelation
    question: str
    """Question this relation answers."""

    example_prompt: str
    """Example to guide generation."""

    typical_patterns: list[str] = field(default_factory=list)
    """Common patterns for this relation."""


RELATION_DEFINITIONS = {
    CommonsenseRelation.X_INTENT: RelationDefinition(
        relation=CommonsenseRelation.X_INTENT,
        question="Why would PersonX do this?",
        example_prompt="PersonX wanted to achieve..., PersonX was trying to...",
        typical_patterns=[
            "to {verb}",
            "because {subject} wanted",
            "in order to",
        ],
    ),
    CommonsenseRelation.X_NEED: RelationDefinition(
        relation=CommonsenseRelation.X_NEED,
        question="What did PersonX need before this?",
        example_prompt="PersonX needed to..., PersonX had to first...",
        typical_patterns=[
            "{subject} needed",
            "required",
            "had to",
        ],
    ),
    CommonsenseRelation.X_ATTR: RelationDefinition(
        relation=CommonsenseRelation.X_ATTR,
        question="How would PersonX be described?",
        example_prompt="PersonX is described as..., PersonX is the type of person who...",
        typical_patterns=[
            "is {adjective}",
            "tends to be",
            "often",
        ],
    ),
    CommonsenseRelation.X_EFFECT: RelationDefinition(
        relation=CommonsenseRelation.X_EFFECT,
        question="What happens to PersonX as a result?",
        example_prompt="PersonX will..., PersonX ends up..., As a result PersonX...",
        typical_patterns=[
            "will {verb}",
            "ends up",
            "as a result",
        ],
    ),
    CommonsenseRelation.X_WANT: RelationDefinition(
        relation=CommonsenseRelation.X_WANT,
        question="What does PersonX want after this?",
        example_prompt="PersonX wants to..., PersonX hopes to...",
        typical_patterns=[
            "wants to",
            "hopes to",
            "would like to",
        ],
    ),
    CommonsenseRelation.X_REACT: RelationDefinition(
        relation=CommonsenseRelation.X_REACT,
        question="How does PersonX feel as a result?",
        example_prompt="PersonX feels..., PersonX is...",
        typical_patterns=[
            "feels {emotion}",
            "is {emotional_state}",
        ],
    ),
    CommonsenseRelation.O_EFFECT: RelationDefinition(
        relation=CommonsenseRelation.O_EFFECT,
        question="What happens to others as a result?",
        example_prompt="Others will..., The other person ends up...",
        typical_patterns=[
            "others will",
            "the recipient",
        ],
    ),
    CommonsenseRelation.O_REACT: RelationDefinition(
        relation=CommonsenseRelation.O_REACT,
        question="How do others feel as a result?",
        example_prompt="Others feel..., The other person is...",
        typical_patterns=[
            "others feel",
            "makes them feel",
        ],
    ),
    CommonsenseRelation.O_WANT: RelationDefinition(
        relation=CommonsenseRelation.O_WANT,
        question="What do others want after this?",
        example_prompt="Others want to..., The other person hopes to...",
        typical_patterns=[
            "others want",
            "they hope to",
        ],
    ),
    CommonsenseRelation.CAUSES: RelationDefinition(
        relation=CommonsenseRelation.CAUSES,
        question="What does this cause to happen?",
        example_prompt="This causes..., This leads to..., This results in...",
        typical_patterns=[
            "causes",
            "leads to",
            "results in",
        ],
    ),
}


# =============================================================================
# Event Extractor
# =============================================================================


@dataclass
class ExtractedEvent:
    """An event extracted from text for inference."""

    text: str
    """The event description."""

    subject: str | None = None
    """The agent/subject of the event."""

    verb: str | None = None
    """The main verb."""

    object: str | None = None
    """The patient/object."""

    span: tuple[int, int] | None = None
    """Character span in source."""


class EventExtractor:
    """Extracts events from text for commonsense inference.

    Identifies the main event and its participants.
    """

    # Common verbs that take sentential complements
    COMPLEMENT_VERBS = {
        "say", "said", "think", "thought", "believe", "believed",
        "know", "knew", "realize", "realized", "discover", "discovered",
        "hope", "hoped", "wish", "wished", "want", "wanted",
        "decide", "decided", "try", "tried", "forget", "forgot",
    }

    def extract(self, text: str) -> list[ExtractedEvent]:
        """Extract events from text.

        Args:
            text: Input text

        Returns:
            List of extracted events
        """
        events = []

        # Simple extraction based on sentence structure
        # In production, would use dependency parsing

        # Try to identify subject-verb-object pattern
        # Pattern: [Subject] [Verb] [Object/Complement]
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            event = self._extract_single_event(sentence)
            if event:
                events.append(event)

        return events if events else [ExtractedEvent(text=text)]

    def _extract_single_event(self, text: str) -> ExtractedEvent | None:
        """Extract a single event from a clause."""
        # Simple heuristic pattern matching
        # Pattern: Subject (noun phrase) + Verb + Rest

        words = text.split()
        if len(words) < 2:
            return None

        # Find first verb-like word (very simplified)
        verb_idx = None
        verb = None
        for i, word in enumerate(words):
            word_lower = word.lower().rstrip('.,!?')
            if self._is_verb_like(word_lower):
                verb_idx = i
                verb = word_lower
                break

        if verb_idx is None:
            return ExtractedEvent(text=text)

        # Subject is before verb
        subject = " ".join(words[:verb_idx]) if verb_idx > 0 else None

        # Object/complement is after verb
        obj = " ".join(words[verb_idx + 1:]) if verb_idx < len(words) - 1 else None

        return ExtractedEvent(
            text=text,
            subject=subject,
            verb=verb,
            object=obj,
        )

    def _is_verb_like(self, word: str) -> bool:
        """Check if word looks like a verb (simplified)."""
        # Past tense
        if word.endswith("ed") and len(word) > 3:
            return True
        # Present participle
        if word.endswith("ing") and len(word) > 4:
            return True
        # Third person singular
        if word.endswith("s") and not word.endswith("ss"):
            # But not nouns - this is very rough
            return True
        # Common irregular verbs
        irregular = {
            "is", "are", "was", "were", "be", "been", "being",
            "has", "have", "had", "do", "does", "did",
            "go", "goes", "went", "gone", "going",
            "get", "gets", "got", "gotten", "getting",
            "say", "says", "said", "make", "makes", "made",
            "know", "knows", "knew", "known", "see", "sees", "saw", "seen",
            "come", "comes", "came", "take", "takes", "took", "taken",
            "think", "thinks", "thought", "find", "finds", "found",
            "give", "gives", "gave", "given", "tell", "tells", "told",
            "become", "becomes", "became", "leave", "leaves", "left",
            "put", "puts", "keep", "keeps", "kept", "let", "lets",
            "begin", "begins", "began", "begun", "seem", "seems", "seemed",
            "help", "helps", "helped", "show", "shows", "showed", "shown",
            "hear", "hears", "heard", "play", "plays", "played",
            "run", "runs", "ran", "move", "moves", "moved",
            "live", "lives", "lived", "believe", "believes", "believed",
            "bring", "brings", "brought", "happen", "happens", "happened",
            "write", "writes", "wrote", "written", "provide", "provides", "provided",
            "sit", "sits", "sat", "stand", "stands", "stood",
            "lose", "loses", "lost", "pay", "pays", "paid",
            "meet", "meets", "met", "include", "includes", "included",
            "continue", "continues", "continued", "set", "sets",
            "learn", "learns", "learned", "change", "changes", "changed",
            "lead", "leads", "led", "understand", "understands", "understood",
            "watch", "watches", "watched", "follow", "follows", "followed",
            "stop", "stops", "stopped", "create", "creates", "created",
            "speak", "speaks", "spoke", "spoken", "read", "reads",
            "spend", "spends", "spent", "grow", "grows", "grew", "grown",
            "open", "opens", "opened", "walk", "walks", "walked",
            "win", "wins", "won", "offer", "offers", "offered",
            "remember", "remembers", "remembered", "love", "loves", "loved",
            "consider", "considers", "considered", "appear", "appears", "appeared",
            "buy", "buys", "bought", "wait", "waits", "waited",
            "serve", "serves", "served", "die", "dies", "died",
            "send", "sends", "sent", "expect", "expects", "expected",
            "build", "builds", "built", "stay", "stays", "stayed",
            "fall", "falls", "fell", "fallen", "cut", "cuts",
            "reach", "reaches", "reached", "kill", "kills", "killed",
            "remain", "remains", "remained", "suggest", "suggests", "suggested",
            "raise", "raises", "raised", "pass", "passes", "passed",
            "sell", "sells", "sold", "require", "requires", "required",
            "report", "reports", "reported", "decide", "decides", "decided",
            "pull", "pulls", "pulled", "forgot", "forget", "forgets",
        }
        return word in irregular


# =============================================================================
# Template-based Generator
# =============================================================================


class TemplateGenerator:
    """Generates commonsense inferences using templates.

    Provides reasonable default inferences without requiring LLM.
    Less accurate but faster and always available.
    """

    # Templates for each relation type
    TEMPLATES: dict[CommonsenseRelation, list[str]] = {
        CommonsenseRelation.X_INTENT: [
            "{subject} wanted to accomplish something",
            "{subject} had a specific goal in mind",
            "{subject} intended to {verb_related}",
        ],
        CommonsenseRelation.X_NEED: [
            "{subject} needed to know how to {verb}",
            "{subject} had to prepare for this",
            "{subject} required the necessary resources",
        ],
        CommonsenseRelation.X_ATTR: [
            "{subject} is someone who {verb}s",
            "{subject} is active in this area",
            "{subject} has experience with this",
        ],
        CommonsenseRelation.X_EFFECT: [
            "{subject} is affected by this action",
            "{subject} may experience consequences",
            "This changes {subject}'s situation",
        ],
        CommonsenseRelation.X_WANT: [
            "{subject} wants to see the results",
            "{subject} hopes for a positive outcome",
            "{subject} desires to continue",
        ],
        CommonsenseRelation.X_REACT: [
            "{subject} feels something about this",
            "{subject} has an emotional response",
            "{subject} experiences the impact",
        ],
        CommonsenseRelation.O_EFFECT: [
            "Others are affected by this",
            "This impacts other people involved",
            "Recipients experience changes",
        ],
        CommonsenseRelation.O_REACT: [
            "Others have feelings about this",
            "People react to this action",
            "This affects others emotionally",
        ],
        CommonsenseRelation.O_WANT: [
            "Others want something from this",
            "People hope for certain outcomes",
            "Others desire specific results",
        ],
        CommonsenseRelation.CAUSES: [
            "This leads to further events",
            "This triggers consequences",
            "This results in changes",
        ],
    }

    # Event-specific templates for common verbs
    # Based on ATOMIC 2020 commonsense categories:
    # - Social-Interaction: emotional/social consequences
    # - Event-Centered: prerequisites, effects, intents
    VERB_SPECIFIC: dict[str, dict[CommonsenseRelation, list[str]]] = {
        # Memory/cognitive verbs
        "forgot": {
            CommonsenseRelation.X_INTENT: [
                "{subject} did not intend to forget",
                "{subject} was distracted by something else",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels embarrassed",
                "{subject} feels guilty",
                "{subject} is worried",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel disappointed",
                "Others feel frustrated",
                "Others feel annoyed",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} may miss important information",
                "{subject} needs to apologize",
                "{subject} has to deal with consequences",
            ],
        },
        "remember": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels relieved",
                "{subject} feels pleased",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} can act on the information",
                "{subject} is prepared",
            ],
        },
        # Achievement verbs
        "won": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to succeed",
                "{subject} aimed to be the best",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels excited",
                "{subject} feels proud",
                "{subject} feels happy",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel impressed",
                "Others feel envious",
                "Others feel happy for {subject}",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} celebrates",
                "{subject} gains recognition",
                "{subject} receives a prize",
            ],
        },
        "lost": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels disappointed",
                "{subject} feels sad",
                "{subject} feels frustrated",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel sympathetic",
                "Others try to comfort {subject}",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} may try again",
                "{subject} reflects on what happened",
            ],
        },
        "failed": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels disappointed",
                "{subject} feels frustrated",
                "{subject} feels upset",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel concerned",
                "Others may offer support",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} needs to try again",
                "{subject} learns from the experience",
            ],
        },
        # Social verbs
        "help": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to be helpful",
                "{subject} wanted to be kind",
                "{subject} wanted to make things easier",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels good",
                "{subject} feels satisfied",
                "{subject} feels fulfilled",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel grateful",
                "Others feel appreciative",
                "Others feel relieved",
            ],
        },
        "thanked": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to show appreciation",
                "{subject} wanted to be polite",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels satisfied",
                "{subject} feels good about expressing gratitude",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel appreciated",
                "Others feel valued",
            ],
        },
        # Physical/consumption verbs
        "ate": {
            CommonsenseRelation.X_INTENT: [
                "{subject} was hungry",
                "{subject} wanted nourishment",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels satisfied",
                "{subject} feels full",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} is no longer hungry",
                "{subject} has energy",
            ],
        },
        "drank": {
            CommonsenseRelation.X_INTENT: [
                "{subject} was thirsty",
                "{subject} wanted refreshment",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels refreshed",
                "{subject} feels satisfied",
            ],
        },
        # Achievement through effort
        "studied": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to learn",
                "{subject} wanted to pass an exam",
                "{subject} wanted to do well",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels prepared",
                "{subject} feels accomplished",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} gains knowledge",
                "{subject} is better prepared",
            ],
        },
        "promoted": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels proud",
                "{subject} feels excited",
                "{subject} feels accomplished",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel happy for {subject}",
                "Others feel proud of {subject}",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} earns more money",
                "{subject} has new responsibilities",
                "{subject} celebrates",
            ],
        },
        # Negative events
        "hurt": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels pain",
                "{subject} feels upset",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel concerned",
                "Others want to help",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} needs medical attention",
                "{subject} is careful",
            ],
        },
        "broke": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels frustrated",
                "{subject} feels annoyed",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} needs to fix or replace something",
                "{subject} deals with the damage",
            ],
        },
        # Communication verbs
        "told": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to inform",
                "{subject} wanted to communicate",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel informed",
                "Others understand better",
            ],
        },
        "asked": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to know something",
                "{subject} wanted clarification",
            ],
            CommonsenseRelation.X_REACT: [
                "{subject} feels curious",
                "{subject} awaits a response",
            ],
        },
        # Movement/departure verbs
        "left": {
            CommonsenseRelation.X_INTENT: [
                "{subject} needed to go elsewhere",
                "{subject} wanted to depart",
            ],
            CommonsenseRelation.X_EFFECT: [
                "{subject} is no longer present",
                "{subject} travels somewhere",
            ],
            CommonsenseRelation.O_REACT: [
                "Others notice the absence",
                "Others may miss {subject}",
            ],
        },
        "arrived": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels relieved to have arrived",
                "{subject} is ready to begin",
            ],
            CommonsenseRelation.O_REACT: [
                "Others notice {subject}",
                "Others may greet {subject}",
            ],
        },
        # Emotional verbs
        "cried": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels sad",
                "{subject} feels emotional",
                "{subject} feels overwhelmed",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel concerned",
                "Others want to comfort {subject}",
            ],
        },
        "laughed": {
            CommonsenseRelation.X_REACT: [
                "{subject} feels happy",
                "{subject} feels amused",
                "{subject} feels joyful",
            ],
            CommonsenseRelation.O_REACT: [
                "Others feel happy too",
                "Others enjoy the moment",
            ],
        },
        # Control/cessation verbs
        "stop": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to end something",
                "{subject} decided to cease the activity",
            ],
            CommonsenseRelation.X_EFFECT: [
                "The activity ends",
                "{subject} has changed behavior",
            ],
        },
        "started": {
            CommonsenseRelation.X_INTENT: [
                "{subject} wanted to begin something new",
                "{subject} was motivated to act",
            ],
            CommonsenseRelation.X_EFFECT: [
                "A new activity is underway",
                "{subject} is now engaged",
            ],
        },
    }

    def generate(
        self,
        event: ExtractedEvent,
        relations: list[CommonsenseRelation],
    ) -> list[CommonsenseInference]:
        """Generate inferences using templates.

        Args:
            event: The extracted event
            relations: Relations to generate

        Returns:
            List of inferences
        """
        inferences = []
        subject = event.subject or "PersonX"
        verb = event.verb or "act"

        # Check for verb-specific templates
        verb_templates = self.VERB_SPECIFIC.get(verb, {})

        for relation in relations:
            # Use verb-specific if available, else general
            templates = verb_templates.get(relation, self.TEMPLATES.get(relation, []))

            if not templates:
                continue

            # Generate from first template (most relevant)
            template = templates[0]
            content = template.format(
                subject=subject,
                verb=verb,
                verb_related=verb,
                object=event.object or "something",
            )

            inferences.append(CommonsenseInference(
                relation=relation,
                head=f"{event.subject} {event.verb}" if event.subject else event.verb,
                tail=content,
                confidence=0.65,  # Lower confidence for templates
                source="template",
            ))

        return inferences


# =============================================================================
# LLM-based Generator
# =============================================================================


class LLMGenerator:
    """Generates commonsense inferences using an LLM.

    More accurate but requires LLM availability.
    """

    def __init__(self, config: CommonsenseConfig, llm: Any):
        """Initialize with config and LLM provider."""
        self.config = config
        self.llm = llm

    async def generate(
        self,
        event: ExtractedEvent,
        relations: list[CommonsenseRelation],
    ) -> list[CommonsenseInference]:
        """Generate inferences using LLM.

        Args:
            event: The extracted event
            relations: Relations to generate

        Returns:
            List of inferences
        """
        # Build relations prompt
        relations_prompt = "\n".join([
            f"- {rel.value}: {RELATION_DEFINITIONS[rel].question}"
            for rel in relations
            if rel in RELATION_DEFINITIONS
        ])

        prompt = self.config.prompt_template.format(
            text=event.text,
            relations_prompt=relations_prompt,
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )
            return self._parse_response(response, event)
        except Exception:
            return []

    def _parse_response(
        self,
        response: str,
        event: ExtractedEvent,
    ) -> list[CommonsenseInference]:
        """Parse LLM response to inferences."""
        inferences = []

        # Parse XML-style inferences
        pattern = r'<inference\s+relation="(\w+)"(?:\s+confidence="([\d.]+)")?\s*>(.*?)</inference>'
        for match in re.finditer(pattern, response, re.IGNORECASE | re.DOTALL):
            relation_str = match.group(1)
            confidence_str = match.group(2)
            content = match.group(3).strip()

            # Map to relation enum
            try:
                relation = CommonsenseRelation(relation_str)
            except ValueError:
                # Try case-insensitive match
                relation = None
                for r in CommonsenseRelation:
                    if r.value.lower() == relation_str.lower():
                        relation = r
                        break
                if relation is None:
                    continue

            # Parse confidence
            try:
                confidence = float(confidence_str) if confidence_str else 0.80
            except ValueError:
                confidence = 0.80

            inferences.append(CommonsenseInference(
                relation=relation,
                head=f"{event.subject} {event.verb}" if event.subject else event.verb,
                tail=content,
                confidence=confidence,
                source="llm",
            ))

        return inferences


# =============================================================================
# Deduplicator
# =============================================================================


class InferenceDeduplicator:
    """Removes duplicate or near-duplicate inferences.

    Uses string similarity to detect redundant inferences.
    """

    def __init__(self, config: CommonsenseConfig):
        """Initialize deduplicator."""
        self.config = config
        self.threshold = config.dedup_similarity_threshold

    def deduplicate(
        self,
        inferences: list[CommonsenseInference],
    ) -> list[CommonsenseInference]:
        """Remove duplicate inferences.

        Args:
            inferences: List of inferences

        Returns:
            Deduplicated list
        """
        if not self.config.deduplicate or len(inferences) <= 1:
            return inferences

        # Group by relation
        by_relation: dict[CommonsenseRelation, list[CommonsenseInference]] = {}
        for inf in inferences:
            if inf.relation not in by_relation:
                by_relation[inf.relation] = []
            by_relation[inf.relation].append(inf)

        # Deduplicate within each relation
        result = []
        for relation, group in by_relation.items():
            deduped = self._dedupe_group(group)
            result.extend(deduped)

        return result

    def _dedupe_group(
        self,
        group: list[CommonsenseInference],
    ) -> list[CommonsenseInference]:
        """Deduplicate a group of same-relation inferences."""
        if len(group) <= 1:
            return group

        result = [group[0]]
        for inf in group[1:]:
            is_duplicate = False
            for kept in result:
                if self._is_similar(inf.tail, kept.tail):
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(inf)

        return result

    def _is_similar(self, a: str, b: str) -> bool:
        """Check if two strings are similar enough to be duplicates."""
        # Simple word overlap similarity
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a or not words_b:
            return False

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        similarity = intersection / union if union > 0 else 0
        return similarity >= self.threshold


# =============================================================================
# Commonsense Extractor
# =============================================================================


class CommonsenseExtractor:
    """Main extractor for commonsense inferences.

    Extracts events and generates commonsense inferences about them.

    Example:
        >>> extractor = CommonsenseExtractor()
        >>> inferences = await extractor.extract("Doug forgot the meeting")
        >>> for inf in inferences:
        ...     print(f"{inf.relation}: {inf.content}")
    """

    def __init__(
        self,
        config: CommonsenseConfig | None = None,
        llm: Any | None = None,
    ):
        """Initialize the commonsense extractor.

        Args:
            config: Extraction configuration
            llm: Optional LLM provider
        """
        self.config = config or CommonsenseConfig()
        self.llm = llm

        self.event_extractor = EventExtractor()
        self.template_generator = TemplateGenerator()
        self.deduplicator = InferenceDeduplicator(self.config)

        if llm:
            self.llm_generator = LLMGenerator(self.config, llm)
        else:
            self.llm_generator = None

    def extract_sync(
        self,
        text: str,
        entity_ids: list[str] | None = None,
        entity_types: dict[str, str] | None = None,
    ) -> list[CommonsenseInference]:
        """Synchronously extract commonsense inferences.

        Uses template-based generation (no LLM).

        Args:
            text: Input text
            entity_ids: Optional entity identifiers
            entity_types: Optional entity type classifications

        Returns:
            List of CommonsenseInference
        """
        # Extract events
        events = self.event_extractor.extract(text)

        # Determine which relations to generate
        relations = self._get_relations(entity_types)

        # Generate inferences for each event
        all_inferences = []
        for event in events:
            inferences = self.template_generator.generate(event, relations)
            all_inferences.extend(inferences)

        # Deduplicate
        all_inferences = self.deduplicator.deduplicate(all_inferences)

        # Limit per relation
        all_inferences = self._limit_per_relation(all_inferences)

        # Filter by confidence
        all_inferences = [
            inf for inf in all_inferences
            if inf.confidence >= self.config.inference_confidence_min
        ]

        return all_inferences

    async def extract(
        self,
        text: str,
        entity_ids: list[str] | None = None,
        entity_types: dict[str, str] | None = None,
    ) -> list[CommonsenseInference]:
        """Asynchronously extract commonsense inferences.

        Uses LLM if available, otherwise falls back to templates.

        Args:
            text: Input text
            entity_ids: Optional entity identifiers
            entity_types: Optional entity type classifications

        Returns:
            List of CommonsenseInference
        """
        # Extract events
        events = self.event_extractor.extract(text)

        # Determine which relations to generate
        relations = self._get_relations(entity_types)

        # Generate inferences
        all_inferences = []
        for event in events:
            if self.llm_generator:
                inferences = await self.llm_generator.generate(event, relations)
                if not inferences:
                    # Fallback to templates
                    inferences = self.template_generator.generate(event, relations)
            else:
                inferences = self.template_generator.generate(event, relations)

            all_inferences.extend(inferences)

        # Deduplicate
        all_inferences = self.deduplicator.deduplicate(all_inferences)

        # Limit per relation
        all_inferences = self._limit_per_relation(all_inferences)

        # Filter by confidence
        all_inferences = [
            inf for inf in all_inferences
            if inf.confidence >= self.config.inference_confidence_min
        ]

        return all_inferences

    def _get_relations(
        self,
        entity_types: dict[str, str] | None,
    ) -> list[CommonsenseRelation]:
        """Determine which relations to generate based on context.

        Uses tiered approach from config. Tier 2 relations are included
        when we have INSTANCE entities (specific people/things) because
        they benefit from richer inference (xNeed, xWant, oReact, Causes).

        For CLASS entities (generic categories), tier 1 relations
        (xIntent, xEffect, xReact, xAttr) are usually sufficient.

        Args:
            entity_types: Dict mapping entity text -> entity type value
                         (e.g., {"Doug": "instance", "person": "class"})

        Returns:
            List of CommonsenseRelation to generate
        """
        relations = []

        # Always include tier 1 (core relations)
        for rel_name in self.config.tier1_relations:
            try:
                relations.append(CommonsenseRelation(rel_name))
            except ValueError:
                pass

        # Include tier 2 if we have INSTANCE entities
        # INSTANCE entities (specific people/things) benefit from richer inference
        # CLASS entities (generic categories) don't need xNeed, xWant, etc.
        include_tier2 = False

        if entity_types:
            # Check if any entity is an INSTANCE
            for entity_text, entity_type in entity_types.items():
                if entity_type.lower() == "instance":
                    include_tier2 = True
                    break
        else:
            # No entity types provided - default to including tier 2
            # to maintain backward compatibility
            include_tier2 = True

        if include_tier2:
            for rel_name in self.config.tier2_relations:
                try:
                    relations.append(CommonsenseRelation(rel_name))
                except ValueError:
                    pass

        return relations

    def _limit_per_relation(
        self,
        inferences: list[CommonsenseInference],
    ) -> list[CommonsenseInference]:
        """Limit inferences per relation type."""
        by_relation: dict[CommonsenseRelation, list[CommonsenseInference]] = {}
        for inf in inferences:
            if inf.relation not in by_relation:
                by_relation[inf.relation] = []
            by_relation[inf.relation].append(inf)

        result = []
        for relation, group in by_relation.items():
            # Sort by confidence and take top N
            group.sort(key=lambda x: x.confidence, reverse=True)
            result.extend(group[:self.config.max_inferences_per_relation])

        return result


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_inferences(
    text: str,
    relations: list[str] | None = None,
) -> list[CommonsenseInference]:
    """Quick commonsense inference generation.

    Args:
        text: Input text
        relations: Optional list of relation names to generate

    Returns:
        List of inferences
    """
    config = CommonsenseConfig()
    if relations:
        config.tier1_relations = relations
        config.tier2_relations = []

    extractor = CommonsenseExtractor(config=config)
    return extractor.extract_sync(text)

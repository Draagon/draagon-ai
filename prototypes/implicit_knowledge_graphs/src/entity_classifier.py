"""Entity Type Classification System.

Classifies text mentions into entity types:
- INSTANCE: Specific real-world entities (Doug, Apple Inc.)
- CLASS: Categories/types (person, company)
- NAMED_CONCEPT: Proper-named categories (Christmas, Agile)
- ROLE: Relational concepts (CEO of Apple)
- ANAPHORA: References needing resolution (he, it)
- GENERIC: Generic references (someone, everyone)

Uses a hybrid approach:
1. Heuristic rules for common cases (fast)
2. LLM fallback for uncertain cases (accurate)

Example:
    >>> from entity_classifier import EntityClassifier, ClassifierConfig
    >>>
    >>> classifier = EntityClassifier()
    >>> result = await classifier.classify("Apple", "Apple announced new products")
    >>> print(result.entity_type)  # EntityType.INSTANCE
    >>> print(result.confidence)  # 0.92
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from identifiers import EntityType


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
# Evolvable Configuration
# =============================================================================


@dataclass
class ClassifierConfig:
    """Evolvable configuration for Entity Type Classification.

    Attributes:
        use_heuristics: Whether to try heuristic classification first
        heuristic_confidence_threshold: Minimum confidence from heuristics to accept
        llm_fallback_enabled: Whether to use LLM when heuristics are uncertain
        llm_temperature: Temperature for LLM calls
        prompt_template: The prompt template for LLM classification
    """

    use_heuristics: bool = True
    heuristic_confidence_threshold: float = 0.7
    llm_fallback_enabled: bool = True
    llm_temperature: float = 0.1

    prompt_template: str = """Classify the entity type for the given text in context.

Text to classify: "{text}"
Context sentence: "{context}"

Entity Types:
- INSTANCE: A specific, unique real-world entity (e.g., "Doug", "Apple Inc.", "The Eiffel Tower")
- CLASS: A category or type of things (e.g., "person", "company", "cat")
- NAMED_CONCEPT: A proper-named category or abstract concept (e.g., "Christmas", "Agile", "The Renaissance")
- ROLE: A relational concept tied to another entity (e.g., "CEO of Apple", "Doug's wife")
- ANAPHORA: A reference needing resolution (e.g., "he", "she", "it", "the company")
- GENERIC: A generic reference to unspecified entities (e.g., "someone", "everyone", "people")

Consider:
1. Is this referring to a specific unique thing (INSTANCE) or a category (CLASS)?
2. If it's capitalized, is it a person/company/place (INSTANCE) or a named concept (NAMED_CONCEPT)?
3. Is this a pronoun or demonstrative reference (ANAPHORA)?
4. Does it describe a role relative to another entity (ROLE)?

Respond in XML:
<classification>
    <entity_type>INSTANCE|CLASS|NAMED_CONCEPT|ROLE|ANAPHORA|GENERIC</entity_type>
    <confidence>0.0-1.0</confidence>
    <reasoning>Brief explanation</reasoning>
</classification>"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for evolution/storage."""
        return {
            "use_heuristics": self.use_heuristics,
            "heuristic_confidence_threshold": self.heuristic_confidence_threshold,
            "llm_fallback_enabled": self.llm_fallback_enabled,
            "llm_temperature": self.llm_temperature,
            "prompt_template": self.prompt_template,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassifierConfig:
        """Deserialize config from dict."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# =============================================================================
# Classification Result
# =============================================================================


@dataclass
class ClassificationResult:
    """Result of entity type classification.

    Attributes:
        text: The original text
        context: The context sentence
        entity_type: The classified EntityType
        confidence: Confidence in the classification (0.0-1.0)
        method: Classification method used (heuristic, llm)
        reasoning: Explanation (if from LLM)
        pos_tag: Part of speech (if available)
    """

    text: str
    context: str
    entity_type: EntityType
    confidence: float = 1.0
    method: str = "heuristic"
    reasoning: str = ""
    pos_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "context": self.context,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "method": self.method,
            "reasoning": self.reasoning,
            "pos_tag": self.pos_tag,
        }


# =============================================================================
# Heuristic Classifier
# =============================================================================


class HeuristicClassifier:
    """Rule-based entity type classification.

    Uses pattern matching and linguistic heuristics for fast classification.
    Falls back to LLM for uncertain cases.
    """

    # Pronouns and their likely types
    PRONOUNS = {
        # Personal pronouns (ANAPHORA)
        "he", "she", "it", "they", "him", "her", "them",
        "his", "hers", "its", "their", "theirs",
        "himself", "herself", "itself", "themselves",
        # Demonstratives (ANAPHORA)
        "this", "that", "these", "those",
        # Relative (ANAPHORA)
        "who", "whom", "whose", "which",
    }

    # Generic quantifiers (GENERIC)
    GENERICS = {
        "someone", "somebody", "anyone", "anybody",
        "everyone", "everybody", "no one", "nobody",
        "something", "anything", "everything", "nothing",
        "people", "things", "stuff", "one",
    }

    # Named concepts (proper nouns that are categories, not individuals)
    NAMED_CONCEPTS = {
        # Holidays
        "christmas", "easter", "thanksgiving", "halloween",
        "hanukkah", "diwali", "ramadan", "eid",
        # Movements/methodologies
        "agile", "scrum", "kanban", "lean",
        "marxism", "capitalism", "socialism",
        # Historical periods
        "renaissance", "enlightenment", "reformation",
        # Languages
        "english", "spanish", "french", "german", "chinese",
        # Religions
        "christianity", "islam", "buddhism", "hinduism",
    }

    # Role indicators (patterns)
    ROLE_PATTERNS = [
        r"(?:the\s+)?(\w+)\s+of\s+(\w+)",  # "CEO of Apple"
        r"(\w+)'s\s+(\w+)",  # "Doug's wife"
        r"(?:the\s+)?(\w+)\s+at\s+(\w+)",  # "manager at Google"
        r"(?:the\s+)?(\w+)\s+for\s+(\w+)",  # "lawyer for the company"
    ]

    # Common role titles
    ROLE_TITLES = {
        "ceo", "cto", "cfo", "coo", "president", "chairman",
        "director", "manager", "head", "chief", "lead",
        "wife", "husband", "father", "mother", "son", "daughter",
        "brother", "sister", "friend", "colleague", "boss",
        "owner", "founder", "creator", "author", "artist",
    }

    # Common classes (generic nouns)
    COMMON_CLASSES = {
        "person", "people", "company", "companies",
        "cat", "cats", "dog", "dogs", "animal", "animals",
        "car", "cars", "book", "books", "phone", "phones",
        "computer", "computers", "house", "houses", "building", "buildings",
        "city", "cities", "country", "countries", "place", "places",
        "time", "day", "week", "month", "year",
        "money", "food", "water", "air", "fire",
        "idea", "thought", "feeling", "emotion",
    }

    def __init__(self, config: ClassifierConfig):
        """Initialize the heuristic classifier.

        Args:
            config: Classifier configuration
        """
        self.config = config

    def classify(
        self,
        text: str,
        context: str,
        pos_tag: str | None = None,
    ) -> ClassificationResult | None:
        """Classify entity type using heuristics.

        Args:
            text: The text to classify
            context: The context sentence
            pos_tag: Part of speech (optional)

        Returns:
            ClassificationResult or None if uncertain
        """
        text_lower = text.lower().strip()
        text_stripped = text.strip()

        # Check for pronouns (ANAPHORA)
        if text_lower in self.PRONOUNS:
            return ClassificationResult(
                text=text,
                context=context,
                entity_type=EntityType.ANAPHORA,
                confidence=0.95,
                method="heuristic",
                reasoning="Recognized as pronoun",
                pos_tag=pos_tag,
            )

        # Check for generics (GENERIC)
        if text_lower in self.GENERICS:
            return ClassificationResult(
                text=text,
                context=context,
                entity_type=EntityType.GENERIC,
                confidence=0.95,
                method="heuristic",
                reasoning="Recognized as generic quantifier",
                pos_tag=pos_tag,
            )

        # Check for role patterns (ROLE)
        for pattern in self.ROLE_PATTERNS:
            if re.search(pattern, text_lower):
                # Check if it's a role title
                words = text_lower.split()
                if any(w in self.ROLE_TITLES for w in words):
                    return ClassificationResult(
                        text=text,
                        context=context,
                        entity_type=EntityType.ROLE,
                        confidence=0.85,
                        method="heuristic",
                        reasoning="Matches role pattern with role title",
                        pos_tag=pos_tag,
                    )

        # Check capitalization for proper nouns
        is_capitalized = text_stripped[0].isupper() if text_stripped else False

        # Check if it's at the start of the sentence
        context_stripped = context.strip()
        is_sentence_start = context_stripped.startswith(text_stripped)

        if is_capitalized:
            # Check for named concepts first (regardless of sentence position)
            if text_lower in self.NAMED_CONCEPTS:
                confidence = 0.90 if not is_sentence_start else 0.75
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.NAMED_CONCEPT,
                    confidence=confidence,
                    method="heuristic",
                    reasoning="Recognized as named concept",
                    pos_tag=pos_tag,
                )

            # Check for multi-word proper noun (e.g., "Apple Inc.", "New York")
            # These are likely instances even at sentence start
            if " " in text_stripped or text_stripped.endswith("."):
                confidence = 0.80 if not is_sentence_start else 0.70
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.INSTANCE,
                    confidence=confidence,
                    method="heuristic",
                    reasoning="Multi-word proper noun",
                    pos_tag=pos_tag,
                )

            # Single capitalized word
            if not is_sentence_start:
                # Not at sentence start = high confidence proper noun
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.INSTANCE,
                    confidence=0.70,
                    method="heuristic",
                    reasoning="Capitalized proper noun (not at sentence start)",
                    pos_tag=pos_tag,
                )
            else:
                # At sentence start - could be proper noun or just capitalized
                # Use lower confidence but still consider it an instance
                # (Most entity queries at sentence start ARE proper nouns)
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.INSTANCE,
                    confidence=0.55,
                    method="heuristic",
                    reasoning="Capitalized at sentence start (may be proper noun)",
                    pos_tag=pos_tag,
                )

        # Check for common class nouns
        if text_lower in self.COMMON_CLASSES:
            return ClassificationResult(
                text=text,
                context=context,
                entity_type=EntityType.CLASS,
                confidence=0.85,
                method="heuristic",
                reasoning="Common class noun",
                pos_tag=pos_tag,
            )

        # Check POS tag if available
        if pos_tag:
            if pos_tag == "PROPN":
                # Proper noun
                if text_lower in self.NAMED_CONCEPTS:
                    return ClassificationResult(
                        text=text,
                        context=context,
                        entity_type=EntityType.NAMED_CONCEPT,
                        confidence=0.85,
                        method="heuristic",
                        reasoning="POS tag PROPN + recognized named concept",
                        pos_tag=pos_tag,
                    )
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.INSTANCE,
                    confidence=0.75,
                    method="heuristic",
                    reasoning="POS tag indicates proper noun",
                    pos_tag=pos_tag,
                )
            elif pos_tag == "NOUN":
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.CLASS,
                    confidence=0.70,
                    method="heuristic",
                    reasoning="POS tag indicates common noun",
                    pos_tag=pos_tag,
                )
            elif pos_tag == "PRON":
                return ClassificationResult(
                    text=text,
                    context=context,
                    entity_type=EntityType.ANAPHORA,
                    confidence=0.90,
                    method="heuristic",
                    reasoning="POS tag indicates pronoun",
                    pos_tag=pos_tag,
                )

        # Default to CLASS for lowercase words (uncertain)
        if not is_capitalized:
            return ClassificationResult(
                text=text,
                context=context,
                entity_type=EntityType.CLASS,
                confidence=0.50,
                method="heuristic",
                reasoning="Default: lowercase word assumed to be class",
                pos_tag=pos_tag,
            )

        # Can't determine with confidence
        return None


# =============================================================================
# LLM Classifier
# =============================================================================


class LLMClassifier:
    """LLM-based entity type classification.

    Uses an LLM to classify entity types when heuristics are uncertain.
    """

    def __init__(self, llm: LLMProvider, config: ClassifierConfig):
        """Initialize the LLM classifier.

        Args:
            llm: LLM provider
            config: Classifier configuration
        """
        self.llm = llm
        self.config = config

    async def classify(
        self,
        text: str,
        context: str,
        pos_tag: str | None = None,
    ) -> ClassificationResult | None:
        """Classify entity type using LLM.

        Args:
            text: The text to classify
            context: The context sentence
            pos_tag: Part of speech (optional)

        Returns:
            ClassificationResult or None if failed
        """
        prompt = self.config.prompt_template.format(
            text=text,
            context=context,
        )

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )

            content = _extract_llm_content(response)
            return self._parse_response(content, text, context, pos_tag)

        except Exception:
            return None

    def _parse_response(
        self,
        response: str,
        text: str,
        context: str,
        pos_tag: str | None,
    ) -> ClassificationResult | None:
        """Parse LLM classification response."""
        # Extract entity_type
        type_match = re.search(r"<entity_type>(\w+)</entity_type>", response)
        if not type_match:
            return None

        type_str = type_match.group(1).strip().upper()

        # Map to EntityType
        type_map = {
            "INSTANCE": EntityType.INSTANCE,
            "CLASS": EntityType.CLASS,
            "NAMED_CONCEPT": EntityType.NAMED_CONCEPT,
            "ROLE": EntityType.ROLE,
            "ANAPHORA": EntityType.ANAPHORA,
            "GENERIC": EntityType.GENERIC,
        }

        entity_type = type_map.get(type_str)
        if entity_type is None:
            return None

        # Extract confidence
        conf_match = re.search(r"<confidence>([0-9.]+)</confidence>", response)
        confidence = float(conf_match.group(1)) if conf_match else 0.8

        # Extract reasoning
        reason_match = re.search(r"<reasoning>([^<]+)</reasoning>", response)
        reasoning = reason_match.group(1).strip() if reason_match else ""

        return ClassificationResult(
            text=text,
            context=context,
            entity_type=entity_type,
            confidence=confidence,
            method="llm",
            reasoning=reasoning,
            pos_tag=pos_tag,
        )


# =============================================================================
# Main Entity Classifier
# =============================================================================


class EntityClassifier:
    """Hybrid entity type classification system.

    Combines heuristic rules with LLM fallback for accurate classification.

    Pipeline:
    1. Try heuristic classification (fast)
    2. If confident enough, return result
    3. Otherwise, use LLM fallback (if enabled)
    4. Return best result or low-confidence heuristic result

    Example:
        >>> config = ClassifierConfig()
        >>> classifier = EntityClassifier(config, llm=my_llm)
        >>>
        >>> result = await classifier.classify("Doug", "Doug went to the store")
        >>> print(result.entity_type)  # EntityType.INSTANCE
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        llm: LLMProvider | None = None,
    ):
        """Initialize the entity classifier.

        Args:
            config: Classifier configuration
            llm: LLM provider for fallback
        """
        self.config = config or ClassifierConfig()
        self.llm = llm
        self.heuristic = HeuristicClassifier(self.config)

        if llm:
            self.llm_classifier = LLMClassifier(llm, self.config)
        else:
            self.llm_classifier = None

        # Metrics
        self.metrics = {
            "total_calls": 0,
            "heuristic_accepted": 0,
            "llm_calls": 0,
            "failures": 0,
        }

    async def classify(
        self,
        text: str,
        context: str,
        pos_tag: str | None = None,
    ) -> ClassificationResult:
        """Classify the entity type of text in context.

        Args:
            text: The text to classify
            context: The context sentence
            pos_tag: Part of speech (optional)

        Returns:
            ClassificationResult (always returns a result, even if uncertain)
        """
        self.metrics["total_calls"] += 1

        # Step 1: Try heuristic classification
        if self.config.use_heuristics:
            heuristic_result = self.heuristic.classify(text, context, pos_tag)

            if heuristic_result:
                if heuristic_result.confidence >= self.config.heuristic_confidence_threshold:
                    self.metrics["heuristic_accepted"] += 1
                    return heuristic_result

        # Step 2: LLM fallback if enabled and available
        if self.config.llm_fallback_enabled and self.llm_classifier:
            self.metrics["llm_calls"] += 1
            llm_result = await self.llm_classifier.classify(text, context, pos_tag)

            if llm_result:
                return llm_result

        # Step 3: Return heuristic result even if low confidence
        if heuristic_result:
            return heuristic_result

        # Step 4: Default to CLASS with very low confidence
        self.metrics["failures"] += 1
        return ClassificationResult(
            text=text,
            context=context,
            entity_type=EntityType.CLASS,
            confidence=0.3,
            method="default",
            reasoning="Could not classify with any method",
            pos_tag=pos_tag,
        )

    async def classify_all(
        self,
        mentions: list[str],
        context: str,
    ) -> dict[str, ClassificationResult]:
        """Classify multiple mentions in the same context.

        Args:
            mentions: List of text mentions to classify
            context: The context sentence

        Returns:
            Dict mapping mentions to classification results
        """
        results = {}
        for mention in mentions:
            result = await self.classify(mention, context)
            results[mention] = result
        return results

    def classify_sync(
        self,
        text: str,
        context: str,
        pos_tag: str | None = None,
    ) -> ClassificationResult:
        """Synchronous classification using only heuristics.

        Args:
            text: The text to classify
            context: The context sentence
            pos_tag: Part of speech (optional)

        Returns:
            ClassificationResult
        """
        self.metrics["total_calls"] += 1

        result = self.heuristic.classify(text, context, pos_tag)
        if result:
            self.metrics["heuristic_accepted"] += 1
            return result

        # Default fallback
        self.metrics["failures"] += 1
        return ClassificationResult(
            text=text,
            context=context,
            entity_type=EntityType.CLASS,
            confidence=0.3,
            method="default",
            reasoning="Could not classify with heuristics",
            pos_tag=pos_tag,
        )

    def get_metrics(self) -> dict[str, int]:
        """Get classification metrics."""
        return dict(self.metrics)

    def reset_metrics(self) -> None:
        """Reset metrics to zero."""
        for key in self.metrics:
            self.metrics[key] = 0


# =============================================================================
# Convenience Functions
# =============================================================================


def is_pronoun(text: str) -> bool:
    """Check if text is a pronoun."""
    return text.lower().strip() in HeuristicClassifier.PRONOUNS


def is_generic(text: str) -> bool:
    """Check if text is a generic quantifier."""
    return text.lower().strip() in HeuristicClassifier.GENERICS


def is_likely_proper_noun(text: str, context: str) -> bool:
    """Check if text is likely a proper noun (not at sentence start)."""
    text_stripped = text.strip()
    if not text_stripped:
        return False

    is_capitalized = text_stripped[0].isupper()
    context_stripped = context.strip()
    is_sentence_start = context_stripped.startswith(text_stripped)

    return is_capitalized and not is_sentence_start


def extract_role_anchor(text: str) -> tuple[str, str] | None:
    """Extract role and anchor from a role phrase.

    Args:
        text: The role phrase (e.g., "CEO of Apple")

    Returns:
        Tuple of (role, anchor) or None if not a role pattern
    """
    # "X of Y" pattern
    match = re.match(r"(?:the\s+)?(\w+)\s+of\s+(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # "Y's X" pattern
    match = re.match(r"(\w+)'s\s+(.+)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip(), match.group(1).strip()

    return None

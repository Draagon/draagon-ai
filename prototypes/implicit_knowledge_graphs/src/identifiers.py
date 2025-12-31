"""Universal Semantic Identifier System.

Provides the foundational identifier system for semantic units including:
- EntityType enum for classifying semantic entities
- UniversalSemanticIdentifier for uniquely identifying any semantic unit
- SynsetInfo for WordNet synset metadata

This module is the foundation for all other phases - proper identification
and disambiguation must happen before any other processing.

Example:
    >>> from identifiers import EntityType, UniversalSemanticIdentifier
    >>>
    >>> # Create an identifier for a financial institution
    >>> bank_id = UniversalSemanticIdentifier(
    ...     entity_type=EntityType.CLASS,
    ...     wordnet_synset="bank.n.01",
    ...     sense_rank=1,
    ...     domain="FINANCE",
    ...     confidence=0.95,
    ... )
    >>>
    >>> # Check if two identifiers refer to the same sense
    >>> other_id = UniversalSemanticIdentifier(
    ...     entity_type=EntityType.CLASS,
    ...     wordnet_synset="bank.n.01",
    ... )
    >>> bank_id.matches_sense(other_id)  # True
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Entity Type Classification
# =============================================================================


class EntityType(str, Enum):
    """Types of semantic entities.

    Every semantic unit must be classified into one of these types:

    INSTANCE: A specific, unique real-world entity
        - Proper nouns referring to specific things
        - Examples: "Doug", "Apple Inc.", "The Eiffel Tower"

    CLASS: A category or type of things
        - Common nouns referring to kinds
        - Examples: "person", "company", "cat", "bank" (financial)

    NAMED_CONCEPT: A proper-named category or abstract concept
        - Proper nouns that name categories, events, or ideas
        - Examples: "Christmas", "Agile", "The Renaissance"

    ROLE: A relational concept tied to another entity
        - Roles are defined by their relation to an anchor entity
        - Examples: "CEO of Apple", "Doug's wife", "the author of the book"

    ANAPHORA: A reference requiring resolution
        - Pronouns and other references that need context
        - Examples: "he", "she", "it", "the company", "that"

    GENERIC: A generic reference to unspecified entities
        - Quantified or indefinite references
        - Examples: "someone", "everyone", "people in general"
    """

    INSTANCE = "instance"
    CLASS = "class"
    NAMED_CONCEPT = "named_concept"
    ROLE = "role"
    ANAPHORA = "anaphora"
    GENERIC = "generic"


# =============================================================================
# Synset Information
# =============================================================================


@dataclass
class SynsetInfo:
    """Information about a WordNet synset.

    Wraps synset data for easier handling, whether from NLTK or mock data.

    Attributes:
        synset_id: The synset identifier (e.g., "bank.n.01")
        pos: Part of speech (n, v, a, r)
        lemmas: List of lemma names in this synset
        definition: The synset's gloss/definition
        examples: Example sentences using this sense
        hypernyms: Direct hypernym synset IDs
        hyponyms: Direct hyponym synset IDs
    """

    synset_id: str
    pos: str
    lemmas: list[str] = field(default_factory=list)
    definition: str = ""
    examples: list[str] = field(default_factory=list)
    hypernyms: list[str] = field(default_factory=list)
    hyponyms: list[str] = field(default_factory=list)

    @property
    def word(self) -> str:
        """Extract the word from the synset ID."""
        parts = self.synset_id.rsplit(".", 2)
        return parts[0] if parts else ""

    @property
    def sense_number(self) -> int:
        """Extract the sense number from the synset ID."""
        parts = self.synset_id.rsplit(".", 2)
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                return 1
        return 1


# =============================================================================
# Universal Semantic Identifier
# =============================================================================


@dataclass
class UniversalSemanticIdentifier:
    """Universal identifier for any semantic unit.

    Provides a unified way to identify and reference any semantic entity,
    whether it's a specific instance, a concept class, a named concept,
    a relational role, or a reference needing resolution.

    Key Design Principles:
    1. Local ID is always present for internal reference
    2. External IDs (WordNet, Wikidata, BabelNet) are optional but preferred
    3. Disambiguation metadata helps with confidence tracking
    4. Type-specific fields support different entity types

    Attributes:
        local_id: UUID for internal reference (auto-generated if not provided)
        entity_type: Classification of this semantic unit
        wordnet_synset: WordNet synset ID (e.g., "bank.n.01")
        wikidata_qid: Wikidata entity ID (e.g., "Q312")
        babelnet_synset: BabelNet synset ID (e.g., "bn:00008364n")
        sense_rank: Which sense number (1 = most common)
        domain: Semantic domain (e.g., "FINANCE", "GEOGRAPHY")
        confidence: Disambiguation confidence (0.0-1.0)
        canonical_name: For INSTANCE type, the canonical name
        aliases: Alternative names for this entity
        hypernym_chain: For CLASS type, the chain of hypernyms

    Example:
        >>> # A named entity (instance)
        >>> doug = UniversalSemanticIdentifier(
        ...     entity_type=EntityType.INSTANCE,
        ...     canonical_name="Doug",
        ...     wikidata_qid=None,  # Local entity, no Wikidata
        ...     confidence=1.0,
        ... )
        >>>
        >>> # A concept class with WordNet sense
        >>> bank_financial = UniversalSemanticIdentifier(
        ...     entity_type=EntityType.CLASS,
        ...     wordnet_synset="bank.n.01",
        ...     domain="FINANCE",
        ...     sense_rank=1,
        ...     confidence=0.92,
        ...     hypernym_chain=["financial_institution.n.01", "institution.n.01"],
        ... )
    """

    # Entity type (required)
    entity_type: EntityType

    # Local identifier (auto-generated if not provided)
    local_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # External identifiers (nullable)
    wordnet_synset: str | None = None
    wikidata_qid: str | None = None
    babelnet_synset: str | None = None

    # Disambiguation metadata
    sense_rank: int = 1
    domain: str | None = None
    confidence: float = 1.0

    # For INSTANCE type
    canonical_name: str | None = None
    aliases: list[str] = field(default_factory=list)

    # For CLASS type
    hypernym_chain: list[str] = field(default_factory=list)

    # For ROLE type
    role_relation: str | None = None  # e.g., "CEO_OF", "WIFE_OF"
    anchor_entity_id: str | None = None  # The entity this role is relative to

    # For ANAPHORA type
    resolved_to: str | None = None  # The local_id of the resolved entity

    def __hash__(self) -> int:
        """Hash based on local_id for set/dict usage."""
        return hash(self.local_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on local_id."""
        if isinstance(other, UniversalSemanticIdentifier):
            return self.local_id == other.local_id
        return False

    def matches_sense(self, other: UniversalSemanticIdentifier) -> bool:
        """Check if two identifiers refer to the same semantic sense.

        This is different from __eq__ which checks identity.
        matches_sense checks if they represent the same concept.

        Priority for matching:
        1. Same WordNet synset
        2. Same Wikidata entity
        3. Same BabelNet synset
        4. Same local_id (fallback)

        Args:
            other: Another identifier to compare

        Returns:
            True if they refer to the same sense
        """
        # Same WordNet synset
        if self.wordnet_synset and other.wordnet_synset:
            return self.wordnet_synset == other.wordnet_synset

        # Same Wikidata entity
        if self.wikidata_qid and other.wikidata_qid:
            return self.wikidata_qid == other.wikidata_qid

        # Same BabelNet synset
        if self.babelnet_synset and other.babelnet_synset:
            return self.babelnet_synset == other.babelnet_synset

        # Fallback to local_id
        return self.local_id == other.local_id

    def is_same_word_different_sense(
        self, other: UniversalSemanticIdentifier
    ) -> bool:
        """Check if this is the same word but a different sense.

        Useful for detecting sense confusion.

        Args:
            other: Another identifier to compare

        Returns:
            True if same word, different sense
        """
        if not self.wordnet_synset or not other.wordnet_synset:
            return False

        # Parse synset IDs (format: word.pos.sense_num)
        self_parts = self.wordnet_synset.rsplit(".", 2)
        other_parts = other.wordnet_synset.rsplit(".", 2)

        if len(self_parts) < 3 or len(other_parts) < 3:
            return False

        # Same word and POS, different sense number
        return (
            self_parts[0] == other_parts[0]
            and self_parts[1] == other_parts[1]
            and self_parts[2] != other_parts[2]
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation of this identifier
        """
        return {
            "local_id": self.local_id,
            "entity_type": self.entity_type.value,
            "wordnet_synset": self.wordnet_synset,
            "wikidata_qid": self.wikidata_qid,
            "babelnet_synset": self.babelnet_synset,
            "sense_rank": self.sense_rank,
            "domain": self.domain,
            "confidence": self.confidence,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "hypernym_chain": self.hypernym_chain,
            "role_relation": self.role_relation,
            "anchor_entity_id": self.anchor_entity_id,
            "resolved_to": self.resolved_to,
        }

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalSemanticIdentifier:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed identifier
        """
        return cls(
            local_id=data.get("local_id", str(uuid.uuid4())),
            entity_type=EntityType(data["entity_type"]),
            wordnet_synset=data.get("wordnet_synset"),
            wikidata_qid=data.get("wikidata_qid"),
            babelnet_synset=data.get("babelnet_synset"),
            sense_rank=data.get("sense_rank", 1),
            domain=data.get("domain"),
            confidence=data.get("confidence", 1.0),
            canonical_name=data.get("canonical_name"),
            aliases=data.get("aliases", []),
            hypernym_chain=data.get("hypernym_chain", []),
            role_relation=data.get("role_relation"),
            anchor_entity_id=data.get("anchor_entity_id"),
            resolved_to=data.get("resolved_to"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> UniversalSemanticIdentifier:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            Reconstructed identifier
        """
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Concise string representation."""
        type_str = self.entity_type.value.upper()

        if self.wordnet_synset:
            return f"USI({type_str}:{self.wordnet_synset}@{self.confidence:.2f})"
        elif self.wikidata_qid:
            return f"USI({type_str}:{self.wikidata_qid}@{self.confidence:.2f})"
        elif self.canonical_name:
            return f"USI({type_str}:{self.canonical_name}@{self.confidence:.2f})"
        else:
            return f"USI({type_str}:{self.local_id[:8]}...@{self.confidence:.2f})"


# =============================================================================
# Factory Functions
# =============================================================================


def create_instance_identifier(
    name: str,
    wikidata_qid: str | None = None,
    aliases: list[str] | None = None,
    confidence: float = 1.0,
) -> UniversalSemanticIdentifier:
    """Create an identifier for a named entity instance.

    Args:
        name: The canonical name of the entity
        wikidata_qid: Optional Wikidata QID
        aliases: Optional list of alternative names
        confidence: Confidence in the identification

    Returns:
        A new identifier for this instance
    """
    return UniversalSemanticIdentifier(
        entity_type=EntityType.INSTANCE,
        canonical_name=name,
        wikidata_qid=wikidata_qid,
        aliases=aliases or [],
        confidence=confidence,
    )


def create_class_identifier(
    synset_id: str,
    definition: str = "",
    domain: str | None = None,
    hypernym_chain: list[str] | None = None,
    confidence: float = 1.0,
) -> UniversalSemanticIdentifier:
    """Create an identifier for a concept class.

    Args:
        synset_id: The WordNet synset ID (e.g., "bank.n.01")
        definition: The definition of this sense
        domain: Optional semantic domain
        hypernym_chain: Optional list of hypernym synset IDs
        confidence: Confidence in the disambiguation

    Returns:
        A new identifier for this class
    """
    # Extract sense rank from synset ID
    sense_rank = 1
    parts = synset_id.rsplit(".", 2)
    if len(parts) >= 3:
        try:
            sense_rank = int(parts[2])
        except ValueError:
            pass

    return UniversalSemanticIdentifier(
        entity_type=EntityType.CLASS,
        wordnet_synset=synset_id,
        sense_rank=sense_rank,
        domain=domain,
        hypernym_chain=hypernym_chain or [],
        confidence=confidence,
    )


def create_role_identifier(
    role_relation: str,
    anchor_entity_id: str,
    confidence: float = 1.0,
) -> UniversalSemanticIdentifier:
    """Create an identifier for a relational role.

    Args:
        role_relation: The role relation type (e.g., "CEO_OF")
        anchor_entity_id: The local_id of the anchor entity
        confidence: Confidence in the role identification

    Returns:
        A new identifier for this role
    """
    return UniversalSemanticIdentifier(
        entity_type=EntityType.ROLE,
        role_relation=role_relation,
        anchor_entity_id=anchor_entity_id,
        confidence=confidence,
    )


def create_anaphora_identifier(
    surface_form: str,
    resolved_to: str | None = None,
    confidence: float = 0.0,
) -> UniversalSemanticIdentifier:
    """Create an identifier for an anaphoric reference.

    Args:
        surface_form: The text of the reference (e.g., "he", "it")
        resolved_to: Optional local_id of the resolved entity
        confidence: Confidence in the resolution (0.0 if unresolved)

    Returns:
        A new identifier for this anaphora
    """
    return UniversalSemanticIdentifier(
        entity_type=EntityType.ANAPHORA,
        canonical_name=surface_form,
        resolved_to=resolved_to,
        confidence=confidence if resolved_to else 0.0,
    )


def create_generic_identifier(
    surface_form: str,
    wordnet_synset: str | None = None,
) -> UniversalSemanticIdentifier:
    """Create an identifier for a generic reference.

    Args:
        surface_form: The text (e.g., "someone", "people")
        wordnet_synset: Optional WordNet synset for the concept

    Returns:
        A new identifier for this generic reference
    """
    return UniversalSemanticIdentifier(
        entity_type=EntityType.GENERIC,
        canonical_name=surface_form,
        wordnet_synset=wordnet_synset,
        confidence=1.0,
    )

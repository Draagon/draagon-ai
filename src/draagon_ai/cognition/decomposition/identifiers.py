"""Universal Semantic Identifier System.

Provides the foundational identifier system for semantic units including:
- EntityType enum for classifying semantic entities
- UniversalSemanticIdentifier for uniquely identifying any semantic unit
- SynsetInfo for WordNet synset metadata
- LearnedSynset for extending WordNet with new terms

This module is the foundation for all other phases - proper identification
and disambiguation must happen before any other processing.

Example:
    >>> from draagon_ai.cognition.decomposition import EntityType, UniversalSemanticIdentifier
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

Based on prototype work in prototypes/implicit_knowledge_graphs/
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
# Learned Synset (Evolving Database)
# =============================================================================


class SynsetSource(str, Enum):
    """Source of a learned synset.

    BOOTSTRAP: Pre-loaded vocabulary from initialization
    USER: User-provided correction or new definition
    LLM: LLM-generated definition (may need verification)
    WORDNET: From WordNet (passthrough, not stored in learned DB)
    """

    BOOTSTRAP = "bootstrap"
    USER = "user"
    LLM = "llm"
    WORDNET = "wordnet"


@dataclass
class LearnedSynset:
    """A synset learned outside of WordNet.

    Extends the concept of SynsetInfo with provenance tracking,
    usage statistics, and reinforcement learning support.

    Used for:
    - Technology terms (kubernetes, docker, terraform)
    - Domain-specific jargon not in WordNet
    - User-defined terms and corrections
    - LLM-generated definitions

    Synset ID Convention for learned terms:
    - Format: {word}.{domain_code}.{sense_number}
    - Examples: kubernetes.tech.01, rag.ai.01
    - This distinguishes from WordNet IDs which use {word}.{pos}.{sense}

    Attributes:
        synset_id: Unique identifier (e.g., "kubernetes.tech.01")
        word: The primary word/term
        pos: Part of speech (n=noun, v=verb, a=adjective, r=adverb)
        definition: Human-readable definition
        examples: Example sentences showing usage
        hypernyms: Parent concept synset IDs
        hyponyms: Child concept synset IDs
        aliases: Alternative names/abbreviations (e.g., ["k8s", "kube"])
        domain: Semantic domain (e.g., "CLOUD_INFRASTRUCTURE", "AI_ML")

        source: Where this synset came from
        confidence: Confidence in the definition (0.0-1.0)
        usage_count: Number of times this synset was used in disambiguation
        success_count: Number of successful disambiguations
        failure_count: Number of failed disambiguations
        last_used: ISO timestamp of last usage
        created_at: ISO timestamp of creation
    """

    # Core synset data
    synset_id: str
    word: str
    pos: str
    definition: str
    examples: list[str] = field(default_factory=list)
    hypernyms: list[str] = field(default_factory=list)
    hyponyms: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    domain: str = ""

    # Provenance
    source: SynsetSource = SynsetSource.BOOTSTRAP
    confidence: float = 1.0

    # Usage statistics for reinforcement learning
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: str | None = None
    created_at: str = field(default_factory=lambda: "")

    def __post_init__(self):
        """Set created_at if not provided."""
        if not self.created_at:
            from datetime import datetime, timezone

            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def success_rate(self) -> float:
        """Calculate the success rate from usage statistics."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0  # No data, assume success
        return self.success_count / total

    @property
    def lemmas(self) -> list[str]:
        """Return word and aliases as lemmas for compatibility with SynsetInfo."""
        return [self.word] + self.aliases

    @property
    def sense_number(self) -> int:
        """Extract sense number from synset_id."""
        parts = self.synset_id.rsplit(".", 2)
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except ValueError:
                return 1
        return 1

    def to_synset_info(self) -> SynsetInfo:
        """Convert to SynsetInfo for compatibility with existing code.

        Returns:
            SynsetInfo with equivalent data
        """
        return SynsetInfo(
            synset_id=self.synset_id,
            pos=self.pos,
            lemmas=self.lemmas,
            definition=self.definition,
            examples=self.examples,
            hypernyms=self.hypernyms,
            hyponyms=self.hyponyms,
        )

    def record_usage(self, success: bool = True) -> None:
        """Record a usage of this synset.

        Args:
            success: Whether the disambiguation was successful
        """
        from datetime import datetime, timezone

        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used = datetime.now(timezone.utc).isoformat()

    def matches_word(self, query: str) -> bool:
        """Check if this synset matches a word query.

        Args:
            query: The word to match (case-insensitive)

        Returns:
            True if the word or any alias matches
        """
        query_lower = query.lower()
        if self.word.lower() == query_lower:
            return True
        return any(alias.lower() == query_lower for alias in self.aliases)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage.

        Returns:
            Dictionary representation
        """
        return {
            "synset_id": self.synset_id,
            "word": self.word,
            "pos": self.pos,
            "definition": self.definition,
            "examples": self.examples,
            "hypernyms": self.hypernyms,
            "hyponyms": self.hyponyms,
            "aliases": self.aliases,
            "domain": self.domain,
            "source": self.source.value if isinstance(self.source, SynsetSource) else self.source,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_used": self.last_used,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedSynset":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed LearnedSynset
        """
        source = data.get("source", "bootstrap")
        if isinstance(source, str):
            source = SynsetSource(source)

        return cls(
            synset_id=data["synset_id"],
            word=data["word"],
            pos=data.get("pos", "n"),
            definition=data.get("definition", ""),
            examples=data.get("examples", []),
            hypernyms=data.get("hypernyms", []),
            hyponyms=data.get("hyponyms", []),
            aliases=data.get("aliases", []),
            domain=data.get("domain", ""),
            source=source,
            confidence=data.get("confidence", 1.0),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            last_used=data.get("last_used"),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LearnedSynset":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"LearnedSynset({self.synset_id}: {self.word}, "
            f"source={self.source.value}, "
            f"success_rate={self.success_rate:.2%})"
        )


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

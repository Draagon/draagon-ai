"""Tests for the Universal Semantic Identifier system."""

import json
import pytest

from identifiers import (
    EntityType,
    SynsetInfo,
    UniversalSemanticIdentifier,
    create_instance_identifier,
    create_class_identifier,
    create_role_identifier,
    create_anaphora_identifier,
    create_generic_identifier,
)


# =============================================================================
# EntityType Tests
# =============================================================================


class TestEntityType:
    """Tests for the EntityType enum."""

    def test_all_types_exist(self):
        """All required entity types should exist."""
        assert EntityType.INSTANCE.value == "instance"
        assert EntityType.CLASS.value == "class"
        assert EntityType.NAMED_CONCEPT.value == "named_concept"
        assert EntityType.ROLE.value == "role"
        assert EntityType.ANAPHORA.value == "anaphora"
        assert EntityType.GENERIC.value == "generic"

    def test_enum_is_string_based(self):
        """EntityType should be string-based for JSON serialization."""
        assert isinstance(EntityType.INSTANCE.value, str)
        assert str(EntityType.INSTANCE) == "EntityType.INSTANCE"

    def test_enum_from_value(self):
        """Should be able to create EntityType from string value."""
        assert EntityType("instance") == EntityType.INSTANCE
        assert EntityType("class") == EntityType.CLASS
        assert EntityType("named_concept") == EntityType.NAMED_CONCEPT


# =============================================================================
# SynsetInfo Tests
# =============================================================================


class TestSynsetInfo:
    """Tests for the SynsetInfo dataclass."""

    def test_basic_creation(self):
        """Should create SynsetInfo with required fields."""
        info = SynsetInfo(
            synset_id="bank.n.01",
            pos="n",
        )
        assert info.synset_id == "bank.n.01"
        assert info.pos == "n"
        assert info.lemmas == []
        assert info.definition == ""

    def test_full_creation(self):
        """Should create SynsetInfo with all fields."""
        info = SynsetInfo(
            synset_id="bank.n.01",
            pos="n",
            lemmas=["bank", "banking_company"],
            definition="a financial institution",
            examples=["he deposited money in the bank"],
            hypernyms=["financial_institution.n.01"],
            hyponyms=["commercial_bank.n.01"],
        )
        assert len(info.lemmas) == 2
        assert "financial" in info.definition
        assert len(info.examples) == 1

    def test_word_property(self):
        """Should extract word from synset ID."""
        info = SynsetInfo(synset_id="bank.n.01", pos="n")
        assert info.word == "bank"

        info2 = SynsetInfo(synset_id="financial_institution.n.01", pos="n")
        assert info2.word == "financial_institution"

    def test_sense_number_property(self):
        """Should extract sense number from synset ID."""
        info1 = SynsetInfo(synset_id="bank.n.01", pos="n")
        assert info1.sense_number == 1

        info2 = SynsetInfo(synset_id="bank.n.02", pos="n")
        assert info2.sense_number == 2

        info3 = SynsetInfo(synset_id="run.v.15", pos="v")
        assert info3.sense_number == 15


# =============================================================================
# UniversalSemanticIdentifier Tests
# =============================================================================


class TestUniversalSemanticIdentifier:
    """Tests for the UniversalSemanticIdentifier dataclass."""

    def test_basic_creation(self):
        """Should create identifier with required fields."""
        usi = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.01",
        )
        assert usi.entity_type == EntityType.CLASS
        assert usi.wordnet_synset == "bank.n.01"
        assert usi.local_id is not None
        assert len(usi.local_id) == 36  # UUID format

    def test_auto_generated_local_id(self):
        """Should auto-generate unique local_ids."""
        usi1 = UniversalSemanticIdentifier(entity_type=EntityType.CLASS)
        usi2 = UniversalSemanticIdentifier(entity_type=EntityType.CLASS)
        assert usi1.local_id != usi2.local_id

    def test_custom_local_id(self):
        """Should accept custom local_id."""
        usi = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            local_id="my-custom-id",
        )
        assert usi.local_id == "my-custom-id"

    def test_hash_and_eq(self):
        """Should be hashable and comparable by local_id."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            local_id="same-id",
        )
        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,  # Different type
            local_id="same-id",  # Same ID
        )
        usi3 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            local_id="different-id",
        )

        # Same local_id = equal
        assert usi1 == usi2
        assert hash(usi1) == hash(usi2)

        # Different local_id = not equal
        assert usi1 != usi3

        # Can be used in sets
        s = {usi1, usi2, usi3}
        assert len(s) == 2  # usi1 and usi2 are same

    def test_matches_sense_wordnet(self):
        """Should match by WordNet synset."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.01",
            local_id="id1",
        )
        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.01",
            local_id="id2",
        )
        usi3 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.02",
            local_id="id3",
        )

        assert usi1.matches_sense(usi2)  # Same synset
        assert not usi1.matches_sense(usi3)  # Different synset

    def test_matches_sense_wikidata(self):
        """Should match by Wikidata QID."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            wikidata_qid="Q312",
            local_id="id1",
        )
        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            wikidata_qid="Q312",
            local_id="id2",
        )
        usi3 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            wikidata_qid="Q123",
            local_id="id3",
        )

        assert usi1.matches_sense(usi2)
        assert not usi1.matches_sense(usi3)

    def test_matches_sense_fallback_to_local_id(self):
        """Should fall back to local_id when no external IDs."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            canonical_name="Doug",
            local_id="id1",
        )
        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            canonical_name="Doug",
            local_id="id1",
        )
        usi3 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            canonical_name="Doug",
            local_id="id3",
        )

        assert usi1.matches_sense(usi2)  # Same local_id
        assert not usi1.matches_sense(usi3)  # Different local_id

    def test_is_same_word_different_sense(self):
        """Should detect same word, different sense."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.01",
        )
        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.02",
        )
        usi3 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="river.n.01",
        )
        usi4 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.v.01",  # Verb, not noun
        )

        assert usi1.is_same_word_different_sense(usi2)  # bank.n.01 vs bank.n.02
        assert not usi1.is_same_word_different_sense(usi3)  # bank vs river
        assert not usi1.is_same_word_different_sense(usi4)  # noun vs verb

    def test_serialization_to_dict(self):
        """Should serialize to dictionary."""
        usi = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            local_id="test-id",
            wordnet_synset="bank.n.01",
            domain="FINANCE",
            confidence=0.95,
            hypernym_chain=["financial_institution.n.01"],
        )

        d = usi.to_dict()
        assert d["entity_type"] == "class"
        assert d["local_id"] == "test-id"
        assert d["wordnet_synset"] == "bank.n.01"
        assert d["domain"] == "FINANCE"
        assert d["confidence"] == 0.95

    def test_serialization_to_json(self):
        """Should serialize to JSON string."""
        usi = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            canonical_name="Doug",
            aliases=["Douglas", "D"],
        )

        json_str = usi.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["entity_type"] == "instance"
        assert data["canonical_name"] == "Doug"
        assert data["aliases"] == ["Douglas", "D"]

    def test_deserialization_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "entity_type": "class",
            "local_id": "test-id",
            "wordnet_synset": "bank.n.01",
            "confidence": 0.85,
        }

        usi = UniversalSemanticIdentifier.from_dict(data)
        assert usi.entity_type == EntityType.CLASS
        assert usi.local_id == "test-id"
        assert usi.wordnet_synset == "bank.n.01"
        assert usi.confidence == 0.85

    def test_deserialization_from_json(self):
        """Should deserialize from JSON string."""
        json_str = '{"entity_type": "instance", "canonical_name": "Apple Inc."}'

        usi = UniversalSemanticIdentifier.from_json(json_str)
        assert usi.entity_type == EntityType.INSTANCE
        assert usi.canonical_name == "Apple Inc."

    def test_roundtrip_serialization(self):
        """Should preserve data through serialization roundtrip."""
        original = UniversalSemanticIdentifier(
            entity_type=EntityType.ROLE,
            role_relation="CEO_OF",
            anchor_entity_id="entity-123",
            confidence=0.9,
        )

        json_str = original.to_json()
        restored = UniversalSemanticIdentifier.from_json(json_str)

        assert restored.entity_type == original.entity_type
        assert restored.role_relation == original.role_relation
        assert restored.anchor_entity_id == original.anchor_entity_id
        assert restored.confidence == original.confidence

    def test_repr(self):
        """Should have informative repr."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.01",
            confidence=0.95,
        )
        assert "CLASS" in repr(usi1)
        assert "bank.n.01" in repr(usi1)
        assert "0.95" in repr(usi1)

        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.INSTANCE,
            wikidata_qid="Q312",
            confidence=1.0,
        )
        assert "INSTANCE" in repr(usi2)
        assert "Q312" in repr(usi2)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for identifier factory functions."""

    def test_create_instance_identifier(self):
        """Should create instance identifier."""
        usi = create_instance_identifier(
            name="Doug",
            wikidata_qid=None,
            aliases=["Douglas"],
            confidence=1.0,
        )

        assert usi.entity_type == EntityType.INSTANCE
        assert usi.canonical_name == "Doug"
        assert "Douglas" in usi.aliases
        assert usi.confidence == 1.0

    def test_create_instance_with_wikidata(self):
        """Should create instance with Wikidata QID."""
        usi = create_instance_identifier(
            name="Apple Inc.",
            wikidata_qid="Q312",
        )

        assert usi.entity_type == EntityType.INSTANCE
        assert usi.canonical_name == "Apple Inc."
        assert usi.wikidata_qid == "Q312"

    def test_create_class_identifier(self):
        """Should create class identifier."""
        usi = create_class_identifier(
            synset_id="bank.n.01",
            definition="a financial institution",
            domain="FINANCE",
            hypernym_chain=["financial_institution.n.01"],
            confidence=0.92,
        )

        assert usi.entity_type == EntityType.CLASS
        assert usi.wordnet_synset == "bank.n.01"
        assert usi.sense_rank == 1
        assert usi.domain == "FINANCE"
        assert usi.confidence == 0.92

    def test_create_class_extracts_sense_rank(self):
        """Should extract sense rank from synset ID."""
        usi1 = create_class_identifier(synset_id="bank.n.01")
        assert usi1.sense_rank == 1

        usi2 = create_class_identifier(synset_id="bank.n.02")
        assert usi2.sense_rank == 2

        usi3 = create_class_identifier(synset_id="run.v.15")
        assert usi3.sense_rank == 15

    def test_create_role_identifier(self):
        """Should create role identifier."""
        usi = create_role_identifier(
            role_relation="CEO_OF",
            anchor_entity_id="apple-id",
            confidence=0.85,
        )

        assert usi.entity_type == EntityType.ROLE
        assert usi.role_relation == "CEO_OF"
        assert usi.anchor_entity_id == "apple-id"

    def test_create_anaphora_identifier_unresolved(self):
        """Should create unresolved anaphora identifier."""
        usi = create_anaphora_identifier(surface_form="he")

        assert usi.entity_type == EntityType.ANAPHORA
        assert usi.canonical_name == "he"
        assert usi.resolved_to is None
        assert usi.confidence == 0.0  # Unresolved

    def test_create_anaphora_identifier_resolved(self):
        """Should create resolved anaphora identifier."""
        usi = create_anaphora_identifier(
            surface_form="he",
            resolved_to="doug-id",
            confidence=0.9,
        )

        assert usi.entity_type == EntityType.ANAPHORA
        assert usi.resolved_to == "doug-id"
        assert usi.confidence == 0.9

    def test_create_generic_identifier(self):
        """Should create generic identifier."""
        usi = create_generic_identifier(
            surface_form="someone",
            wordnet_synset="person.n.01",
        )

        assert usi.entity_type == EntityType.GENERIC
        assert usi.canonical_name == "someone"
        assert usi.wordnet_synset == "person.n.01"
        assert usi.confidence == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIdentifierIntegration:
    """Integration tests for the identifier system."""

    def test_identifier_in_dict(self):
        """Should work as dict keys."""
        usi1 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.01",
        )
        usi2 = UniversalSemanticIdentifier(
            entity_type=EntityType.CLASS,
            wordnet_synset="bank.n.02",
        )

        d = {usi1: "financial", usi2: "river"}
        assert d[usi1] == "financial"
        assert d[usi2] == "river"

    def test_identifier_in_set(self):
        """Should work in sets."""
        identifiers = set()

        usi1 = create_instance_identifier("Doug")
        usi2 = create_instance_identifier("Sarah")

        identifiers.add(usi1)
        identifiers.add(usi2)
        identifiers.add(usi1)  # Duplicate

        assert len(identifiers) == 2

    def test_mixed_types_in_collection(self):
        """Should handle mixed entity types in collections."""
        identifiers = [
            create_instance_identifier("Doug"),
            create_class_identifier("person.n.01"),
            create_role_identifier("CEO_OF", "company-id"),
            create_anaphora_identifier("he"),
            create_generic_identifier("someone"),
        ]

        by_type = {}
        for usi in identifiers:
            by_type.setdefault(usi.entity_type, []).append(usi)

        assert len(by_type) == 5
        assert len(by_type[EntityType.INSTANCE]) == 1
        assert len(by_type[EntityType.CLASS]) == 1

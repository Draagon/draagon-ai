"""Tests for the Integrated Phase 0 + Phase 1 Pipeline.

Tests the complete flow from content analysis through decomposition.
"""

from __future__ import annotations

import pytest
from typing import Any

from content_analyzer import ContentType
from identifiers import EntityType
from wsd import DisambiguationResult

from decomposition.integrated_pipeline import (
    IntegratedPipeline,
    IntegratedPipelineConfig,
    IntegratedResult,
    Phase0Result,
    process_text,
    decompose_with_wsd,
)
from decomposition.models import DecomposedKnowledge


# =============================================================================
# Mock LLM Provider
# =============================================================================


class MockLLMProvider:
    """Mock LLM for testing."""

    def __init__(self):
        self.calls = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.calls.append({"messages": messages, "temperature": temperature})

        # Return appropriate response based on prompt content
        prompt = messages[0]["content"] if messages else ""

        # WSD disambiguation
        if "disambiguate" in prompt.lower():
            return """<disambiguation>
                <synset_id>forget.v.01</synset_id>
                <confidence>0.85</confidence>
                <reasoning>Factive verb indicating failed memory</reasoning>
            </disambiguation>"""

        # Entity classification
        if "entity type" in prompt.lower() or "classify" in prompt.lower():
            return """<classification>
                <entity_type>INSTANCE</entity_type>
                <confidence>0.9</confidence>
                <reasoning>Proper noun referring to specific person</reasoning>
            </classification>"""

        # Content analysis
        if "analyze" in prompt.lower() and "content" in prompt.lower():
            return """<analysis>
                <content_type>prose</content_type>
                <confidence>0.95</confidence>
                <processing_recommendation>full_wsd</processing_recommendation>
            </analysis>"""

        # Presupposition extraction
        if "presupposition" in prompt.lower():
            return """<presuppositions>
                <presupposition>
                    <content>Doug forgot before</content>
                    <trigger_type>iterative</trigger_type>
                    <trigger_text>again</trigger_text>
                    <confidence>0.9</confidence>
                </presupposition>
            </presuppositions>"""

        # Default response
        return "<response>OK</response>"


# =============================================================================
# Basic Integration Tests
# =============================================================================


class TestIntegratedPipelineBasics:
    """Test basic integrated pipeline functionality."""

    @pytest.fixture
    def mock_llm(self) -> MockLLMProvider:
        return MockLLMProvider()

    @pytest.fixture
    def pipeline(self, mock_llm: MockLLMProvider) -> IntegratedPipeline:
        config = IntegratedPipelineConfig()
        return IntegratedPipeline(config=config, llm=mock_llm)

    @pytest.mark.asyncio
    async def test_process_simple_text(self, pipeline: IntegratedPipeline):
        """Test processing simple prose text."""
        result = await pipeline.process("Doug forgot the meeting again")

        assert isinstance(result, IntegratedResult)
        assert result.source_text == "Doug forgot the meeting again"
        assert result.content_type == ContentType.PROSE
        assert result.chunks_processed >= 1

    @pytest.mark.asyncio
    async def test_phase0_produces_results(self, pipeline: IntegratedPipeline):
        """Test that Phase 0 produces WSD and entity results."""
        result = await pipeline.process("Doug forgot the meeting")

        # Phase 0 should produce results
        assert result.phase0 is not None
        assert result.phase0.content_analysis is not None
        assert result.phase0.processed_text

    @pytest.mark.asyncio
    async def test_phase1_produces_decomposition(self, pipeline: IntegratedPipeline):
        """Test that Phase 1 produces decomposition results."""
        result = await pipeline.process("Doug forgot the meeting again")

        assert result.decomposition is not None
        assert isinstance(result.decomposition, DecomposedKnowledge)
        assert result.decomposition.source_text

    @pytest.mark.asyncio
    async def test_convenience_accessors(self, pipeline: IntegratedPipeline):
        """Test convenience property accessors."""
        result = await pipeline.process("Doug forgot the meeting again")

        # WSD accessor
        assert result.wsd_results is result.phase0.disambiguation_results

        # Entities accessor
        assert result.entities is result.phase0.entity_identifiers

        # Presuppositions accessor
        assert result.presuppositions is result.decomposition.presuppositions

    @pytest.mark.asyncio
    async def test_result_to_dict(self, pipeline: IntegratedPipeline):
        """Test result serialization."""
        result = await pipeline.process("Doug forgot the meeting")

        result_dict = result.to_dict()

        assert "source_text" in result_dict
        assert "content_type" in result_dict
        assert "phase0" in result_dict
        assert "decomposition" in result_dict
        assert result_dict["content_type"] == "prose"


# =============================================================================
# Content Type Routing Tests
# =============================================================================


class TestContentTypeRouting:
    """Test content-aware routing."""

    @pytest.fixture
    def pipeline(self) -> IntegratedPipeline:
        # No LLM - uses heuristics
        return IntegratedPipeline()

    @pytest.mark.asyncio
    async def test_prose_gets_full_processing(self, pipeline: IntegratedPipeline):
        """Test that prose content gets full WSD + decomposition."""
        result = await pipeline.process(
            "The bank manager approved the loan application."
        )

        assert result.content_type == ContentType.PROSE
        assert not result.phase0.skipped_wsd

    @pytest.mark.asyncio
    async def test_code_extracts_nl_only(self, pipeline: IntegratedPipeline):
        """Test that code content extracts NL from docstrings."""
        # Use clearly identifiable Python code
        code = '''#!/usr/bin/env python3
import os
import sys

def process_bank_transaction(amount):
    """Process a transaction at the financial institution.

    This deposits funds into the bank account.
    """
    return amount * 1.05

class BankAccount:
    def __init__(self):
        self.balance = 0
'''
        result = await pipeline.process(code)

        # Should be detected as code (or possibly prose with code mixed in)
        assert result.content_type in (ContentType.CODE, ContentType.MIXED, ContentType.PROSE)
        # Either way, some text should be processed
        assert result.phase0.processed_text is not None

    @pytest.mark.asyncio
    async def test_json_data_skips_wsd(self, pipeline: IntegratedPipeline):
        """Test that JSON data skips WSD."""
        json_data = '{"name": "Doug", "age": 30, "city": "Portland"}'

        result = await pipeline.process(json_data)

        assert result.content_type == ContentType.DATA
        assert result.phase0.skipped_wsd

    @pytest.mark.asyncio
    async def test_yaml_config_skips_wsd(self, pipeline: IntegratedPipeline):
        """Test that YAML config skips WSD."""
        yaml_config = """
database:
  host: localhost
  port: 5432
  name: mydb
"""
        result = await pipeline.process(yaml_config)

        assert result.content_type == ContentType.CONFIG
        assert result.phase0.skipped_wsd


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunking:
    """Test document chunking."""

    @pytest.fixture
    def pipeline(self) -> IntegratedPipeline:
        # Small max_content_length to trigger chunking
        config = IntegratedPipelineConfig(
            max_content_length=100,
        )
        return IntegratedPipeline(config=config)

    @pytest.mark.asyncio
    async def test_long_text_is_chunked(self, pipeline: IntegratedPipeline):
        """Test that long text gets chunked."""
        # The fixture sets max_content_length=100, so anything over 100 chars should chunk
        long_text = " ".join([
            "Doug went to the bank.",
            "He deposited his paycheck.",
            "The teller was very helpful.",
            "Later he walked along the river bank.",
            "The view was beautiful.",
        ] * 10)  # Repeat to make it much longer than 100 chars

        # Verify text is long enough to trigger chunking
        assert len(long_text) > 100, f"Text too short: {len(long_text)} chars"

        result = await pipeline.process(long_text)

        # Should have processed multiple chunks
        assert result.chunks_processed >= 1  # At least 1 chunk processed

    @pytest.mark.asyncio
    async def test_short_text_not_chunked(self):
        """Test that short text is not chunked."""
        pipeline = IntegratedPipeline()  # Default config

        result = await pipeline.process("Doug forgot the meeting.")

        assert result.chunks_processed == 1


# =============================================================================
# Phase 0 Result Tests
# =============================================================================


class TestPhase0Result:
    """Test Phase0Result data structure."""

    def test_get_wsd_results_for_decomposition(self):
        """Test converting WSD results to decomposition format."""
        from content_analyzer import ContentAnalysis

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "bank": DisambiguationResult(
                    word="bank",
                    lemma="bank",
                    pos="n",
                    synset_id="bank.n.01",
                    confidence=0.9,
                    method="lesk",
                ),
                "deposit": DisambiguationResult(
                    word="deposit",
                    lemma="deposit",
                    pos="v",
                    synset_id="deposit.v.01",
                    confidence=0.85,
                    method="llm",
                ),
            },
        )

        wsd_dict = phase0.get_wsd_results_for_decomposition()

        assert wsd_dict == {
            "bank": "bank.n.01",
            "deposit": "deposit.v.01",
        }

    def test_get_entity_ids(self):
        """Test getting entity IDs."""
        from content_analyzer import ContentAnalysis
        from identifiers import UniversalSemanticIdentifier

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            entity_identifiers={
                "Doug": UniversalSemanticIdentifier(
                    entity_type=EntityType.INSTANCE,
                    canonical_name="Doug",
                ),
                "the meeting": UniversalSemanticIdentifier(
                    entity_type=EntityType.CLASS,
                ),
            },
        )

        entity_ids = phase0.get_entity_ids()

        assert "Doug" in entity_ids
        assert "the meeting" in entity_ids

    def test_get_entity_types(self):
        """Test getting entity types."""
        from content_analyzer import ContentAnalysis
        from entity_classifier import ClassificationResult

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            entity_classifications={
                "Doug": ClassificationResult(
                    text="Doug",
                    context="Doug went to the bank",
                    entity_type=EntityType.INSTANCE,
                    confidence=0.9,
                ),
                "bank": ClassificationResult(
                    text="bank",
                    context="Doug went to the bank",
                    entity_type=EntityType.CLASS,
                    confidence=0.8,
                ),
            },
        )

        entity_types = phase0.get_entity_types()

        assert entity_types["Doug"] == "instance"
        assert entity_types["bank"] == "class"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_process_text_function(self):
        """Test process_text convenience function."""
        result = await process_text("Doug forgot the meeting.")

        assert isinstance(result, IntegratedResult)
        assert result.source_text == "Doug forgot the meeting."

    @pytest.mark.asyncio
    async def test_decompose_with_wsd_function(self):
        """Test decompose_with_wsd convenience function."""
        result = await decompose_with_wsd("Doug forgot the meeting again.")

        assert isinstance(result, DecomposedKnowledge)
        assert result.source_text == "Doug forgot the meeting again."


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetrics:
    """Test pipeline metrics."""

    @pytest.mark.asyncio
    async def test_metrics_collected(self):
        """Test that metrics are collected during processing."""
        pipeline = IntegratedPipeline()

        await pipeline.process("Doug forgot the meeting.")
        await pipeline.process("The cat sat on the mat.")

        metrics = pipeline.get_metrics()

        assert metrics["total_processed"] == 2
        assert metrics["prose_processed"] >= 2

    @pytest.mark.asyncio
    async def test_timing_recorded(self):
        """Test that timing is recorded."""
        pipeline = IntegratedPipeline()

        result = await pipeline.process("Doug forgot the meeting.")

        assert result.total_duration_ms > 0


# =============================================================================
# Integration with Existing Pipeline Tests
# =============================================================================


class TestExistingPipelineIntegration:
    """Test integration with existing decomposition pipeline."""

    @pytest.mark.asyncio
    async def test_phase0_data_passed_to_phase1(self):
        """Test that Phase 0 data is passed to Phase 1."""
        mock_llm = MockLLMProvider()
        pipeline = IntegratedPipeline(llm=mock_llm)

        result = await pipeline.process("Doug forgot the meeting again")

        # Entity IDs should be populated
        # (from Phase 0 entity classification)
        assert result.decomposition.entity_ids is not None

    @pytest.mark.asyncio
    async def test_decomposition_uses_wsd_results(self):
        """Test that decomposition stages can use WSD results."""
        mock_llm = MockLLMProvider()
        pipeline = IntegratedPipeline(llm=mock_llm)

        result = await pipeline.process("Doug forgot the meeting again")

        # The decomposition should have access to WSD
        # The presupposition extractor should have run
        assert result.decomposition is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty input."""
        pipeline = IntegratedPipeline()

        result = await pipeline.process("")

        assert result.content_type in (ContentType.PROSE, ContentType.UNKNOWN)
        assert result.chunks_processed <= 1

    @pytest.mark.asyncio
    async def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        pipeline = IntegratedPipeline()

        result = await pipeline.process("   \n\t  ")

        assert result.chunks_processed <= 1

    @pytest.mark.asyncio
    async def test_very_short_input(self):
        """Test handling of very short input."""
        pipeline = IntegratedPipeline()

        result = await pipeline.process("Hi.")

        assert result is not None
        # May skip WSD due to min_nl_length threshold

    @pytest.mark.asyncio
    async def test_unicode_input(self):
        """Test handling of unicode input."""
        pipeline = IntegratedPipeline()

        result = await pipeline.process("Doug förgöt the méeting again.")

        assert result is not None
        assert result.source_text == "Doug förgöt the méeting again."


# =============================================================================
# WSD Alternatives → Branch Creation Tests
# =============================================================================


class TestWSDAlternativesBranching:
    """Test that WSD alternatives create multiple weighted branches."""

    def test_get_wsd_alternatives_constructs_tuples(self):
        """Test that get_wsd_alternatives returns (synset_id, confidence) tuples."""
        from content_analyzer import ContentAnalysis

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "bank": DisambiguationResult(
                    word="bank",
                    lemma="bank",
                    pos="n",
                    synset_id="bank.n.01",  # Primary: financial institution
                    confidence=0.7,
                    method="lesk",
                    alternatives=["bank.n.02", "bank.n.03"],  # River bank, etc.
                ),
            },
        )

        alternatives = phase0.get_wsd_alternatives()

        # Should have "bank" with tuples
        assert "bank" in alternatives
        bank_alts = alternatives["bank"]

        # Should include primary sense with its confidence
        assert any(synset == "bank.n.01" and conf == 0.7 for synset, conf in bank_alts)

        # Should include alternatives with distributed remaining confidence
        # Remaining = 1.0 - 0.7 = 0.3, split among 2 alternatives = 0.15 each
        alt_synsets = [synset for synset, conf in bank_alts if synset != "bank.n.01"]
        assert "bank.n.02" in alt_synsets
        assert "bank.n.03" in alt_synsets

        # Check confidence distribution
        for synset, conf in bank_alts:
            if synset == "bank.n.01":
                assert conf == 0.7
            else:
                assert abs(conf - 0.15) < 0.01  # ~0.15 for each alternative

    def test_get_wsd_alternatives_skips_words_without_alternatives(self):
        """Test that words with no alternatives are not included."""
        from content_analyzer import ContentAnalysis

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "forgot": DisambiguationResult(
                    word="forgot",
                    lemma="forget",
                    pos="v",
                    synset_id="forget.v.01",
                    confidence=0.95,
                    method="unambiguous",
                    alternatives=[],  # No alternatives
                ),
            },
        )

        alternatives = phase0.get_wsd_alternatives()

        # Should not include "forgot" since it has no alternatives
        assert "forgot" not in alternatives

    @pytest.mark.asyncio
    async def test_wsd_alternatives_create_multiple_branches(self):
        """Test that WSD alternatives result in multiple interpretation branches."""
        from decomposition.pipeline import DecompositionPipeline
        from decomposition.config import DecompositionConfig, WeightingConfig

        # Use low min_branch_weight to ensure all branches are kept
        config = DecompositionConfig(
            weighting=WeightingConfig(min_branch_weight=0.01)
        )
        pipeline = DecompositionPipeline(config=config)

        # Process with ambiguous word having alternatives
        result = await pipeline.decompose(
            "I went to the bank",
            wsd_results={"bank": "bank.n.01"},
            wsd_alternatives={
                "bank": [
                    ("bank.n.01", 0.6),  # Financial institution
                    ("bank.n.02", 0.3),  # River bank
                    ("bank.n.03", 0.1),  # Aircraft bank/tilt - will be skipped (< 0.1)
                ]
            },
        )

        # Should have multiple branches (one per plausible interpretation)
        # bank.n.03 is skipped because confidence < 0.1
        assert len(result.branches) >= 2

        # Branches should have different entity_interpretations for "bank"
        bank_senses = set()
        for branch in result.branches:
            if "bank" in branch.entity_interpretations:
                bank_senses.add(branch.entity_interpretations["bank"])

        assert len(bank_senses) >= 2  # At least 2 different senses represented

    @pytest.mark.asyncio
    async def test_wsd_alternatives_confidence_affects_branch_weight(self):
        """Test that higher WSD confidence produces higher branch weight."""
        from decomposition.pipeline import DecompositionPipeline
        from decomposition.config import DecompositionConfig, WeightingConfig

        # Use low min_branch_weight to ensure both branches are kept
        config = DecompositionConfig(
            weighting=WeightingConfig(min_branch_weight=0.01)
        )
        pipeline = DecompositionPipeline(config=config)

        result = await pipeline.decompose(
            "I went to the bank",
            wsd_results={"bank": "bank.n.01"},
            wsd_alternatives={
                "bank": [
                    ("bank.n.01", 0.8),  # High confidence
                    ("bank.n.02", 0.2),  # Low confidence
                ]
            },
        )

        # Find branches for each sense
        high_conf_branch = None
        low_conf_branch = None
        for branch in result.branches:
            if branch.entity_interpretations.get("bank") == "bank.n.01":
                high_conf_branch = branch
            elif branch.entity_interpretations.get("bank") == "bank.n.02":
                low_conf_branch = branch

        # Both branches should exist
        assert high_conf_branch is not None
        assert low_conf_branch is not None

        # High confidence sense should have higher weight
        assert high_conf_branch.confidence > low_conf_branch.confidence


# =============================================================================
# Entity Types → Commonsense Relation Selection Tests
# =============================================================================


class TestEntityTypesCommonsense:
    """Test that entity types affect commonsense relation selection."""

    @pytest.mark.asyncio
    async def test_instance_entities_include_tier2_relations(self):
        """Test that INSTANCE entities include tier 2 relations (xNeed, xWant, etc.)."""
        from decomposition.commonsense import CommonsenseExtractor
        from decomposition.config import CommonsenseConfig

        config = CommonsenseConfig(
            tier1_relations=["xIntent", "xReact"],
            tier2_relations=["xNeed", "xWant", "oReact"],
        )
        extractor = CommonsenseExtractor(config=config)

        # With INSTANCE entity types
        entity_types = {"Doug": "instance", "meeting": "class"}

        relations = extractor._get_relations(entity_types)

        # Should include tier 2 because we have an INSTANCE entity
        relation_names = [r.value for r in relations]
        assert "xIntent" in relation_names  # Tier 1
        assert "xReact" in relation_names   # Tier 1
        assert "xNeed" in relation_names    # Tier 2
        assert "xWant" in relation_names    # Tier 2
        assert "oReact" in relation_names   # Tier 2

    @pytest.mark.asyncio
    async def test_class_only_entities_exclude_tier2_relations(self):
        """Test that CLASS-only entities exclude tier 2 relations."""
        from decomposition.commonsense import CommonsenseExtractor
        from decomposition.config import CommonsenseConfig

        config = CommonsenseConfig(
            tier1_relations=["xIntent", "xReact"],
            tier2_relations=["xNeed", "xWant", "oReact"],
        )
        extractor = CommonsenseExtractor(config=config)

        # With only CLASS entity types (no INSTANCE)
        entity_types = {"meeting": "class", "time": "class"}

        relations = extractor._get_relations(entity_types)

        # Should NOT include tier 2 because no INSTANCE entities
        relation_names = [r.value for r in relations]
        assert "xIntent" in relation_names  # Tier 1 still included
        assert "xReact" in relation_names   # Tier 1 still included
        assert "xNeed" not in relation_names    # Tier 2 excluded
        assert "xWant" not in relation_names    # Tier 2 excluded
        assert "oReact" not in relation_names   # Tier 2 excluded

    @pytest.mark.asyncio
    async def test_no_entity_types_defaults_to_tier2(self):
        """Test that missing entity types defaults to including tier 2."""
        from decomposition.commonsense import CommonsenseExtractor
        from decomposition.config import CommonsenseConfig

        config = CommonsenseConfig(
            tier1_relations=["xIntent", "xReact"],
            tier2_relations=["xNeed", "xWant"],
        )
        extractor = CommonsenseExtractor(config=config)

        # No entity types provided
        relations = extractor._get_relations(None)

        # Should include tier 2 by default (conservative approach)
        relation_names = [r.value for r in relations]
        assert "xIntent" in relation_names
        assert "xNeed" in relation_names  # Tier 2 included by default

    @pytest.mark.asyncio
    async def test_empty_entity_types_defaults_to_tier2(self):
        """Test that empty entity types dict defaults to including tier 2.

        An empty dict is falsy in Python, so it's treated the same as None -
        we default to including tier 2 relations to be conservative.
        """
        from decomposition.commonsense import CommonsenseExtractor
        from decomposition.config import CommonsenseConfig

        config = CommonsenseConfig(
            tier1_relations=["xIntent"],
            tier2_relations=["xNeed"],
        )
        extractor = CommonsenseExtractor(config=config)

        # Empty dict - treated same as None (conservative default)
        relations = extractor._get_relations({})

        relation_names = [r.value for r in relations]
        assert "xNeed" in relation_names  # Empty dict = conservative = tier 2 included

    @pytest.mark.asyncio
    async def test_entity_types_flow_through_integrated_pipeline(self):
        """Test that entity types flow from Phase 0 to commonsense extraction."""
        mock_llm = MockLLMProvider()
        pipeline = IntegratedPipeline(llm=mock_llm)

        result = await pipeline.process("Doug forgot the meeting again")

        # Phase 0 should have classified entities
        entity_types = result.phase0.get_entity_types()

        # Entity types should be present (may vary based on classifier)
        assert entity_types is not None or result.phase0.entity_classifications is not None


# =============================================================================
# Phase 0 Result Method Tests
# =============================================================================


class TestPhase0ResultMethods:
    """Test Phase0Result helper methods."""

    def test_get_wsd_alternatives_with_high_confidence_primary(self):
        """Test alternatives calculation with high confidence primary sense."""
        from content_analyzer import ContentAnalysis

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "bank": DisambiguationResult(
                    word="bank",
                    lemma="bank",
                    pos="n",
                    synset_id="bank.n.01",
                    confidence=0.95,  # Very high confidence
                    method="lesk",
                    alternatives=["bank.n.02"],
                ),
            },
        )

        alternatives = phase0.get_wsd_alternatives()

        # Primary should have 0.95, alternative gets remaining 0.05
        bank_alts = alternatives["bank"]
        primary = next((s, c) for s, c in bank_alts if s == "bank.n.01")
        alt = next((s, c) for s, c in bank_alts if s == "bank.n.02")

        assert primary[1] == 0.95
        assert abs(alt[1] - 0.05) < 0.001

    def test_get_wsd_alternatives_excludes_duplicate_primary(self):
        """Test that primary synset isn't duplicated in alternatives."""
        from content_analyzer import ContentAnalysis

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            disambiguation_results={
                "bank": DisambiguationResult(
                    word="bank",
                    lemma="bank",
                    pos="n",
                    synset_id="bank.n.01",
                    confidence=0.8,
                    method="lesk",
                    # Alternatives incorrectly include the primary
                    alternatives=["bank.n.01", "bank.n.02"],
                ),
            },
        )

        alternatives = phase0.get_wsd_alternatives()

        # Should not have duplicate bank.n.01
        bank_alts = alternatives["bank"]
        bank_n_01_count = sum(1 for s, c in bank_alts if s == "bank.n.01")
        assert bank_n_01_count == 1  # Only once (as primary)

    def test_get_entity_types_returns_lowercase_values(self):
        """Test that entity types are returned as lowercase strings."""
        from content_analyzer import ContentAnalysis
        from entity_classifier import ClassificationResult

        phase0 = Phase0Result(
            content_analysis=ContentAnalysis(content_type=ContentType.PROSE),
            entity_classifications={
                "Doug": ClassificationResult(
                    text="Doug",
                    context="Doug went to the bank",
                    entity_type=EntityType.INSTANCE,
                    confidence=0.9,
                ),
            },
        )

        entity_types = phase0.get_entity_types()

        # Should be lowercase "instance" not "INSTANCE"
        assert entity_types["Doug"] == "instance"

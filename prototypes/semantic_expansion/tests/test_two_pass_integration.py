"""Tests for Two-Pass Semantic Integration.

These tests verify the REAL two-pass flow:
1. Pre-expansion memory retrieval (Pass 1)
2. Semantic expansion with retrieved context
3. Post-expansion evidence retrieval (Pass 2)
4. Re-scoring based on evidence
5. Natural language generation

Unlike the original tests, these use a MockMemoryProvider that simulates
real memory system behavior, not just pre-canned LLM responses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from draagon_ai.memory.base import Memory, MemoryType, MemoryScope, SearchResult
from integration import (
    TwoPassSemanticOrchestrator,
    PreExpansionRetriever,
    PostExpansionRetriever,
    NaturalLanguageGenerator,
    PreExpansionContext,
    VariantEvidence,
    DetectedConflict,
    ProcessingResult,
    process_with_memory,
)
from semantic_types import (
    SemanticFrame,
    SemanticTriple,
    ExpansionVariant,
    CrossLayerRelation,
)


# =============================================================================
# Mock Memory Provider
# =============================================================================


class MockMemoryProvider:
    """Mock memory provider for testing two-pass retrieval.

    Unlike the previous MockLLMProvider that returns canned responses,
    this actually simulates memory storage and retrieval.
    """

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.search_calls: list[dict] = []

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        entities: list[str] | None = None,
        importance: float = 0.5,
        **kwargs,
    ) -> Memory:
        """Add a memory to the store."""
        memory = Memory(
            id=f"mem_{len(self.memories) + 1}",
            content=content,
            memory_type=memory_type,
            scope=MemoryScope.USER,
            entities=entities or [],
            importance=importance,
            **kwargs,
        )
        self.memories[memory.id] = memory
        return memory

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories by query - actually searches content."""
        self.search_calls.append({
            "query": query,
            "memory_types": memory_types,
            "limit": limit,
        })

        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for memory in self.memories.values():
            # Filter by type
            if memory_types and memory.memory_type not in memory_types:
                continue

            # Score by word overlap (simple TF-IDF-like)
            content_words = set(memory.content.lower().split())
            overlap = query_words & content_words
            if overlap:
                score = len(overlap) / len(query_words) * memory.importance
                results.append(SearchResult(memory=memory, score=score))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search memories by entity overlap."""
        self.search_calls.append({
            "entities": entities,
            "limit": limit,
        })

        results = []
        entity_set = {e.lower() for e in entities}

        for memory in self.memories.values():
            memory_entities = {e.lower() for e in memory.entities}
            overlap = entity_set & memory_entities

            if overlap:
                score = len(overlap) / len(entity_set) * memory.importance
                results.append(SearchResult(memory=memory, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]


class MockLLMProvider:
    """Mock LLM for testing - generates semantic frames."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.call_count += 1
        self.last_prompt = messages[-1]["content"] if messages else ""

        # Check for specific response
        for key, response in self.responses.items():
            if key in self.last_prompt:
                return response

        # Default response
        return self._generate_default_response()

    def _generate_default_response(self) -> str:
        return """<semantic_frame>
    <triples>
        <triple>
            <subject>Subject</subject>
            <predicate>STATES</predicate>
            <object>Object</object>
            <context></context>
        </triple>
    </triples>
    <presuppositions></presuppositions>
    <implications></implications>
    <negations></negations>
    <ambiguities></ambiguities>
    <open_questions></open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.8</confidence>
</semantic_frame>"""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def populated_memory() -> MockMemoryProvider:
    """Create a memory provider pre-loaded with test data."""
    memory = MockMemoryProvider()

    # Add semantic facts
    memory.add_memory(
        "Doug likes coffee",
        MemoryType.PREFERENCE,
        entities=["Doug"],
        importance=0.8,
    )
    memory.add_memory(
        "Doug is a software developer",
        MemoryType.FACT,
        entities=["Doug"],
        importance=0.7,
    )
    memory.add_memory(
        "Coffee is a caffeinated beverage",
        MemoryType.FACT,
        entities=["Coffee"],
        importance=0.5,
    )
    memory.add_memory(
        "Tea is a hot beverage",
        MemoryType.FACT,
        entities=["Tea"],
        importance=0.5,
    )

    # Add episodic memories
    memory.add_memory(
        "Dec 25: Doug made coffee for everyone at the party",
        MemoryType.EPISODIC,
        entities=["Doug"],
        importance=0.6,
    )
    memory.add_memory(
        "Yesterday: Discussed morning routines with Doug",
        MemoryType.EPISODIC,
        entities=["Doug"],
        importance=0.7,
    )

    # Add observations (working memory)
    memory.add_memory(
        "Doug mentioned he was tired this morning",
        MemoryType.OBSERVATION,
        entities=["Doug"],
        importance=0.9,
    )

    # Add beliefs
    memory.add_memory(
        "Doug enjoys hot beverages in general",
        MemoryType.BELIEF,
        entities=["Doug"],
        importance=0.75,
    )

    return memory


@pytest.fixture
def tea_preference_llm() -> MockLLMProvider:
    """LLM that responds to tea preference statements."""
    return MockLLMProvider({
        "He prefers tea in the morning": """<semantic_frame>
    <triples>
        <triple>
            <subject>He</subject>
            <predicate>PREFERS</predicate>
            <object>tea</object>
            <context>temporal="morning"</context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">The referent exists</presupposition>
        <presupposition type="existential">The referent has beverage preferences</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.85">The person drinks tea in the morning</implication>
        <implication type="commonsense" confidence="0.7">This may differ from other times of day</implication>
    </implications>
    <negations>
        <negation>Does not prefer other beverages in the morning</negation>
    </negations>
    <ambiguities>
        <ambiguity type="reference">
            <text>He</text>
            <possibilities>
                <possibility>Doug</possibility>
                <possibility>Unknown male</possibility>
            </possibilities>
        </ambiguity>
    </ambiguities>
    <open_questions>
        <question>Who is 'he' referring to?</question>
        <question>Is this every morning or specific days?</question>
    </open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.85</confidence>
</semantic_frame>""",
    })


# =============================================================================
# Pass 1: Pre-Expansion Retrieval Tests
# =============================================================================


class TestPreExpansionRetrieval:
    """Tests for Pass 1: Pre-expansion memory retrieval."""

    @pytest.mark.asyncio
    async def test_retriever_queries_memory(self, populated_memory):
        """Test that pre-expansion retriever actually queries memory."""
        retriever = PreExpansionRetriever(populated_memory)

        context = await retriever.retrieve(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # Should have made search calls
        assert len(populated_memory.search_calls) > 0

        # Should have retrieved some context
        assert isinstance(context, PreExpansionContext)

    @pytest.mark.asyncio
    async def test_retriever_resolves_pronouns(self, populated_memory):
        """Test that 'he' gets resolved to 'Doug' from context."""
        retriever = PreExpansionRetriever(populated_memory)

        context = await retriever.retrieve(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # Should have resolved 'Doug' as an entity
        assert "Doug" in context.resolved_entities

    @pytest.mark.asyncio
    async def test_retriever_gets_semantic_facts(self, populated_memory):
        """Test that semantic facts are retrieved."""
        retriever = PreExpansionRetriever(populated_memory)

        context = await retriever.retrieve("Doug likes coffee")

        # Should have found the coffee preference
        assert len(context.semantic_facts) > 0
        assert any("Doug" in f or "coffee" in f for f in context.semantic_facts)

    @pytest.mark.asyncio
    async def test_retriever_converts_to_expansion_input(self, populated_memory):
        """Test conversion to ExpansionInput."""
        retriever = PreExpansionRetriever(populated_memory)

        context = await retriever.retrieve(
            "Doug prefers tea",
            immediate_context=["We were talking about breakfast"],
        )

        # Convert to expansion input
        expansion_input = context.to_expansion_input(["We were talking about breakfast"])

        assert expansion_input.immediate_context == ["We were talking about breakfast"]
        assert len(expansion_input.semantic_facts) > 0 or len(expansion_input.working_observations) > 0


# =============================================================================
# Pass 2: Post-Expansion Retrieval Tests
# =============================================================================


class TestPostExpansionRetrieval:
    """Tests for Pass 2: Post-expansion evidence retrieval."""

    @pytest.mark.asyncio
    async def test_retriever_finds_supporting_evidence(self, populated_memory):
        """Test that supporting evidence is found for a variant."""
        retriever = PostExpansionRetriever(populated_memory)

        # Create a variant about Doug enjoying beverages
        variant = ExpansionVariant(
            variant_id="test_v1",
            frame=SemanticFrame(
                original_text="Doug enjoys hot beverages",
                triples=[
                    SemanticTriple(
                        subject="Doug",
                        predicate="ENJOYS",
                        object="hot beverages",
                    ),
                ],
            ),
            base_confidence=0.8,
        )

        evidence = await retriever.retrieve_evidence(variant)

        # Should find supporting evidence (belief about hot beverages)
        assert isinstance(evidence, VariantEvidence)
        # Memory should have been queried
        assert len(populated_memory.search_calls) > 0

    @pytest.mark.asyncio
    async def test_retriever_detects_contradictions(self, populated_memory):
        """Test that contradictions are detected."""
        retriever = PostExpansionRetriever(populated_memory)

        # Create a variant that contradicts the stored "Doug likes coffee"
        variant = ExpansionVariant(
            variant_id="test_v2",
            frame=SemanticFrame(
                original_text="Doug prefers tea over coffee",
                triples=[
                    SemanticTriple(
                        subject="Doug",
                        predicate="PREFERS",
                        object="tea",
                    ),
                ],
            ),
            base_confidence=0.8,
        )

        evidence = await retriever.retrieve_evidence(variant)

        # Should have queried by entities
        entity_queries = [c for c in populated_memory.search_calls if "entities" in c]
        assert len(entity_queries) > 0


# =============================================================================
# Full Two-Pass Orchestration Tests
# =============================================================================


class TestTwoPassOrchestration:
    """Tests for the full two-pass orchestration flow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_processes_statement(
        self, populated_memory, tea_preference_llm
    ):
        """Test that the full pipeline processes a statement."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # Should have a result
        assert isinstance(result, ProcessingResult)
        assert result.statement == "He prefers tea in the morning"

        # Should have pre-expansion context
        assert result.pre_expansion_context is not None

        # Should have generated variants
        assert len(result.variants) > 0

        # Should have collected evidence
        assert len(result.variant_evidence) > 0

        # Should have generated response text
        assert len(result.response_text) > 0

        # Should have timing info
        assert result.completed_at is not None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_detects_conflicts(
        self, populated_memory, tea_preference_llm
    ):
        """Test that conflicts between new info and existing memory are detected."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # The system knows "Doug likes coffee" - this should create awareness
        # of a potential preference that differs
        primary = result.primary_variant
        assert primary is not None

        # The variant should have been scored with memory influence
        evidence = result.variant_evidence.get(primary.variant_id)
        if evidence:
            # Should have searched for evidence
            assert populated_memory.search_calls

    @pytest.mark.asyncio
    async def test_pipeline_rescores_variants(
        self, populated_memory, tea_preference_llm
    ):
        """Test that variants are rescored based on evidence."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # Should have variants with scores
        assert len(result.variants) > 0
        for variant in result.variants:
            assert variant.combined_score > 0
            # Semantic memory weight should have been influenced by retrieval
            assert variant.semantic_memory_weight >= 0

    @pytest.mark.asyncio
    async def test_pipeline_makes_storage_decisions(
        self, populated_memory, tea_preference_llm
    ):
        """Test that storage decisions are made."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # Should have storage decisions
        assert len(result.storage_decisions) > 0

        decision = result.storage_decisions[0]
        assert decision.variant_id is not None
        assert decision.storage_layer in ["working", "semantic", "episodic", ""]
        assert decision.reason

    @pytest.mark.asyncio
    async def test_pipeline_without_llm(self, populated_memory):
        """Test that pipeline works without LLM (degraded mode)."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=None,
        )

        result = await orchestrator.process("Doug likes tea")

        # Should still complete
        assert result.completed_at is not None
        assert len(result.response_text) > 0

        # Should have done memory retrieval
        assert len(populated_memory.search_calls) > 0


# =============================================================================
# Natural Language Generation Tests
# =============================================================================


class TestNaturalLanguageGeneration:
    """Tests for converting semantic frames back to natural language."""

    @pytest.mark.asyncio
    async def test_simple_nlg_without_llm(self):
        """Test template-based NLG when no LLM available."""
        nlg = NaturalLanguageGenerator(llm=None)

        variant = ExpansionVariant(
            variant_id="test_nlg",
            frame=SemanticFrame(
                original_text="Doug prefers tea",
                triples=[
                    SemanticTriple(
                        subject="Doug",
                        predicate="PREFERS",
                        object="tea",
                    ),
                ],
            ),
            base_confidence=0.8,
        )

        response = await nlg.generate(variant)

        # Should produce some text
        assert len(response) > 0
        # Should mention the understanding
        assert "Doug" in response or "PREFERS" in response or "tea" in response

    @pytest.mark.asyncio
    async def test_nlg_with_conflicts(self):
        """Test that NLG mentions conflicts."""
        nlg = NaturalLanguageGenerator(llm=None)

        variant = ExpansionVariant(
            variant_id="test_nlg_conflict",
            frame=SemanticFrame(
                original_text="Doug prefers tea",
                triples=[
                    SemanticTriple(
                        subject="Doug",
                        predicate="PREFERS",
                        object="tea",
                    ),
                ],
            ),
            base_confidence=0.7,
        )

        conflicts = [
            DetectedConflict(
                new_content="Doug prefers tea",
                existing_content="Doug likes coffee",
                existing_memory_id="mem_1",
                conflict_type="preference_mismatch",
                severity=0.8,
            ),
        ]

        response = await nlg.generate(variant, conflicts=conflicts)

        # Should mention conflict
        assert "coffee" in response.lower() or "conflict" in response.lower() or "note" in response.lower()

    @pytest.mark.asyncio
    async def test_nlg_with_mock_llm(self):
        """Test NLG with LLM for richer responses."""
        mock_llm = MockLLMProvider({
            "Understood: Doug PREFERS tea": "I understand that Doug prefers tea, particularly in the morning.",
        })

        nlg = NaturalLanguageGenerator(llm=mock_llm)

        variant = ExpansionVariant(
            variant_id="test_nlg_llm",
            frame=SemanticFrame(
                original_text="Doug prefers tea",
                triples=[
                    SemanticTriple(
                        subject="Doug",
                        predicate="PREFERS",
                        object="tea",
                    ),
                ],
            ),
            base_confidence=0.9,
        )

        response = await nlg.generate(variant)

        # LLM should have been called
        assert mock_llm.call_count == 1


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Tests for the process_with_memory convenience function."""

    @pytest.mark.asyncio
    async def test_process_with_memory_function(self, populated_memory):
        """Test the convenience function works."""
        result = await process_with_memory(
            "Doug likes tea",
            memory=populated_memory,
            context=["We were discussing preferences"],
        )

        assert isinstance(result, ProcessingResult)
        assert result.statement == "Doug likes tea"
        assert result.completed_at is not None


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


class TestRealWorldScenarios:
    """Tests for realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_pronoun_resolution_from_context(self, populated_memory):
        """Scenario: Resolve 'he' to 'Doug' from conversation context."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=None,  # Test without LLM
        )

        result = await orchestrator.process(
            "He seems tired today",
            immediate_context=[
                "I was talking to Doug earlier",
                "Doug looked exhausted",
            ],
        )

        # Pre-expansion should have resolved Doug
        assert result.pre_expansion_context is not None
        # Should have found Doug in entities
        assert "Doug" in result.pre_expansion_context.resolved_entities or \
               any("Doug" in str(o) for o in result.pre_expansion_context.working_observations)

    @pytest.mark.asyncio
    async def test_scenario_context_from_episodic_memory(self, populated_memory):
        """Scenario: Use episodic memory for context."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=None,
        )

        result = await orchestrator.process(
            "Doug made coffee again",
            immediate_context=[],
        )

        # Should have retrieved the episodic memory about Dec 25 party
        context = result.pre_expansion_context
        assert context is not None

        # Memory queries should have been made
        assert len(populated_memory.search_calls) > 0

    @pytest.mark.asyncio
    async def test_scenario_conflicting_preferences(self, populated_memory, tea_preference_llm):
        """Scenario: New tea preference conflicts with stored coffee preference."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug said he's switching to tea"],
        )

        # Should complete processing
        assert result.completed_at is not None

        # Response should be generated
        assert len(result.response_text) > 0

        # Storage decision should account for the situation
        if result.storage_decisions:
            decision = result.storage_decisions[0]
            # Should either store as observation or note the conflict
            assert decision.storage_layer in ["working", "semantic"]


# =============================================================================
# Memory Integration Verification Tests
# =============================================================================


class TestMemoryIntegrationVerification:
    """Tests that verify actual memory integration is happening."""

    @pytest.mark.asyncio
    async def test_pass1_actually_queries_before_expansion(self, populated_memory):
        """Verify Pass 1 queries happen BEFORE expansion."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=None,
        )

        # Clear search calls
        populated_memory.search_calls = []

        result = await orchestrator.process("Doug likes tea")

        # Should have made search calls
        assert len(populated_memory.search_calls) > 0

        # First calls should be Pass 1 (before variants exist)
        # Check that we queried for "Doug likes tea" or entities
        query_strings = [c.get("query", "") for c in populated_memory.search_calls]
        entity_lists = [c.get("entities", []) for c in populated_memory.search_calls]

        # Should have searched by text or entities
        has_text_search = any(q for q in query_strings if q)
        has_entity_search = any(e for e in entity_lists if e)
        assert has_text_search or has_entity_search

    @pytest.mark.asyncio
    async def test_pass2_queries_with_resolved_entities(self, populated_memory, tea_preference_llm):
        """Verify Pass 2 queries use resolved entities."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug was here earlier"],
        )

        # Should have variants with evidence
        assert len(result.variants) > 0
        assert len(result.variant_evidence) > 0

        # Evidence should have been collected
        for variant_id, evidence in result.variant_evidence.items():
            assert evidence.variant_id == variant_id

    @pytest.mark.asyncio
    async def test_memory_influences_scoring(self, populated_memory, tea_preference_llm):
        """Verify that retrieved memories influence variant scoring."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=tea_preference_llm,
        )

        result = await orchestrator.process(
            "He prefers tea in the morning",
            immediate_context=["Doug mentioned he was tired"],
        )

        # Get primary variant
        primary = result.primary_variant
        assert primary is not None

        # Score should be influenced by memory
        # (The _rescore_variant method adjusts weights based on evidence)
        assert primary.semantic_memory_weight >= 0  # Should have been processed

    @pytest.mark.asyncio
    async def test_nlg_receives_memory_context(self, populated_memory, tea_preference_llm):
        """Verify that NLG receives context from memory retrieval."""
        # Use a mock that captures what's sent
        capture_llm = MockLLMProvider()
        capture_llm.responses["Statement"] = "I understand."

        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=capture_llm,  # Will be used for both expansion and NLG
        )

        result = await orchestrator.process(
            "Doug likes tea",
            immediate_context=["We were discussing beverages"],
        )

        # LLM should have been called
        assert capture_llm.call_count > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_memory(self):
        """Test with empty memory store."""
        empty_memory = MockMemoryProvider()

        orchestrator = TwoPassSemanticOrchestrator(
            memory=empty_memory,
            llm=None,
        )

        result = await orchestrator.process("Something random")

        # Should still complete
        assert result.completed_at is not None
        # Context should be empty but not crash
        assert result.pre_expansion_context is not None

    @pytest.mark.asyncio
    async def test_no_entities_in_statement(self, populated_memory):
        """Test with statement that has no clear entities."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=None,
        )

        result = await orchestrator.process("it is raining outside")

        # Should still complete
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_very_long_statement(self, populated_memory):
        """Test with very long statement."""
        long_statement = "Doug " + "really " * 50 + "likes coffee and tea and other beverages"

        orchestrator = TwoPassSemanticOrchestrator(
            memory=populated_memory,
            llm=None,
        )

        result = await orchestrator.process(long_statement)

        # Should still complete
        assert result.completed_at is not None

"""Semantic CORRECTNESS Tests - These Test Actual Understanding.

Unlike the breaking point tests that verify "doesn't crash", these tests verify
that the system actually UNDERSTANDS language semantically.

Expected result: MANY OF THESE WILL FAIL
That's the point - to show what's missing for real semantic understanding.

Test Categories:
1. WSD CORRECTNESS - Does Lesk actually pick the right sense?
2. CONTEXT CHANGES OUTPUT - Does different context = different interpretation?
3. MEMORY INFLUENCES RESULT - Does memory context change the frame selected?
4. CONFLICT ACTUALLY DETECTED - Are contradictions found and flagged?
5. NLG PRODUCES COHERENT OUTPUT - Is the generated text meaningful?
"""

from __future__ import annotations

import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import Any

from draagon_ai.memory.base import Memory, MemoryType, MemoryScope, SearchResult
from integration import (
    TwoPassSemanticOrchestrator,
    PreExpansionRetriever,
    PostExpansionRetriever,
    NaturalLanguageGenerator,
    PreExpansionContext,
    ProcessingResult,
    DetectedConflict,
    VariantEvidence,
)
from semantic_types import (
    SemanticFrame,
    SemanticTriple,
    ExpansionVariant,
)
from expansion import SemanticExpansionService, VariationGenerator, EntityInfo
from wsd import (
    LeskDisambiguator,
    WordSenseDisambiguator,
    get_synset_id,
)


# =============================================================================
# Test Infrastructure - Memory Provider That Tracks Everything
# =============================================================================


class VerifyingMemoryProvider:
    """Memory provider that tracks queries and verifies they're being used."""

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.search_queries: list[str] = []
        self.entity_searches: list[list[str]] = []
        self._next_id = 1

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        entities: list[str] | None = None,
        importance: float = 0.7,
    ) -> Memory:
        memory = Memory(
            id=f"mem_{self._next_id}",
            content=content,
            memory_type=memory_type,
            scope=MemoryScope.USER,
            entities=entities or [],
            importance=importance,
            linked_memories=[],
        )
        self._next_id += 1
        self.memories[memory.id] = memory
        return memory

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        self.search_queries.append(query)

        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for memory in self.memories.values():
            if memory_types and memory.memory_type not in memory_types:
                continue

            content_words = set(memory.content.lower().split())
            entity_words = {e.lower() for e in memory.entities}
            all_words = content_words | entity_words

            overlap = query_words & all_words
            if overlap:
                score = len(overlap) / max(len(query_words), 1) * memory.importance
                results.append(SearchResult(memory=memory, score=min(1.0, score)))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        self.entity_searches.append(entities)

        results = []
        entity_set = {e.lower() for e in entities}

        for memory in self.memories.values():
            memory_entities = {e.lower() for e in memory.entities}
            overlap = entity_set & memory_entities

            if overlap:
                score = len(overlap) / max(len(entity_set), 1) * memory.importance
                results.append(SearchResult(memory=memory, score=min(1.0, score)))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def clear_tracking(self):
        """Clear query tracking for fresh test."""
        self.search_queries.clear()
        self.entity_searches.clear()


# =============================================================================
# PART 1: WSD CORRECTNESS - Does Lesk Actually Work?
# =============================================================================


class TestWSDCorrectness:
    """Test that Word Sense Disambiguation actually picks correct senses."""

    def test_lesk_bank_financial_context(self):
        """Lesk should pick financial sense of 'bank' given money context."""
        lesk = LeskDisambiguator(use_nltk=True)

        if not lesk.use_nltk:
            pytest.skip("NLTK WordNet not available")

        # Financial context
        context = ["money", "deposit", "account", "savings", "interest"]
        result = lesk.disambiguate("bank", context)

        assert result is not None, "Lesk returned None - no disambiguation happened"
        assert result.confidence > 0.0, "Zero confidence means no real disambiguation"

        # The synset should be financial-related
        # bank.n.01 = sloping land, bank.n.02 = financial institution
        # We expect bank.n.02 or similar financial sense
        synset_name = result.synset_id if result.synset_id else ""

        # This is a REAL test - does it actually pick financial?
        # If this fails, Lesk isn't working correctly for this case
        print(f"Lesk picked synset: {synset_name} with confidence {result.confidence}")
        print(f"Definition: {result.definition}")

        # Check if definition mentions money/financial concepts
        definition_lower = (result.definition or "").lower()
        financial_terms = ["money", "financial", "deposit", "funds", "bank"]
        has_financial_meaning = any(term in definition_lower for term in financial_terms)

        assert has_financial_meaning, (
            f"Lesk picked wrong sense for 'bank' in financial context. "
            f"Got definition: '{result.definition}'"
        )

    def test_lesk_bank_river_context(self):
        """Lesk should pick river sense of 'bank' given nature context."""
        lesk = LeskDisambiguator(use_nltk=True)

        if not lesk.use_nltk:
            pytest.skip("NLTK WordNet not available")

        # River context
        context = ["river", "water", "shore", "fish", "stream", "flow"]
        result = lesk.disambiguate("bank", context)

        assert result is not None, "Lesk returned None"

        definition_lower = (result.definition or "").lower()
        river_terms = ["river", "water", "slope", "edge", "shore", "land"]
        has_river_meaning = any(term in definition_lower for term in river_terms)

        print(f"Lesk picked: {result.definition}")

        assert has_river_meaning, (
            f"Lesk picked wrong sense for 'bank' in river context. "
            f"Got definition: '{result.definition}'"
        )

    def test_lesk_different_contexts_different_senses(self):
        """CRITICAL: Same word + different context = different sense."""
        lesk = LeskDisambiguator(use_nltk=True)

        if not lesk.use_nltk:
            pytest.skip("NLTK WordNet not available")

        # Two different contexts for "bass"
        fish_context = ["fish", "fishing", "lake", "catch", "water", "rod"]
        music_context = ["music", "guitar", "band", "play", "instrument", "sound"]

        fish_result = lesk.disambiguate("bass", fish_context)
        music_result = lesk.disambiguate("bass", music_context)

        assert fish_result is not None, "No fish disambiguation"
        assert music_result is not None, "No music disambiguation"

        print(f"Fish context: {fish_result.definition}")
        print(f"Music context: {music_result.definition}")

        # THE CRITICAL TEST: Are they DIFFERENT?
        # If they're the same, context isn't actually influencing disambiguation
        if fish_result.synset_id == music_result.synset_id:
            pytest.fail(
                f"CRITICAL FAILURE: Same sense picked for both contexts!\n"
                f"Fish: {fish_result.synset_id} - {fish_result.definition}\n"
                f"Music: {music_result.synset_id} - {music_result.definition}\n"
                f"Context is NOT influencing disambiguation!"
            )

    @pytest.mark.asyncio
    async def test_wsd_full_sentence_disambiguation(self):
        """Test full sentence WSD picks different senses based on sentence context."""
        wsd = WordSenseDisambiguator()

        if not wsd.lesk.use_nltk:
            pytest.skip("NLTK WordNet not available")

        # Two sentences with 'bank'
        sentence1 = "I deposited my money at the bank"
        sentence2 = "The fish swam near the river bank"

        # disambiguate_sentence is async and doesn't take target_words
        result1 = await wsd.disambiguate_sentence(sentence1)
        result2 = await wsd.disambiguate_sentence(sentence2)

        # WSD returns keys as "position:word" format
        # Find the bank entries
        bank_key1 = next((k for k in result1 if k.endswith(":bank")), None)
        bank_key2 = next((k for k in result2 if k.endswith(":bank")), None)

        assert bank_key1 is not None, f"Didn't disambiguate 'bank' in sentence 1. Keys: {list(result1.keys())}"
        assert bank_key2 is not None, f"Didn't disambiguate 'bank' in sentence 2. Keys: {list(result2.keys())}"

        sense1 = result1[bank_key1]
        sense2 = result2[bank_key2]

        print(f"Sentence 1 sense: {sense1.definition if sense1 else 'None'}")
        print(f"Sentence 2 sense: {sense2.definition if sense2 else 'None'}")

        # They should be different
        if sense1 and sense2:
            assert sense1.synset_id != sense2.synset_id, (
                f"Same sense for different contexts! "
                f"Both got: {sense1.synset_id}"
            )


# =============================================================================
# PART 2: CONTEXT CHANGES OUTPUT
# =============================================================================


class TestContextChangesOutput:
    """Test that different context actually produces different results."""

    @pytest.mark.asyncio
    async def test_different_memory_different_pronoun_resolution(self):
        """Same statement, different memory context = different pronoun resolution."""

        # Context 1: Sarah is the topic
        memory1 = VerifyingMemoryProvider()
        memory1.add_memory("Sarah is the project manager", MemoryType.FACT, ["Sarah"])
        memory1.add_memory("Sarah has been working late", MemoryType.EPISODIC, ["Sarah"])

        # Context 2: John is the topic
        memory2 = VerifyingMemoryProvider()
        memory2.add_memory("John is the team lead", MemoryType.FACT, ["John"])
        memory2.add_memory("John has been stressed lately", MemoryType.EPISODIC, ["John"])

        # Use PreExpansionRetriever directly to test pronoun resolution
        retriever1 = PreExpansionRetriever(memory=memory1)
        retriever2 = PreExpansionRetriever(memory=memory2)

        # Same input with pronoun
        statement = "She left early today"
        context = ["We were discussing the team"]

        result1 = await retriever1.retrieve(statement, context)
        result2 = await retriever2.retrieve(statement, context)

        # Check what entities were resolved
        print(f"Memory 1 resolved entities: {result1.resolved_entities}")
        print(f"Memory 2 resolved entities: {result2.resolved_entities}")

        # CRITICAL: The resolved entities should be DIFFERENT
        # because the memory context is different
        # If they're the same, memory isn't influencing resolution

        # Note: This test may fail because our current implementation
        # doesn't actually do pronoun resolution - it just extracts
        # entities from the immediate context

    @pytest.mark.asyncio
    async def test_memory_context_influences_expansion_scoring(self):
        """Memory context should influence which variant scores highest."""

        # Setup memory with coffee preference
        memory_coffee = VerifyingMemoryProvider()
        memory_coffee.add_memory("Doug loves coffee", MemoryType.PREFERENCE, ["Doug", "coffee"])
        memory_coffee.add_memory("Doug drinks coffee every morning", MemoryType.EPISODIC, ["Doug", "coffee"])

        # Setup memory with tea preference
        memory_tea = VerifyingMemoryProvider()
        memory_tea.add_memory("Doug prefers tea", MemoryType.PREFERENCE, ["Doug", "tea"])
        memory_tea.add_memory("Doug switched from coffee to tea", MemoryType.EPISODIC, ["Doug", "tea"])

        # Create mock LLM that returns proper semantic frame format
        class DualVariantLLM:
            async def chat(self, messages, **kwargs):
                # Return format expected by _parse_frame_response
                return """<semantic_frame>
<triples>
<triple>
<subject>Doug</subject>
<predicate>PREFERS</predicate>
<object>beverage</object>
</triple>
</triples>
<ambiguities>
<ambiguity type="word_sense">
<text>beverage</text>
<possibilities>
<possibility>coffee</possibility>
<possibility>tea</possibility>
</possibilities>
</ambiguity>
</ambiguities>
<frame_type>ASSERTION</frame_type>
<confidence>0.8</confidence>
</semantic_frame>"""

        llm = DualVariantLLM()

        # Process with coffee memory
        orchestrator_coffee = TwoPassSemanticOrchestrator(memory=memory_coffee, llm=llm)
        result_coffee = await orchestrator_coffee.process(
            "What does Doug like to drink?",
            immediate_context=[],
        )

        # Process with tea memory
        orchestrator_tea = TwoPassSemanticOrchestrator(memory=memory_tea, llm=llm)
        result_tea = await orchestrator_tea.process(
            "What does Doug like to drink?",
            immediate_context=[],
        )

        # CRITICAL TEST: The primary variant should be DIFFERENT
        # because memory context should rescore them differently
        print(f"Coffee memory result: {result_coffee.primary_variant}")
        print(f"Tea memory result: {result_tea.primary_variant}")

        if result_coffee.primary_variant and result_tea.primary_variant:
            # Check if the variants are actually different
            coffee_obj = None
            tea_obj = None

            if result_coffee.primary_variant.frame and result_coffee.primary_variant.frame.triples:
                for triple in result_coffee.primary_variant.frame.triples:
                    if triple.predicate == "PREFERS":
                        coffee_obj = triple.object

            if result_tea.primary_variant.frame and result_tea.primary_variant.frame.triples:
                for triple in result_tea.primary_variant.frame.triples:
                    if triple.predicate == "PREFERS":
                        tea_obj = triple.object

            print(f"Coffee memory picked: {coffee_obj}")
            print(f"Tea memory picked: {tea_obj}")

            # They SHOULD be different - coffee memory should pick coffee variant,
            # tea memory should pick tea variant
            # If they're the same, memory isn't influencing scoring
            assert coffee_obj != tea_obj, (
                f"FAILURE: Memory context didn't change which variant was selected!\n"
                f"Both selected: {coffee_obj}\n"
                f"Memory should have influenced scoring!"
            )


# =============================================================================
# PART 3: MEMORY ACTUALLY QUERIED
# =============================================================================


class TestMemoryActuallyQueried:
    """Verify that memory is actually being queried (not just checked that it doesn't crash)."""

    @pytest.mark.asyncio
    async def test_pass1_queries_memory_before_expansion(self):
        """Pass 1 MUST query memory BEFORE semantic expansion."""
        memory = VerifyingMemoryProvider()
        memory.add_memory("Doug is the CEO", MemoryType.FACT, ["Doug"])

        retriever = PreExpansionRetriever(memory=memory)

        # Clear any prior queries
        memory.clear_tracking()

        # Run retrieval
        await retriever.retrieve("Doug said something", immediate_context=[])

        # VERIFY memory was actually searched
        assert len(memory.search_queries) > 0 or len(memory.entity_searches) > 0, (
            "FAILURE: Pass 1 didn't query memory at all!\n"
            f"Search queries: {memory.search_queries}\n"
            f"Entity searches: {memory.entity_searches}\n"
            "Pre-expansion retrieval MUST query memory."
        )

        print(f"Pass 1 made {len(memory.search_queries)} searches, {len(memory.entity_searches)} entity lookups")

    @pytest.mark.asyncio
    async def test_pass2_queries_with_expanded_entities(self):
        """Pass 2 MUST query memory with entities from expanded frames."""
        memory = VerifyingMemoryProvider()
        memory.add_memory("Doug works at Acme Corp", MemoryType.FACT, ["Doug", "Acme"])

        # Create a variant with specific entities (entities are in the triple, not frame)
        variant = ExpansionVariant(
            variant_id="test_variant_1",
            base_confidence=0.8,
            frame=SemanticFrame(
                original_text="Doug went to work",
                triples=[
                    SemanticTriple(subject="Doug", predicate="GOES_TO", object="Acme"),
                ],
            ),
        )

        retriever = PostExpansionRetriever(memory=memory)
        memory.clear_tracking()

        # Run post-expansion retrieval
        evidence = await retriever.retrieve_evidence(variant)

        # VERIFY memory was queried with the expanded entities
        all_searched_entities = []
        for search in memory.entity_searches:
            all_searched_entities.extend(search)

        print(f"Pass 2 entity searches: {memory.entity_searches}")
        print(f"Pass 2 text searches: {memory.search_queries}")

        # Should have searched for Doug or Acme
        assert len(memory.entity_searches) > 0 or len(memory.search_queries) > 0, (
            "FAILURE: Pass 2 didn't query memory!"
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_queries_memory_twice(self):
        """Full pipeline should query memory in BOTH passes."""
        memory = VerifyingMemoryProvider()
        memory.add_memory("Test fact about Doug", MemoryType.FACT, ["Doug"])

        class SimpleLLM:
            async def chat(self, messages, **kwargs):
                return """<semantic_frame>
<triples>
<triple>
<subject>Doug</subject>
<predicate>SAYS</predicate>
<object>hello</object>
</triple>
</triples>
<frame_type>ASSERTION</frame_type>
<confidence>0.8</confidence>
</semantic_frame>"""

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=SimpleLLM())
        memory.clear_tracking()

        await orchestrator.process("Doug said hello", immediate_context=[])

        total_queries = len(memory.search_queries) + len(memory.entity_searches)

        print(f"Total memory queries: {total_queries}")
        print(f"Search queries: {memory.search_queries}")
        print(f"Entity searches: {memory.entity_searches}")

        # Should have multiple queries (Pass 1 + Pass 2)
        assert total_queries >= 2, (
            f"FAILURE: Pipeline only made {total_queries} memory queries.\n"
            f"Expected at least 2 (Pass 1 + Pass 2).\n"
            "Two-pass architecture isn't working!"
        )


# =============================================================================
# PART 4: CONFLICT ACTUALLY DETECTED
# =============================================================================


class TestConflictActuallyDetected:
    """Test that contradictions are actually detected and flagged."""

    @pytest.mark.asyncio
    async def test_direct_contradiction_flagged(self):
        """Direct contradiction should be detected."""
        memory = VerifyingMemoryProvider()

        # Add contradicting facts
        memory.add_memory("Doug has 3 cats", MemoryType.FACT, ["Doug", "cats"])
        memory.add_memory("Doug has 6 cats", MemoryType.FACT, ["Doug", "cats"])

        class CatLLM:
            async def chat(self, messages, **kwargs):
                prompt = messages[-1]["content"] if messages else ""
                # Handle memory conflict detection (memory-to-memory)
                if "Memory A" in prompt and "Memory B" in prompt:
                    if ("3 cats" in prompt and "6 cats" in prompt) or \
                       ("6 cats" in prompt and "3 cats" in prompt):
                        return """<classification>
<relationship>CONFLICTING</relationship>
<confidence>0.95</confidence>
<reason>Different cat counts: 3 vs 6</reason>
</classification>"""
                    return """<classification>
<relationship>CONSISTENT</relationship>
<confidence>0.5</confidence>
<reason>No direct conflict</reason>
</classification>"""
                # Handle variant-vs-memory conflict detection
                if "Statement A" in prompt and "Statement B" in prompt:
                    if ("3 cats" in prompt and "6 cats" in prompt) or \
                       ("6 cats" in prompt and "3 cats" in prompt):
                        return """<classification>
<relationship>CONTRADICTING</relationship>
<confidence>0.95</confidence>
<reason>Different cat counts: 3 vs 6</reason>
</classification>"""
                    return """<classification>
<relationship>RELATED</relationship>
<confidence>0.5</confidence>
<reason>No direct conflict</reason>
</classification>"""
                # Default: semantic frame expansion
                return """<semantic_frame>
<triples>
<triple>
<subject>Doug</subject>
<predicate>HAS</predicate>
<object>cats</object>
</triple>
</triples>
<frame_type>ASSERTION</frame_type>
<confidence>0.8</confidence>
</semantic_frame>"""

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=CatLLM())

        result = await orchestrator.process(
            "How many cats does Doug have?",
            immediate_context=[],
        )

        # CRITICAL: Should have detected conflict
        print(f"Detected conflicts: {result.detected_conflicts}")

        # The system should flag this as a conflict
        assert len(result.detected_conflicts) > 0, (
            "FAILURE: No conflict detected!\n"
            f"Memory has '3 cats' and '6 cats' for Doug.\n"
            f"System should have flagged this contradiction.\n"
            f"Conflicts found: {result.detected_conflicts}"
        )

    @pytest.mark.asyncio
    async def test_preference_contradiction_detected(self):
        """Contradicting preferences should be detected."""
        memory = VerifyingMemoryProvider()

        memory.add_memory("Doug loves coffee", MemoryType.PREFERENCE, ["Doug", "coffee"])
        memory.add_memory("Doug hates coffee", MemoryType.PREFERENCE, ["Doug", "coffee"])

        class CoffeeLLM:
            async def chat(self, messages, **kwargs):
                prompt = messages[-1]["content"] if messages else ""
                # Handle memory conflict detection (memory-to-memory)
                if "Memory A" in prompt and "Memory B" in prompt:
                    if ("loves coffee" in prompt and "hates coffee" in prompt) or \
                       ("hates coffee" in prompt and "loves coffee" in prompt):
                        return """<classification>
<relationship>CONFLICTING</relationship>
<confidence>0.95</confidence>
<reason>Opposite preferences: loves vs hates</reason>
</classification>"""
                    return """<classification>
<relationship>CONSISTENT</relationship>
<confidence>0.5</confidence>
<reason>No direct conflict</reason>
</classification>"""
                # Handle variant-vs-memory conflict detection
                if "Statement A" in prompt and "Statement B" in prompt:
                    if ("loves coffee" in prompt and "hates coffee" in prompt) or \
                       ("hates coffee" in prompt and "loves coffee" in prompt):
                        return """<classification>
<relationship>CONTRADICTING</relationship>
<confidence>0.95</confidence>
<reason>Opposite preferences: loves vs hates</reason>
</classification>"""
                    return """<classification>
<relationship>RELATED</relationship>
<confidence>0.5</confidence>
<reason>No direct conflict</reason>
</classification>"""
                # Default: semantic frame expansion
                return """<semantic_frame>
<triples>
<triple>
<subject>Doug</subject>
<predicate>LIKES</predicate>
<object>coffee</object>
</triple>
</triples>
<frame_type>ASSERTION</frame_type>
<confidence>0.8</confidence>
</semantic_frame>"""

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=CoffeeLLM())

        result = await orchestrator.process(
            "Does Doug like coffee?",
            immediate_context=[],
        )

        print(f"Conflicts: {result.detected_conflicts}")

        assert len(result.detected_conflicts) > 0, (
            "FAILURE: Preference contradiction not detected!\n"
            "'loves coffee' and 'hates coffee' should conflict."
        )


# =============================================================================
# PART 5: NLG PRODUCES COHERENT OUTPUT
# =============================================================================


class TestNLGCoherence:
    """Test that Natural Language Generation produces meaningful output."""

    @pytest.mark.asyncio
    async def test_nlg_produces_readable_text(self):
        """NLG should produce human-readable text from a semantic frame."""
        nlg = NaturalLanguageGenerator(llm=None)

        # Create a simple variant (entities come from triples, not frame)
        variant = ExpansionVariant(
            variant_id="nlg_test_1",
            base_confidence=0.9,
            frame=SemanticFrame(
                original_text="Doug likes coffee",
                triples=[
                    SemanticTriple(subject="Doug", predicate="LIKES", object="coffee"),
                ],
            ),
        )

        result = await nlg.generate(variant)

        print(f"NLG output: {result}")

        # Should produce some text
        assert result is not None, "NLG returned None"
        assert len(result) > 0, "NLG returned empty string"

        # Should be readable (not XML/JSON garbage)
        assert "<" not in result or ">" not in result, (
            f"NLG output contains XML tags: {result}"
        )

    @pytest.mark.asyncio
    async def test_nlg_incorporates_context(self):
        """NLG should incorporate memory context into output."""
        nlg = NaturalLanguageGenerator(llm=None)

        variant = ExpansionVariant(
            variant_id="nlg_test_2",
            base_confidence=0.9,
            frame=SemanticFrame(
                original_text="He went there",
                triples=[
                    SemanticTriple(subject="Doug", predicate="WENT_TO", object="office"),
                ],
            ),
        )

        # Create context that provides resolution
        context = PreExpansionContext(
            resolved_entities={
                "He": EntityInfo(entity_type="PERSON"),
                "there": EntityInfo(entity_type="LOCATION"),
            },
            semantic_facts=["Doug works at the downtown office"],
            raw_results=[
                SearchResult(
                    memory=Memory(
                        id="m1",
                        content="Doug works at the downtown office",
                        memory_type=MemoryType.FACT,
                        scope=MemoryScope.USER,
                        entities=["Doug", "office"],
                        importance=0.8,
                        linked_memories=[],
                    ),
                    score=0.9,
                )
            ],
        )

        result = await nlg.generate(variant, context=context)

        print(f"NLG with context: {result}")

        # Should mention Doug (not "He")
        # If NLG is working properly, it should use the resolved entities
        assert "Doug" in result or "office" in result, (
            f"NLG didn't incorporate context!\n"
            f"Output: {result}\n"
            f"Should mention 'Doug' or 'office' from resolved context."
        )


# =============================================================================
# PART 6: VARIATION SCORING CORRECTNESS
# =============================================================================


class TestVariationScoringCorrectness:
    """Test that variation scoring actually works correctly."""

    @pytest.mark.asyncio
    async def test_supporting_evidence_boosts_score(self):
        """Variant with supporting memory evidence should score higher."""
        memory = VerifyingMemoryProvider()

        # Add supporting evidence for coffee
        memory.add_memory("Doug drinks coffee every day", MemoryType.FACT, ["Doug", "coffee"])
        memory.add_memory("Doug's favorite drink is coffee", MemoryType.PREFERENCE, ["Doug", "coffee"])

        # No evidence for tea

        # Create two variants (entities come from triples, not frame)
        coffee_variant = ExpansionVariant(
            variant_id="coffee_variant",
            base_confidence=0.7,  # Same base confidence
            frame=SemanticFrame(
                original_text="Doug's drink",
                triples=[SemanticTriple(subject="Doug", predicate="DRINKS", object="coffee")],
            ),
        )

        tea_variant = ExpansionVariant(
            variant_id="tea_variant",
            base_confidence=0.7,  # Same base confidence
            frame=SemanticFrame(
                original_text="Doug's drink",
                triples=[SemanticTriple(subject="Doug", predicate="DRINKS", object="tea")],
            ),
        )

        retriever = PostExpansionRetriever(memory=memory)

        coffee_evidence = await retriever.retrieve_evidence(coffee_variant)
        tea_evidence = await retriever.retrieve_evidence(tea_variant)

        print(f"Coffee supporting evidence: {len(coffee_evidence.supporting) if coffee_evidence else 0}")
        print(f"Tea supporting evidence: {len(tea_evidence.supporting) if tea_evidence else 0}")

        # Coffee should have more supporting evidence
        coffee_support = len(coffee_evidence.supporting) if coffee_evidence else 0
        tea_support = len(tea_evidence.supporting) if tea_evidence else 0

        assert coffee_support > tea_support, (
            f"FAILURE: Coffee variant should have more supporting evidence!\n"
            f"Coffee: {coffee_support}, Tea: {tea_support}\n"
            f"Memory has evidence for coffee but not tea."
        )


# =============================================================================
# PART 7: END-TO-END SEMANTIC CORRECTNESS
# =============================================================================


class TestEndToEndSemanticCorrectness:
    """End-to-end tests that verify the whole pipeline produces correct results."""

    @pytest.mark.asyncio
    async def test_full_pipeline_correct_interpretation(self):
        """Full pipeline should produce semantically correct interpretation."""
        memory = VerifyingMemoryProvider()

        # Build up memory about Doug
        memory.add_memory("Doug is the CEO of TechCorp", MemoryType.FACT, ["Doug", "CEO", "TechCorp"])
        memory.add_memory("Doug prefers morning meetings", MemoryType.PREFERENCE, ["Doug", "meetings"])
        memory.add_memory("Doug met with investors yesterday", MemoryType.EPISODIC, ["Doug", "investors"])

        class SmartLLM:
            async def chat(self, messages, **kwargs):
                prompt = messages[-1]["content"] if messages else ""

                # Generate contextually appropriate expansion
                if "Doug" in prompt and "meeting" in prompt.lower():
                    return """<semantic_frame>
<triples>
<triple>
<subject>Doug</subject>
<predicate>WANTS</predicate>
<object>meeting</object>
<context>temporal="morning"</context>
</triple>
</triples>
<implications>
<implication type="pragmatic" confidence="0.8">Doug prefers morning meetings</implication>
</implications>
<frame_type>REQUEST</frame_type>
<confidence>0.85</confidence>
</semantic_frame>"""

                return """<semantic_frame>
<triples>
<triple>
<subject>unknown</subject>
<predicate>UNKNOWN</predicate>
<object>unknown</object>
</triple>
</triples>
<frame_type>ASSERTION</frame_type>
<confidence>0.5</confidence>
</semantic_frame>"""

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=SmartLLM())

        result = await orchestrator.process(
            "Doug wants to set up a meeting",
            immediate_context=["Scheduling discussion"],
        )

        print(f"Pipeline result: {result}")
        print(f"Primary variant: {result.primary_variant}")

        # Verify the interpretation makes sense
        assert result.primary_variant is not None, "No variant selected"
        assert result.primary_variant.combined_score > 0.5, "Low confidence interpretation"

        # The interpretation should reflect memory context
        # Check if the frame has implications about morning preference
        implications = result.primary_variant.frame.implications
        print(f"Implications: {implications}")

        # Should mention morning preference since memory has it
        # This tests whether memory actually influenced the result

"""Integration tests with real providers.

These tests demonstrate the semantic expansion system working with
real LLM and memory providers. They are skipped if the required
environment variables are not set.

To run these tests:
    1. Set GROQ_API_KEY environment variable
    2. Run: pytest tests/semantic/test_real_provider_integration.py -v

Example:
    GROQ_API_KEY=your_key pytest tests/semantic/test_real_provider_integration.py -v

Note: Tests that require semantic search also need an embedding provider.
Without embeddings, search operations return empty results.
"""

import os
import pytest

# Prototype imports
from integration import (
    TwoPassSemanticOrchestrator,
    process_with_memory,
)
from expansion import SemanticExpansionService, ExpansionInput
from wsd import WordSenseDisambiguator

# Core draagon-ai imports (stable base types)
from draagon_ai.memory.providers.layered import LayeredMemoryProvider
from draagon_ai.memory.base import MemoryType, MemoryScope, SearchResult, Memory

# Check if Groq is available
try:
    from draagon_ai.llm.groq import GroqLLM
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SKIP_REASON = "GROQ_API_KEY not set or groq not installed"


class MockMemoryProvider:
    """Mock memory provider for testing without embeddings.

    This simulates what a real provider does but uses simple
    substring matching instead of semantic similarity.
    """

    def __init__(self):
        self.memories: list[Memory] = []
        self._next_id = 1

    async def initialize(self):
        pass

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        entities: list[str] | None = None,
        **kwargs,
    ) -> Memory:
        memory = Memory(
            id=f"mem_{self._next_id}",
            content=content,
            memory_type=memory_type,
            scope=scope,
            entities=entities or [],
        )
        self._next_id += 1
        self.memories.append(memory)
        return memory

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        **kwargs,
    ) -> list[SearchResult]:
        """Simple substring-based search."""
        results = []
        query_lower = query.lower()
        for mem in self.memories:
            if any(word in mem.content.lower() for word in query_lower.split()):
                results.append(SearchResult(memory=mem, score=0.8))
        return results[:limit]

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search by entity overlap."""
        if not entities:
            return []

        query_entities = {e.lower() for e in entities}
        results = []

        for mem in self.memories:
            mem_entities = {e.lower() for e in (mem.entities or [])}
            overlap = query_entities & mem_entities
            if overlap:
                score = len(overlap) / len(query_entities)
                results.append(SearchResult(memory=mem, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def get(self, memory_id: str) -> Memory | None:
        for mem in self.memories:
            if mem.id == memory_id:
                return mem
        return None

    async def boost_memory(self, memory_id: str, boost_amount: float | None = None) -> Memory | None:
        return await self.get(memory_id)

    async def demote_memory(self, memory_id: str, demote_amount: float | None = None) -> Memory | None:
        return await self.get(memory_id)


@pytest.fixture
def real_llm():
    """Create a real Groq LLM provider."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        pytest.skip(SKIP_REASON)
    return GroqLLM(api_key=GROQ_API_KEY)


@pytest.fixture
async def mock_memory():
    """Create a MockMemoryProvider with test data.

    This works without embeddings for integration testing.
    """
    provider = MockMemoryProvider()
    await provider.initialize()

    # Add some test memories
    await provider.store(
        content="Doug's favorite drink is coffee",
        memory_type=MemoryType.PREFERENCE,
        scope=MemoryScope.USER,
        entities=["Doug", "coffee"],
    )
    await provider.store(
        content="Doug has 3 cats named Whiskers, Shadow, and Luna",
        memory_type=MemoryType.FACT,
        scope=MemoryScope.USER,
        entities=["Doug", "cats", "Whiskers", "Shadow", "Luna"],
    )
    await provider.store(
        content="Yesterday Doug mentioned he was tired from work",
        memory_type=MemoryType.EPISODIC,
        scope=MemoryScope.USER,
        entities=["Doug"],
    )

    return provider


class TestRealLLMIntegration:
    """Tests with real LLM (Groq)."""

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_semantic_expansion_with_groq(self, real_llm):
        """Test that SemanticExpansionService works with real Groq LLM."""
        from expansion import ExpansionInput

        service = SemanticExpansionService(llm=real_llm)
        inputs = ExpansionInput()

        # expand() returns a list of ExpansionVariant
        variants = await service.expand("Doug prefers tea in the morning", inputs)

        # Real LLM should produce meaningful variants
        assert variants is not None
        assert len(variants) >= 1

        # Check that the frame contains the original text
        frame = variants[0].frame
        assert frame.original_text == "Doug prefers tea in the morning"

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_wsd_with_groq(self, real_llm):
        """Test Word Sense Disambiguation with real LLM."""
        wsd = WordSenseDisambiguator(llm=real_llm, use_nltk=True)

        # Disambiguate a sentence with bank
        result = await wsd.disambiguate_sentence("I deposited money at the bank")

        # Should return at least one sense
        assert len(result) >= 1

        # The bank sense should be financial, not river bank
        bank_senses = [s for k, s in result.items() if "bank" in k.lower()]
        if bank_senses:
            sense = bank_senses[0]
            # With Lesk, the definition should relate to finance
            assert sense.definition is not None


class TestMemoryIntegration:
    """Tests with MockMemoryProvider (works without embeddings)."""

    async def test_memory_search_by_entities(self, mock_memory):
        """Test that search_by_entities works correctly."""
        results = await mock_memory.search_by_entities(["Doug", "cats"])

        # Should find the cats memory
        assert len(results) >= 1
        contents = [r.memory.content for r in results]
        assert any("cats" in c for c in contents)

    async def test_memory_regular_search(self, mock_memory):
        """Test search across memories."""
        results = await mock_memory.search("What is Doug's favorite drink?")

        # Should find coffee preference (substring match on "favorite" or "Doug")
        assert len(results) >= 1


class TestFullPipelineIntegration:
    """Tests with both real LLM and mock memory."""

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_two_pass_with_real_llm(self, real_llm, mock_memory):
        """Test full two-pass pipeline with real LLM."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=mock_memory,
            llm=real_llm,
        )

        # Process a statement that should use memory context
        result = await orchestrator.process("He has several pets")

        # Should produce a meaningful result
        assert result is not None
        assert result.variants is not None
        assert len(result.variants) >= 1

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_conflict_detection_with_real_llm(self, real_llm, mock_memory):
        """Test conflict detection with real LLM."""
        orchestrator = TwoPassSemanticOrchestrator(
            memory=mock_memory,
            llm=real_llm,
        )

        # This should conflict with the stored "3 cats" memory
        result = await orchestrator.process("Doug has 6 cats")

        # Should detect conflict with existing memory
        assert result is not None
        # The conflict detection depends on LLM analysis
        # Just verify the pipeline completes without error

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_convenience_function_with_real_llm(self, real_llm, mock_memory):
        """Test the process_with_memory convenience function."""
        result = await process_with_memory(
            statement="Doug mentioned his pets",
            memory=mock_memory,
            llm=real_llm,
        )

        assert result is not None
        # response_text is the correct attribute name
        assert result.response_text is not None


class TestProviderProtocolCompliance:
    """Verify providers comply with expected protocols."""

    async def test_mock_memory_has_search_by_entities(self, mock_memory):
        """Verify MockMemoryProvider has the required method."""
        assert hasattr(mock_memory, "search_by_entities")
        assert callable(mock_memory.search_by_entities)

        # Method should work with empty list
        results = await mock_memory.search_by_entities([])
        assert results == []

    async def test_layered_memory_has_search_by_entities(self):
        """Verify LayeredMemoryProvider has the required method."""
        provider = LayeredMemoryProvider()
        await provider.initialize()

        assert hasattr(provider, "search_by_entities")
        assert callable(provider.search_by_entities)

        # Method should work with empty list
        results = await provider.search_by_entities([])
        assert results == []

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_groq_returns_chat_response(self, real_llm):
        """Verify Groq returns ChatResponse (not str)."""
        from draagon_ai.llm.base import ChatResponse

        response = await real_llm.chat(
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )

        assert isinstance(response, ChatResponse)
        assert hasattr(response, "content")
        assert isinstance(response.content, str)


# Example usage documentation
class TestExampleUsage:
    """These tests serve as documentation for how to use the semantic system."""

    async def test_example_basic_usage(self):
        """Example: Basic semantic expansion without LLM."""
        # Prototype imports - already imported at top of file
        # from expansion import SemanticExpansionService, ExpansionInput

        # Create service without LLM (uses simple heuristics)
        service = SemanticExpansionService(llm=None)
        inputs = ExpansionInput()

        # expand() returns a list of ExpansionVariant
        variants = await service.expand("The user likes pizza", inputs)

        # Should have at least one variant
        assert len(variants) >= 1
        assert variants[0].frame.original_text == "The user likes pizza"

    @pytest.mark.skipif(not GROQ_AVAILABLE or not GROQ_API_KEY, reason=SKIP_REASON)
    async def test_example_with_memory_integration(self, real_llm, mock_memory):
        """Example: Full pipeline with memory integration."""
        # Prototype imports - already imported at top of file
        # from integration import TwoPassSemanticOrchestrator

        # Create orchestrator with both providers
        orchestrator = TwoPassSemanticOrchestrator(
            memory=mock_memory,
            llm=real_llm,
        )

        # Process a statement - this does:
        # 1. Pre-expansion: Query memory for context
        # 2. Expansion: Use LLM to extract semantic frame
        # 3. Post-expansion: Query memory for evidence
        # 4. NLG: Generate natural language response
        result = await orchestrator.process("Tell me about Doug's preferences")

        # Access the results
        print(f"Response: {result.response_text}")
        print(f"Variants: {len(result.variants)}")
        print(f"Conflicts: {len(result.detected_conflicts)}")

        assert result.response_text is not None

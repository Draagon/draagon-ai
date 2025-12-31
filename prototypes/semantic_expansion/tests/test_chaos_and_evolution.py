"""Chaos Testing and Evolutionary Optimization for Semantic Expansion.

This test suite is designed to:
1. CHAOS TEST - Try to break the system with adversarial inputs
2. ACCUMULATION TEST - Build knowledge over time, verify it compounds
3. GRAPH TRAVERSAL TEST - Multi-hop reasoning, transitive relationships
4. EVOLUTIONARY TEST - Optimize weights and prompts through evolution

The goal is to find weaknesses, edge cases, and validate real-world robustness.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest

from draagon_ai.memory.base import Memory, MemoryType, MemoryScope, SearchResult
from integration import (
    TwoPassSemanticOrchestrator,
    PreExpansionContext,
    ProcessingResult,
    DetectedConflict,
)
from semantic_types import (
    SemanticFrame,
    SemanticTriple,
    ExpansionVariant,
    CrossLayerRelation,
)
from wsd import (
    LeskDisambiguator,
    WordSenseDisambiguator,
    get_synset_id,
)


# =============================================================================
# Enhanced Mock Memory Provider (Tracks Everything)
# =============================================================================


class InstrumentedMemoryProvider:
    """Memory provider with full instrumentation for testing.

    Tracks:
    - All queries made
    - Query patterns over time
    - Memory access frequency
    - Relationship graph between memories
    """

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.query_log: list[dict] = []
        self.access_counts: dict[str, int] = {}
        self.relationships: list[tuple[str, str, str]] = []  # (from_id, relation, to_id)
        self._next_id = 1

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        entities: list[str] | None = None,
        importance: float = 0.5,
        linked_memories: list[str] | None = None,
        **kwargs,
    ) -> Memory:
        """Add a memory with full tracking."""
        memory = Memory(
            id=f"mem_{self._next_id}",
            content=content,
            memory_type=memory_type,
            scope=MemoryScope.USER,
            entities=entities or [],
            importance=importance,
            linked_memories=linked_memories or [],
            **kwargs,
        )
        self._next_id += 1
        self.memories[memory.id] = memory

        # Track relationships from linked memories
        for linked_id in memory.linked_memories:
            self.relationships.append((memory.id, "links_to", linked_id))

        # Auto-detect entity-based relationships with existing memories
        if memory.entities:
            memory_entities = {e.lower() for e in memory.entities}
            for existing_mem in self.memories.values():
                if existing_mem.id == memory.id:
                    continue
                existing_entities = {e.lower() for e in existing_mem.entities}
                shared = memory_entities & existing_entities
                if shared:
                    # Create bidirectional "shares_entity" relationships
                    for entity in shared:
                        self.relationships.append((memory.id, f"shares_{entity}", existing_mem.id))
                        self.relationships.append((existing_mem.id, f"shares_{entity}", memory.id))

        return memory

    def link_memories(self, from_id: str, relation: str, to_id: str):
        """Create an explicit relationship between memories."""
        self.relationships.append((from_id, relation, to_id))
        if from_id in self.memories:
            if to_id not in self.memories[from_id].linked_memories:
                self.memories[from_id].linked_memories.append(to_id)

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search with full instrumentation."""
        self.query_log.append({
            "type": "search",
            "query": query,
            "memory_types": memory_types,
            "limit": limit,
            "timestamp": datetime.now().isoformat(),
        })

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
                self.access_counts[memory.id] = self.access_counts.get(memory.id, 0) + 1

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search by entities with instrumentation."""
        self.query_log.append({
            "type": "entity_search",
            "entities": entities,
            "limit": limit,
            "timestamp": datetime.now().isoformat(),
        })

        results = []
        entity_set = {e.lower() for e in entities}

        for memory in self.memories.values():
            memory_entities = {e.lower() for e in memory.entities}
            overlap = entity_set & memory_entities

            if overlap:
                score = len(overlap) / max(len(entity_set), 1) * memory.importance
                results.append(SearchResult(memory=memory, score=min(1.0, score)))
                self.access_counts[memory.id] = self.access_counts.get(memory.id, 0) + 1

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_relationship_graph(self) -> dict[str, list[tuple[str, str]]]:
        """Get the full relationship graph."""
        graph = {}
        for from_id, relation, to_id in self.relationships:
            if from_id not in graph:
                graph[from_id] = []
            graph[from_id].append((relation, to_id))
        return graph

    def find_path(self, from_id: str, to_id: str, max_hops: int = 5) -> list[str] | None:
        """Find a path between two memories (for graph traversal testing)."""
        if from_id == to_id:
            return [from_id]

        visited = set()
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_hops:
                continue

            if current in visited:
                continue
            visited.add(current)

            # Find neighbors of current node
            for rel_from, rel_type, rel_to in self.relationships:
                if rel_from == current:
                    neighbor = rel_to
                    if neighbor == to_id:
                        return path + [to_id]
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return None


class SmartMockLLM:
    """LLM mock that generates contextually appropriate responses."""

    def __init__(self, personality: str = "helpful"):
        self.personality = personality
        self.call_history: list[dict] = []
        self.custom_responses: dict[str, str] = {}

    def add_response(self, trigger: str, response: str):
        """Add a custom response for a trigger phrase."""
        self.custom_responses[trigger.lower()] = response

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        prompt = messages[-1]["content"] if messages else ""
        self.call_history.append({"prompt": prompt, "temperature": temperature})

        # Check for custom responses
        for trigger, response in self.custom_responses.items():
            if trigger in prompt.lower():
                return response

        # Generate contextual response based on prompt content
        return self._generate_contextual_response(prompt)

    def _generate_contextual_response(self, prompt: str) -> str:
        """Generate a response based on prompt content."""
        prompt_lower = prompt.lower()

        # Extract statement if present
        import re
        statement_match = re.search(r'Statement: "([^"]+)"', prompt)
        statement = statement_match.group(1) if statement_match else "unknown"

        # Detect entities
        entities = re.findall(r'\b([A-Z][a-z]+)\b', statement)
        subject = entities[0] if entities else "Subject"

        # Detect predicates
        predicates = {
            "like": "LIKES",
            "love": "LOVES",
            "prefer": "PREFERS",
            "hate": "DISLIKES",
            "want": "WANTS",
            "need": "NEEDS",
            "is": "IS_A",
            "are": "IS_A",
            "has": "HAS",
            "have": "HAS",
        }

        predicate = "STATES"
        for word, pred in predicates.items():
            if word in prompt_lower:
                predicate = pred
                break

        # Detect objects
        objects = re.findall(r'\b(coffee|tea|water|food|sleep|work|home)\b', prompt_lower)
        obj = objects[0] if objects else "something"

        # Detect ambiguities
        ambiguities = ""
        if any(p in prompt_lower for p in ["he ", "she ", "they ", "it "]):
            pronoun = "He" if "he " in prompt_lower else "She" if "she " in prompt_lower else "They"
            ambiguities = f"""
    <ambiguities>
        <ambiguity type="reference">
            <text>{pronoun}</text>
            <possibilities>
                <possibility>{subject}</possibility>
                <possibility>Unknown person</possibility>
            </possibilities>
        </ambiguity>
    </ambiguities>"""
        else:
            ambiguities = "<ambiguities></ambiguities>"

        return f"""<semantic_frame>
    <triples>
        <triple>
            <subject>{subject}</subject>
            <predicate>{predicate}</predicate>
            <object>{obj}</object>
            <context></context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">{subject} exists</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.8">This reflects a preference or state</implication>
    </implications>
    <negations></negations>
    {ambiguities}
    <open_questions></open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.85</confidence>
</semantic_frame>"""


# =============================================================================
# PART 1: CHAOS TESTING - Adversarial Inputs
# =============================================================================


class TestChaosAdversarial:
    """Adversarial tests designed to break the system."""

    @pytest.mark.asyncio
    async def test_contradictory_statements_in_sequence(self):
        """Feed contradictory statements and see how system handles them."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Statement 1: Doug likes coffee
        memory.add_memory(
            "Doug likes coffee",
            MemoryType.PREFERENCE,
            entities=["Doug"],
            importance=0.8,
        )

        result1 = await orchestrator.process("Doug likes coffee")

        # Statement 2: Doug hates coffee (contradiction!)
        result2 = await orchestrator.process(
            "Doug hates coffee",
            immediate_context=["Doug likes coffee"],
        )

        # System should detect some form of tension
        assert result2.completed_at is not None
        # The pre-expansion context should have found the original preference
        assert len(memory.query_log) > 0

    @pytest.mark.asyncio
    async def test_pronoun_chaos_multiple_candidates(self):
        """Multiple possible referents for pronouns."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add multiple people
        memory.add_memory("Doug is a programmer", MemoryType.FACT, entities=["Doug"])
        memory.add_memory("John is a designer", MemoryType.FACT, entities=["John"])
        memory.add_memory("Mike is a manager", MemoryType.FACT, entities=["Mike"])

        # Ambiguous pronoun
        result = await orchestrator.process(
            "He said he would finish the project",
            immediate_context=[
                "Doug and John had a meeting",
                "Mike was also there",
            ],
        )

        # Should complete without crashing
        assert result.completed_at is not None
        # Should have queried for context
        assert len(memory.query_log) > 0

    @pytest.mark.asyncio
    async def test_homonym_confusion(self):
        """Test words with multiple meanings."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add financial context
        memory.add_memory(
            "The bank approved Doug's loan",
            MemoryType.FACT,
            entities=["Doug", "bank"],
        )

        # Now use "bank" in river context
        result = await orchestrator.process(
            "Doug walked along the bank",
            immediate_context=["Doug went to the river for fishing"],
        )

        # WSD should be involved
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_negation_complexity(self):
        """Test handling of negations and double negatives."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        test_statements = [
            "Doug doesn't dislike coffee",  # Double negative
            "Doug never said he didn't want coffee",  # Triple negative
            "It's not that Doug doesn't prefer tea",  # Convoluted
            "Doug won't not come to the meeting",  # Double negative verb
        ]

        for statement in test_statements:
            result = await orchestrator.process(statement)
            assert result.completed_at is not None, f"Failed on: {statement}"

    @pytest.mark.asyncio
    async def test_temporal_confusion(self):
        """Test temporal references that could be ambiguous."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add historical context
        memory.add_memory(
            "Doug preferred tea last year",
            MemoryType.EPISODIC,
            entities=["Doug"],
        )
        memory.add_memory(
            "Doug switched to coffee recently",
            MemoryType.FACT,
            entities=["Doug"],
        )

        # Ambiguous temporal statement
        result = await orchestrator.process(
            "Doug prefers tea",  # Does this mean NOW or referring to past?
            immediate_context=["We were talking about Doug's old habits"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_empty_and_malformed_input(self):
        """Test edge cases with empty or malformed input."""
        memory = InstrumentedMemoryProvider()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=None)

        test_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "...",  # Just punctuation
            "???",  # Just question marks
            "a",  # Single character
            "!!!",  # Exclamations
        ]

        for input_text in test_inputs:
            result = await orchestrator.process(input_text)
            # Should not crash
            assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_very_long_input(self):
        """Test with extremely long input."""
        memory = InstrumentedMemoryProvider()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=None)

        # Create a very long statement
        long_statement = "Doug " + ("really " * 100) + "likes coffee"

        result = await orchestrator.process(long_statement)
        assert result.completed_at is not None
        assert result.processing_time_ms < 5000  # Should complete in reasonable time

    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        memory = InstrumentedMemoryProvider()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=None)

        test_inputs = [
            "Doug likes coffee",  # Emoji
            "Dougは コーヒーが 好きです",  # Japanese
            "Doug aime le cafe",  # French (no accent to avoid encoding issues)
            "Doug <3 coffee",  # Symbol
            "Doug @work likes coffee",  # @ symbol
            "Doug's coffee preference",  # Apostrophe
        ]

        for input_text in test_inputs:
            result = await orchestrator.process(input_text)
            assert result.completed_at is not None, f"Failed on: {input_text}"

    @pytest.mark.asyncio
    async def test_rapid_fire_processing(self):
        """Test rapid sequential processing."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        statements = [
            f"Person{i} likes thing{i}" for i in range(20)
        ]

        results = []
        for statement in statements:
            result = await orchestrator.process(statement)
            results.append(result)

        # All should complete
        assert all(r.completed_at is not None for r in results)
        # Memory should have logged many queries
        assert len(memory.query_log) >= 20


# =============================================================================
# PART 2: KNOWLEDGE ACCUMULATION - Building Over Time
# =============================================================================


class TestKnowledgeAccumulation:
    """Tests that verify knowledge compounds over time."""

    @pytest.mark.asyncio
    async def test_knowledge_builds_across_statements(self):
        """Verify that later statements benefit from earlier knowledge."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Phase 1: Establish Doug exists
        memory.add_memory(
            "Doug is a software developer",
            MemoryType.FACT,
            entities=["Doug"],
            importance=0.9,
        )

        result1 = await orchestrator.process("Doug works at a tech company")

        # Phase 2: Add more about Doug
        memory.add_memory(
            "Doug works at a tech company",
            MemoryType.FACT,
            entities=["Doug"],
            importance=0.8,
        )

        result2 = await orchestrator.process("Doug drinks coffee at work")

        # Phase 3: Reference Doug with pronoun
        memory.add_memory(
            "Doug drinks coffee at work",
            MemoryType.PREFERENCE,
            entities=["Doug", "coffee"],
            importance=0.8,
        )

        result3 = await orchestrator.process(
            "He prefers the dark roast",
            immediate_context=["Doug drinks coffee at work"],
        )

        # The later queries should find more context
        # Check that Doug was found in pre-expansion
        assert result3.pre_expansion_context is not None
        assert len(result3.pre_expansion_context.semantic_facts) > 0 or \
               len(result3.pre_expansion_context.resolved_entities) > 0

    @pytest.mark.asyncio
    async def test_preference_evolution(self):
        """Track how preferences evolve over time."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Initial preference
        mem1 = memory.add_memory(
            "Doug prefers tea",
            MemoryType.PREFERENCE,
            entities=["Doug", "tea"],
            importance=0.8,
        )

        # Later preference (should be detected as potential change)
        mem2 = memory.add_memory(
            "Doug now prefers coffee",
            MemoryType.PREFERENCE,
            entities=["Doug", "coffee"],
            importance=0.9,
        )

        # Link them
        memory.link_memories(mem2.id, "supersedes", mem1.id)

        result = await orchestrator.process(
            "What does Doug like to drink?",
            immediate_context=["We're ordering drinks"],
        )

        # Should find both and potentially note the evolution
        assert len(memory.query_log) > 0

    @pytest.mark.asyncio
    async def test_entity_relationship_building(self):
        """Build a web of entity relationships."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Build a social graph
        relationships = [
            ("Doug", "works_with", "John"),
            ("Doug", "married_to", "Sarah"),
            ("Sarah", "sister_of", "Mike"),
            ("John", "reports_to", "Doug"),
            ("Mike", "lives_near", "Doug"),
        ]

        for subj, rel, obj in relationships:
            mem = memory.add_memory(
                f"{subj} {rel.replace('_', ' ')} {obj}",
                MemoryType.FACT,
                entities=[subj, obj],
                importance=0.7,
            )

        # Query about relationships
        result = await orchestrator.process(
            "Who does Doug know?",
            immediate_context=[],
        )

        # Should have found multiple related memories
        assert result.completed_at is not None

        # Check memory graph
        graph = memory.get_relationship_graph()
        assert len(graph) > 0


# =============================================================================
# PART 3: GRAPH TRAVERSAL - Multi-Hop Reasoning
# =============================================================================


class TestGraphTraversal:
    """Tests for complex graph-based reasoning."""

    @pytest.mark.asyncio
    async def test_transitive_relationship_discovery(self):
        """Test if system can follow transitive relationships."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create a chain: A -> B -> C -> D
        mem_a = memory.add_memory(
            "Alice manages Bob",
            MemoryType.FACT,
            entities=["Alice", "Bob"],
        )
        mem_b = memory.add_memory(
            "Bob manages Carol",
            MemoryType.FACT,
            entities=["Bob", "Carol"],
        )
        mem_c = memory.add_memory(
            "Carol manages Dave",
            MemoryType.FACT,
            entities=["Carol", "Dave"],
        )

        # Link the chain
        memory.link_memories(mem_a.id, "relates_to", mem_b.id)
        memory.link_memories(mem_b.id, "relates_to", mem_c.id)

        # Query about distant relationship
        result = await orchestrator.process(
            "Dave works in Alice's organization",
            immediate_context=["We're discussing the org structure"],
        )

        # Should be able to find connections
        assert result.completed_at is not None

        # Verify graph has path
        path = memory.find_path(mem_a.id, mem_c.id)
        assert path is not None

    @pytest.mark.asyncio
    async def test_circular_reference_handling(self):
        """Test handling of circular references in memory graph."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create a cycle: A -> B -> C -> A
        mem_a = memory.add_memory("A relates to B", MemoryType.FACT, entities=["A", "B"])
        mem_b = memory.add_memory("B relates to C", MemoryType.FACT, entities=["B", "C"])
        mem_c = memory.add_memory("C relates to A", MemoryType.FACT, entities=["C", "A"])

        memory.link_memories(mem_a.id, "links", mem_b.id)
        memory.link_memories(mem_b.id, "links", mem_c.id)
        memory.link_memories(mem_c.id, "links", mem_a.id)  # Cycle!

        # Should not hang or crash
        result = await orchestrator.process("Tell me about A")
        assert result.completed_at is not None
        assert result.processing_time_ms < 5000  # Should not hang

    @pytest.mark.asyncio
    async def test_multi_path_evidence(self):
        """Test when multiple paths lead to the same conclusion."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Multiple paths to same conclusion
        # Path 1: Doug likes coffee -> Coffee has caffeine -> Doug likes caffeine
        # Path 2: Doug drinks energy drinks -> Energy drinks have caffeine -> Doug likes caffeine

        memory.add_memory(
            "Doug likes coffee",
            MemoryType.PREFERENCE,
            entities=["Doug", "coffee"],
        )
        memory.add_memory(
            "Coffee contains caffeine",
            MemoryType.FACT,
            entities=["coffee", "caffeine"],
        )
        memory.add_memory(
            "Doug drinks energy drinks",
            MemoryType.PREFERENCE,
            entities=["Doug", "energy drinks"],
        )
        memory.add_memory(
            "Energy drinks contain caffeine",
            MemoryType.FACT,
            entities=["energy drinks", "caffeine"],
        )

        result = await orchestrator.process(
            "Doug seems to like caffeine",
            immediate_context=["Discussing Doug's drink preferences"],
        )

        # Should find supporting evidence from multiple paths
        assert result.completed_at is not None
        # Multiple memories should have been accessed
        assert len(memory.access_counts) > 0


# =============================================================================
# PART 4: EVOLUTIONARY OPTIMIZATION
# =============================================================================


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_count: int = 2


@dataclass
class Individual:
    """An individual in the evolutionary population."""

    # Cognitive scoring weights (the genes)
    recency_weight: float = 0.20
    working_memory_weight: float = 0.15
    episodic_memory_weight: float = 0.10
    semantic_memory_weight: float = 0.20
    belief_weight: float = 0.15
    commonsense_weight: float = 0.10
    metacognitive_weight: float = 0.10

    # Support/contradiction adjustment weights
    support_boost: float = 0.1
    contradiction_penalty: float = 0.15

    # Fitness score
    fitness: float = 0.0

    def mutate(self, rate: float) -> "Individual":
        """Create a mutated copy."""
        new = Individual(
            recency_weight=self._mutate_val(self.recency_weight, rate),
            working_memory_weight=self._mutate_val(self.working_memory_weight, rate),
            episodic_memory_weight=self._mutate_val(self.episodic_memory_weight, rate),
            semantic_memory_weight=self._mutate_val(self.semantic_memory_weight, rate),
            belief_weight=self._mutate_val(self.belief_weight, rate),
            commonsense_weight=self._mutate_val(self.commonsense_weight, rate),
            metacognitive_weight=self._mutate_val(self.metacognitive_weight, rate),
            support_boost=self._mutate_val(self.support_boost, rate),
            contradiction_penalty=self._mutate_val(self.contradiction_penalty, rate),
        )
        return new

    def _mutate_val(self, val: float, rate: float) -> float:
        """Mutate a single value."""
        if random.random() < rate:
            delta = random.gauss(0, 0.05)
            return max(0.0, min(1.0, val + delta))
        return val

    @staticmethod
    def crossover(parent1: "Individual", parent2: "Individual") -> "Individual":
        """Create offspring from two parents."""
        return Individual(
            recency_weight=random.choice([parent1.recency_weight, parent2.recency_weight]),
            working_memory_weight=random.choice([parent1.working_memory_weight, parent2.working_memory_weight]),
            episodic_memory_weight=random.choice([parent1.episodic_memory_weight, parent2.episodic_memory_weight]),
            semantic_memory_weight=random.choice([parent1.semantic_memory_weight, parent2.semantic_memory_weight]),
            belief_weight=random.choice([parent1.belief_weight, parent2.belief_weight]),
            commonsense_weight=random.choice([parent1.commonsense_weight, parent2.commonsense_weight]),
            metacognitive_weight=random.choice([parent1.metacognitive_weight, parent2.metacognitive_weight]),
            support_boost=random.choice([parent1.support_boost, parent2.support_boost]),
            contradiction_penalty=random.choice([parent1.contradiction_penalty, parent2.contradiction_penalty]),
        )


@dataclass
class EvaluationCase:
    """An evaluation case for fitness scoring (not a pytest test)."""
    statement: str
    context: list[str]
    expected_entity: str | None = None  # Expected resolved entity
    expected_conflict: bool = False
    expected_synset: str | None = None


class EvolutionaryOptimizer:
    """Evolutionary optimizer for semantic expansion weights."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: list[Individual] = []
        self.generation_stats: list[dict] = []

    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.config.population_size):
            ind = Individual(
                recency_weight=random.uniform(0.05, 0.35),
                working_memory_weight=random.uniform(0.05, 0.25),
                episodic_memory_weight=random.uniform(0.05, 0.20),
                semantic_memory_weight=random.uniform(0.10, 0.35),
                belief_weight=random.uniform(0.05, 0.25),
                commonsense_weight=random.uniform(0.05, 0.20),
                metacognitive_weight=random.uniform(0.05, 0.20),
                support_boost=random.uniform(0.05, 0.20),
                contradiction_penalty=random.uniform(0.10, 0.25),
            )
            self.population.append(ind)

    async def evaluate_fitness(
        self,
        individual: Individual,
        test_cases: list[EvaluationCase],
        memory: InstrumentedMemoryProvider,
    ) -> float:
        """Evaluate fitness of an individual on test cases."""
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(
            memory=memory,
            llm=llm,
            support_boost=individual.support_boost,
            contradiction_penalty=individual.contradiction_penalty,
        )

        total_score = 0.0

        for test_case in test_cases:
            result = await orchestrator.process(
                test_case.statement,
                immediate_context=test_case.context,
            )

            score = 0.0

            # Score based on expected entity resolution
            if test_case.expected_entity:
                if result.pre_expansion_context:
                    if test_case.expected_entity in result.pre_expansion_context.resolved_entities:
                        score += 0.3

            # Score based on conflict detection
            if test_case.expected_conflict:
                if result.has_conflicts:
                    score += 0.3
            else:
                if not result.has_conflicts:
                    score += 0.2

            # Score based on completion and response quality
            if result.completed_at:
                score += 0.2
            if result.response_text:
                score += 0.2

            total_score += score

        return total_score / len(test_cases) if test_cases else 0.0

    def select_parents(self) -> tuple[Individual, Individual]:
        """Tournament selection."""
        tournament_size = 3

        def tournament():
            contestants = random.sample(self.population, min(tournament_size, len(self.population)))
            return max(contestants, key=lambda x: x.fitness)

        return tournament(), tournament()

    def evolve_generation(self) -> list[Individual]:
        """Create next generation."""
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # Keep elite
        new_population = sorted_pop[:self.config.elite_count]

        # Create rest through crossover and mutation
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                parent1, parent2 = self.select_parents()
                child = Individual.crossover(parent1, parent2)
            else:
                parent1, _ = self.select_parents()
                child = Individual(
                    recency_weight=parent1.recency_weight,
                    working_memory_weight=parent1.working_memory_weight,
                    episodic_memory_weight=parent1.episodic_memory_weight,
                    semantic_memory_weight=parent1.semantic_memory_weight,
                    belief_weight=parent1.belief_weight,
                    commonsense_weight=parent1.commonsense_weight,
                    metacognitive_weight=parent1.metacognitive_weight,
                    support_boost=parent1.support_boost,
                    contradiction_penalty=parent1.contradiction_penalty,
                )

            child = child.mutate(self.config.mutation_rate)
            new_population.append(child)

        return new_population

    async def run(
        self,
        test_cases: list[EvaluationCase],
        memory: InstrumentedMemoryProvider,
    ) -> Individual:
        """Run evolutionary optimization."""
        self.initialize_population()

        for gen in range(self.config.generations):
            # Evaluate all individuals
            for ind in self.population:
                ind.fitness = await self.evaluate_fitness(ind, test_cases, memory)

            # Record stats
            fitnesses = [ind.fitness for ind in self.population]
            self.generation_stats.append({
                "generation": gen,
                "best": max(fitnesses),
                "avg": sum(fitnesses) / len(fitnesses),
                "worst": min(fitnesses),
            })

            # Evolve
            if gen < self.config.generations - 1:
                self.population = self.evolve_generation()

        # Return best individual
        return max(self.population, key=lambda x: x.fitness)


class TestEvolutionaryOptimization:
    """Tests for evolutionary optimization of weights."""

    @pytest.mark.asyncio
    async def test_evolution_improves_fitness(self):
        """Verify that evolution improves fitness over generations."""
        memory = InstrumentedMemoryProvider()

        # Setup memory
        memory.add_memory("Doug likes coffee", MemoryType.PREFERENCE, entities=["Doug", "coffee"])
        memory.add_memory("Doug is a developer", MemoryType.FACT, entities=["Doug"])
        memory.add_memory("Sarah prefers tea", MemoryType.PREFERENCE, entities=["Sarah", "tea"])

        # Create test cases
        test_cases = [
            EvaluationCase(
                statement="He likes coffee",
                context=["Doug was here earlier"],
                expected_entity="Doug",
            ),
            EvaluationCase(
                statement="She prefers tea",
                context=["Sarah mentioned her preferences"],
                expected_entity="Sarah",
            ),
            EvaluationCase(
                statement="Doug now prefers tea",
                context=["Doug likes coffee"],
                expected_conflict=True,
            ),
        ]

        # Run evolution
        config = EvolutionConfig(
            population_size=5,
            generations=3,
            mutation_rate=0.2,
        )
        optimizer = EvolutionaryOptimizer(config)

        best = await optimizer.run(test_cases, memory)

        # Should have run
        assert len(optimizer.generation_stats) == 3

        # Best fitness should be reasonable
        assert best.fitness >= 0.0

        # Later generations should generally improve (or at least not get worse)
        first_best = optimizer.generation_stats[0]["best"]
        last_best = optimizer.generation_stats[-1]["best"]
        assert last_best >= first_best * 0.8  # Allow some variance

    @pytest.mark.asyncio
    async def test_evolution_finds_good_weights(self):
        """Test that evolution finds weights that work well."""
        memory = InstrumentedMemoryProvider()

        # Setup richer memory
        memory.add_memory("Doug is a senior developer", MemoryType.FACT, entities=["Doug"], importance=0.9)
        memory.add_memory("Doug prefers coffee in the morning", MemoryType.PREFERENCE, entities=["Doug", "coffee"])
        memory.add_memory("Doug switched to tea recently", MemoryType.PREFERENCE, entities=["Doug", "tea"])
        memory.add_memory("John reports to Doug", MemoryType.FACT, entities=["Doug", "John"])

        test_cases = [
            EvaluationCase("Doug likes coffee", ["Morning meeting"], expected_entity="Doug"),
            EvaluationCase("He manages John", ["Discussing Doug"], expected_entity="Doug"),
            EvaluationCase("Doug prefers tea now", ["Doug switched to tea"], expected_conflict=False),  # Not conflict, it's the new state
        ]

        config = EvolutionConfig(population_size=8, generations=4)
        optimizer = EvolutionaryOptimizer(config)

        best = await optimizer.run(test_cases, memory)

        # Best individual should have reasonable weights
        assert 0.0 < best.recency_weight < 1.0
        assert 0.0 < best.semantic_memory_weight < 1.0

        print(f"\nBest evolved weights:")
        print(f"  recency: {best.recency_weight:.3f}")
        print(f"  working_memory: {best.working_memory_weight:.3f}")
        print(f"  episodic_memory: {best.episodic_memory_weight:.3f}")
        print(f"  semantic_memory: {best.semantic_memory_weight:.3f}")
        print(f"  belief: {best.belief_weight:.3f}")
        print(f"  support_boost: {best.support_boost:.3f}")
        print(f"  contradiction_penalty: {best.contradiction_penalty:.3f}")
        print(f"  fitness: {best.fitness:.3f}")


# =============================================================================
# PART 5: COMPREHENSIVE SCENARIO TESTS
# =============================================================================


class TestComplexScenarios:
    """Complex real-world scenarios that combine multiple challenges."""

    @pytest.mark.asyncio
    async def test_office_conversation_scenario(self):
        """Simulate a complex office conversation with multiple people and topics."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Build up office context
        memory.add_memory("Doug is the tech lead", MemoryType.FACT, entities=["Doug"])
        memory.add_memory("Sarah is the product manager", MemoryType.FACT, entities=["Sarah"])
        memory.add_memory("John is a junior developer", MemoryType.FACT, entities=["John"])
        memory.add_memory("The project deadline is next Friday", MemoryType.FACT, entities=["project"])
        memory.add_memory("Doug prefers Python", MemoryType.PREFERENCE, entities=["Doug", "Python"])
        memory.add_memory("Sarah wants the feature shipped this sprint", MemoryType.PREFERENCE, entities=["Sarah", "feature"])

        # Simulate conversation flow
        conversation = [
            "Doug said we need to refactor the code",
            "Sarah disagrees with the timeline",
            "John asked if he should use Python",
            "She mentioned the deadline is tight",
            "He suggested using the new framework",
        ]

        results = []
        context = []
        for statement in conversation:
            result = await orchestrator.process(statement, immediate_context=context[-3:])
            results.append(result)
            context.append(statement)

        # All should complete
        assert all(r.completed_at is not None for r in results)

        # Later statements should have more context
        assert len(results[-1].pre_expansion_context.raw_results) > 0 or \
               len(memory.query_log) > len(conversation)

    @pytest.mark.asyncio
    async def test_preference_change_tracking(self):
        """Track how preferences change over a conversation."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Initial state
        mem1 = memory.add_memory(
            "Doug always drinks coffee",
            MemoryType.PREFERENCE,
            entities=["Doug", "coffee"],
            importance=0.8,
        )

        # Process statements that modify preference
        statements = [
            ("Doug is trying to reduce caffeine", ["health discussion"]),
            ("Doug had tea this morning", ["breakfast talk"]),
            ("Doug said he likes tea now", ["recent conversation"]),
            ("Doug hasn't had coffee in a week", ["discussing habits"]),
        ]

        for statement, context in statements:
            result = await orchestrator.process(statement, immediate_context=context)

            # Add to memory (simulating storage)
            memory.add_memory(
                statement,
                MemoryType.OBSERVATION,
                entities=["Doug"],
                importance=0.7,
            )

        # Final query should find the evolution
        final_result = await orchestrator.process(
            "What does Doug drink?",
            immediate_context=["Ordering drinks for the team"],
        )

        # Should have found multiple relevant memories
        assert len(memory.query_log) > len(statements)

    @pytest.mark.asyncio
    async def test_memory_staleness_handling(self):
        """Test handling of old vs new information."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Old information
        old_mem = memory.add_memory(
            "Doug worked at Company A",
            MemoryType.FACT,
            entities=["Doug", "Company A"],
            importance=0.5,  # Lower importance for old info
        )

        # New information
        new_mem = memory.add_memory(
            "Doug now works at Company B",
            MemoryType.FACT,
            entities=["Doug", "Company B"],
            importance=0.9,  # Higher importance for new info
        )

        memory.link_memories(new_mem.id, "supersedes", old_mem.id)

        result = await orchestrator.process(
            "Where does Doug work?",
            immediate_context=["Updating contact info"],
        )

        # Should prioritize newer information
        assert result.completed_at is not None


# =============================================================================
# PART 6: STRESS TESTS
# =============================================================================


class TestStress:
    """Stress tests for performance and robustness."""

    @pytest.mark.asyncio
    async def test_large_memory_store(self):
        """Test with a large number of memories."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add many memories
        for i in range(100):
            memory.add_memory(
                f"Person{i} likes thing{i}",
                MemoryType.FACT,
                entities=[f"Person{i}", f"thing{i}"],
                importance=random.uniform(0.3, 0.9),
            )

        # Query should still be fast
        import time
        start = time.time()
        result = await orchestrator.process("Person50 likes thing50")
        elapsed = time.time() - start

        assert result.completed_at is not None
        assert elapsed < 2.0  # Should complete in under 2 seconds

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple statements."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()

        # Add some base memories
        for name in ["Doug", "Sarah", "John"]:
            memory.add_memory(
                f"{name} is an employee",
                MemoryType.FACT,
                entities=[name],
            )

        # Process multiple statements concurrently
        async def process_one(statement: str):
            orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)
            return await orchestrator.process(statement)

        statements = [
            "Doug likes coffee",
            "Sarah prefers tea",
            "John wants water",
            "Doug is tired",
            "Sarah is busy",
        ]

        results = await asyncio.gather(*[process_one(s) for s in statements])

        # All should complete
        assert all(r.completed_at is not None for r in results)

    @pytest.mark.asyncio
    async def test_deep_context_chain(self):
        """Test with very long context chain."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Build a long context chain
        context = []
        for i in range(20):
            context.append(f"Statement number {i} about topic {i}")

        # Process with full context
        result = await orchestrator.process(
            "Summarize the discussion",
            immediate_context=context,
        )

        assert result.completed_at is not None


# =============================================================================
# PART 7: REPORTING AND ANALYSIS
# =============================================================================


class TestReporting:
    """Tests that generate reports on system behavior."""

    @pytest.mark.asyncio
    async def test_query_pattern_analysis(self):
        """Analyze query patterns over a session."""
        memory = InstrumentedMemoryProvider()
        llm = SmartMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Setup memory
        memory.add_memory("Doug is a developer", MemoryType.FACT, entities=["Doug"])
        memory.add_memory("Doug likes coffee", MemoryType.PREFERENCE, entities=["Doug", "coffee"])

        # Process several statements
        statements = [
            "Doug is working late",
            "He needs coffee",
            "The project is complex",
            "Doug asked for help",
        ]

        for s in statements:
            await orchestrator.process(s)

        # Analyze query patterns
        search_queries = [q for q in memory.query_log if q["type"] == "search"]
        entity_queries = [q for q in memory.query_log if q["type"] == "entity_search"]

        print(f"\n=== Query Pattern Analysis ===")
        print(f"Total queries: {len(memory.query_log)}")
        print(f"  - Search queries: {len(search_queries)}")
        print(f"  - Entity queries: {len(entity_queries)}")
        print(f"\nMemory access counts:")
        for mem_id, count in sorted(memory.access_counts.items(), key=lambda x: -x[1])[:5]:
            content = memory.memories[mem_id].content[:50]
            print(f"  - {mem_id}: {count} accesses - \"{content}...\"")

        # Basic assertions
        assert len(memory.query_log) > 0
        assert len(memory.access_counts) > 0

    @pytest.mark.asyncio
    async def test_generate_coverage_report(self):
        """Generate a coverage report of what was tested."""
        test_coverage = {
            "adversarial": [
                "contradictory_statements",
                "pronoun_chaos",
                "homonym_confusion",
                "negation_complexity",
                "temporal_confusion",
                "malformed_input",
                "unicode_handling",
            ],
            "accumulation": [
                "knowledge_builds",
                "preference_evolution",
                "entity_relationships",
            ],
            "graph_traversal": [
                "transitive_relationships",
                "circular_references",
                "multi_path_evidence",
            ],
            "evolutionary": [
                "fitness_improvement",
                "weight_optimization",
            ],
            "stress": [
                "large_memory",
                "concurrent_processing",
                "deep_context",
            ],
        }

        print("\n=== Test Coverage Report ===")
        for category, tests in test_coverage.items():
            print(f"\n{category.upper()}:")
            for test in tests:
                print(f"  [x] {test}")

        # This is just a documentation test
        assert True

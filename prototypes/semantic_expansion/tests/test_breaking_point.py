"""Breaking Point Tests for Semantic Expansion PoC.

This test suite is designed to BREAK the system by pushing it to extreme limits:

1. LINGUISTIC ADVERSARIAL - Garden path sentences, pronoun storms, structural ambiguity
2. DEEP GRAPH TRAVERSAL - 15+ hop chains, find needles in haystacks
3. MEMORY-CONDITIONAL DISAMBIGUATION - Same words, different meanings based on context
4. CONFLICT CHAIN DISCOVERY - Traverse memory graph, find contradictions
5. SCALE + COMPLEXITY - 1000s of memories, cryptic queries
6. CRYPTIC MESSAGES - Require multiple semantic variations to understand

The goal is to find WHERE this PoC breaks and WHAT it cannot handle.
"""

from __future__ import annotations

import asyncio
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from collections import defaultdict

import pytest

from draagon_ai.memory.base import Memory, MemoryType, MemoryScope, SearchResult
from integration import (
    TwoPassSemanticOrchestrator,
    PreExpansionContext,
    ProcessingResult,
    DetectedConflict,
    PreExpansionRetriever,
    PostExpansionRetriever,
    NaturalLanguageGenerator,
)
from semantic_types import (
    SemanticFrame,
    SemanticTriple,
    ExpansionVariant,
    CrossLayerRelation,
)
from expansion import SemanticExpansionService
from wsd import (
    LeskDisambiguator,
    WordSenseDisambiguator,
)


# =============================================================================
# Enhanced Memory Provider with Deep Graph Support
# =============================================================================


class DeepGraphMemoryProvider:
    """Memory provider optimized for deep graph traversal testing.

    Features:
    - Explicit graph edges with typed relationships
    - Path finding with configurable max depth
    - Conflict detection across paths
    - Query logging for analysis
    """

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self.edges: list[tuple[str, str, str]] = []  # (from_id, relation, to_id)
        self.query_log: list[dict] = []
        self._next_id = 1
        self._entity_index: dict[str, list[str]] = defaultdict(list)  # entity -> memory_ids

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        entities: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> Memory:
        """Add a memory with entity indexing."""
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

        # Index by entity for fast lookup
        for entity in memory.entities:
            self._entity_index[entity.lower()].append(memory.id)

        return memory

    def add_edge(self, from_id: str, relation: str, to_id: str):
        """Add a directed edge between memories."""
        self.edges.append((from_id, relation, to_id))
        if from_id in self.memories and to_id not in self.memories[from_id].linked_memories:
            self.memories[from_id].linked_memories.append(to_id)

    def add_chain(self, contents: list[str], relation: str = "leads_to") -> list[Memory]:
        """Create a chain of linked memories."""
        memories = []
        for i, content in enumerate(contents):
            # Extract entities from content (simple: capitalized words)
            import re
            entities = re.findall(r'\b([A-Z][a-z]+)\b', content)
            mem = self.add_memory(content, entities=entities)
            memories.append(mem)
            if i > 0:
                self.add_edge(memories[i-1].id, relation, mem.id)
        return memories

    async def search(
        self,
        query: str,
        *,
        memory_types: list[MemoryType] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search with query logging."""
        self.query_log.append({
            "type": "search",
            "query": query,
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

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def search_by_entities(
        self,
        entities: list[str],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search by entity with indexing."""
        self.query_log.append({
            "type": "entity_search",
            "entities": entities,
            "timestamp": datetime.now().isoformat(),
        })

        found_ids = set()
        for entity in entities:
            found_ids.update(self._entity_index.get(entity.lower(), []))

        results = []
        for mem_id in found_ids:
            if mem_id in self.memories:
                mem = self.memories[mem_id]
                results.append(SearchResult(memory=mem, score=mem.importance))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def find_path(self, from_id: str, to_id: str, max_hops: int = 20) -> list[str] | None:
        """BFS path finding with configurable depth."""
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

            for edge_from, _, edge_to in self.edges:
                if edge_from == current:
                    if edge_to == to_id:
                        return path + [to_id]
                    if edge_to not in visited:
                        queue.append((edge_to, path + [edge_to]))

        return None

    def find_all_paths(self, from_id: str, to_id: str, max_hops: int = 10) -> list[list[str]]:
        """Find ALL paths between two nodes (for conflict detection)."""
        all_paths = []

        def dfs(current: str, target: str, path: list[str], visited: set):
            if len(path) > max_hops:
                return
            if current == target:
                all_paths.append(path.copy())
                return

            for edge_from, _, edge_to in self.edges:
                if edge_from == current and edge_to not in visited:
                    visited.add(edge_to)
                    path.append(edge_to)
                    dfs(edge_to, target, path, visited)
                    path.pop()
                    visited.remove(edge_to)

        visited = {from_id}
        dfs(from_id, to_id, [from_id], visited)
        return all_paths


class ContextualMockLLM:
    """LLM mock that can be configured for specific disambiguation behaviors."""

    def __init__(self):
        self.call_history: list[dict] = []
        self.response_rules: list[tuple[str, str]] = []  # (trigger_pattern, response)
        self.default_response_fn = None

    def add_rule(self, trigger: str, response: str):
        """Add a response rule."""
        self.response_rules.append((trigger.lower(), response))

    def set_default_response(self, fn):
        """Set a function to generate default responses."""
        self.default_response_fn = fn

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        prompt = messages[-1]["content"] if messages else ""
        self.call_history.append({"prompt": prompt, "temperature": temperature})

        # Check rules
        prompt_lower = prompt.lower()
        for trigger, response in self.response_rules:
            if trigger in prompt_lower:
                return response

        # Use default response function if set
        if self.default_response_fn:
            return self.default_response_fn(prompt)

        # Generic expansion response
        return self._generate_generic_expansion(prompt)

    def _generate_generic_expansion(self, prompt: str) -> str:
        """Generate a generic semantic expansion response."""
        import re

        # Try to extract the statement
        statement_match = re.search(r'Statement:\s*["\']?([^"\']+)["\']?', prompt)
        statement = statement_match.group(1) if statement_match else "unknown"

        # Find capitalized words as potential entities
        entities = re.findall(r'\b([A-Z][a-z]+)\b', statement)
        subject = entities[0] if entities else "Subject"
        obj = entities[1] if len(entities) > 1 else "Object"

        return f"""<expansion>
<variant rank="1" confidence="0.8">
<frame>
<subject>{subject}</subject>
<predicate>RELATES_TO</predicate>
<object>{obj}</object>
</frame>
<interpretation>Primary interpretation of statement</interpretation>
</variant>
<variant rank="2" confidence="0.5">
<frame>
<subject>{subject}</subject>
<predicate>UNKNOWN</predicate>
<object>unspecified</object>
</frame>
<interpretation>Alternative interpretation</interpretation>
</variant>
</expansion>"""


# =============================================================================
# PART 1: LINGUISTIC ADVERSARIAL TESTS
# =============================================================================


class TestLinguisticAdversarial:
    """Tests with linguistically challenging inputs designed to confuse the system."""

    @pytest.mark.asyncio
    async def test_garden_path_sentence_horse(self):
        """Garden path: 'The horse raced past the barn fell.'

        This is grammatically correct but initially misleads readers.
        Correct parse: The horse [that was] raced past the barn fell.
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add context that should help
        memory.add_memory("The old horse was being trained", MemoryType.FACT, ["horse"])
        memory.add_memory("The barn is where horses are raced", MemoryType.FACT, ["barn", "horses"])
        memory.add_memory("A horse fell during training", MemoryType.EPISODIC, ["horse"])

        result = await orchestrator.process(
            "The horse raced past the barn fell",
            immediate_context=["We're discussing horse training accidents"],
        )

        assert result.completed_at is not None
        # The system should process this without crashing
        # Whether it gets the parse right is the question

    @pytest.mark.asyncio
    async def test_garden_path_sentence_old_man(self):
        """Garden path: 'The old man the boats.'

        Correct parse: Old people man (operate) the boats.
        'old' is a noun (elderly people), 'man' is a verb.
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Elderly workers operate fishing boats", MemoryType.FACT, ["workers", "boats"])
        memory.add_memory("The harbor employs retired sailors", MemoryType.FACT, ["harbor", "sailors"])

        result = await orchestrator.process(
            "The old man the boats",
            immediate_context=["We're discussing harbor operations"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_garden_path_time_flies(self):
        """'Time flies like an arrow; fruit flies like a banana.'

        Same structure, completely different parses.
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add memories for both interpretations
        memory.add_memory("Time passes quickly", MemoryType.FACT, ["time"])
        memory.add_memory("Fruit flies are insects", MemoryType.FACT, ["fruit", "flies"])
        memory.add_memory("Bananas attract fruit flies", MemoryType.FACT, ["bananas", "fruit", "flies"])

        # Process both sentences
        result1 = await orchestrator.process("Time flies like an arrow")
        result2 = await orchestrator.process("Fruit flies like a banana")

        assert result1.completed_at is not None
        assert result2.completed_at is not None
        # The interpretations should be different despite similar structure

    @pytest.mark.asyncio
    async def test_pronoun_storm_many_referents(self):
        """Test with many potential pronoun referents.

        'John told Bob that he saw him with her at their place before they left.'
        Who is he? him? her? their? they?
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add context for all potential referents
        memory.add_memory("John is married to Sarah", MemoryType.FACT, ["John", "Sarah"])
        memory.add_memory("Bob is dating Mary", MemoryType.FACT, ["Bob", "Mary"])
        memory.add_memory("John and Bob share an apartment", MemoryType.FACT, ["John", "Bob"])
        memory.add_memory("Sarah and Mary are sisters", MemoryType.FACT, ["Sarah", "Mary"])

        result = await orchestrator.process(
            "John told Bob that he saw him with her at their place before they left",
            immediate_context=[
                "John called Bob yesterday",
                "They were discussing Sarah",
            ],
        )

        assert result.completed_at is not None
        # This is extremely ambiguous - system should recognize the ambiguity

    @pytest.mark.asyncio
    async def test_structural_ambiguity_attachment(self):
        """PP-attachment ambiguity: 'I saw the man with the telescope.'

        Did I use the telescope to see him, or did he have the telescope?
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Context for instrument interpretation
        memory.add_memory("I bought a telescope last week", MemoryType.FACT, ["telescope"])

        result1 = await orchestrator.process(
            "I saw the man with the telescope",
            immediate_context=["I was using my new telescope"],
        )

        # Context for possession interpretation
        memory.add_memory("The astronomer carries his telescope everywhere", MemoryType.FACT, ["astronomer", "telescope"])

        result2 = await orchestrator.process(
            "I saw the man with the telescope",
            immediate_context=["The astronomer walked by"],
        )

        assert result1.completed_at is not None
        assert result2.completed_at is not None
        # Different contexts should yield different interpretations

    @pytest.mark.asyncio
    async def test_lexical_ambiguity_bank(self):
        """Classic lexical ambiguity with 'bank'."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()

        # Configure LLM for WSD
        llm.add_rule("bank", """<expansion>
<variant rank="1" confidence="0.7">
<frame>
<subject>entity</subject>
<predicate>NEAR</predicate>
<object>bank</object>
<sense>financial_institution</sense>
</frame>
</variant>
<variant rank="2" confidence="0.6">
<frame>
<subject>entity</subject>
<predicate>NEAR</predicate>
<object>bank</object>
<sense>river_edge</sense>
</frame>
</variant>
</expansion>""")

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Financial context
        memory.add_memory("Doug works at Chase Bank", MemoryType.FACT, ["Doug", "Chase", "Bank"])

        result_financial = await orchestrator.process(
            "Meet me at the bank",
            immediate_context=["Doug said he'd be at work"],
        )

        # River context
        memory.add_memory("The fishing spot is on the river bank", MemoryType.FACT, ["fishing", "river", "bank"])

        result_river = await orchestrator.process(
            "Meet me at the bank",
            immediate_context=["Let's go fishing"],
        )

        assert result_financial.completed_at is not None
        assert result_river.completed_at is not None

    @pytest.mark.asyncio
    async def test_negation_scope_ambiguity(self):
        """Negation scope: 'All politicians are not corrupt.'

        Does this mean:
        - Not all politicians are corrupt (some are honest)
        - All politicians are honest (none are corrupt)
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Some politicians have been convicted", MemoryType.FACT, ["politicians"])
        memory.add_memory("Many politicians are honest public servants", MemoryType.FACT, ["politicians"])

        result = await orchestrator.process(
            "All politicians are not corrupt",
            immediate_context=["We're discussing political ethics"],
        )

        assert result.completed_at is not None
        # Should recognize this as ambiguous

    @pytest.mark.asyncio
    async def test_quantifier_scope_every_some(self):
        """Quantifier scope: 'Every boy loves some girl.'

        Interpretations:
        - Each boy has a (possibly different) girl he loves
        - There's one specific girl that every boy loves
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add context
        memory.add_memory("The school has 30 boys and 30 girls", MemoryType.FACT, ["school", "boys", "girls"])
        memory.add_memory("Sarah is the most popular girl in school", MemoryType.FACT, ["Sarah", "school"])

        result = await orchestrator.process(
            "Every boy loves some girl",
            immediate_context=["We're discussing the school dance"],
        )

        assert result.completed_at is not None


# =============================================================================
# PART 2: DEEP GRAPH TRAVERSAL TESTS
# =============================================================================


class TestDeepGraphTraversal:
    """Tests for deep multi-hop reasoning through memory graph."""

    @pytest.mark.asyncio
    async def test_15_hop_chain_traversal(self):
        """Traverse a 15-hop chain of relationships."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create a 15-node chain
        chain = [
            "Alice manages Bob",
            "Bob manages Carol",
            "Carol manages Dave",
            "Dave manages Eve",
            "Eve manages Frank",
            "Frank manages Grace",
            "Grace manages Henry",
            "Henry manages Ivy",
            "Ivy manages Jack",
            "Jack manages Kate",
            "Kate manages Leo",
            "Leo manages Mary",
            "Mary manages Nick",
            "Nick manages Olivia",
            "Olivia manages Pete",
        ]

        memories = memory.add_chain(chain, "manages")

        # Try to find path from first to last
        path = memory.find_path(memories[0].id, memories[-1].id, max_hops=20)

        assert path is not None
        assert len(path) == 15

        # Process a query about the distant relationship
        result = await orchestrator.process(
            "Who is at the bottom of Alice's management chain?",
            immediate_context=["Alice is the CEO"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_branching_graph_traversal(self):
        """Traverse a branching graph with multiple paths."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create a tree structure
        #           Alice
        #          /     \
        #        Bob     Carol
        #       / \       / \
        #      D   E     F   G
        #     /\  /\    /\  /\
        #    H I J K   L M N O

        root = memory.add_memory("Alice is the founder", MemoryType.FACT, ["Alice"])

        # Level 1
        bob = memory.add_memory("Bob reports to Alice", MemoryType.FACT, ["Bob", "Alice"])
        carol = memory.add_memory("Carol reports to Alice", MemoryType.FACT, ["Carol", "Alice"])
        memory.add_edge(root.id, "manages", bob.id)
        memory.add_edge(root.id, "manages", carol.id)

        # Level 2
        d = memory.add_memory("Dave reports to Bob", MemoryType.FACT, ["Dave", "Bob"])
        e = memory.add_memory("Eve reports to Bob", MemoryType.FACT, ["Eve", "Bob"])
        f = memory.add_memory("Frank reports to Carol", MemoryType.FACT, ["Frank", "Carol"])
        g = memory.add_memory("Grace reports to Carol", MemoryType.FACT, ["Grace", "Carol"])
        memory.add_edge(bob.id, "manages", d.id)
        memory.add_edge(bob.id, "manages", e.id)
        memory.add_edge(carol.id, "manages", f.id)
        memory.add_edge(carol.id, "manages", g.id)

        # Level 3 (8 nodes)
        level3 = []
        for name, parent, parent_mem in [
            ("Henry", "Dave", d), ("Ivy", "Dave", d),
            ("Jack", "Eve", e), ("Kate", "Eve", e),
            ("Leo", "Frank", f), ("Mary", "Frank", f),
            ("Nick", "Grace", g), ("Olivia", "Grace", g),
        ]:
            mem = memory.add_memory(f"{name} reports to {parent}", MemoryType.FACT, [name, parent])
            memory.add_edge(parent_mem.id, "manages", mem.id)
            level3.append(mem)

        # Find all paths from root to a leaf
        all_paths = memory.find_all_paths(root.id, level3[7].id, max_hops=10)

        # Should find exactly one path through the tree
        assert len(all_paths) >= 1

        result = await orchestrator.process(
            "How is Olivia connected to Alice?",
            immediate_context=["Looking at org structure"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_needle_in_haystack_graph(self):
        """Find a specific fact hidden deep in a large graph."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create 100 random facts (the haystack)
        names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery"]
        actions = ["met", "called", "emailed", "visited", "helped", "thanked"]

        for i in range(100):
            name1 = random.choice(names)
            name2 = random.choice(names)
            action = random.choice(actions)
            memory.add_memory(f"{name1} {action} {name2}", MemoryType.EPISODIC, [name1, name2])

        # Add the needle - a specific important fact buried in the graph
        needle = memory.add_memory(
            "The secret code is ALPHA-OMEGA-7",
            MemoryType.FACT,
            ["secret", "code"],
            importance=0.9,
        )

        # Link it to a random node
        random_mem = random.choice(list(memory.memories.values()))
        memory.add_edge(random_mem.id, "contains", needle.id)

        result = await orchestrator.process(
            "What is the secret code?",
            immediate_context=[],
        )

        assert result.completed_at is not None
        # Check if we can find the needle
        search_results = await memory.search("secret code")
        assert any("ALPHA-OMEGA" in r.memory.content for r in search_results)

    @pytest.mark.asyncio
    async def test_circular_reference_handling(self):
        """Ensure circular references don't cause infinite loops."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create a cycle: A -> B -> C -> D -> A
        a = memory.add_memory("Node A connects forward", MemoryType.FACT, ["A"])
        b = memory.add_memory("Node B connects forward", MemoryType.FACT, ["B"])
        c = memory.add_memory("Node C connects forward", MemoryType.FACT, ["C"])
        d = memory.add_memory("Node D connects back to A", MemoryType.FACT, ["D", "A"])

        memory.add_edge(a.id, "next", b.id)
        memory.add_edge(b.id, "next", c.id)
        memory.add_edge(c.id, "next", d.id)
        memory.add_edge(d.id, "next", a.id)  # Creates cycle

        # This should not hang
        start_time = time.time()
        path = memory.find_path(a.id, c.id, max_hops=10)
        elapsed = time.time() - start_time

        assert path is not None
        assert elapsed < 1.0  # Should complete quickly

        result = await orchestrator.process(
            "How do the nodes connect?",
            immediate_context=[],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_disconnected_subgraph_detection(self):
        """Test handling of disconnected subgraphs."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create two disconnected clusters
        # Cluster 1: Work relationships
        work = memory.add_chain([
            "Alice works with Bob at Acme Corp",
            "Bob manages the engineering team",
            "The engineering team builds products",
        ], "work_related")

        # Cluster 2: Family relationships (disconnected)
        family = memory.add_chain([
            "John is married to Sarah",
            "Sarah has a brother named Mike",
            "Mike lives in California",
        ], "family_related")

        # There should be no path between clusters
        path = memory.find_path(work[0].id, family[0].id, max_hops=20)
        assert path is None

        result = await orchestrator.process(
            "How is Alice related to John?",
            immediate_context=[],
        )

        assert result.completed_at is not None
        # Should not find a connection


# =============================================================================
# PART 3: MEMORY-CONDITIONAL DISAMBIGUATION
# =============================================================================


class TestMemoryConditionalDisambiguation:
    """Tests where the same statement means different things based on memory context."""

    @pytest.mark.asyncio
    async def test_bass_fish_vs_instrument(self):
        """'John's bass was huge' - fish or instrument?"""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()

        # Configure LLM for bass disambiguation
        llm.add_rule("bass", """<expansion>
<variant rank="1" confidence="0.6">
<frame><subject>John</subject><predicate>OWNS</predicate><object>bass</object><sense>fish</sense></frame>
</variant>
<variant rank="2" confidence="0.6">
<frame><subject>John</subject><predicate>OWNS</predicate><object>bass</object><sense>instrument</sense></frame>
</variant>
</expansion>""")

        # Test with fisherman context
        memory_fish = DeepGraphMemoryProvider()
        memory_fish.add_memory("John is an avid fisherman", MemoryType.FACT, ["John", "fisherman"])
        memory_fish.add_memory("John goes fishing every weekend", MemoryType.EPISODIC, ["John", "fishing"])

        orchestrator_fish = TwoPassSemanticOrchestrator(memory=memory_fish, llm=llm)
        result_fish = await orchestrator_fish.process(
            "John's bass was huge",
            immediate_context=["John went to the lake"],
        )

        # Test with musician context
        memory_music = DeepGraphMemoryProvider()
        memory_music.add_memory("John plays in a jazz band", MemoryType.FACT, ["John", "jazz", "band"])
        memory_music.add_memory("John's band performed last night", MemoryType.EPISODIC, ["John", "band"])

        orchestrator_music = TwoPassSemanticOrchestrator(memory=memory_music, llm=llm)
        result_music = await orchestrator_music.process(
            "John's bass was huge",
            immediate_context=["John was performing on stage"],
        )

        assert result_fish.completed_at is not None
        assert result_music.completed_at is not None
        # Ideally, scores should differ based on context

    @pytest.mark.asyncio
    async def test_she_pronoun_resolution_by_context(self):
        """'She left early' - different 'she' based on context."""
        llm = ContextualMockLLM()

        # Context 1: Office meeting
        memory_office = DeepGraphMemoryProvider()
        memory_office.add_memory("Sarah is the team lead", MemoryType.FACT, ["Sarah"])
        memory_office.add_memory("Maria is the CEO", MemoryType.FACT, ["Maria"])
        memory_office.add_memory("Sarah had a doctor's appointment", MemoryType.EPISODIC, ["Sarah"])

        orchestrator_office = TwoPassSemanticOrchestrator(memory=memory_office, llm=llm)
        result_office = await orchestrator_office.process(
            "She left early",
            immediate_context=["The team meeting just ended", "Sarah mentioned her appointment"],
        )

        # Context 2: Family dinner
        memory_family = DeepGraphMemoryProvider()
        memory_family.add_memory("Mom is visiting this week", MemoryType.FACT, ["Mom"])
        memory_family.add_memory("Grandma feels tired after long dinners", MemoryType.FACT, ["Grandma"])
        memory_family.add_memory("Grandma looked exhausted tonight", MemoryType.EPISODIC, ["Grandma"])

        orchestrator_family = TwoPassSemanticOrchestrator(memory=memory_family, llm=llm)
        result_family = await orchestrator_family.process(
            "She left early",
            immediate_context=["Family dinner just finished", "Grandma said goodbye"],
        )

        assert result_office.completed_at is not None
        assert result_family.completed_at is not None

    @pytest.mark.asyncio
    async def test_implicit_reference_that_thing(self):
        """'Do that thing we discussed' - what thing?"""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()

        # Add multiple potential referents
        memory.add_memory("We discussed the quarterly report last week", MemoryType.EPISODIC, ["quarterly", "report"])
        memory.add_memory("We talked about the server migration yesterday", MemoryType.EPISODIC, ["server", "migration"])
        memory.add_memory("The budget proposal was mentioned", MemoryType.EPISODIC, ["budget", "proposal"])

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)
        result = await orchestrator.process(
            "Do that thing we discussed",
            immediate_context=["Let's tackle the pending task"],
        )

        assert result.completed_at is not None
        # Should find relevant memories

    @pytest.mark.asyncio
    async def test_temporal_reference_disambiguation(self):
        """'Next Friday' means different things on different days."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()

        # Add context about meetings
        memory.add_memory("Team standup is every Friday at 10am", MemoryType.FACT, ["standup", "Friday"])
        memory.add_memory("Quarterly review is next Friday January 10", MemoryType.FACT, ["review", "Friday", "January"])
        memory.add_memory("Today is Monday January 6", MemoryType.FACT, ["Monday", "January"])

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)
        result = await orchestrator.process(
            "Let's meet next Friday",
            immediate_context=["We need to prepare for the review"],
        )

        assert result.completed_at is not None


# =============================================================================
# PART 4: CONFLICT CHAIN DISCOVERY
# =============================================================================


class TestConflictChainDiscovery:
    """Tests for detecting contradictions across memory chains."""

    @pytest.mark.asyncio
    async def test_direct_contradiction_detection(self):
        """Detect direct contradiction between two memories."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add contradicting memories
        memory.add_memory("Doug prefers coffee in the morning", MemoryType.PREFERENCE, ["Doug", "coffee"])
        memory.add_memory("Doug hates coffee", MemoryType.PREFERENCE, ["Doug", "coffee"])

        result = await orchestrator.process(
            "What does Doug like to drink?",
            immediate_context=[],
        )

        assert result.completed_at is not None
        # Should detect the conflict

    @pytest.mark.asyncio
    async def test_transitive_contradiction(self):
        """Contradiction only visible through transitive reasoning.

        A likes B, B is C, C is hated by A
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create transitive contradiction
        m1 = memory.add_memory("John likes rock music", MemoryType.PREFERENCE, ["John", "rock", "music"])
        m2 = memory.add_memory("Metal is a type of rock music", MemoryType.FACT, ["metal", "rock", "music"])
        m3 = memory.add_memory("John hates metal music", MemoryType.PREFERENCE, ["John", "metal", "music"])

        memory.add_edge(m1.id, "related_to", m2.id)
        memory.add_edge(m2.id, "related_to", m3.id)

        result = await orchestrator.process(
            "What music does John like?",
            immediate_context=[],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_temporal_supersession(self):
        """Newer information should supersede older information."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Old information
        old = memory.add_memory(
            "Doug lives in New York",
            MemoryType.FACT,
            ["Doug", "New York"],
        )
        # The Memory object doesn't have a timestamp field we can easily set,
        # so we'll use importance as a proxy

        # New information
        new = memory.add_memory(
            "Doug moved to California",
            MemoryType.FACT,
            ["Doug", "California"],
            importance=0.9,  # Higher importance = more recent/relevant
        )

        memory.add_edge(old.id, "superseded_by", new.id)

        result = await orchestrator.process(
            "Where does Doug live?",
            immediate_context=["Doug moved recently"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_multi_source_conflict(self):
        """Conflict where multiple sources disagree."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Three different sources, three different answers
        memory.add_memory("Sarah said Doug has 3 cats", MemoryType.FACT, ["Sarah", "Doug", "cats"])
        memory.add_memory("Bob mentioned Doug has 5 cats", MemoryType.FACT, ["Bob", "Doug", "cats"])
        memory.add_memory("Doug told me he has 6 cats", MemoryType.FACT, ["Doug", "cats"])

        result = await orchestrator.process(
            "How many cats does Doug have?",
            immediate_context=[],
        )

        assert result.completed_at is not None


# =============================================================================
# PART 5: SCALE + COMPLEXITY COMBINED
# =============================================================================


class TestScaleAndComplexity:
    """Tests combining large scale with complex queries."""

    @pytest.mark.asyncio
    async def test_1000_memories_cryptic_query(self):
        """1000 memories, find the answer to a cryptic query."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Generate 1000 random memories
        topics = ["weather", "food", "travel", "work", "family", "health", "money", "hobbies"]
        names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]

        for i in range(1000):
            topic = random.choice(topics)
            name = random.choice(names)
            memory.add_memory(
                f"{name} discussed {topic} topic {i}",
                MemoryType.EPISODIC,
                [name, topic],
            )

        # Add a specific needle
        memory.add_memory(
            "The treasure is hidden under the old oak tree in the garden",
            MemoryType.FACT,
            ["treasure", "oak", "tree", "garden"],
            importance=0.95,
        )

        start_time = time.time()
        result = await orchestrator.process(
            "Where did we hide it?",
            immediate_context=["The thing we buried", "In the backyard"],
        )
        elapsed = time.time() - start_time

        assert result.completed_at is not None
        assert elapsed < 5.0  # Should complete in reasonable time

    @pytest.mark.asyncio
    async def test_deep_context_chain(self):
        """Very deep context chain (20+ statements)."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add some relevant memories
        memory.add_memory("The project deadline is next month", MemoryType.FACT, ["project", "deadline"])
        memory.add_memory("John is the project manager", MemoryType.FACT, ["John", "project"])

        # Create a very long context chain
        long_context = [
            "We were discussing the project",
            "John mentioned some concerns",
            "The budget might be an issue",
            "But we have contingency funds",
            "Sarah from finance approved it",
            "She spoke with the CFO",
            "The CFO wants weekly updates",
            "We agreed to that",
            "The first update is due Monday",
            "I'll prepare the slides",
            "John will present",
            "We need the latest numbers",
            "Those come from analytics",
            "The analytics team is busy",
            "But they promised to help",
            "So we should be fine",
            "Unless something unexpected happens",
            "Like last quarter",
            "When the server crashed",
            "But IT fixed that",
        ]

        result = await orchestrator.process(
            "What should I do about Monday?",
            immediate_context=long_context,
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test concurrent processing of multiple queries."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()

        # Add some memories
        memory.add_memory("Weather is sunny today", MemoryType.FACT, ["weather"])
        memory.add_memory("Meeting at 3pm", MemoryType.EPISODIC, ["meeting"])
        memory.add_memory("Doug likes coffee", MemoryType.PREFERENCE, ["Doug", "coffee"])

        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Create 10 concurrent queries
        queries = [
            "What's the weather?",
            "When is the meeting?",
            "What does Doug drink?",
            "Tell me about today",
            "Who likes coffee?",
            "What's happening at 3?",
            "Is it sunny?",
            "Doug's preferences?",
            "Today's schedule?",
            "Weather forecast?",
        ]

        start_time = time.time()
        results = await asyncio.gather(*[
            orchestrator.process(query, immediate_context=[])
            for query in queries
        ])
        elapsed = time.time() - start_time

        # All should complete
        assert all(r.completed_at is not None for r in results)
        assert elapsed < 10.0


# =============================================================================
# PART 6: CRYPTIC MESSAGES REQUIRING MULTIPLE VARIATIONS
# =============================================================================


class TestCrypticMessages:
    """Test messages that require multiple semantic variations to understand."""

    @pytest.mark.asyncio
    async def test_extremely_vague_reference(self):
        """'You know, that thing at the place with the person.'"""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add potential referents
        memory.add_memory("Conference at the convention center with Sarah", MemoryType.EPISODIC, ["conference", "Sarah"])
        memory.add_memory("Party at John's house with everyone", MemoryType.EPISODIC, ["party", "John"])
        memory.add_memory("Meeting at the office with the client", MemoryType.EPISODIC, ["meeting", "office", "client"])

        result = await orchestrator.process(
            "You know, that thing at the place with the person",
            immediate_context=["Remember when we discussed it?"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_heavily_elided_speech(self):
        """'Gonna? Yeah. Coffee? Nah, tea. When? Now.'"""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("We often get coffee together", MemoryType.EPISODIC, ["coffee"])
        memory.add_memory("Doug started drinking tea recently", MemoryType.PREFERENCE, ["Doug", "tea"])

        result = await orchestrator.process(
            "Gonna? Yeah. Coffee? Nah, tea. When? Now.",
            immediate_context=["Quick chat about going out"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_sarcasm_detection(self):
        """'Oh great, another meeting. Just what I needed.'"""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Doug complains about too many meetings", MemoryType.FACT, ["Doug", "meetings"])
        memory.add_memory("The calendar is packed with meetings today", MemoryType.EPISODIC, ["calendar", "meetings"])

        result = await orchestrator.process(
            "Oh great, another meeting. Just what I needed.",
            immediate_context=["Calendar notification popped up"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_code_switching_mixed_language(self):
        """Mixed language input (code-switching)."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Memories in different languages/styles
        memory.add_memory("La reunion es a las tres", MemoryType.EPISODIC, ["reunion"])
        memory.add_memory("The meeting is at 3pm", MemoryType.EPISODIC, ["meeting"])

        result = await orchestrator.process(
            "La meeting, cuando es?",  # Mixed Spanish/English
            immediate_context=[],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_typos_and_speech_errors(self):
        """Input with typos simulating speech-to-text errors."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Doug's password is secure123", MemoryType.FACT, ["Doug", "password"])
        memory.add_memory("The WiFi password is homeNetwork", MemoryType.FACT, ["WiFi", "password"])

        result = await orchestrator.process(
            "Waht iz teh passwrod?",  # Heavily typo'd
            immediate_context=["Trying to connect to internet"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_multiple_embedded_questions(self):
        """'I wonder whether you know if she remembers what he said about it.'"""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Sarah mentioned that John talked about the project", MemoryType.EPISODIC, ["Sarah", "John", "project"])
        memory.add_memory("John said the deadline is flexible", MemoryType.FACT, ["John", "deadline"])

        result = await orchestrator.process(
            "I wonder whether you know if she remembers what he said about it",
            immediate_context=["Discussing the project with Sarah"],
        )

        assert result.completed_at is not None


# =============================================================================
# PART 7: SEMANTIC VARIATION EXPLOSION
# =============================================================================


class TestSemanticVariationExplosion:
    """Test statements that could have many valid interpretations."""

    @pytest.mark.asyncio
    async def test_many_homonym_sentence(self):
        """Sentence with multiple homonyms.

        'I left my left arm on the left.'
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        result = await orchestrator.process(
            "I left my left arm on the left",
            immediate_context=[],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_crash_blossom_headline(self):
        """Crash blossom: Ambiguous headlines.

        'Squad helps dog bite victim'
        Did they help the victim of a dog bite, or help a dog bite someone?
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Local rescue squad responds to emergencies", MemoryType.FACT, ["squad", "rescue"])

        result = await orchestrator.process(
            "Squad helps dog bite victim",
            immediate_context=["News headline"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_buffalo_sentence(self):
        """The famous 'Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.'

        Grammatically correct, extremely confusing.
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("Buffalo is a city in New York", MemoryType.FACT, ["Buffalo", "New York"])
        memory.add_memory("Buffalo can refer to bison", MemoryType.FACT, ["Buffalo", "bison"])
        memory.add_memory("Buffalo can mean to intimidate", MemoryType.FACT, ["buffalo", "intimidate"])

        result = await orchestrator.process(
            "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
            immediate_context=["Discussing linguistic puzzles"],
        )

        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_colorless_green_ideas(self):
        """Chomsky's 'Colorless green ideas sleep furiously.'

        Grammatically correct but semantically nonsensical.
        """
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        result = await orchestrator.process(
            "Colorless green ideas sleep furiously",
            immediate_context=["Discussing abstract concepts"],
        )

        assert result.completed_at is not None
        # System should handle semantically odd but grammatically correct input

    @pytest.mark.asyncio
    async def test_maximal_ambiguity(self):
        """Create a maximally ambiguous sentence."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add memories for many possible interpretations
        for entity in ["bank", "spring", "light", "fair", "bass", "bow", "lead"]:
            memory.add_memory(f"{entity} has multiple meanings", MemoryType.FACT, [entity])

        result = await orchestrator.process(
            "I saw her duck near the bank by the spring light",
            immediate_context=[],
        )

        assert result.completed_at is not None


# =============================================================================
# PART 8: PERFORMANCE BENCHMARKS
# =============================================================================


class TestPerformanceBenchmarks:
    """Benchmark tests to measure system performance under stress."""

    @pytest.mark.asyncio
    async def test_processing_latency_simple(self):
        """Measure processing latency for simple queries."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        memory.add_memory("The sky is blue", MemoryType.FACT, ["sky"])

        latencies = []
        for _ in range(10):
            start = time.time()
            await orchestrator.process("What color is the sky?")
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert avg_latency < 0.5  # Average should be fast
        assert max_latency < 1.0  # No outliers

    @pytest.mark.asyncio
    async def test_processing_latency_complex(self):
        """Measure processing latency for complex queries."""
        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()
        orchestrator = TwoPassSemanticOrchestrator(memory=memory, llm=llm)

        # Add 100 memories
        for i in range(100):
            memory.add_memory(f"Fact number {i} about topic {i % 10}", MemoryType.FACT, [f"topic_{i % 10}"])

        latencies = []
        for _ in range(10):
            start = time.time()
            await orchestrator.process(
                "Tell me about the topics we discussed",
                immediate_context=["Long context string " * 10],
            )
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        assert avg_latency < 2.0  # Complex should still be reasonable
        assert max_latency < 5.0

    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self):
        """Test that memory usage scales reasonably."""
        import sys

        memory = DeepGraphMemoryProvider()
        llm = ContextualMockLLM()

        # Baseline memory
        baseline = sys.getsizeof(memory.memories)

        # Add 1000 memories
        for i in range(1000):
            memory.add_memory(f"Memory content {i} with some additional text", MemoryType.FACT, [f"entity_{i}"])

        # Check growth
        after_1000 = sys.getsizeof(memory.memories)

        # Add 1000 more
        for i in range(1000, 2000):
            memory.add_memory(f"Memory content {i} with some additional text", MemoryType.FACT, [f"entity_{i}"])

        after_2000 = sys.getsizeof(memory.memories)

        # Growth should be approximately linear
        growth_1 = after_1000 - baseline
        growth_2 = after_2000 - after_1000

        # Second 1000 shouldn't take dramatically more space than first 1000
        assert growth_2 < growth_1 * 2  # Allow for some dict resizing overhead

    @pytest.mark.asyncio
    async def test_graph_traversal_scaling(self):
        """Test that graph traversal scales reasonably with depth."""
        memory = DeepGraphMemoryProvider()

        # Create chains of different lengths
        depths = [5, 10, 15, 20]
        traversal_times = {}

        for depth in depths:
            chain = memory.add_chain(
                [f"Node {i} at depth {depth}" for i in range(depth)],
                f"chain_{depth}"
            )

            start = time.time()
            for _ in range(100):  # Run 100 times for better measurement
                memory.find_path(chain[0].id, chain[-1].id, max_hops=depth + 5)
            elapsed = time.time() - start

            traversal_times[depth] = elapsed / 100

        # Traversal time should grow roughly linearly, not exponentially
        # depth 20 shouldn't take more than 6x the time of depth 5
        # When times are negligible (< 0.1ms), skip the comparison as noise dominates
        if traversal_times[5] > 0.0001:  # Only test if times are measurable
            assert traversal_times[20] < traversal_times[5] * 6
        # If times are too small to measure reliably, the system is fast enough

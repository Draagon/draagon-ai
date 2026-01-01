# TASK-011: Implement Memory Integration Tests (FR-010.2)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P0 (Critical - validates cognitive memory architecture)
**Effort**: 3 days
**Status**: Pending
**Dependencies**: TASK-009 (Real agent fixtures), TASK-010 (Core agent tests)

## Description

Implement integration tests for memory storage, retrieval, reinforcement, and layer promotion with real Neo4jMemoryProvider. Validates the 4-layer cognitive memory architecture works correctly in practice.

**Core Principle:** Test real memory behavior (persistence, search, reinforcement, promotion) with real database.

## Acceptance Criteria

- [ ] Test memory storage and recall across agent sessions
- [ ] Test semantic search returns relevant memories
- [ ] Test memory reinforcement (boost on success, demote on failure)
- [ ] Test layer promotion (working → episodic → semantic → metacognitive)
- [ ] Test layer demotion (failed memories move down)
- [ ] Test TTL enforcement and expiration
- [ ] Test importance-based memory ranking
- [ ] All tests use real Neo4j database (not mocks)

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_memory.py`

**Example Tests:**
```python
@pytest.mark.memory_integration
class TestAgentMemory:
    """Test memory persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_store_and_recall_fact(self, agent, memory_provider, evaluator):
        """Agent stores fact and recalls it later."""

        # Teach agent a fact
        response1 = await agent.process("My birthday is March 15")

        # Verify storage
        memories = await memory_provider.search(
            query="birthday",
            user_id="test_user",
            limit=5,
        )
        assert len(memories) > 0
        assert any("March 15" in m.content for m in memories)

        # Verify recall
        response2 = await agent.process("When is my birthday?")

        result = await evaluator.evaluate_correctness(
            query="When is my birthday?",
            expected_outcome="Agent says March 15",
            actual_response=response2.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_memory_reinforcement_boost(self, agent, memory_provider):
        """Using a memory successfully boosts its importance."""

        # Store initial memory
        memory_id = await memory_provider.store(
            content="Doug prefers Celsius",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.5,
        )

        # Agent uses memory successfully
        response = await agent.process("What temperature unit do I prefer?")

        # Record success
        await memory_provider.record_usage(memory_id, outcome="success")

        # Check importance increased
        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance > 0.5  # Boosted by 0.05

    @pytest.mark.asyncio
    async def test_layer_promotion_working_to_episodic(self, agent, memory_provider):
        """Repeated successful use promotes memory to episodic layer."""

        memory_id = await memory_provider.store(
            content="Restart command: sudo systemctl restart myservice",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            importance=0.6,  # Starts in working memory
        )

        # Use skill multiple times successfully
        for _ in range(5):
            await memory_provider.record_usage(memory_id, outcome="success")

        # Check promoted to episodic layer
        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance >= 0.7  # Promoted to episodic

    @pytest.mark.asyncio
    async def test_layer_promotion_to_semantic(self, agent, memory_provider):
        """Continued success promotes to semantic layer."""

        memory_id = await memory_provider.store(
            content="Common pattern: sudo systemctl restart <service>",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            importance=0.7,  # Starts in episodic
        )

        # Many successful uses
        for _ in range(15):
            await memory_provider.record_usage(memory_id, outcome="success")

        # Check promoted to semantic
        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance >= 0.8

    @pytest.mark.asyncio
    async def test_memory_demotion_on_failure(self, agent, memory_provider):
        """Failed memories get demoted."""

        memory_id = await memory_provider.store(
            content="Broken command: sudo rm -rf everything",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            importance=0.8,  # Starts high
        )

        # Use fails multiple times
        for _ in range(5):
            await memory_provider.record_usage(memory_id, outcome="failure")

        # Check demoted
        updated_memory = await memory_provider.get(memory_id)
        assert updated_memory.importance < 0.8

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_expiration_by_ttl(self, agent, memory_provider, advance_time):
        """Memories expire based on TTL and layer."""

        # Store short-lived working memory
        memory_id = await memory_provider.store(
            content="Temporary note: Call back at 3pm",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.SESSION,
            importance=0.3,  # Working memory (5 min TTL)
        )

        # Fast-forward time
        await advance_time(minutes=6)

        # Memory should be expired/deleted
        memory = await memory_provider.get(memory_id)
        assert memory is None or memory.expired

    @pytest.mark.asyncio
    async def test_semantic_search_relevance(self, agent, memory_provider):
        """Semantic search returns relevant memories, not just keyword matches."""

        # Store memories with different semantic meanings
        await memory_provider.store(
            content="I love cats",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        await memory_provider.store(
            content="I have 3 felines at home",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        await memory_provider.store(
            content="I hate mice",  # Unrelated
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="test_user",
        )

        # Search for "cats" - should find both cat-related memories
        results = await memory_provider.search(
            query="pets cats",
            user_id="test_user",
            limit=5,
        )

        # Should find both cat/feline memories (semantic match)
        assert len(results) >= 2
        assert any("cats" in m.content.lower() or "feline" in m.content.lower() for m in results)

    @pytest.mark.asyncio
    async def test_importance_based_ranking(self, agent, memory_provider):
        """Higher importance memories rank higher in search."""

        # Store memories with different importance
        low_id = await memory_provider.store(
            content="Random fact about cats",
            memory_type=MemoryType.FACT,
            importance=0.3,
            user_id="test_user",
        )

        high_id = await memory_provider.store(
            content="Important cat allergy information",
            memory_type=MemoryType.FACT,
            importance=0.9,
            user_id="test_user",
        )

        # Search
        results = await memory_provider.search(
            query="cats",
            user_id="test_user",
            limit=5,
        )

        # High importance should rank higher
        result_ids = [m.id for m in results]
        assert result_ids.index(high_id) < result_ids.index(low_id)
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_store_and_recall_fact` - Basic persistence
- [ ] `test_memory_reinforcement_boost` - Boost on success
- [ ] `test_layer_promotion_working_to_episodic` - Promotion at 0.7
- [ ] `test_layer_promotion_to_semantic` - Promotion at 0.8
- [ ] `test_memory_demotion_on_failure` - Demotion on failures
- [ ] `test_memory_expiration_by_ttl` - TTL enforcement
- [ ] `test_semantic_search_relevance` - Semantic vector search
- [ ] `test_importance_based_ranking` - Importance ranking

**Performance Tests:**
- [ ] Memory store: <100ms
- [ ] Memory search: <500ms
- [ ] Reinforcement update: <50ms

**Cognitive Tests:**
- [ ] Working memory capacity: Miller's Law (7±2 items accessible)
- [ ] Episodic consolidation: 2-week retention for important memories
- [ ] Semantic permanence: >6 months for frequently used skills
- [ ] Metacognitive stability: Permanent for verified self-knowledge

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_memory.py` - Memory tests

**Modify:**
- `tests/integration/agents/conftest.py` - Add `advance_time` fixture

## Pre-Implementation Work

**CRITICAL: Wire Memory Usage Tracking**

Before implementing these tests, add memory usage tracking to AgentLoop:

1. Add `used_memory_ids` to AgentResponse:
```python
@dataclass
class AgentResponse:
    answer: str
    confidence: float
    used_memory_ids: list[str]  # NEW - which memories were accessed
    # ... other fields
```

2. Track memory access in DecisionEngine or middleware:
```python
# In DecisionEngine.decide() or memory retrieval middleware
async def get_context_memories(query: str) -> tuple[list[Memory], list[str]]:
    memories = await self.memory.search(query, limit=10)
    memory_ids = [m.id for m in memories]
    return memories, memory_ids
```

3. Wire to AgentResponse:
```python
response = AgentResponse(
    answer=final_answer,
    confidence=confidence,
    used_memory_ids=memory_ids,  # Track which memories contributed
)
```

**Estimated Effort:** 4 hours to wire memory tracking

## Success Metrics

- ✅ Search recall@5: >80% for relevant memories
- ✅ Reinforcement: +0.05 importance per success, -0.08 per failure
- ✅ Promotion thresholds: working→episodic at 0.7, episodic→semantic at 0.8
- ✅ TTL accuracy: ±1 minute
- ✅ Semantic search: >70% precision for synonyms
- ✅ All tests pass with >95% success rate

## Notes

**TTL Testing Strategy:**
- Option 1: Mock time (fast, deterministic)
- Option 2: Real delays with `@pytest.mark.slow` (realistic)
- Recommended: Mock for CI, real delays for manual validation

**Cost Control:**
- Memory tests use database operations (free)
- LLM calls only for semantic decomposition (already in Neo4jMemoryProvider)
- Estimated cost: <$0.05 per test suite run

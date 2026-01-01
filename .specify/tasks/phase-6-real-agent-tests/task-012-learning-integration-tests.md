# TASK-012: Implement Learning Integration Tests (FR-010.3)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P1 (Important - validates learning pipeline)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-009, TASK-011 (Memory integration tests)

## Description

Implement integration tests for agent learning capabilities: skill extraction from successful interactions, fact learning from user statements, correction acceptance, and skill verification.

**Core Principle:** Test that agents actually learn from experience and update beliefs based on feedback.

## Acceptance Criteria

- [ ] Test autonomous skill extraction from successful tool executions
- [ ] Test fact learning from user statements
- [ ] Test correction acceptance and belief updates
- [ ] Test skill verification (demote broken skills)
- [ ] Test multi-user knowledge scoping
- [ ] All learning uses LLM semantic analysis (not regex patterns)

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_learning.py`

**Example Tests:**
```python
from draagon_ai.cognition.learning import LearningService

@pytest.mark.learning_integration
class TestAgentLearning:
    """Test agent learning from interactions."""

    @pytest.mark.asyncio
    async def test_extract_skill_from_success(self, agent, memory_provider):
        """Agent learns skill from successful execution."""

        # Simulate successful tool execution
        # (This assumes agent has tool execution tracking)
        response = await agent.process(
            "Restart the web server using nginx"
        )

        # Learning service should extract skill
        # (Trigger may be automatic or manual)
        skills = await memory_provider.search(
            query="restart nginx",
            memory_types=[MemoryType.SKILL],
            limit=5,
        )

        assert len(skills) > 0
        assert any("nginx" in s.content.lower() for s in skills)

    @pytest.mark.asyncio
    async def test_learn_fact_from_statement(self, agent, memory_provider, evaluator):
        """Agent learns and stores facts from user statements."""

        # User states fact (LLM detects learning opportunity)
        response = await agent.process("I have 3 cats named Whiskers, Mittens, and Shadow")

        # Fact should be stored
        facts = await memory_provider.search(
            query="cats",
            memory_types=[MemoryType.FACT],
            user_id="test_user",
        )

        assert len(facts) > 0

        # Agent should recall
        response2 = await agent.process("What are my cats' names?")
        result = await evaluator.evaluate_correctness(
            query="What are my cats' names?",
            expected_outcome="Lists Whiskers, Mittens, Shadow",
            actual_response=response2.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_apply_correction(self, agent, memory_provider, evaluator):
        """Agent updates beliefs when corrected."""

        # Initial statement
        await agent.process("I have 3 cats")

        # User corrects (LLM detects correction semantically - NO REGEX!)
        await agent.process("Actually, I have 4 cats now. Got a new one!")

        # Check belief updated
        beliefs = await memory_provider.search(
            query="how many cats",
            memory_types=[MemoryType.FACT, MemoryType.BELIEF],
            user_id="test_user",
        )

        # Should reflect correction
        assert len(beliefs) > 0
        # Use LLM to validate correction was applied
        response = await agent.process("How many cats do I have?")
        result = await evaluator.evaluate_correctness(
            query="How many cats do I have?",
            expected_outcome="Says 4 cats",
            actual_response=response.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_skill_verification_demotes_broken_skills(self, agent, memory_provider):
        """Agent verifies learned skills and demotes broken ones."""

        # Store potentially broken skill
        memory_id = await memory_provider.store(
            content="To restart: sudo reboot-everything",
            memory_type=MemoryType.SKILL,
            importance=0.7,
            user_id="test_user",
        )

        # Agent tries to use it (tool execution fails)
        # This requires tool execution tracking
        response = await agent.process("Restart the system")

        # If skill fails, importance should drop
        updated = await memory_provider.get(memory_id)
        if "fail" in response.answer.lower() or response.confidence < 0.5:
            # Skill was demoted due to failure
            assert updated.importance < 0.7

    @pytest.mark.asyncio
    async def test_multi_user_knowledge_scoping(self, agent, memory_provider):
        """Facts learned from one user don't leak to others."""

        # User 1 teaches fact
        await agent.process("My favorite color is blue", user_id="alice")

        # User 2 asks (shouldn't know)
        response = await agent.process("What's my favorite color?", user_id="bob")

        # Should not have access to Alice's preference
        assert response.confidence < 0.6  # Uncertain

        # But Alice should still have access
        response2 = await agent.process("What's my favorite color?", user_id="alice")
        assert response2.confidence > 0.7
        assert "blue" in response2.answer.lower()

    @pytest.mark.asyncio
    async def test_learning_from_correction_improves_accuracy(self, agent, evaluator):
        """Agent becomes more accurate after correction."""

        # Initial wrong answer
        response1 = await agent.process("How many cats do I have?", user_id="test_user")
        # (Assume agent doesn't know yet)

        # User provides correction
        await agent.process("I have 3 cats", user_id="test_user")

        # Ask again - should now be correct
        response2 = await agent.process("How many cats do I have?", user_id="test_user")

        result = await evaluator.evaluate_correctness(
            query="How many cats do I have?",
            expected_outcome="Says 3 cats",
            actual_response=response2.answer,
        )
        assert result.correct
        assert response2.confidence > 0.8

    @pytest.mark.asyncio
    async def test_fact_vs_skill_classification(self, agent, memory_provider):
        """Agent correctly classifies facts vs skills when learning."""

        # Teach a fact
        await agent.process("Paris is the capital of France")

        # Teach a skill
        await agent.process("To restart Docker, run: sudo systemctl restart docker")

        # Check correct memory types
        facts = await memory_provider.search(
            query="Paris capital",
            memory_types=[MemoryType.FACT],
            limit=5,
        )
        assert len(facts) > 0

        skills = await memory_provider.search(
            query="restart Docker",
            memory_types=[MemoryType.SKILL],
            limit=5,
        )
        assert len(skills) > 0
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_extract_skill_from_success` - Skill extraction
- [ ] `test_learn_fact_from_statement` - Fact learning
- [ ] `test_apply_correction` - Correction acceptance
- [ ] `test_skill_verification_demotes_broken_skills` - Skill verification
- [ ] `test_multi_user_knowledge_scoping` - User scoping
- [ ] `test_learning_from_correction_improves_accuracy` - Learning efficacy
- [ ] `test_fact_vs_skill_classification` - Memory type classification

**Performance Tests:**
- [ ] Skill extraction: <3s per interaction
- [ ] Fact learning: <2s per statement
- [ ] Correction processing: <2s

**Cognitive Tests:**
- [ ] Learning detects correction semantically (NOT via regex like "actually|no,")
- [ ] Skills stored with procedural context (how-to)
- [ ] Facts stored with declarative context (what-is)
- [ ] User scoping prevents knowledge leakage

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_learning.py` - Learning tests

**Modify:**
- None (uses existing LearningService from cognition/)

## Pre-Implementation Work

**Verify LearningService Integration:**

Check that `LearningService` from `src/draagon_ai/cognition/learning.py` is wired into AgentLoop:

1. **Post-Response Learning Hook:**
```python
# In AgentLoop.process() - after response generation
if self.config.enable_learning:
    await self.learning_service.extract_learnings(
        query=query,
        response=response,
        used_memories=response.used_memory_ids,
        outcome=outcome,  # success/failure
    )
```

2. **Correction Detection:**
```python
# In LearningService - use LLM to detect corrections (NO REGEX!)
async def detect_correction(self, current_query: str, history: list) -> bool:
    """Use LLM to detect if user is correcting previous statement."""
    prompt = f"""
    <correction_detection>
      <conversation_history>{history}</conversation_history>
      <current_query>{current_query}</current_query>
      <question>Is the user correcting a previous statement?</question>
      <output>
        <is_correction>true/false</is_correction>
        <reasoning>Why this is/isn't a correction</reasoning>
      </output>
    </correction_detection>
    """
    # Parse LLM response
    return is_correction
```

**Estimated Effort:** 6 hours to wire learning hooks if not already implemented

## Success Metrics

- ✅ Skill extraction accuracy: >70%
- ✅ Fact recall after learning: >90%
- ✅ Correction acceptance: >95%
- ✅ Broken skill detection: >80%
- ✅ User scoping: 100% (no leakage)
- ✅ Fact vs skill classification: >85%

## Notes

**CRITICAL: NO REGEX FOR SEMANTIC DETECTION**

The following are **FORBIDDEN**:
```python
# ❌ WRONG - semantic detection via regex
if re.match(r"actually|no,|wrong", query):
    is_correction = True

# ✅ RIGHT - semantic detection via LLM
is_correction = await self.llm.detect_correction(query, history)
```

**Learning Triggers:**
- Post-response hook (automatic)
- User explicit teaching ("remember that...")
- Correction detection (LLM semantic analysis)
- Skill success/failure tracking

**Cost Control:**
- Learning detection adds ~1 LLM call per interaction
- Correction detection adds ~1 LLM call when triggered
- Estimated cost: ~$0.10 for full test suite

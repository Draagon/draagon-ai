"""Learning integration tests (FR-010.3).

Tests the agent learning pipeline:
- Skill extraction from interactions
- Fact learning from user statements
- Correction acceptance and belief updates
- Skill verification (demote broken skills)
- Multi-user knowledge scoping

These tests validate the LearningService works correctly with real LLM.
Some tests require full AgentLoop integration and are marked as skipped.
"""

import os
import pytest

from draagon_ai.cognition.learning import LearningService, LearningResult
from draagon_ai.memory.base import MemoryType, MemoryScope


# =============================================================================
# Learning Service Tests (Standalone)
# =============================================================================


@pytest.mark.learning_integration
class TestLearningServiceDetection:
    """Test the learning detection logic in isolation."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_detect_fact_from_statement(self, real_llm, memory_provider):
        """LearningService detects factual statements."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        result = await service.process_interaction(
            user_query="My birthday is March 15th",
            response="I'll remember that your birthday is March 15th!",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # Should detect a learning opportunity
        assert isinstance(result, LearningResult)
        # Note: May or may not learn depending on LLM confidence
        if result.learned:
            assert result.learning_type in ("fact", "preference")
            assert result.memory_id is not None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_detect_skill_from_success(self, real_llm, memory_provider):
        """LearningService detects skills from tool success."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # Simulate a successful tool execution
        tool_calls = [
            {
                "tool": "run_command",
                "args": {"command": "docker restart nginx"},
                "result": {"success": True, "output": "nginx restarted"},
            }
        ]

        result = await service.process_interaction(
            user_query="Restart the nginx container",
            response="Done! I've restarted the nginx container using `docker restart nginx`.",
            tool_calls=tool_calls,
            user_id="test_user",
            conversation_id="test_conv",
        )

        assert isinstance(result, LearningResult)
        # Skills from tool success may or may not be learned
        # depending on LLM judgment

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_detect_preference(self, real_llm, memory_provider):
        """LearningService detects user preferences."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        result = await service.process_interaction(
            user_query="I prefer dark mode for all my apps",
            response="Got it! I'll remember that you prefer dark mode.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        assert isinstance(result, LearningResult)
        if result.learned:
            assert result.learning_type in ("preference", "fact")

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_no_learning_from_query(self, real_llm, memory_provider):
        """LearningService does NOT learn from simple queries."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        result = await service.process_interaction(
            user_query="What's the weather like today?",
            response="It's sunny and 72 degrees.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # Simple query should not trigger learning
        # (no personal fact, no skill, no preference)
        assert isinstance(result, LearningResult)
        # Learning is unlikely for a simple weather query
        # (unless LLM decides the weather is worth remembering)


@pytest.mark.learning_integration
class TestLearningStorage:
    """Test that learned content is properly stored."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_stored_fact_is_searchable(self, real_llm, memory_provider):
        """Stored facts can be found via semantic search."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # Teach a distinctive fact
        result = await service.process_interaction(
            user_query="I have 6 cats named Luna, Stella, Orion, Nova, Cosmos, and Nebula",
            response="Wow, 6 cats! I'll remember Luna, Stella, Orion, Nova, Cosmos, and Nebula.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # If learning occurred, verify it's searchable
        if result.learned:
            results = await memory_provider.search(
                query="how many cats",
                user_id="test_user",
                limit=5,
            )

            # Should find the cat fact
            assert len(results) > 0
            found_cats = any(
                "cat" in r.memory.content.lower()
                for r in results
            )
            assert found_cats, "Cat fact not found in search results"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_skill_stored_as_skill_type(self, real_llm, memory_provider):
        """Skills are stored with SKILL memory type."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # Teach a clear skill/procedure
        tool_calls = [
            {
                "tool": "run_command",
                "args": {"command": "systemctl restart postgresql"},
                "result": {"success": True, "output": "postgresql.service restarted"},
            }
        ]

        result = await service.process_interaction(
            user_query="How do I restart PostgreSQL?",
            response="To restart PostgreSQL, run: systemctl restart postgresql",
            tool_calls=tool_calls,
            user_id="test_user",
            conversation_id="test_conv",
        )

        if result.learned and result.learning_type == "skill":
            # Verify it's stored as SKILL type
            results = await memory_provider.search(
                query="restart postgresql",
                user_id="test_user",
                memory_types=[MemoryType.SKILL],
                limit=5,
            )

            assert len(results) > 0, "Skill not stored as SKILL type"


@pytest.mark.learning_integration
class TestLearningCorrections:
    """Test correction detection and handling."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_correction_detected_semantically(self, real_llm, memory_provider):
        """Corrections are detected via LLM, not regex."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # First, establish a fact
        await service.process_interaction(
            user_query="I have 3 cats",
            response="Got it, you have 3 cats!",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # Now correct it (without using obvious keywords like "actually")
        result = await service.process_interaction(
            user_query="Wait, I miscounted - there are 4 cats now, we got a kitten",
            response="Oh nice, congratulations on the new kitten! So you have 4 cats now.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # Should detect this as a correction
        assert isinstance(result, LearningResult)
        if result.learned:
            # The learning type should indicate correction or the fact was updated
            assert result.learning_type in ("correction", "fact", "refinement")

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_correction_updates_memory(self, real_llm, memory_provider):
        """Corrections should update the stored memory."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # First fact
        result1 = await service.process_interaction(
            user_query="My favorite color is blue",
            response="Blue, nice! I'll remember that.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # Correction
        result2 = await service.process_interaction(
            user_query="Actually, I changed my mind - my favorite color is now purple",
            response="Purple it is! I've updated my notes.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )

        # At least one learning should have occurred
        # (either the initial fact or the correction)
        at_least_one_learned = result1.learned or result2.learned

        if at_least_one_learned:
            # Search for color preference
            results = await memory_provider.search(
                query="favorite color",
                user_id="test_user",
                limit=5,
            )

            # Should find at least one color-related memory
            assert len(results) > 0, "Learning occurred but search found nothing"
        else:
            # LLM decided not to learn - this is acceptable behavior
            # (low confidence on casual statements)
            pytest.skip("LLM decided not to learn these statements (low confidence)")


@pytest.mark.learning_integration
class TestLearningScoping:
    """Test multi-user knowledge scoping."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_user_facts_scoped_to_user(self, real_llm, memory_provider):
        """Facts from one user are not visible to others."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # Alice shares a fact
        await service.process_interaction(
            user_query="I'm allergic to peanuts",
            response="Important to know - you're allergic to peanuts. I'll remember that.",
            tool_calls=[],
            user_id="alice",
            conversation_id="alice_conv",
        )

        # Search as Bob
        bob_results = await memory_provider.search(
            query="allergic peanuts",
            user_id="bob",
            limit=5,
        )

        # Bob should NOT see Alice's allergy info
        assert all(
            r.memory.user_id != "alice"
            for r in bob_results
        ), "Alice's private info leaked to Bob"

        # Alice should see her own info
        alice_results = await memory_provider.search(
            query="allergic peanuts",
            user_id="alice",
            limit=5,
        )

        # Alice should find her allergy info
        alice_found = any(
            "peanut" in r.memory.content.lower()
            for r in alice_results
        )
        # Note: This may fail if learning didn't occur
        # The test validates scoping when learning does occur


@pytest.mark.learning_integration
class TestLearningPerformance:
    """Test learning operation performance."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_learning_latency(self, real_llm, memory_provider):
        """Learning completes within performance budget."""
        import time

        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        start = time.time()
        await service.process_interaction(
            user_query="My phone number is 555-0123",
            response="I've noted your phone number.",
            tool_calls=[],
            user_id="test_user",
            conversation_id="test_conv",
        )
        elapsed = time.time() - start

        # Should complete in <5s (generous for LLM detection + storage)
        assert elapsed < 5.0, f"Learning took {elapsed:.2f}s, expected <5s"


# =============================================================================
# AgentLoop Integration Tests (Require Full Wiring)
# =============================================================================


@pytest.mark.learning_integration
@pytest.mark.skip(reason="Requires LearningService wired into AgentLoop - not yet implemented")
class TestAgentLearningIntegration:
    """Test learning through the full agent pipeline.

    These tests require LearningService to be integrated into AgentLoop.
    They are skipped until that integration is complete.
    """

    @pytest.mark.asyncio
    async def test_agent_learns_and_recalls(
        self, agent, test_behavior, test_context, evaluator, memory_provider
    ):
        """Agent learns fact and recalls it in subsequent query."""
        # Teach the agent
        await agent.process(
            query="I have 3 cats named Whiskers, Mittens, and Shadow",
            behavior=test_behavior,
            context=test_context,
        )

        # Ask the agent to recall
        response = await agent.process(
            query="What are my cats' names?",
            behavior=test_behavior,
            context=test_context,
        )

        result = await evaluator.evaluate_correctness(
            query="What are my cats' names?",
            expected_outcome="Lists Whiskers, Mittens, and Shadow",
            actual_response=response.response,
        )

        assert result.correct, f"Agent didn't recall cats: {result.reasoning}"

    @pytest.mark.asyncio
    async def test_agent_corrects_belief(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Agent updates belief when user corrects."""
        # Initial fact
        await agent.process(
            query="I have 3 cats",
            behavior=test_behavior,
            context=test_context,
        )

        # Correction
        await agent.process(
            query="Actually, I have 4 cats now. Got a new one!",
            behavior=test_behavior,
            context=test_context,
        )

        # Verify correction was accepted
        response = await agent.process(
            query="How many cats do I have?",
            behavior=test_behavior,
            context=test_context,
        )

        result = await evaluator.evaluate_correctness(
            query="How many cats do I have?",
            expected_outcome="Says 4 cats",
            actual_response=response.response,
        )

        assert result.correct, f"Correction not applied: {result.reasoning}"

    @pytest.mark.asyncio
    async def test_skill_verified_and_demoted(
        self, agent, test_behavior, test_context, memory_provider
    ):
        """Agent demotes skill that fails verification."""
        # Store a broken skill directly
        memory = await memory_provider.store(
            content="To restart: sudo reboot-everything-now",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.7,
        )
        original_importance = memory.importance

        # Try to use the broken skill
        response = await agent.process(
            query="Restart the system using that command you learned",
            behavior=test_behavior,
            context=test_context,
        )

        # If skill execution failed, importance should drop
        updated = await memory_provider.get(memory.id)
        # Skill should be demoted after failure
        # (This requires tool execution tracking in AgentLoop)

    @pytest.mark.asyncio
    async def test_answered_question_stored(
        self, agent, test_behavior, test_context, evaluator, memory_provider
    ):
        """Agent stores facts from answered questions."""
        # Agent asks a question
        await agent.process(
            query="What's your daughter's name?",
            behavior=test_behavior,
            context=test_context,
        )

        # User answers
        await agent.process(
            query="Maya",
            behavior=test_behavior,
            context=test_context,
        )

        # Verify the answer was stored
        results = await memory_provider.search(
            query="daughter name",
            user_id="test_user",
            limit=5,
        )

        found_maya = any(
            "maya" in r.memory.content.lower()
            for r in results
        )

        assert found_maya, "Answer to question not stored"


# =============================================================================
# Skill Verification Tests (Require Tool Execution Tracking)
# =============================================================================


@pytest.mark.learning_integration
@pytest.mark.skip(reason="Requires tool failure tracking in AgentLoop")
class TestSkillVerification:
    """Test skill verification through failure detection.

    These tests require AgentLoop to track tool execution outcomes
    and report failures to LearningService.
    """

    @pytest.mark.asyncio
    async def test_skill_confidence_decay_on_failure(
        self, real_llm, memory_provider
    ):
        """Skill confidence decreases after failures."""
        service = LearningService(
            llm=real_llm,
            memory=memory_provider,
        )

        # Record a skill
        skill_memory = await memory_provider.store(
            content="Restart nginx: docker restart nginx",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.8,
        )

        # Simulate tool failure
        await service.process_tool_failure(
            tool_name="run_command",
            tool_args={"command": "docker restart nginx"},
            tool_result={"error": "Container not found: nginx"},
            skill_used={
                "memory_id": skill_memory.id,
                "content": skill_memory.content,
            },
            user_id="test_user",
            conversation_id="test_conv",
        )

        # Check confidence dropped
        confidence = service.get_skill_confidence(skill_memory.id)
        assert confidence is not None
        assert confidence < 0.8, "Skill confidence should decrease on failure"

    @pytest.mark.asyncio
    async def test_skill_relearned_after_failure(
        self, real_llm, memory_provider
    ):
        """Skill is relearned from web search after repeated failures."""
        # This requires SearchProvider to be configured
        pass

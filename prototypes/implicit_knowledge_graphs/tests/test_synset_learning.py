"""Tests for runtime synset learning service.

Tests cover:
1. Unknown term recording and tracking
2. LLM-driven definition extraction
3. Confidence thresholds and reinforcement
4. Integration with EvolvingSynsetDatabase
5. Integration with WSD pipeline
6. Qdrant storage (stub)
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from synset_learning import (
    SynsetLearningConfig,
    SynsetLearningService,
    QdrantSynsetStore,
    UnknownTermRecord,
    DefinitionExtraction,
    ReinforcementResult,
    ResolutionSource,
)
from evolving_synsets import EvolvingSynsetDatabase, EvolvingDBConfig
from identifiers import LearnedSynset, SynsetSource
from wsd import WordSenseDisambiguator, WSDConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def evolving_db():
    """Create an empty evolving synset database."""
    return EvolvingSynsetDatabase(
        config=EvolvingDBConfig(min_confidence_threshold=0.0)
    )


@pytest.fixture
def learning_config():
    """Create learning service configuration."""
    return SynsetLearningConfig(
        confidence_threshold=0.8,
        min_confidence_for_use=0.5,
        reinforcement_boost=0.05,
        reinforcement_penalty=0.08,
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def mock_memory():
    """Create a mock memory provider."""
    memory = MagicMock()
    memory.store = AsyncMock()
    memory.search = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def learning_service(evolving_db, learning_config, mock_llm, mock_memory):
    """Create a learning service with mocks."""
    return SynsetLearningService(
        config=learning_config,
        evolving_db=evolving_db,
        llm=mock_llm,
        memory=mock_memory,
    )


# =============================================================================
# Test: Unknown Term Recording
# =============================================================================


class TestUnknownTermRecording:
    """Tests for recording unknown terms."""

    @pytest.mark.asyncio
    async def test_record_unknown_term_basic(self, learning_service):
        """Recording an unknown term creates a record."""
        record = await learning_service.record_unknown_term(
            term="ArgoCD",
            context="We use ArgoCD for GitOps deployment",
        )

        assert record.term == "ArgoCD"
        assert record.context == "We use ArgoCD for GitOps deployment"
        assert record.attempts == 1
        assert record.resolved is False
        assert record.resolution_source == ResolutionSource.UNRESOLVED

    @pytest.mark.asyncio
    async def test_record_same_term_increments_attempts(self, learning_service):
        """Recording the same term increments attempts."""
        await learning_service.record_unknown_term(
            term="ArgoCD",
            context="First context",
        )
        record = await learning_service.record_unknown_term(
            term="ArgoCD",
            context="Second context (longer)",
        )

        assert record.attempts == 2
        # Should keep longer context
        assert record.context == "Second context (longer)"

    @pytest.mark.asyncio
    async def test_get_pending_terms(self, learning_service):
        """Get list of unresolved term words."""
        await learning_service.record_unknown_term("ArgoCD", "context1")
        await learning_service.record_unknown_term("FluxCD", "context2")
        await learning_service.record_unknown_term("Pulumi", "context3")

        pending = learning_service.get_pending_term_words()

        assert len(pending) == 3
        assert "ArgoCD" in pending
        assert "FluxCD" in pending
        assert "Pulumi" in pending

    @pytest.mark.asyncio
    async def test_records_to_memory(self, learning_service, mock_memory):
        """Unknown terms are stored in memory."""
        await learning_service.record_unknown_term(
            term="ArgoCD",
            context="GitOps context",
        )

        mock_memory.store.assert_called_once()
        # Check the content was passed correctly
        call_args = mock_memory.store.call_args
        content = call_args.kwargs.get("content") or call_args.args[0]
        assert "Unknown term encountered: ArgoCD" in content
        assert call_args.kwargs["memory_type"] == "observation"

    @pytest.mark.asyncio
    async def test_case_insensitive_dedup(self, learning_service):
        """Terms are deduplicated case-insensitively."""
        await learning_service.record_unknown_term("ArgoCD", "context1")
        record = await learning_service.record_unknown_term("argocd", "context2")

        assert record.attempts == 2
        pending = learning_service.get_pending_term_words()
        assert len(pending) == 1


# =============================================================================
# Test: Definition Extraction
# =============================================================================


class TestDefinitionExtraction:
    """Tests for LLM-driven definition extraction."""

    @pytest.mark.asyncio
    async def test_extract_definitions_from_output(self, learning_service, mock_llm):
        """Extract definitions from agent output using LLM."""
        # Setup LLM response
        mock_llm.chat.return_value = """
        <definitions>
          <definition>
            <term>ArgoCD</term>
            <explanation>A declarative GitOps continuous delivery tool for Kubernetes</explanation>
            <pos>n</pos>
            <example>ArgoCD syncs cluster state with Git repos</example>
            <domain>DEVOPS_CICD</domain>
            <confidence>0.9</confidence>
          </definition>
        </definitions>
        """

        # Record unknown term first
        await learning_service.record_unknown_term("ArgoCD", "GitOps context")

        # Extract from output
        extractions = await learning_service.extract_definitions_from_output(
            output="ArgoCD is a declarative GitOps continuous delivery tool for Kubernetes.",
            unknown_terms=["ArgoCD"],
        )

        assert len(extractions) == 1
        assert extractions[0].term == "ArgoCD"
        assert "GitOps" in extractions[0].explanation
        assert extractions[0].pos == "n"
        assert extractions[0].domain == "DEVOPS_CICD"
        assert extractions[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_extract_marks_resolved(self, learning_service, mock_llm):
        """Successful extraction marks term as resolved."""
        mock_llm.chat.return_value = """
        <definitions>
          <definition>
            <term>ArgoCD</term>
            <explanation>GitOps CD tool</explanation>
            <pos>n</pos>
            <example>Use ArgoCD</example>
            <domain>DEVOPS_CICD</domain>
            <confidence>0.85</confidence>
          </definition>
        </definitions>
        """

        await learning_service.record_unknown_term("ArgoCD", "context")

        await learning_service.extract_definitions_from_output(
            output="ArgoCD is a GitOps tool",
            unknown_terms=["ArgoCD"],
        )

        # Term should be resolved
        records = learning_service.get_unknown_terms(include_resolved=True)
        assert records[0].resolved is True
        assert records[0].resolution_source == ResolutionSource.AGENT_OUTPUT

    @pytest.mark.asyncio
    async def test_no_llm_returns_empty(self, evolving_db, learning_config):
        """Without LLM, extraction returns empty list."""
        service = SynsetLearningService(
            config=learning_config,
            evolving_db=evolving_db,
            llm=None,  # No LLM
        )

        extractions = await service.extract_definitions_from_output(
            output="Some output",
            unknown_terms=["ArgoCD"],
        )

        assert extractions == []

    @pytest.mark.asyncio
    async def test_extraction_to_learned_synset(self, learning_service, mock_llm):
        """Extracted definitions convert to LearnedSynset."""
        mock_llm.chat.return_value = """
        <definitions>
          <definition>
            <term>Pulumi</term>
            <explanation>Infrastructure as code using general-purpose languages</explanation>
            <pos>n</pos>
            <example>Define cloud resources with Pulumi</example>
            <domain>CLOUD_INFRASTRUCTURE</domain>
            <confidence>0.88</confidence>
          </definition>
        </definitions>
        """

        extractions = await learning_service.extract_definitions_from_output(
            output="Pulumi is an IaC tool...",
            unknown_terms=["Pulumi"],
        )

        synset = extractions[0].to_learned_synset()

        assert synset.synset_id == "pulumi.cloud.01"
        assert synset.word == "pulumi"
        assert synset.pos == "n"
        assert synset.source == SynsetSource.LLM
        assert synset.confidence == 0.88


# =============================================================================
# Test: User Explanation Learning
# =============================================================================


class TestUserExplanationLearning:
    """Tests for learning from user explanations."""

    @pytest.mark.asyncio
    async def test_learn_from_user_explanation(self, learning_service, evolving_db):
        """User explanations create high-confidence synsets."""
        synset = await learning_service.learn_from_user_explanation(
            term="ArgoCD",
            explanation="It's a GitOps tool that syncs Kubernetes with Git",
            user_id="doug",
        )

        assert synset is not None
        assert synset.word == "argocd"
        assert synset.source == SynsetSource.USER
        assert synset.confidence == 0.95  # High confidence for user-provided

        # Should be in evolving DB
        assert evolving_db.has_word("argocd")

    @pytest.mark.asyncio
    async def test_user_explanation_marks_resolved(self, learning_service):
        """User explanation marks term as resolved."""
        await learning_service.record_unknown_term("ArgoCD", "context")

        await learning_service.learn_from_user_explanation(
            term="ArgoCD",
            explanation="GitOps tool",
            user_id="doug",
        )

        records = learning_service.get_unknown_terms(include_resolved=True)
        assert records[0].resolved is True
        assert records[0].resolution_source == ResolutionSource.USER_EXPLANATION


# =============================================================================
# Test: Confidence and Reinforcement
# =============================================================================


class TestConfidenceAndReinforcement:
    """Tests for confidence tracking and reinforcement."""

    @pytest.mark.asyncio
    async def test_reinforce_success_increases_confidence(
        self, learning_service, evolving_db
    ):
        """Successful usage increases confidence."""
        # Add a synset with 0.7 confidence
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            source=SynsetSource.LLM,
            confidence=0.7,
        )
        evolving_db.add_synset(synset)

        result = await learning_service.reinforce("argocd.tech.01", success=True)

        assert result is not None
        assert result.old_confidence == 0.7
        assert result.new_confidence == 0.75  # +0.05 boost
        assert result.success is True

    @pytest.mark.asyncio
    async def test_reinforce_failure_decreases_confidence(
        self, learning_service, evolving_db
    ):
        """Failed usage decreases confidence."""
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            source=SynsetSource.LLM,
            confidence=0.7,
        )
        evolving_db.add_synset(synset)

        result = await learning_service.reinforce("argocd.tech.01", success=False)

        assert result is not None
        assert result.old_confidence == 0.7
        assert result.new_confidence == 0.62  # -0.08 penalty
        assert result.success is False

    @pytest.mark.asyncio
    async def test_confidence_capped_at_max(self, learning_service, evolving_db):
        """Confidence doesn't exceed max_confidence (0.99)."""
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            confidence=0.98,
        )
        evolving_db.add_synset(synset)

        result = await learning_service.reinforce("argocd.tech.01", success=True)

        assert result.new_confidence == 0.99  # Capped

    @pytest.mark.asyncio
    async def test_confidence_floored_at_min(self, learning_service, evolving_db):
        """Confidence doesn't go below min_confidence (0.1)."""
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            confidence=0.12,
        )
        evolving_db.add_synset(synset)

        result = await learning_service.reinforce("argocd.tech.01", success=False)

        assert result.new_confidence == 0.1  # Floored

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_research_flag(
        self, learning_service, evolving_db
    ):
        """When confidence drops below threshold, needs_research is True."""
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="Maybe wrong definition",
            confidence=0.55,  # Just above min_confidence_for_use (0.5)
        )
        evolving_db.add_synset(synset)

        result = await learning_service.reinforce("argocd.tech.01", success=False)

        assert abs(result.new_confidence - 0.47) < 0.001  # Below 0.5 threshold
        assert result.needs_research is True

    @pytest.mark.asyncio
    async def test_get_synset_with_confidence_check(
        self, learning_service, evolving_db
    ):
        """Query returns synset with confidence assessment."""
        # High confidence synset
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration",
            confidence=0.95,
        )
        evolving_db.add_synset(synset)

        result, is_confident = await learning_service.get_synset_with_confidence_check(
            "kubernetes"
        )

        assert result is not None
        assert result.word == "kubernetes"
        assert is_confident is True  # 0.95 >= 0.8 threshold

    @pytest.mark.asyncio
    async def test_get_uncertain_synsets(self, learning_service, evolving_db):
        """Get synsets below confidence threshold."""
        # Add various confidence levels
        evolving_db.add_synset(LearnedSynset(
            synset_id="high.tech.01", word="high", pos="n", definition="", confidence=0.95
        ))
        evolving_db.add_synset(LearnedSynset(
            synset_id="medium.tech.01", word="medium", pos="n", definition="", confidence=0.7
        ))
        evolving_db.add_synset(LearnedSynset(
            synset_id="low.tech.01", word="low", pos="n", definition="", confidence=0.4
        ))

        uncertain = learning_service.get_uncertain_synsets()

        # Only medium and low are below 0.8 threshold
        assert len(uncertain) == 2
        assert uncertain[0].synset_id == "low.tech.01"  # Sorted ascending
        assert uncertain[1].synset_id == "medium.tech.01"


# =============================================================================
# Test: Integration with EvolvingSynsetDatabase
# =============================================================================


class TestEvolvingDBIntegration:
    """Tests for integration with EvolvingSynsetDatabase."""

    @pytest.mark.asyncio
    async def test_persist_adds_to_evolving_db(self, learning_service, evolving_db):
        """Persisting a synset adds it to the evolving database."""
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
        )

        await learning_service.persist_learned_synset(synset)

        assert evolving_db.has_word("argocd")
        retrieved = evolving_db.get_synset_by_id("argocd.tech.01")
        assert retrieved is not None
        assert retrieved.definition == "GitOps tool"

    @pytest.mark.asyncio
    async def test_persist_records_to_memory(
        self, learning_service, mock_memory
    ):
        """Persisting records to memory provider."""
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            confidence=0.85,
        )

        await learning_service.persist_learned_synset(synset)

        mock_memory.store.assert_called_once()
        call_args = mock_memory.store.call_args
        content = call_args.kwargs.get("content") or call_args.args[0]
        assert "Learned definition: argocd" in content
        assert call_args.kwargs["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_get_statistics(self, learning_service, evolving_db):
        """Get learning statistics."""
        # Add some synsets
        evolving_db.add_synset(LearnedSynset(
            synset_id="high.tech.01", word="high", pos="n", definition="", confidence=0.95
        ))
        evolving_db.add_synset(LearnedSynset(
            synset_id="low.tech.01", word="low", pos="n", definition="", confidence=0.4
        ))

        # Record some unknown terms
        await learning_service.record_unknown_term("ArgoCD", "context")

        stats = learning_service.get_statistics()

        assert stats["total_learned_synsets"] == 2
        assert stats["synsets_above_threshold"] == 1
        assert stats["synsets_below_threshold"] == 1
        assert stats["pending_unknown_terms"] == 1
        assert stats["terms_recorded"] == 1


# =============================================================================
# Test: Integration with WSD Pipeline
# =============================================================================


class TestWSDIntegration:
    """Tests for integration with WSD pipeline."""

    @pytest.mark.asyncio
    async def test_wsd_records_unknown_terms(self, learning_service):
        """WSD pipeline records unknown terms when synset learner is set."""
        wsd = WordSenseDisambiguator(
            config=WSDConfig(),
            require_wordnet=True,
            synset_learner=learning_service,
        )

        # Try to disambiguate a term not in WordNet or evolving DB
        result = await wsd.disambiguate(
            word="ArgoCD",
            sentence="Deploy with ArgoCD",
        )

        # Should fail (not in WordNet)
        assert result is None

        # But should have recorded the term
        pending = learning_service.get_pending_term_words()
        assert "ArgoCD" in pending

    @pytest.mark.asyncio
    async def test_wsd_metrics_track_unknown_terms(self, learning_service):
        """WSD tracks unknown terms in metrics."""
        wsd = WordSenseDisambiguator(
            config=WSDConfig(),
            synset_learner=learning_service,
        )

        await wsd.disambiguate("ArgoCD", "Deploy with ArgoCD")
        await wsd.disambiguate("FluxCD", "GitOps with FluxCD")

        assert wsd.metrics["unknown_terms_recorded"] == 2

    @pytest.mark.asyncio
    async def test_wsd_with_evolving_db_finds_tech_terms(self, learning_service, evolving_db):
        """WSD finds terms in evolving DB."""
        # Add a tech term to evolving DB
        evolving_db.add_synset(LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps CD tool",
            confidence=0.9,
        ))

        # Configure WSD with evolving DB
        wsd = WordSenseDisambiguator(
            config=WSDConfig(),
            synset_learner=learning_service,
        )
        wsd.wordnet.set_evolving_db(evolving_db)

        result = await wsd.disambiguate(
            word="ArgoCD",
            sentence="Deploy with ArgoCD",
        )

        # Should find it in evolving DB
        assert result is not None
        assert result.synset_id == "argocd.tech.01"

        # Should NOT have recorded as unknown
        pending = learning_service.get_pending_term_words()
        assert "ArgoCD" not in pending


# =============================================================================
# Test: QdrantSynsetStore (Stub)
# =============================================================================


class TestQdrantSynsetStore:
    """Tests for Qdrant synset storage (stub implementation)."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        """Store and retrieve synset."""
        store = QdrantSynsetStore(client=MagicMock())

        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
        )

        synset_id = await store.store_synset(synset)
        assert synset_id == "argocd.tech.01"

        results = await store.search_by_word("argocd")
        assert len(results) == 1
        assert results[0].synset_id == "argocd.tech.01"

    @pytest.mark.asyncio
    async def test_update_confidence(self):
        """Update synset confidence."""
        store = QdrantSynsetStore(client=MagicMock())

        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            confidence=0.7,
        )

        await store.store_synset(synset)
        success = await store.update_synset_confidence("argocd.tech.01", 0.85)

        assert success is True
        results = await store.search_by_word("argocd")
        assert results[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_load_all(self):
        """Load all synsets."""
        store = QdrantSynsetStore(client=MagicMock())

        await store.store_synset(LearnedSynset(
            synset_id="a.tech.01", word="a", pos="n", definition=""
        ))
        await store.store_synset(LearnedSynset(
            synset_id="b.tech.01", word="b", pos="n", definition=""
        ))

        all_synsets = await store.load_all()
        assert len(all_synsets) == 2


# =============================================================================
# Test: Full Learning Flow
# =============================================================================


class TestFullLearningFlow:
    """End-to-end tests for the full learning flow."""

    @pytest.mark.asyncio
    async def test_unknown_to_learned_flow(self, learning_service, evolving_db, mock_llm):
        """Test flow: unknown term → extract → persist → query."""
        # 1. Record unknown term
        await learning_service.record_unknown_term(
            term="ArgoCD",
            context="We use ArgoCD for GitOps",
        )

        assert len(learning_service.get_pending_term_words()) == 1

        # 2. Extract from agent output
        mock_llm.chat.return_value = """
        <definitions>
          <definition>
            <term>ArgoCD</term>
            <explanation>Declarative GitOps CD tool for Kubernetes</explanation>
            <pos>n</pos>
            <example>Sync with ArgoCD</example>
            <domain>DEVOPS_CICD</domain>
            <confidence>0.88</confidence>
          </definition>
        </definitions>
        """

        extractions = await learning_service.extract_definitions_from_output(
            output="ArgoCD is a declarative GitOps CD tool for Kubernetes",
            unknown_terms=["ArgoCD"],
        )

        assert len(extractions) == 1

        # 3. Persist
        synset = extractions[0].to_learned_synset()
        await learning_service.persist_learned_synset(synset)

        # 4. Verify in database
        assert evolving_db.has_word("argocd")

        # 5. Query with confidence check
        result, is_confident = await learning_service.get_synset_with_confidence_check(
            "argocd"
        )
        assert result is not None
        assert is_confident is True  # 0.88 >= 0.8

    @pytest.mark.asyncio
    async def test_reinforcement_lifecycle(self, learning_service, evolving_db):
        """Test reinforcement over multiple uses."""
        # Add synset with moderate confidence
        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            source=SynsetSource.LLM,
            confidence=0.7,
        )
        evolving_db.add_synset(synset)

        # Initially below threshold
        _, is_confident = await learning_service.get_synset_with_confidence_check("argocd")
        assert is_confident is False  # 0.7 < 0.8

        # Success → 0.75
        await learning_service.reinforce("argocd.tech.01", success=True)
        # Success → 0.80
        await learning_service.reinforce("argocd.tech.01", success=True)

        # Now at threshold
        _, is_confident = await learning_service.get_synset_with_confidence_check("argocd")
        assert is_confident is True  # 0.8 >= 0.8

        # Failure → 0.72
        await learning_service.reinforce("argocd.tech.01", success=False)

        _, is_confident = await learning_service.get_synset_with_confidence_check("argocd")
        assert is_confident is False  # Back below

    @pytest.mark.asyncio
    async def test_synsets_needing_reinforcement(self, learning_service, evolving_db):
        """Identify synsets that need more reinforcement."""
        # LLM-learned with low confidence and usage
        llm_synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            pos="n",
            definition="GitOps tool",
            source=SynsetSource.LLM,
            confidence=0.6,
        )
        llm_synset.record_usage(success=True)  # Has been used
        evolving_db.add_synset(llm_synset)

        # Bootstrap synset (shouldn't be flagged)
        bootstrap = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration",
            source=SynsetSource.BOOTSTRAP,
            confidence=1.0,
        )
        evolving_db.add_synset(bootstrap)

        needing = learning_service.get_synsets_needing_reinforcement()

        assert len(needing) == 1
        assert needing[0].synset_id == "argocd.tech.01"

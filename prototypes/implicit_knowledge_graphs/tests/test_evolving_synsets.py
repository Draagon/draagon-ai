"""Tests for the Evolving Synset Database."""

import json
import tempfile
from pathlib import Path

import pytest

from identifiers import LearnedSynset, SynsetSource, SynsetInfo
from evolving_synsets import (
    EvolvingDBConfig,
    EvolvingSynsetDatabase,
    create_evolving_database,
)


# =============================================================================
# LearnedSynset Tests
# =============================================================================


class TestLearnedSynset:
    """Tests for the LearnedSynset dataclass."""

    def test_basic_creation(self):
        """Should create a learned synset with required fields."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
        )
        assert synset.synset_id == "kubernetes.tech.01"
        assert synset.word == "kubernetes"
        assert synset.pos == "n"
        assert synset.confidence == 1.0
        assert synset.source == SynsetSource.BOOTSTRAP

    def test_with_aliases(self):
        """Should include aliases."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
            aliases=["k8s", "kube"],
        )
        assert synset.aliases == ["k8s", "kube"]
        assert synset.lemmas == ["kubernetes", "k8s", "kube"]

    def test_matches_word(self):
        """Should match word and aliases."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
            aliases=["k8s", "kube"],
        )
        assert synset.matches_word("kubernetes")
        assert synset.matches_word("Kubernetes")
        assert synset.matches_word("k8s")
        assert synset.matches_word("K8S")
        assert not synset.matches_word("docker")

    def test_success_rate(self):
        """Should calculate success rate correctly."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
            success_count=8,
            failure_count=2,
        )
        assert synset.success_rate == 0.8

    def test_success_rate_no_data(self):
        """Should return 1.0 when no usage data."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
        )
        assert synset.success_rate == 1.0

    def test_record_usage(self):
        """Should record usage correctly."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
        )
        synset.record_usage(success=True)
        synset.record_usage(success=True)
        synset.record_usage(success=False)

        assert synset.usage_count == 3
        assert synset.success_count == 2
        assert synset.failure_count == 1
        assert synset.last_used is not None

    def test_to_synset_info(self):
        """Should convert to SynsetInfo correctly."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
            examples=["Deploy to Kubernetes"],
            hypernyms=["container_orchestration.tech.01"],
            aliases=["k8s"],
        )
        info = synset.to_synset_info()

        assert isinstance(info, SynsetInfo)
        assert info.synset_id == "kubernetes.tech.01"
        assert info.pos == "n"
        assert info.definition == "Container orchestration platform"
        assert info.lemmas == ["kubernetes", "k8s"]
        assert info.examples == ["Deploy to Kubernetes"]
        assert info.hypernyms == ["container_orchestration.tech.01"]

    def test_serialization(self):
        """Should serialize and deserialize correctly."""
        original = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
            aliases=["k8s"],
            domain="CLOUD_INFRASTRUCTURE",
            source=SynsetSource.BOOTSTRAP,
            confidence=0.95,
        )

        d = original.to_dict()
        restored = LearnedSynset.from_dict(d)

        assert restored.synset_id == original.synset_id
        assert restored.word == original.word
        assert restored.aliases == original.aliases
        assert restored.source == original.source
        assert restored.confidence == original.confidence

    def test_json_serialization(self):
        """Should serialize to JSON and back."""
        original = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
        )

        json_str = original.to_json()
        restored = LearnedSynset.from_json(json_str)

        assert restored.synset_id == original.synset_id
        assert restored.word == original.word

    def test_sense_number_extraction(self):
        """Should extract sense number from synset_id."""
        synset = LearnedSynset(
            synset_id="kubernetes.tech.01",
            word="kubernetes",
            pos="n",
            definition="Container orchestration platform",
        )
        assert synset.sense_number == 1

        synset2 = LearnedSynset(
            synset_id="docker.tech.02",
            word="docker",
            pos="n",
            definition="Docker second sense",
        )
        assert synset2.sense_number == 2


# =============================================================================
# EvolvingDBConfig Tests
# =============================================================================


class TestEvolvingDBConfig:
    """Tests for EvolvingDBConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = EvolvingDBConfig()
        assert config.auto_save == False
        assert config.prefer_learned == True
        assert config.min_confidence_threshold == 0.3
        assert config.success_rate_threshold == 0.5

    def test_serialization(self):
        """Should serialize and deserialize."""
        original = EvolvingDBConfig(
            data_directory="/tmp/synsets",
            auto_save=True,
        )
        d = original.to_dict()
        restored = EvolvingDBConfig.from_dict(d)

        assert restored.data_directory == "/tmp/synsets"
        assert restored.auto_save == True


# =============================================================================
# EvolvingSynsetDatabase Tests
# =============================================================================


class TestEvolvingSynsetDatabase:
    """Tests for the EvolvingSynsetDatabase class."""

    @pytest.fixture
    def sample_synsets(self):
        """Create sample synsets for testing."""
        return [
            LearnedSynset(
                synset_id="kubernetes.tech.01",
                word="kubernetes",
                pos="n",
                definition="Container orchestration platform",
                aliases=["k8s", "kube"],
                domain="CLOUD_INFRASTRUCTURE",
            ),
            LearnedSynset(
                synset_id="docker.tech.01",
                word="docker",
                pos="n",
                definition="Container runtime platform",
                domain="CLOUD_INFRASTRUCTURE",
            ),
            LearnedSynset(
                synset_id="llm.ai.01",
                word="llm",
                pos="n",
                definition="Large language model",
                aliases=["large_language_model"],
                domain="AI_ML",
            ),
        ]

    @pytest.fixture
    def temp_synset_dir(self, sample_synsets):
        """Create a temp directory with synset files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write cloud infrastructure file
            cloud_file = Path(tmpdir) / "cloud_infrastructure.json"
            cloud_data = {
                "domain": "CLOUD_INFRASTRUCTURE",
                "synsets": [s.to_dict() for s in sample_synsets[:2]],
            }
            with open(cloud_file, "w") as f:
                json.dump(cloud_data, f)

            # Write AI/ML file
            ai_file = Path(tmpdir) / "ai_ml.json"
            ai_data = {
                "domain": "AI_ML",
                "synsets": [sample_synsets[2].to_dict()],
            }
            with open(ai_file, "w") as f:
                json.dump(ai_data, f)

            yield tmpdir

    def test_load_from_directory(self, temp_synset_dir):
        """Should load synsets from directory."""
        db = EvolvingSynsetDatabase()
        count = db.load_from_directory(temp_synset_dir)

        assert count == 3
        assert db.size == 3

    def test_get_synsets_by_word(self, temp_synset_dir):
        """Should get synsets by word."""
        db = create_evolving_database(temp_synset_dir)

        synsets = db.get_synsets("kubernetes")
        assert len(synsets) == 1
        assert synsets[0].word == "kubernetes"

    def test_get_synsets_by_alias(self, temp_synset_dir):
        """Should get synsets by alias."""
        db = create_evolving_database(temp_synset_dir)

        synsets = db.get_synsets("k8s")
        assert len(synsets) == 1
        assert synsets[0].word == "kubernetes"

    def test_get_synsets_case_insensitive(self, temp_synset_dir):
        """Should be case insensitive."""
        db = create_evolving_database(temp_synset_dir)

        synsets = db.get_synsets("KUBERNETES")
        assert len(synsets) == 1
        assert synsets[0].word == "kubernetes"

    def test_get_synset_by_id(self, temp_synset_dir):
        """Should get specific synset by ID."""
        db = create_evolving_database(temp_synset_dir)

        synset = db.get_synset_by_id("kubernetes.tech.01")
        assert synset is not None
        assert synset.word == "kubernetes"

    def test_get_synset_by_id_not_found(self, temp_synset_dir):
        """Should return None for unknown ID."""
        db = create_evolving_database(temp_synset_dir)

        synset = db.get_synset_by_id("nonexistent.tech.01")
        assert synset is None

    def test_get_synsets_by_domain(self, temp_synset_dir):
        """Should get synsets by domain."""
        db = create_evolving_database(temp_synset_dir)

        cloud_synsets = db.get_synsets_by_domain("CLOUD_INFRASTRUCTURE")
        assert len(cloud_synsets) == 2

        ai_synsets = db.get_synsets_by_domain("AI_ML")
        assert len(ai_synsets) == 1

    def test_has_word(self, temp_synset_dir):
        """Should check if word exists."""
        db = create_evolving_database(temp_synset_dir)

        assert db.has_word("kubernetes")
        assert db.has_word("k8s")
        assert not db.has_word("nonexistent")

    def test_resolve_alias(self, temp_synset_dir):
        """Should resolve alias to canonical word."""
        db = create_evolving_database(temp_synset_dir)

        canonical = db.resolve_alias("k8s")
        assert canonical == "kubernetes"

    def test_add_synset(self, temp_synset_dir):
        """Should add new synset."""
        db = create_evolving_database(temp_synset_dir)
        initial_size = db.size

        new_synset = LearnedSynset(
            synset_id="terraform.tech.01",
            word="terraform",
            pos="n",
            definition="Infrastructure as code tool",
            domain="CLOUD_INFRASTRUCTURE",
        )
        db.add_synset(new_synset)

        assert db.size == initial_size + 1
        assert db.has_word("terraform")

    def test_remove_synset(self, temp_synset_dir):
        """Should remove synset."""
        db = create_evolving_database(temp_synset_dir)

        assert db.has_word("kubernetes")
        result = db.remove_synset("kubernetes.tech.01")

        assert result == True
        assert not db.has_word("kubernetes")

    def test_remove_synset_not_found(self, temp_synset_dir):
        """Should return False for unknown synset."""
        db = create_evolving_database(temp_synset_dir)

        result = db.remove_synset("nonexistent.tech.01")
        assert result == False

    def test_update_synset(self, temp_synset_dir):
        """Should update synset fields."""
        db = create_evolving_database(temp_synset_dir)

        result = db.update_synset(
            "kubernetes.tech.01",
            {"confidence": 0.95, "definition": "Updated definition"},
        )

        assert result == True
        synset = db.get_synset_by_id("kubernetes.tech.01")
        assert synset.confidence == 0.95
        assert synset.definition == "Updated definition"

    def test_reinforce_success(self, temp_synset_dir):
        """Should record successful usage."""
        db = create_evolving_database(temp_synset_dir)

        synset_before = db.get_synset_by_id("kubernetes.tech.01")
        initial_count = synset_before.usage_count

        db.reinforce("kubernetes.tech.01", success=True)

        synset_after = db.get_synset_by_id("kubernetes.tech.01")
        assert synset_after.usage_count == initial_count + 1
        assert synset_after.success_count == 1

    def test_reinforce_failure(self, temp_synset_dir):
        """Should record failed usage."""
        db = create_evolving_database(temp_synset_dir)

        db.reinforce("kubernetes.tech.01", success=False)

        synset = db.get_synset_by_id("kubernetes.tech.01")
        assert synset.failure_count == 1

    def test_boost_synset(self, temp_synset_dir):
        """Should boost confidence."""
        db = create_evolving_database(temp_synset_dir)

        synset_before = db.get_synset_by_id("kubernetes.tech.01")
        initial_confidence = synset_before.confidence

        db.boost_synset("kubernetes.tech.01", amount=0.05)

        synset_after = db.get_synset_by_id("kubernetes.tech.01")
        assert synset_after.confidence == min(1.0, initial_confidence + 0.05)

    def test_demote_synset(self, temp_synset_dir):
        """Should demote confidence."""
        db = create_evolving_database(temp_synset_dir)

        db.demote_synset("kubernetes.tech.01", amount=0.1)

        synset = db.get_synset_by_id("kubernetes.tech.01")
        assert synset.confidence == 0.9

    def test_save(self, temp_synset_dir):
        """Should save modified synsets."""
        db = create_evolving_database(temp_synset_dir)

        # Add a new synset
        new_synset = LearnedSynset(
            synset_id="terraform.tech.01",
            word="terraform",
            pos="n",
            definition="Infrastructure as code tool",
            domain="CLOUD_INFRASTRUCTURE",
        )
        db.add_synset(new_synset)

        # Save
        count = db.save()
        assert count >= 1

        # Reload and verify
        db2 = create_evolving_database(temp_synset_dir)
        assert db2.has_word("terraform")

    def test_get_stats(self, temp_synset_dir):
        """Should return statistics."""
        db = create_evolving_database(temp_synset_dir)
        stats = db.get_stats()

        assert stats["total_synsets"] == 3
        assert "by_source" in stats
        assert "by_domain" in stats
        assert stats["by_domain"]["CLOUD_INFRASTRUCTURE"] == 2
        assert stats["by_domain"]["AI_ML"] == 1

    def test_words_property(self, temp_synset_dir):
        """Should return all indexed words."""
        db = create_evolving_database(temp_synset_dir)
        words = db.words

        assert "kubernetes" in words
        assert "k8s" in words
        assert "docker" in words
        assert "llm" in words

    def test_domains_property(self, temp_synset_dir):
        """Should return all domains."""
        db = create_evolving_database(temp_synset_dir)
        domains = db.domains

        assert "CLOUD_INFRASTRUCTURE" in domains
        assert "AI_ML" in domains

    def test_empty_directory(self):
        """Should handle empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = EvolvingSynsetDatabase()
            count = db.load_from_directory(tmpdir)
            assert count == 0
            assert db.size == 0

    def test_nonexistent_directory(self):
        """Should handle nonexistent directory gracefully."""
        db = EvolvingSynsetDatabase()
        count = db.load_from_directory("/nonexistent/path")
        assert count == 0

    def test_confidence_threshold_filtering(self, temp_synset_dir):
        """Should filter by confidence threshold."""
        db = create_evolving_database(temp_synset_dir)

        # Demote one synset below threshold
        db.update_synset("docker.tech.01", {"confidence": 0.2})

        # Set threshold
        db.config.min_confidence_threshold = 0.5

        # Docker should be filtered out
        cloud_synsets = db.get_synsets("docker")
        assert len(cloud_synsets) == 0

    def test_success_rate_threshold_filtering(self, temp_synset_dir):
        """Should filter by success rate threshold."""
        db = create_evolving_database(temp_synset_dir)

        # Record many failures
        synset = db.get_synset_by_id("docker.tech.01")
        synset.success_count = 1
        synset.failure_count = 9  # 10% success rate

        # Set threshold
        db.config.success_rate_threshold = 0.5

        # Docker should be filtered out
        synsets = db.get_synsets("docker")
        assert len(synsets) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationWithBootstrapData:
    """Tests using the real bootstrap data files."""

    @pytest.fixture
    def data_dir(self):
        """Get the path to the data directory."""
        return Path(__file__).parent.parent / "data" / "synsets"

    def test_load_bootstrap_data(self, data_dir):
        """Should load all bootstrap data files."""
        if not data_dir.exists():
            pytest.skip("Bootstrap data directory not found")

        db = create_evolving_database(data_dir)

        # Should have loaded many synsets
        assert db.size > 100, f"Expected >100 synsets, got {db.size}"

    def test_kubernetes_lookup(self, data_dir):
        """Should find kubernetes and its aliases."""
        if not data_dir.exists():
            pytest.skip("Bootstrap data directory not found")

        db = create_evolving_database(data_dir)

        # Lookup by name
        synsets = db.get_synsets("kubernetes")
        assert len(synsets) >= 1

        # Lookup by alias
        k8s = db.get_synsets("k8s")
        assert len(k8s) >= 1
        assert k8s[0].word == "kubernetes"

    def test_llm_lookup(self, data_dir):
        """Should find LLM-related terms."""
        if not data_dir.exists():
            pytest.skip("Bootstrap data directory not found")

        db = create_evolving_database(data_dir)

        synsets = db.get_synsets("llm")
        assert len(synsets) >= 1
        assert "large language model" in synsets[0].definition.lower()

    def test_all_domains_have_synsets(self, data_dir):
        """Should have synsets in all expected domains."""
        if not data_dir.exists():
            pytest.skip("Bootstrap data directory not found")

        db = create_evolving_database(data_dir)
        domains = db.domains

        expected_domains = [
            "CLOUD_INFRASTRUCTURE",
            "AI_ML",
            "PROGRAMMING_FRAMEWORKS",
            "DEVOPS_CICD",
            "DATABASES",
            "WEB_APIS",
            "OBSERVABILITY",
            "GENERAL_TECH",
        ]

        for domain in expected_domains:
            assert domain in domains, f"Missing domain: {domain}"
            synsets = db.get_synsets_by_domain(domain)
            assert len(synsets) > 0, f"No synsets in domain: {domain}"

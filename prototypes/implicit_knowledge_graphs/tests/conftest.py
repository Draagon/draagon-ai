"""Pytest configuration and fixtures for Implicit Knowledge Graphs prototype."""

import os
import sys
from pathlib import Path

import pytest

# Add src directory to path for direct imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add tests directory to path for test fixtures like ground_truth
tests_path = Path(__file__).parent
sys.path.insert(0, str(tests_path))

# Also add project root for draagon-ai imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "evolution: marks evolution tests (run with '--run-evolution')"
    )
    config.addinivalue_line(
        "markers", "eval: marks evaluation tests requiring Opus 4.5 (run with '--run-eval')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-evolution",
        action="store_true",
        default=False,
        help="Run evolution tests (slow)",
    )
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="Run evaluation tests requiring Opus 4.5 API",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and options."""
    if not config.getoption("--run-evolution"):
        skip_evolution = pytest.mark.skip(reason="need --run-evolution option to run")
        for item in items:
            if "evolution" in item.keywords:
                item.add_marker(skip_evolution)

    if not config.getoption("--run-eval"):
        skip_eval = pytest.mark.skip(reason="need --run-eval option and ANTHROPIC_API_KEY")
        for item in items:
            if "eval" in item.keywords:
                item.add_marker(skip_eval)


# =============================================================================
# Common Fixtures
# =============================================================================


@pytest.fixture
def sample_sentences():
    """Sample sentences for testing."""
    return [
        "Doug went to the bank to deposit money.",
        "The bank of the river was muddy.",
        "I caught a large bass in the lake.",
        "The bass line in that song is amazing.",
        "Apple announced new products today.",
        "I ate an apple for lunch.",
        "Doug forgot the meeting again.",
        "She said the food was great.",
    ]


@pytest.fixture
def wsd_test_cases():
    """Test cases for word sense disambiguation."""
    return [
        {
            "sentence": "I deposited money in the bank",
            "word": "bank",
            "expected_synset": "bank.n.01",
            "domain": "FINANCE",
        },
        {
            "sentence": "We walked along the bank of the river",
            "word": "bank",
            "expected_synset": "bank.n.02",
            "domain": "GEOGRAPHY",
        },
        {
            "sentence": "I caught a large bass in the lake",
            "word": "bass",
            "expected_synset": "bass.n.01",
            "domain": "FISH",
        },
        {
            "sentence": "The bass line in that song is amazing",
            "word": "bass",
            "expected_synset": "bass.n.07",
            "domain": "MUSIC",
        },
    ]


@pytest.fixture
def entity_type_test_cases():
    """Test cases for entity type classification."""
    return [
        {
            "text": "Doug",
            "context": "Doug went to the store",
            "expected_type": "INSTANCE",
        },
        {
            "text": "Apple",
            "context": "Apple announced new products",
            "expected_type": "INSTANCE",
        },
        {
            "text": "apple",
            "context": "I ate an apple",
            "expected_type": "CLASS",
        },
        {
            "text": "Christmas",
            "context": "We celebrate Christmas every year",
            "expected_type": "NAMED_CONCEPT",
        },
        {
            "text": "CEO of Apple",
            "context": "The CEO of Apple spoke today",
            "expected_type": "ROLE",
        },
        {
            "text": "he",
            "context": "Doug left. He forgot his keys.",
            "expected_type": "ANAPHORA",
        },
    ]


@pytest.fixture
def decomposition_test_cases():
    """Test cases for decomposition."""
    return [
        {
            "text": "Doug forgot the meeting again",
            "expected_presuppositions": [
                "Doug forgot before",
                "A specific meeting exists",
                "Doug was supposed to remember the meeting",
            ],
            "expected_inferences": [
                "Doug missed the meeting",
                "Doug may be forgetful",
            ],
        },
        {
            "text": "Doug stopped smoking",
            "expected_presuppositions": [
                "Doug used to smoke",
            ],
            "expected_inferences": [
                "Doug no longer smokes",
                "Doug may be healthier",
            ],
        },
    ]


@pytest.fixture
def knowledge_base_fixture():
    """Sample knowledge base for retrieval tests."""
    return [
        "Doug has 3 cats named Whiskers, Mittens, and Shadow.",
        "Doug prefers tea in the morning and coffee in the afternoon.",
        "Doug works as a software engineer at TechCorp.",
        "Sarah is Doug's wife. They have been married for 5 years.",
        "Doug's favorite color is blue.",
        "The meeting is scheduled for 3pm on Tuesdays.",
        "Doug often forgets appointments when he's busy with coding.",
        "Whiskers is a tabby cat who loves to sit on keyboards.",
        "TechCorp is located in downtown Seattle.",
        "Doug and Sarah met at a coffee shop in 2018.",
    ]


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Mock LLM provider for testing without API calls."""

    class MockLLM:
        async def chat(self, messages, **kwargs):
            # Return sensible defaults based on message content
            content = str(messages)

            if "disambiguate" in content.lower() or "synset" in content.lower():
                return """<disambiguation>
                    <synset_id>bank.n.01</synset_id>
                    <confidence>0.85</confidence>
                    <reasoning>Financial context</reasoning>
                </disambiguation>"""

            if "presupposition" in content.lower():
                return """<presuppositions>
                    <item>A meeting exists</item>
                    <item>Doug forgot before</item>
                </presuppositions>"""

            return "Mock response"

    return MockLLM()


@pytest.fixture
def mock_opus_client():
    """Mock Opus 4.5 client for evaluation tests."""

    class MockOpusClient:
        class Messages:
            async def create(self, **kwargs):
                class Response:
                    content = [
                        type(
                            "ContentBlock",
                            (),
                            {
                                "text": """<evaluation>
                                    <factual_accuracy>0.85</factual_accuracy>
                                    <completeness>0.80</completeness>
                                    <relevance>0.90</relevance>
                                    <coherence>0.85</coherence>
                                    <context_efficiency>0.75</context_efficiency>
                                    <rationale>Good response overall</rationale>
                                    <improvement_suggestions>Could be more specific</improvement_suggestions>
                                </evaluation>"""
                            },
                        )()
                    ]

                return Response()

        messages = Messages()

    return MockOpusClient()


# =============================================================================
# Environment Fixtures
# =============================================================================


def _check_wordnet_available() -> bool:
    """Check if WordNet is available."""
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")
        return True
    except (ImportError, LookupError):
        return False


# Cache the result
WORDNET_AVAILABLE = _check_wordnet_available()


@pytest.fixture
def has_wordnet():
    """Check if WordNet is available."""
    return WORDNET_AVAILABLE


@pytest.fixture
def require_wordnet():
    """Skip test if WordNet is not available.

    Use this fixture to skip tests that require WordNet:

        def test_something(require_wordnet):
            # This test will be skipped if WordNet is not installed
            ...
    """
    if not WORDNET_AVAILABLE:
        pytest.skip(
            "WordNet not available. Install with:\n"
            "  pip install nltk\n"
            "  python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        )


@pytest.fixture
def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# =============================================================================
# Evolution Fixtures
# =============================================================================


@pytest.fixture
def sample_config():
    """Sample evolvable configuration."""
    return {
        "wsd": {
            "lesk_context_window": 5,
            "lesk_high_confidence": 0.8,
            "llm_fallback_threshold": 0.5,
        },
        "decomposition": {
            "max_presuppositions": 5,
            "min_confidence": 0.3,
        },
        "weighting": {
            "recency_weight": 0.2,
            "memory_weight": 0.2,
            "belief_weight": 0.15,
            "commonsense_weight": 0.1,
        },
        "retrieval": {
            "max_triples": 20,
            "synset_filter_strictness": 0.5,
        },
    }


@pytest.fixture
def train_holdout_split(knowledge_base_fixture, decomposition_test_cases):
    """Split test data into train and holdout sets."""
    # Simple split for testing
    return {
        "train": decomposition_test_cases[:1],
        "holdout": decomposition_test_cases[1:],
    }

"""Fixtures for benchmark tests."""

import pytest

from draagon_ai.testing.benchmarks import (
    BenchmarkDocument,
    DocumentCategory,
    DocumentSource,
    DocumentCorpus,
)


@pytest.fixture
def sample_document() -> BenchmarkDocument:
    """Create a sample benchmark document."""
    return BenchmarkDocument(
        doc_id="doc_001",
        source=DocumentSource.LOCAL,
        category=DocumentCategory.TECHNICAL,
        domain="python",
        file_path="/home/user/project/README.md",
        content="This is a sample Python project README with documentation.",
        chunk_ids=["chunk_001", "chunk_002"],
        metadata={"author": "test_user", "lines": 42},
        is_distractor=False,
        semantic_tags=["python", "documentation", "readme"],
    )


@pytest.fixture
def sample_distractor() -> BenchmarkDocument:
    """Create a sample distractor document."""
    return BenchmarkDocument(
        doc_id="distractor_001",
        source=DocumentSource.SYNTHETIC,
        category=DocumentCategory.SYNTHETIC,
        domain="random",
        file_path="synthetic://generated",
        content="This is synthetic content designed to be a distractor.",
        is_distractor=True,
        semantic_tags=["distractor", "noise"],
    )


@pytest.fixture
def sample_legal_document() -> BenchmarkDocument:
    """Create a sample legal document."""
    return BenchmarkDocument(
        doc_id="legal_001",
        source=DocumentSource.ONLINE,
        category=DocumentCategory.LEGAL,
        domain="contract_law",
        file_path="https://example.com/terms-of-service",
        content="TERMS OF SERVICE. By using this service, you agree to...",
        semantic_tags=["legal", "tos", "agreement"],
    )


@pytest.fixture
def diverse_corpus(
    sample_document: BenchmarkDocument,
    sample_distractor: BenchmarkDocument,
    sample_legal_document: BenchmarkDocument,
) -> DocumentCorpus:
    """Create a corpus with diverse document types."""
    return DocumentCorpus(
        documents=[sample_document, sample_distractor, sample_legal_document],
        version="1.0.0",
        description="Test corpus with diverse content",
    )

#!/usr/bin/env python3
"""Build a real benchmark corpus by downloading documentation from the web.

This script assembles a production-grade benchmark corpus by:
1. Scanning local development directories
2. Fetching online documentation (Python, TypeScript, Wikipedia, etc.)
3. Collecting legal documents (licenses, ToS, privacy policies)
4. Generating synthetic distractors

All documents include source provenance metadata for attribution.

IMPORTANT: This script filters out files that may contain secrets:
- .env files
- Files containing API keys, tokens, credentials
- Config files with sensitive data

Usage:
    python scripts/build_benchmark_corpus.py [--output corpus.json] [--min-docs 500]
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from draagon_ai.testing.benchmarks import (
    BenchmarkDocument,
    CorpusBuilder,
    CorpusBuilderConfig,
    SourceConfig,
    DocumentCorpus,
    OnlineSource,
    DocumentCategory,
    TECHNICAL_SOURCES,
    NARRATIVE_SOURCES,
    ACADEMIC_SOURCES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Extended online sources with better coverage
EXTENDED_TECHNICAL_SOURCES = [
    # Python documentation
    OnlineSource(
        name="python_tutorial",
        base_url="https://docs.python.org/3/tutorial/",
        patterns=["index.html", "introduction.html", "controlflow.html", "datastructures.html"],
        category=DocumentCategory.TECHNICAL,
        domain="python",
    ),
    OnlineSource(
        name="python_library",
        base_url="https://docs.python.org/3/library/",
        patterns=["functions.html", "stdtypes.html", "collections.html", "itertools.html", "asyncio.html"],
        category=DocumentCategory.TECHNICAL,
        domain="python",
    ),
    # FastAPI
    OnlineSource(
        name="fastapi_docs",
        base_url="https://fastapi.tiangolo.com/",
        patterns=["tutorial/first-steps/", "tutorial/path-params/", "tutorial/query-params/"],
        category=DocumentCategory.TECHNICAL,
        domain="python_web",
    ),
    # TypeScript
    OnlineSource(
        name="typescript_handbook",
        base_url="https://www.typescriptlang.org/docs/handbook/",
        patterns=["2/basic-types.html", "2/everyday-types.html", "2/functions.html"],
        category=DocumentCategory.TECHNICAL,
        domain="typescript",
    ),
]

EXTENDED_NARRATIVE_SOURCES = [
    # Wikipedia - various topics for diversity
    OnlineSource(
        name="wikipedia_cs",
        base_url="https://en.wikipedia.org/wiki/",
        patterns=[
            "Computer_science",
            "Artificial_intelligence",
            "Machine_learning",
            "Neural_network",
            "Natural_language_processing",
        ],
        category=DocumentCategory.NARRATIVE,
        domain="wikipedia_cs",
    ),
    OnlineSource(
        name="wikipedia_history",
        base_url="https://en.wikipedia.org/wiki/",
        patterns=[
            "History_of_computing",
            "History_of_the_Internet",
            "History_of_programming_languages",
        ],
        category=DocumentCategory.NARRATIVE,
        domain="wikipedia_history",
    ),
    OnlineSource(
        name="wikipedia_science",
        base_url="https://en.wikipedia.org/wiki/",
        patterns=[
            "Physics",
            "Chemistry",
            "Biology",
            "Mathematics",
        ],
        category=DocumentCategory.NARRATIVE,
        domain="wikipedia_science",
    ),
]

EXTENDED_ACADEMIC_SOURCES = [
    # arXiv abstracts (these are public)
    OnlineSource(
        name="arxiv_ai",
        base_url="https://arxiv.org/abs/",
        patterns=[
            "2301.00774",  # Real paper: "Constitutional AI: Harmlessness from AI Feedback"
            "2303.08774",  # Real paper: "GPT-4 Technical Report"
            "2302.13971",  # Real paper: "LLaMA"
        ],
        category=DocumentCategory.ACADEMIC,
        domain="arxiv_ai",
    ),
]


# Patterns that indicate potential secrets in content
SECRET_PATTERNS = [
    r'sk-ant-[a-zA-Z0-9-]+',  # Anthropic API key
    r'sk-[a-zA-Z0-9]{32,}',  # OpenAI API key
    r'gsk_[a-zA-Z0-9]+',  # Groq API key
    r'AC[a-f0-9]{32}',  # Twilio Account SID
    r'[a-zA-Z0-9]{32,}',  # Generic long tokens (be careful)
    r'ANTHROPIC_API_KEY\s*=',
    r'OPENAI_API_KEY\s*=',
    r'GROQ_API_KEY\s*=',
    r'API_KEY\s*=\s*["\'][^"\']+["\']',
    r'password\s*=\s*["\'][^"\']+["\']',
    r'secret\s*=\s*["\'][^"\']+["\']',
    r'token\s*=\s*["\'][^"\']+["\']',
]


def contains_secrets(content: str) -> bool:
    """Check if content appears to contain secrets."""
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def filter_documents_with_secrets(
    documents: list[BenchmarkDocument],
) -> tuple[list[BenchmarkDocument], int]:
    """Filter out documents that may contain secrets.

    Returns:
        Tuple of (filtered_docs, removed_count)
    """
    filtered = []
    removed = 0

    for doc in documents:
        # Check file path for sensitive files
        path_lower = doc.file_path.lower()
        if any(s in path_lower for s in ['.env', 'credentials', 'secrets', '.secret']):
            logger.warning(f"Filtered out potential secret file: {doc.file_path}")
            removed += 1
            continue

        # Check content for secret patterns
        if contains_secrets(doc.content):
            logger.warning(f"Filtered out document with potential secrets: {doc.doc_id}")
            removed += 1
            continue

        filtered.append(doc)

    return filtered, removed


async def build_corpus(
    output_path: Path,
    min_documents: int = 500,
    distractor_ratio: float = 0.4,
    local_paths: list[Path] | None = None,
    cache_dir: Path | None = None,
) -> DocumentCorpus:
    """Build a complete benchmark corpus.

    Args:
        output_path: Where to save the corpus JSON
        min_documents: Minimum target document count
        distractor_ratio: Ratio of synthetic distractors (0.3-0.5 recommended)
        local_paths: Local directories to scan
        cache_dir: Cache directory for downloaded content

    Returns:
        The assembled DocumentCorpus
    """
    logger.info("Starting corpus build...")
    logger.info(f"Target: {min_documents}+ documents, {distractor_ratio:.0%} distractors")

    # Default local paths
    if local_paths is None:
        local_paths = []

        # Add Development folder
        dev_path = Path.home() / "Development"
        if dev_path.exists():
            local_paths.append(dev_path)
            logger.info(f"Scanning local path: {dev_path}")

        # Add Books folder for diverse document categories
        books_path = Path.home() / "Family" / "Shared" / "Books_2"
        if books_path.exists():
            local_paths.append(books_path)
            logger.info(f"Scanning books path: {books_path}")

        if not local_paths:
            logger.warning("No local paths found to scan")

    # Default cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "draagon_benchmark"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")

    # Configure builder (output_path=None - we'll save manually after filtering)
    config = CorpusBuilderConfig(
        min_documents=min_documents,
        distractor_ratio=distractor_ratio,
        local=SourceConfig(enabled=bool(local_paths), max_docs=300),  # Room for multiple paths
        online=SourceConfig(enabled=True, max_docs=200),
        legal=SourceConfig(enabled=True, max_docs=50),
        distractors=SourceConfig(enabled=True),
        cache_dir=cache_dir,
        output_path=None,  # Don't auto-save, we filter first
    )

    builder = CorpusBuilder(
        local_paths=local_paths,
        config=config,
    )

    # Combine all online sources
    all_online_sources = (
        EXTENDED_TECHNICAL_SOURCES +
        EXTENDED_NARRATIVE_SOURCES +
        EXTENDED_ACADEMIC_SOURCES
    )

    logger.info(f"Online sources configured: {len(all_online_sources)}")
    for source in all_online_sources:
        logger.info(f"  - {source.name}: {source.base_url}")

    # Build corpus
    logger.info("Building corpus (this may take a few minutes)...")
    corpus = await builder.build(
        online_sources=all_online_sources,
        version="1.0.0",
        description=f"Benchmark corpus built on {datetime.now().isoformat()}",
    )

    # Filter out documents with potential secrets
    logger.info("\nFiltering documents for potential secrets...")
    original_count = len(corpus.documents)
    filtered_docs, removed_count = filter_documents_with_secrets(corpus.documents)

    if removed_count > 0:
        logger.warning(f"Filtered out {removed_count} documents with potential secrets")
        # Create new corpus with filtered documents
        corpus = DocumentCorpus(
            documents=filtered_docs,
            version=corpus.version,
            description=corpus.description,
        )

    logger.info(f"Documents after filtering: {len(corpus)} (removed {removed_count})")

    # Save the filtered corpus
    corpus.save(output_path)
    logger.info(f"Saved filtered corpus to {output_path}")

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("CORPUS BUILD COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents: {len(corpus)}")
    logger.info(f"Saved to: {output_path}")

    # Log distribution
    logger.info("\nSource distribution:")
    for source, count in corpus.metadata.source_distribution.items():
        pct = count / len(corpus) * 100 if len(corpus) > 0 else 0
        logger.info(f"  {source}: {count} ({pct:.1f}%)")

    logger.info("\nCategory distribution:")
    for category, count in corpus.metadata.category_distribution.items():
        pct = count / len(corpus) * 100 if len(corpus) > 0 else 0
        logger.info(f"  {category}: {count} ({pct:.1f}%)")

    logger.info(f"\nDistractors: {corpus.metadata.distractor_count} ({corpus.metadata.distractor_ratio:.1%})")
    logger.info(f"Size stats: min={corpus.metadata.size_stats['min']}, max={corpus.metadata.size_stats['max']}, mean={corpus.metadata.size_stats['mean']:.0f}")

    # Validate
    issues = builder.validate_corpus(corpus)
    if issues:
        logger.warning("\nValidation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\nValidation: PASSED ✓")

    # Save provenance report
    provenance_path = output_path.with_suffix(".provenance.json")
    save_provenance_report(corpus, provenance_path)
    logger.info(f"\nProvenance report saved to: {provenance_path}")

    return corpus


def save_provenance_report(corpus: DocumentCorpus, path: Path) -> None:
    """Save a detailed provenance report for all documents.

    This tracks the source of every document for attribution and reproducibility.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "corpus_version": corpus.version,
        "total_documents": len(corpus),
        "sources": {},
        "documents": [],
    }

    # Group by source
    source_groups: dict[str, list] = {}

    for doc in corpus.documents:
        # Build document record
        doc_record = {
            "doc_id": doc.doc_id,
            "source_type": doc.source.value,
            "category": doc.category.value,
            "domain": doc.domain,
            "file_path": doc.file_path,
            "content_hash": doc.content_hash,
            "size_bytes": doc.size_bytes,
            "created_at": doc.created_at.isoformat(),
            "semantic_tags": doc.semantic_tags,
            "is_distractor": doc.is_distractor,
            "metadata": doc.metadata,
        }
        report["documents"].append(doc_record)

        # Group by domain for summary
        domain = doc.domain
        if domain not in source_groups:
            source_groups[domain] = []
        source_groups[domain].append({
            "doc_id": doc.doc_id,
            "file_path": doc.file_path,
            "source_type": doc.source.value,
        })

    # Summary by domain
    for domain, docs in source_groups.items():
        report["sources"][domain] = {
            "count": len(docs),
            "source_types": list(set(d["source_type"] for d in docs)),
            "sample_paths": [d["file_path"] for d in docs[:5]],
        }

    # Write report
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build a benchmark corpus by downloading documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build default corpus (500+ docs)
    python scripts/build_benchmark_corpus.py

    # Build smaller corpus for testing
    python scripts/build_benchmark_corpus.py --min-docs 100 --output test_corpus.json

    # Skip local scanning
    python scripts/build_benchmark_corpus.py --no-local
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("benchmark_corpus.json"),
        help="Output path for corpus JSON (default: benchmark_corpus.json)",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=500,
        help="Minimum document count target (default: 500)",
    )
    parser.add_argument(
        "--distractor-ratio",
        type=float,
        default=0.4,
        help="Ratio of synthetic distractors (default: 0.4)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded content",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip local directory scanning",
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        action="append",
        dest="local_paths",
        help="Local directory to scan (can be specified multiple times)",
    )

    args = parser.parse_args()

    # Handle local paths
    local_paths = args.local_paths
    if args.no_local:
        local_paths = []

    # Run async build
    corpus = asyncio.run(
        build_corpus(
            output_path=args.output,
            min_documents=args.min_docs,
            distractor_ratio=args.distractor_ratio,
            local_paths=local_paths,
            cache_dir=args.cache_dir,
        )
    )

    print(f"\n✓ Corpus built successfully: {len(corpus)} documents")
    print(f"  Output: {args.output}")
    print(f"  Provenance: {args.output.with_suffix('.provenance.json')}")


if __name__ == "__main__":
    main()

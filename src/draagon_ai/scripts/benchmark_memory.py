#!/usr/bin/env python3
"""Memory system performance benchmarks.

Benchmarks the draagon-ai memory system against target performance metrics.

Requirements:
    - Running Qdrant instance (default: http://192.168.168.216:6333)
    - qdrant-client package installed

Target Metrics (from REQ-001-09):
    | Operation                    | Target  | Max Acceptable |
    |------------------------------|---------|----------------|
    | Store single memory          | <50ms   | 100ms          |
    | Search (top-5)               | <100ms  | 200ms          |
    | Promotion cycle (100 items)  | <5s     | 10s            |
    | Load graph (1000 nodes)      | <2s     | 5s             |

Usage:
    python -m draagon_ai.scripts.benchmark_memory [options]

Options:
    --qdrant-url URL    Qdrant instance URL (default: http://192.168.168.216:6333)
    --iterations N      Number of iterations per benchmark (default: 10)
    --warmup N          Warmup iterations (default: 3)
    --json              Output results as JSON
    --verbose           Show detailed output
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.models import (
        VectorParams,
        Distance,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    AsyncQdrantClient = None  # type: ignore

from draagon_ai.memory import (
    LayeredMemoryProvider,
    TemporalCognitiveGraph,
)
from draagon_ai.memory.providers.layered import LayeredMemoryConfig
from draagon_ai.memory.providers.qdrant_graph import QdrantGraphStore, QdrantGraphConfig
from draagon_ai.memory.base import MemoryType, MemoryScope
from draagon_ai.memory.temporal_nodes import NodeType, MemoryLayer, EdgeType


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    qdrant_url: str = "http://192.168.168.216:6333"
    iterations: int = 10
    warmup: int = 3
    embedding_dimension: int = 768
    verbose: bool = False
    json_output: bool = False


@dataclass
class BenchmarkTarget:
    """Target performance metrics."""
    target_ms: float
    max_acceptable_ms: float


TARGETS = {
    "store_single": BenchmarkTarget(target_ms=50, max_acceptable_ms=100),
    "search_top_5": BenchmarkTarget(target_ms=100, max_acceptable_ms=200),
    "promotion_100": BenchmarkTarget(target_ms=5000, max_acceptable_ms=10000),
    "load_graph_1000": BenchmarkTarget(target_ms=2000, max_acceptable_ms=5000),
}


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    iterations: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0

    @property
    def stddev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def meets_target(self, target: BenchmarkTarget) -> bool:
        """Check if result meets target."""
        return self.median_ms <= target.target_ms

    def within_acceptable(self, target: BenchmarkTarget) -> bool:
        """Check if result is within acceptable range."""
        return self.median_ms <= target.max_acceptable_ms

    def to_dict(self) -> dict[str, Any]:
        target = TARGETS.get(self.name)
        result = {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "stddev_ms": round(self.stddev_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }
        if target:
            result["target_ms"] = target.target_ms
            result["max_acceptable_ms"] = target.max_acceptable_ms
            result["meets_target"] = self.meets_target(target)
            result["within_acceptable"] = self.within_acceptable(target)
        return result


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    config: BenchmarkConfig
    results: list[BenchmarkResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def add_error(self, error: str):
        self.errors.append(error)

    @property
    def all_passed(self) -> bool:
        """Check if all benchmarks passed."""
        for result in self.results:
            target = TARGETS.get(result.name)
            if target and not result.within_acceptable(target):
                return False
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
            "summary": self._summary(),
        }

    def _summary(self) -> dict[str, int]:
        passed = 0
        failed = 0
        for result in self.results:
            target = TARGETS.get(result.name)
            if target:
                if result.within_acceptable(target):
                    passed += 1
                else:
                    failed += 1
        return {"passed": passed, "failed": failed, "errors": len(self.errors)}


# =============================================================================
# Embedding Provider
# =============================================================================

class WordBasedEmbeddingProvider:
    """Word-based embedding provider for benchmarks.

    Generates consistent embeddings based on word content.
    Similar texts produce similar vectors.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    def _get_word_idx(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_idx
            self._next_idx += 1
        return self._vocab[word]

    async def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        words = re.findall(r'\b\w+\b', text.lower())
        embedding = [0.0] * self.dimension

        for word in words:
            idx = self._get_word_idx(word)
            for offset in range(5):
                dim_idx = (idx * 7 + offset * 13) % self.dimension
                embedding[dim_idx] += 0.5

        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        else:
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [(hash_bytes[i % 32] / 255.0) * 0.1 for i in range(self.dimension)]

        self._cache[text] = embedding
        return embedding


# =============================================================================
# Benchmark Runner
# =============================================================================

class MemoryBenchmarkRunner:
    """Runs memory system benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.embedder = WordBasedEmbeddingProvider(config.embedding_dimension)
        self._collection_name = f"benchmark_{uuid4().hex[:8]}"
        self._client: AsyncQdrantClient | None = None
        self._provider: LayeredMemoryProvider | None = None
        self._graph: QdrantGraphStore | None = None

    async def setup(self):
        """Initialize benchmark environment."""
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client package not installed")

        self._client = AsyncQdrantClient(url=self.config.qdrant_url, timeout=30.0)

        # Test connection
        try:
            await self._client.get_collections()
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Qdrant at {self.config.qdrant_url}: {e}")

        if self.config.verbose:
            print(f"Connected to Qdrant at {self.config.qdrant_url}")
            print(f"Using collection: {self._collection_name}")

    async def teardown(self):
        """Clean up benchmark environment."""
        if self._client:
            # Delete all test collections
            collections = await self._client.get_collections()
            for col in collections.collections:
                if col.name.startswith("benchmark_"):
                    try:
                        await self._client.delete_collection(col.name)
                    except Exception:
                        pass
            await self._client.close()

    async def _create_provider(self) -> LayeredMemoryProvider:
        """Create a fresh LayeredMemoryProvider."""
        collection_prefix = f"benchmark_{uuid4().hex[:8]}"
        config = LayeredMemoryConfig(
            qdrant_url=self.config.qdrant_url,
            qdrant_nodes_collection=f"{collection_prefix}_nodes",
            qdrant_edges_collection=f"{collection_prefix}_edges",
            embedding_dimension=self.config.embedding_dimension,
        )
        provider = LayeredMemoryProvider(config=config, embedding_provider=self.embedder)
        await provider.initialize()
        return provider

    async def _create_graph(self) -> QdrantGraphStore:
        """Create a fresh QdrantGraphStore."""
        collection_name = f"benchmark_{uuid4().hex[:8]}"
        config = QdrantGraphConfig(
            url=self.config.qdrant_url,
            nodes_collection=f"{collection_name}_nodes",
            edges_collection=f"{collection_name}_edges",
            embedding_dimension=self.config.embedding_dimension,
        )
        graph = QdrantGraphStore(config, self.embedder)
        await graph.initialize()
        return graph

    def _log(self, msg: str):
        """Log message if verbose."""
        if self.config.verbose:
            print(f"  {msg}")

    async def benchmark_store_single(self) -> BenchmarkResult:
        """Benchmark storing a single memory."""
        result = BenchmarkResult(name="store_single", iterations=self.config.iterations)

        # Create provider for this benchmark
        provider = await self._create_provider()

        try:
            # Warmup
            for i in range(self.config.warmup):
                await provider.store(
                    content=f"Warmup memory {i}",
                    memory_type=MemoryType.FACT,
                    scope=MemoryScope.USER,
                    user_id="benchmark_user",
                )

            # Benchmark
            for i in range(self.config.iterations):
                content = f"Benchmark memory {i} - This is a test fact about something important at {datetime.now().isoformat()}"

                start = time.perf_counter()
                await provider.store(
                    content=content,
                    memory_type=MemoryType.FACT,
                    scope=MemoryScope.USER,
                    user_id="benchmark_user",
                    metadata={"iteration": i, "timestamp": datetime.now().isoformat()},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.times_ms.append(elapsed_ms)
                self._log(f"Store iteration {i+1}: {elapsed_ms:.2f}ms")

        finally:
            # Cleanup
            await provider.close()

        return result

    async def benchmark_search_top_5(self) -> BenchmarkResult:
        """Benchmark searching for top-5 results."""
        result = BenchmarkResult(name="search_top_5", iterations=self.config.iterations)

        # Create provider and populate with data
        provider = await self._create_provider()

        try:
            # Create 100 memories for searching
            test_topics = [
                "artificial intelligence machine learning",
                "python programming language",
                "database management systems",
                "cloud computing infrastructure",
                "network security protocols",
                "web development frameworks",
                "mobile application design",
                "data science analytics",
                "software engineering practices",
                "operating system kernels",
            ]

            for i in range(100):
                topic = test_topics[i % len(test_topics)]
                content = f"Memory about {topic} - iteration {i} with detailed information"
                await provider.store(
                    content=content,
                    memory_type=MemoryType.FACT,
                    scope=MemoryScope.USER,
                    user_id="benchmark_user",
                )

            # Warmup searches
            for i in range(self.config.warmup):
                await provider.search(
                    query="artificial intelligence",
                    limit=5,
                    user_id="benchmark_user",
                )

            # Benchmark searches
            search_queries = [
                "machine learning algorithms",
                "python code examples",
                "database optimization",
                "cloud services deployment",
                "security best practices",
                "frontend development",
                "mobile apps iOS Android",
                "data visualization",
                "agile development",
                "linux kernel modules",
            ]

            for i in range(self.config.iterations):
                query = search_queries[i % len(search_queries)]

                start = time.perf_counter()
                results = await provider.search(
                    query=query,
                    limit=5,
                    user_id="benchmark_user",
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.times_ms.append(elapsed_ms)
                self._log(f"Search iteration {i+1}: {elapsed_ms:.2f}ms (found {len(results)} results)")

        finally:
            await provider.close()

        return result

    async def benchmark_promotion_100(self) -> BenchmarkResult:
        """Benchmark promoting 100 items through layers."""
        result = BenchmarkResult(name="promotion_100", iterations=self.config.iterations)

        for iteration in range(self.config.iterations):
            # Create fresh provider for each iteration
            provider = await self._create_provider()

            try:
                # Create 100 working memories with high importance (to ensure promotion)
                for i in range(100):
                    await provider.store(
                        content=f"Important fact {i} that should be promoted to episodic memory",
                        memory_type=MemoryType.FACT,
                        scope=MemoryScope.USER,
                        user_id="benchmark_user",
                        importance=0.8,  # High importance to trigger promotion
                        metadata={"iteration": i},
                    )

                # Benchmark promotion cycle
                start = time.perf_counter()
                stats = await provider.promote_all()
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.times_ms.append(elapsed_ms)

                promoted_count = (
                    stats.working_to_episodic +
                    stats.episodic_to_semantic +
                    stats.semantic_to_metacognitive
                )
                self._log(f"Promotion iteration {iteration+1}: {elapsed_ms:.2f}ms (promoted {promoted_count} items)")

            finally:
                await provider.close()

        return result

    async def benchmark_load_graph_1000(self) -> BenchmarkResult:
        """Benchmark loading a graph with 1000 nodes."""
        result = BenchmarkResult(name="load_graph_1000", iterations=self.config.iterations)

        for iteration in range(self.config.iterations):
            # Create a graph and populate it
            graph = await self._create_graph()

            try:
                # Create 1000 nodes
                self._log(f"Creating 1000 nodes for iteration {iteration+1}...")
                node_ids = []
                for i in range(1000):
                    content = f"Node {i} with content about topic {i % 50}"
                    embedding = await self.embedder.embed(content)

                    node = await graph.add_node(
                        content=content,
                        node_type=NodeType.FACT,
                        embedding=embedding,
                        importance=0.5 + (i % 10) * 0.05,
                        metadata={"index": i},
                    )
                    node_ids.append(node.node_id)

                    # Add some edges (every 10th node links to previous)
                    if i > 0 and i % 10 == 0:
                        await graph.add_edge(
                            source_id=node_ids[-1],
                            target_id=node_ids[-2],
                            edge_type=EdgeType.RELATED_TO,
                        )

                # Close and reopen to simulate fresh load
                await graph._client.close()

                # Create new graph instance pointing to same collection
                load_config = QdrantGraphConfig(
                    url=self.config.qdrant_url,
                    nodes_collection=graph.config.nodes_collection,
                    edges_collection=graph.config.edges_collection,
                    embedding_dimension=self.config.embedding_dimension,
                )

                # Benchmark the load
                start = time.perf_counter()
                new_graph = QdrantGraphStore(load_config, self.embedder)
                await new_graph.initialize()
                # Force load by accessing nodes
                await new_graph.load_from_qdrant()
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.times_ms.append(elapsed_ms)

                node_count = len(new_graph._nodes)
                self._log(f"Load iteration {iteration+1}: {elapsed_ms:.2f}ms (loaded {node_count} nodes)")

                await new_graph._client.close()

            finally:
                # Cleanup collections
                if self._client:
                    try:
                        await self._client.delete_collection(graph.config.nodes_collection)
                        await self._client.delete_collection(graph.config.edges_collection)
                    except Exception:
                        pass

        return result

    async def benchmark_concurrent_stores(self) -> BenchmarkResult:
        """Benchmark concurrent store operations."""
        result = BenchmarkResult(name="concurrent_stores_10", iterations=self.config.iterations)

        provider = await self._create_provider()

        try:
            for iteration in range(self.config.iterations):
                # Create 10 concurrent store operations
                async def store_memory(idx: int):
                    await provider.store(
                        content=f"Concurrent memory {idx} at {datetime.now().isoformat()}",
                        memory_type=MemoryType.FACT,
                        scope=MemoryScope.USER,
                        user_id="benchmark_user",
                    )

                start = time.perf_counter()
                await asyncio.gather(*[store_memory(i) for i in range(10)])
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.times_ms.append(elapsed_ms)
                self._log(f"Concurrent stores iteration {iteration+1}: {elapsed_ms:.2f}ms")

        finally:
            await provider.close()

        return result

    async def benchmark_concurrent_searches(self) -> BenchmarkResult:
        """Benchmark concurrent search operations."""
        result = BenchmarkResult(name="concurrent_searches_10", iterations=self.config.iterations)

        provider = await self._create_provider()

        try:
            # Populate with data first
            for i in range(50):
                await provider.store(
                    content=f"Test memory {i} about various topics including AI and databases",
                    memory_type=MemoryType.FACT,
                    scope=MemoryScope.USER,
                    user_id="benchmark_user",
                )

            for iteration in range(self.config.iterations):
                # Create 10 concurrent search operations
                queries = [
                    "artificial intelligence",
                    "database systems",
                    "cloud computing",
                    "machine learning",
                    "software development",
                    "data science",
                    "web applications",
                    "network security",
                    "mobile development",
                    "operating systems",
                ]

                async def search_memory(query: str):
                    return await provider.search(
                        query=query,
                        limit=5,
                        user_id="benchmark_user",
                    )

                start = time.perf_counter()
                await asyncio.gather(*[search_memory(q) for q in queries])
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.times_ms.append(elapsed_ms)
                self._log(f"Concurrent searches iteration {iteration+1}: {elapsed_ms:.2f}ms")

        finally:
            await provider.close()

        return result

    async def run_all(self) -> BenchmarkReport:
        """Run all benchmarks."""
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            config=self.config,
        )

        benchmarks = [
            ("store_single", self.benchmark_store_single),
            ("search_top_5", self.benchmark_search_top_5),
            ("promotion_100", self.benchmark_promotion_100),
            ("load_graph_1000", self.benchmark_load_graph_1000),
            ("concurrent_stores_10", self.benchmark_concurrent_stores),
            ("concurrent_searches_10", self.benchmark_concurrent_searches),
        ]

        for name, benchmark_fn in benchmarks:
            if not self.config.json_output:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print(f"{'='*60}")

            try:
                result = await benchmark_fn()
                report.add_result(result)

                if not self.config.json_output:
                    target = TARGETS.get(name)
                    status = "✓" if (not target or result.within_acceptable(target)) else "✗"
                    print(f"\nResult: {status}")
                    print(f"  Median: {result.median_ms:.2f}ms")
                    print(f"  Mean:   {result.mean_ms:.2f}ms")
                    print(f"  Min:    {result.min_ms:.2f}ms")
                    print(f"  Max:    {result.max_ms:.2f}ms")
                    print(f"  P95:    {result.p95_ms:.2f}ms")
                    if target:
                        print(f"  Target: {target.target_ms}ms (max: {target.max_acceptable_ms}ms)")
                        if result.meets_target(target):
                            print(f"  Status: MEETS TARGET ✓")
                        elif result.within_acceptable(target):
                            print(f"  Status: WITHIN ACCEPTABLE ✓")
                        else:
                            print(f"  Status: EXCEEDS MAXIMUM ✗")

            except Exception as e:
                error_msg = f"{name}: {str(e)}"
                report.add_error(error_msg)
                if not self.config.json_output:
                    print(f"\nError: {e}")

        return report


# =============================================================================
# Output Formatting
# =============================================================================

def print_report(report: BenchmarkReport):
    """Print formatted benchmark report."""
    print("\n" + "="*70)
    print("MEMORY SYSTEM BENCHMARK REPORT")
    print("="*70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Qdrant URL: {report.config.qdrant_url}")
    print(f"Iterations: {report.config.iterations}")

    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)

    # Table header
    print(f"{'Benchmark':<25} {'Median':>10} {'Target':>10} {'Max':>10} {'Status':>10}")
    print("-"*70)

    for result in report.results:
        target = TARGETS.get(result.name)
        target_str = f"{target.target_ms}ms" if target else "N/A"
        max_str = f"{target.max_acceptable_ms}ms" if target else "N/A"

        if target:
            if result.meets_target(target):
                status = "✓ TARGET"
            elif result.within_acceptable(target):
                status = "✓ OK"
            else:
                status = "✗ FAIL"
        else:
            status = "N/A"

        print(f"{result.name:<25} {result.median_ms:>8.1f}ms {target_str:>10} {max_str:>10} {status:>10}")

    if report.errors:
        print("\n" + "-"*70)
        print("ERRORS")
        print("-"*70)
        for error in report.errors:
            print(f"  ✗ {error}")

    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    summary = report._summary()
    total = summary["passed"] + summary["failed"]
    if total > 0:
        print(f"  Passed: {summary['passed']}/{total}")
        print(f"  Failed: {summary['failed']}/{total}")
    if summary["errors"] > 0:
        print(f"  Errors: {summary['errors']}")

    if report.all_passed:
        print("\n  ✓ ALL BENCHMARKS PASSED")
    else:
        print("\n  ✗ SOME BENCHMARKS FAILED")

    print("="*70)


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Memory system performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target Metrics:
  store_single      <50ms (max 100ms)
  search_top_5      <100ms (max 200ms)
  promotion_100     <5s (max 10s)
  load_graph_1000   <2s (max 5s)

Examples:
  python -m draagon_ai.scripts.benchmark_memory
  python -m draagon_ai.scripts.benchmark_memory --iterations 20 --verbose
  python -m draagon_ai.scripts.benchmark_memory --json > results.json
        """
    )

    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://192.168.168.216:6333"),
        help="Qdrant instance URL"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        qdrant_url=args.qdrant_url,
        iterations=args.iterations,
        warmup=args.warmup,
        verbose=args.verbose,
        json_output=args.json,
    )

    if not args.json:
        print("Memory System Benchmarks")
        print(f"Qdrant: {config.qdrant_url}")
        print(f"Iterations: {config.iterations} (warmup: {config.warmup})")

    runner = MemoryBenchmarkRunner(config)

    try:
        await runner.setup()
        report = await runner.run_all()

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print_report(report)

        # Exit with error code if benchmarks failed
        sys.exit(0 if report.all_passed else 1)

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}")
        sys.exit(1)

    finally:
        await runner.teardown()


if __name__ == "__main__":
    asyncio.run(main())

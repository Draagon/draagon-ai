"""Scripts and utilities for draagon-ai.

Available scripts:
    - migrate_roxy_memories: Migrate Roxy memories to layered structure
    - benchmark_memory: Performance benchmarks for memory system

Usage:
    python -m draagon_ai.scripts.migrate_roxy_memories --help
    python -m draagon_ai.scripts.benchmark_memory --help
"""

from draagon_ai.scripts.migrate_roxy_memories import (
    MigrationConfig,
    MigrationEngine,
    MigrationRecord,
    MigrationStats,
    MigrationStatus,
    rollback,
)
from draagon_ai.scripts.benchmark_memory import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkReport,
    MemoryBenchmarkRunner,
    TARGETS,
)

__all__ = [
    # Migration
    "MigrationConfig",
    "MigrationEngine",
    "MigrationRecord",
    "MigrationStats",
    "MigrationStatus",
    "rollback",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkReport",
    "MemoryBenchmarkRunner",
    "TARGETS",
]

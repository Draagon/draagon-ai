# TASK-087: Checkpointing & Progress Tracking

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Prevents lost work on long runs)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-086 (BenchmarkRunner)

---

## Description

Implement checkpointing and progress tracking for long-running benchmarks:
- Save intermediate results periodically
- Resume from checkpoint after crash/interrupt
- Real-time progress display
- Estimated time remaining

For 250+ queries × 5 runs, benchmarks can take 30+ minutes. Don't lose progress.

**Location:** `src/draagon_ai/testing/benchmarks/checkpointing.py`

---

## Acceptance Criteria

### Checkpointing
- [ ] `Checkpoint` dataclass with run state
- [ ] Save checkpoint every N queries (configurable)
- [ ] Save on graceful shutdown (SIGINT/SIGTERM)
- [ ] Resume from latest checkpoint

### State Preservation
- [ ] Completed queries and their results
- [ ] Current run number and seed
- [ ] Partial evaluation results
- [ ] Configuration used

### Progress Tracking
- [ ] Real-time progress bar (tqdm or similar)
- [ ] Queries completed / total
- [ ] Current run / total runs
- [ ] Estimated time remaining
- [ ] Current query being processed

### Recovery
- [ ] Detect existing checkpoints on startup
- [ ] Validate checkpoint integrity
- [ ] Skip already-completed queries
- [ ] Merge checkpoint with new results

---

## Technical Notes

### Checkpoint Format

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

@dataclass
class Checkpoint:
    # Identification
    benchmark_id: str
    created_at: datetime
    updated_at: datetime

    # Configuration
    config_hash: str  # SHA256 of config for validation
    config: dict

    # Progress
    current_run: int
    total_runs: int
    current_query_index: int
    total_queries: int

    # Results
    completed_queries: dict[str, dict]  # query_id -> results
    run_results: list[dict]  # Completed runs

    # Timing
    elapsed_seconds: float
    estimated_remaining_seconds: float

    def save(self, path: Path):
        """Save checkpoint to file."""
        data = {
            "benchmark_id": self.benchmark_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config_hash": self.config_hash,
            "config": self.config,
            "current_run": self.current_run,
            "total_runs": self.total_runs,
            "current_query_index": self.current_query_index,
            "total_queries": self.total_queries,
            "completed_queries": self.completed_queries,
            "run_results": self.run_results,
            "elapsed_seconds": self.elapsed_seconds,
        }

        # Atomic write (write to temp, then rename)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(path)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            benchmark_id=data["benchmark_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            config_hash=data["config_hash"],
            config=data["config"],
            current_run=data["current_run"],
            total_runs=data["total_runs"],
            current_query_index=data["current_query_index"],
            total_queries=data["total_queries"],
            completed_queries=data["completed_queries"],
            run_results=data["run_results"],
            elapsed_seconds=data["elapsed_seconds"],
            estimated_remaining_seconds=0,
        )

    def is_valid_for_config(self, config_hash: str) -> bool:
        """Check if checkpoint matches current config."""
        return self.config_hash == config_hash
```

### Checkpoint Manager

```python
import hashlib
import signal
import atexit
from typing import Optional, Callable

class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_interval: int = 50,
        config: BenchmarkConfig = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.config = config
        self.config_hash = self._hash_config(config)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "checkpoint.json"

        self.checkpoint: Optional[Checkpoint] = None
        self._setup_signal_handlers()

    def _hash_config(self, config: BenchmarkConfig) -> str:
        """Create hash of config for validation."""
        import json
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _setup_signal_handlers(self):
        """Setup handlers to save on interrupt."""
        def save_on_exit(signum=None, frame=None):
            if self.checkpoint:
                logger.info("Saving checkpoint before exit...")
                self.save()

        signal.signal(signal.SIGINT, save_on_exit)
        signal.signal(signal.SIGTERM, save_on_exit)
        atexit.register(save_on_exit)

    def load_or_create(
        self,
        benchmark_id: str,
        total_runs: int,
        total_queries: int,
    ) -> Checkpoint:
        """Load existing checkpoint or create new one."""
        if self.checkpoint_path.exists():
            try:
                checkpoint = Checkpoint.load(self.checkpoint_path)

                if checkpoint.is_valid_for_config(self.config_hash):
                    logger.info(
                        f"Resuming from checkpoint: run {checkpoint.current_run}, "
                        f"query {checkpoint.current_query_index}"
                    )
                    self.checkpoint = checkpoint
                    return checkpoint
                else:
                    logger.warning("Checkpoint config mismatch, starting fresh")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        # Create new checkpoint
        self.checkpoint = Checkpoint(
            benchmark_id=benchmark_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config_hash=self.config_hash,
            config=self.config.__dict__,
            current_run=0,
            total_runs=total_runs,
            current_query_index=0,
            total_queries=total_queries,
            completed_queries={},
            run_results=[],
            elapsed_seconds=0,
            estimated_remaining_seconds=0,
        )

        return self.checkpoint

    def update(
        self,
        run: int,
        query_index: int,
        query_id: str,
        result: dict,
    ):
        """Update checkpoint with query result."""
        self.checkpoint.current_run = run
        self.checkpoint.current_query_index = query_index
        self.checkpoint.completed_queries[query_id] = result

        # Save periodically
        if query_index % self.checkpoint_interval == 0:
            self.save()

    def complete_run(self, run: int, run_result: dict):
        """Mark run as complete."""
        self.checkpoint.run_results.append(run_result)
        self.checkpoint.current_run = run + 1
        self.checkpoint.current_query_index = 0
        self.save()

    def save(self):
        """Save current checkpoint."""
        if self.checkpoint:
            self.checkpoint.save(self.checkpoint_path)

    def is_query_completed(self, run: int, query_id: str) -> bool:
        """Check if query was already processed in current run."""
        return query_id in self.checkpoint.completed_queries

    def get_completed_result(self, query_id: str) -> Optional[dict]:
        """Get result for completed query."""
        return self.checkpoint.completed_queries.get(query_id)

    def clear(self):
        """Remove checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
```

### Progress Display

```python
from tqdm import tqdm
import time

class ProgressTracker:
    def __init__(
        self,
        total_runs: int,
        total_queries: int,
        description: str = "Benchmark",
    ):
        self.total_runs = total_runs
        self.total_queries = total_queries
        self.total_operations = total_runs * total_queries

        self.pbar = tqdm(
            total=self.total_operations,
            desc=description,
            unit="query",
            dynamic_ncols=True,
        )

        self.start_time = time.time()
        self.current_run = 0
        self.current_query = 0

    def update(self, run: int, query_index: int, query_id: str = None):
        """Update progress with current position."""
        self.current_run = run
        self.current_query = query_index

        # Calculate progress
        completed = run * self.total_queries + query_index
        self.pbar.n = completed
        self.pbar.refresh()

        # Update description with current status
        elapsed = time.time() - self.start_time
        if completed > 0:
            rate = completed / elapsed
            remaining = (self.total_operations - completed) / rate
            self.pbar.set_postfix({
                "run": f"{run+1}/{self.total_runs}",
                "query": query_id[:20] if query_id else "",
                "eta": f"{remaining/60:.1f}m",
            })

    def complete_run(self, run: int):
        """Mark run as complete."""
        self.pbar.set_description(f"Run {run+1}/{self.total_runs} complete")

    def close(self):
        """Close progress bar."""
        self.pbar.close()
```

---

## Testing Requirements

### Unit Tests
```python
def test_checkpoint_save_load(tmp_path):
    """Checkpoint round-trips correctly."""
    checkpoint = Checkpoint(
        benchmark_id="test",
        current_run=2,
        current_query_index=50,
        completed_queries={"q1": {"score": 0.9}},
        ...
    )

    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)

    loaded = Checkpoint.load(path)
    assert loaded.current_run == 2
    assert loaded.current_query_index == 50
    assert "q1" in loaded.completed_queries

def test_checkpoint_config_validation():
    """Checkpoint rejected if config changed."""
    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
        config=BenchmarkConfig(num_runs=5, ...),
    )

    # Create checkpoint with different config
    old_checkpoint = Checkpoint(config_hash="different", ...)
    old_checkpoint.save(manager.checkpoint_path)

    # Should create new checkpoint, not load old
    checkpoint = manager.load_or_create("test", 5, 100)
    assert checkpoint.config_hash == manager.config_hash

def test_resume_from_checkpoint(tmp_path):
    """Resume skips completed queries."""
    # Create checkpoint with some completed queries
    manager = CheckpointManager(tmp_path, config=config)
    checkpoint = manager.load_or_create("test", 5, 100)

    manager.update(0, 25, "q25", {"score": 0.9})
    manager.save()

    # New manager should resume
    manager2 = CheckpointManager(tmp_path, config=config)
    checkpoint2 = manager2.load_or_create("test", 5, 100)

    assert manager2.is_query_completed(0, "q25")
    assert manager2.get_completed_result("q25") == {"score": 0.9}
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_on_interrupt(tmp_path):
    """Checkpoint saved on interrupt."""
    import subprocess
    import os

    # Start benchmark in subprocess
    proc = subprocess.Popen(
        ["python", "-m", "draagon_ai.testing.benchmarks", "--config", str(config_path)],
        cwd=str(tmp_path),
    )

    # Wait a bit, then interrupt
    await asyncio.sleep(5)
    proc.send_signal(signal.SIGINT)
    proc.wait()

    # Checkpoint should exist
    assert (tmp_path / "checkpoint.json").exists()
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/checkpointing.py`
- `src/draagon_ai/testing/benchmarks/progress.py`
- Integrate with `runner.py`
- Add tests to `tests/benchmarks/test_checkpointing.py`

---

## Definition of Done

- [ ] Checkpoint dataclass with all state
- [ ] Atomic checkpoint saves
- [ ] Load and validate checkpoints
- [ ] Config hash validation
- [ ] Signal handlers (SIGINT, SIGTERM)
- [ ] Resume from checkpoint
- [ ] Skip completed queries
- [ ] Progress bar with ETA
- [ ] Periodic checkpoint saves
- [ ] Integration test for interrupt handling

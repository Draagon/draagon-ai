"""Task persistence layer for the scheduling system."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .models import ScheduledTask, TaskStatus, TriggerType

logger = logging.getLogger(__name__)


class TaskPersistence(ABC):
    """Abstract base class for task persistence.

    Implementations can use Qdrant, SQLite, file-based storage, or in-memory.
    """

    @abstractmethod
    async def save(self, task: ScheduledTask) -> None:
        """Save or update a task."""
        pass

    @abstractmethod
    async def get(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        pass

    @abstractmethod
    async def delete(self, task_id: str) -> bool:
        """Delete a task by ID. Returns True if deleted."""
        pass

    @abstractmethod
    async def list_tasks(
        self,
        owner_id: str | None = None,
        owner_type: str | None = None,
        status: TaskStatus | None = None,
        trigger_type: TriggerType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ScheduledTask]:
        """List tasks with optional filtering."""
        pass

    @abstractmethod
    async def get_pending(
        self,
        trigger_type: TriggerType | None = None,
    ) -> list[ScheduledTask]:
        """Get all pending tasks, optionally filtered by trigger type."""
        pass

    @abstractmethod
    async def get_due_tasks(
        self,
        before: datetime | None = None,
        trigger_type: TriggerType | None = None,
    ) -> list[ScheduledTask]:
        """Get tasks that are due to run (next_run <= before)."""
        pass

    @abstractmethod
    async def get_by_event(self, event_type: str) -> list[ScheduledTask]:
        """Get tasks that are triggered by a specific event type."""
        pass


class InMemoryPersistence(TaskPersistence):
    """In-memory persistence for testing and development."""

    def __init__(self):
        self._tasks: dict[str, ScheduledTask] = {}

    async def save(self, task: ScheduledTask) -> None:
        """Save or update a task."""
        self._tasks[task.task_id] = task

    async def get(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    async def delete(self, task_id: str) -> bool:
        """Delete a task by ID."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    async def list_tasks(
        self,
        owner_id: str | None = None,
        owner_type: str | None = None,
        status: TaskStatus | None = None,
        trigger_type: TriggerType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ScheduledTask]:
        """List tasks with optional filtering."""
        result = []
        for task in self._tasks.values():
            if owner_id and task.owner_id != owner_id:
                continue
            if owner_type and task.owner_type != owner_type:
                continue
            if status and task.status != status:
                continue
            if trigger_type and task.trigger_type != trigger_type:
                continue
            if tags and not all(t in task.tags for t in tags):
                continue
            result.append(task)
            if len(result) >= limit:
                break
        return result

    async def get_pending(
        self,
        trigger_type: TriggerType | None = None,
    ) -> list[ScheduledTask]:
        """Get all pending tasks."""
        result = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if trigger_type and task.trigger_type != trigger_type:
                continue
            result.append(task)
        return result

    async def get_due_tasks(
        self,
        before: datetime | None = None,
        trigger_type: TriggerType | None = None,
    ) -> list[ScheduledTask]:
        """Get tasks that are due to run."""
        now = before or datetime.now()
        result = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if task.next_run is None:
                continue
            if task.next_run > now:
                continue
            if trigger_type and task.trigger_type != trigger_type:
                continue
            result.append(task)
        # Sort by priority then next_run
        result.sort(key=lambda t: (t.priority.value, t.next_run or now))
        return result

    async def get_by_event(self, event_type: str) -> list[ScheduledTask]:
        """Get tasks triggered by a specific event type."""
        result = []
        for task in self._tasks.values():
            if task.trigger_type != TriggerType.EVENT:
                continue
            if task.status not in (TaskStatus.PENDING, TaskStatus.PAUSED):
                continue
            if task.trigger_config.get("event_type") == event_type:
                result.append(task)
        return result

    def clear(self) -> None:
        """Clear all tasks (for testing)."""
        self._tasks.clear()


class QdrantPersistence(TaskPersistence):
    """Qdrant-based persistence for production use.

    Stores tasks in a Qdrant collection with a dummy vector.
    Uses payload filtering for queries.
    """

    RECORD_TYPE = "scheduled_task"

    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        embedding_dimensions: int = 768,
    ):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.embedding_dimensions = embedding_dimensions
        self._http_client: Any = None

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is available."""
        if self._http_client is None:
            try:
                import aiohttp

                self._http_client = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("aiohttp is required for QdrantPersistence")

    async def _qdrant_post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make POST request to Qdrant."""
        await self._ensure_client()
        url = f"{self.qdrant_url}/{endpoint}"
        async with self._http_client.post(url, json=data) as resp:
            return await resp.json()

    async def _qdrant_put(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make PUT request to Qdrant."""
        await self._ensure_client()
        url = f"{self.qdrant_url}/{endpoint}"
        async with self._http_client.put(url, json=data) as resp:
            return await resp.json()

    def _zero_vector(self) -> list[float]:
        """Return a zero vector for storage."""
        return [0.0] * self.embedding_dimensions

    async def save(self, task: ScheduledTask) -> None:
        """Save or update a task."""
        payload = task.to_dict()
        payload["record_type"] = self.RECORD_TYPE

        # Use task_id as point ID for easy updates
        import hashlib

        point_id = hashlib.md5(task.task_id.encode()).hexdigest()

        await self._qdrant_put(
            f"collections/{self.collection}/points",
            data={
                "points": [
                    {
                        "id": point_id,
                        "vector": self._zero_vector(),
                        "payload": payload,
                    }
                ]
            },
        )

    async def get(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        result = await self._qdrant_post(
            f"collections/{self.collection}/points/scroll",
            data={
                "filter": {
                    "must": [
                        {"key": "record_type", "match": {"value": self.RECORD_TYPE}},
                        {"key": "task_id", "match": {"value": task_id}},
                    ]
                },
                "limit": 1,
                "with_payload": True,
            },
        )

        points = result.get("result", {}).get("points", [])
        if points:
            return ScheduledTask.from_dict(points[0]["payload"])
        return None

    async def delete(self, task_id: str) -> bool:
        """Delete a task by ID."""
        # First find the point
        result = await self._qdrant_post(
            f"collections/{self.collection}/points/scroll",
            data={
                "filter": {
                    "must": [
                        {"key": "record_type", "match": {"value": self.RECORD_TYPE}},
                        {"key": "task_id", "match": {"value": task_id}},
                    ]
                },
                "limit": 1,
                "with_payload": False,
            },
        )

        points = result.get("result", {}).get("points", [])
        if not points:
            return False

        point_id = points[0]["id"]
        await self._qdrant_post(
            f"collections/{self.collection}/points/delete",
            data={"points": [point_id]},
        )
        return True

    async def list_tasks(
        self,
        owner_id: str | None = None,
        owner_type: str | None = None,
        status: TaskStatus | None = None,
        trigger_type: TriggerType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ScheduledTask]:
        """List tasks with optional filtering."""
        must_filters = [{"key": "record_type", "match": {"value": self.RECORD_TYPE}}]

        if owner_id:
            must_filters.append({"key": "owner_id", "match": {"value": owner_id}})
        if owner_type:
            must_filters.append({"key": "owner_type", "match": {"value": owner_type}})
        if status:
            must_filters.append({"key": "status", "match": {"value": status.value}})
        if trigger_type:
            must_filters.append(
                {"key": "trigger_type", "match": {"value": trigger_type.value}}
            )

        result = await self._qdrant_post(
            f"collections/{self.collection}/points/scroll",
            data={
                "filter": {"must": must_filters},
                "limit": limit,
                "with_payload": True,
            },
        )

        points = result.get("result", {}).get("points", [])
        tasks = [ScheduledTask.from_dict(p["payload"]) for p in points]

        # Filter by tags in Python (Qdrant array matching is limited)
        if tags:
            tasks = [t for t in tasks if all(tag in t.tags for tag in tags)]

        return tasks

    async def get_pending(
        self,
        trigger_type: TriggerType | None = None,
    ) -> list[ScheduledTask]:
        """Get all pending tasks."""
        must_filters = [
            {"key": "record_type", "match": {"value": self.RECORD_TYPE}},
            {"key": "status", "match": {"value": TaskStatus.PENDING.value}},
        ]

        if trigger_type:
            must_filters.append(
                {"key": "trigger_type", "match": {"value": trigger_type.value}}
            )

        result = await self._qdrant_post(
            f"collections/{self.collection}/points/scroll",
            data={
                "filter": {"must": must_filters},
                "limit": 1000,
                "with_payload": True,
            },
        )

        points = result.get("result", {}).get("points", [])
        return [ScheduledTask.from_dict(p["payload"]) for p in points]

    async def get_due_tasks(
        self,
        before: datetime | None = None,
        trigger_type: TriggerType | None = None,
    ) -> list[ScheduledTask]:
        """Get tasks that are due to run."""
        now = before or datetime.now()

        must_filters = [
            {"key": "record_type", "match": {"value": self.RECORD_TYPE}},
            {"key": "status", "match": {"value": TaskStatus.PENDING.value}},
        ]

        if trigger_type:
            must_filters.append(
                {"key": "trigger_type", "match": {"value": trigger_type.value}}
            )

        result = await self._qdrant_post(
            f"collections/{self.collection}/points/scroll",
            data={
                "filter": {"must": must_filters},
                "limit": 1000,
                "with_payload": True,
            },
        )

        points = result.get("result", {}).get("points", [])
        tasks = []

        for point in points:
            task = ScheduledTask.from_dict(point["payload"])
            if task.next_run and task.next_run <= now:
                tasks.append(task)

        # Sort by priority then next_run
        tasks.sort(key=lambda t: (t.priority.value, t.next_run or now))
        return tasks

    async def get_by_event(self, event_type: str) -> list[ScheduledTask]:
        """Get tasks triggered by a specific event type."""
        result = await self._qdrant_post(
            f"collections/{self.collection}/points/scroll",
            data={
                "filter": {
                    "must": [
                        {"key": "record_type", "match": {"value": self.RECORD_TYPE}},
                        {
                            "key": "trigger_type",
                            "match": {"value": TriggerType.EVENT.value},
                        },
                    ],
                    "should": [
                        {"key": "status", "match": {"value": TaskStatus.PENDING.value}},
                        {"key": "status", "match": {"value": TaskStatus.PAUSED.value}},
                    ],
                },
                "limit": 1000,
                "with_payload": True,
            },
        )

        points = result.get("result", {}).get("points", [])
        tasks = []

        for point in points:
            task = ScheduledTask.from_dict(point["payload"])
            if task.trigger_config.get("event_type") == event_type:
                tasks.append(task)

        return tasks

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None

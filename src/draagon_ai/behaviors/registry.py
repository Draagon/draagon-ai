"""Behavior registry for managing all behaviors.

The registry is the central store for all behaviors across tiers.
It handles registration, lookup, persistence, and trigger matching.
"""

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import (
    Behavior,
    BehaviorStatus,
    BehaviorTier,
    Trigger,
)


class BehaviorRegistry:
    """Central registry for all behaviors."""

    def __init__(
        self,
        core_behaviors_path: Path | None = None,
        addon_behaviors_path: Path | None = None,
        storage_path: Path | None = None,
    ):
        """Initialize the behavior registry.

        Args:
            core_behaviors_path: Path to built-in core behaviors
            addon_behaviors_path: Path to add-on behaviors
            storage_path: Path for persisting generated/custom behaviors
        """
        self._behaviors: dict[str, Behavior] = {}
        self._by_tier: dict[BehaviorTier, list[str]] = {t: [] for t in BehaviorTier}
        self._storage_path = storage_path

        # Load core behaviors from package
        if core_behaviors_path and core_behaviors_path.exists():
            self._load_from_directory(core_behaviors_path, BehaviorTier.CORE)

        # Load add-on behaviors
        if addon_behaviors_path and addon_behaviors_path.exists():
            self._load_from_directory(addon_behaviors_path, BehaviorTier.ADDON)

        # Load generated/custom behaviors from storage
        if storage_path:
            self._load_from_storage()

    # =========================================================================
    # Registration
    # =========================================================================

    def register(self, behavior: Behavior) -> None:
        """Register a behavior.

        Args:
            behavior: The behavior to register

        Raises:
            ValueError: If behavior_id is already registered
        """
        if behavior.behavior_id in self._behaviors:
            # Update existing behavior
            old_tier = self._behaviors[behavior.behavior_id].tier
            if behavior.behavior_id in self._by_tier[old_tier]:
                self._by_tier[old_tier].remove(behavior.behavior_id)

        self._behaviors[behavior.behavior_id] = behavior
        if behavior.behavior_id not in self._by_tier[behavior.tier]:
            self._by_tier[behavior.tier].append(behavior.behavior_id)

    def register_from_application(
        self,
        behavior: Behavior,
        application_id: str,
    ) -> None:
        """Register a behavior from a host application.

        Args:
            behavior: The behavior to register
            application_id: ID of the application providing this behavior
        """
        behavior.tier = BehaviorTier.APPLICATION
        behavior.author = f"app:{application_id}"
        self.register(behavior)

    def unregister(self, behavior_id: str) -> bool:
        """Remove a behavior.

        Args:
            behavior_id: ID of the behavior to remove

        Returns:
            True if removed, False if not found
        """
        if behavior_id not in self._behaviors:
            return False

        behavior = self._behaviors[behavior_id]
        if behavior.behavior_id in self._by_tier[behavior.tier]:
            self._by_tier[behavior.tier].remove(behavior.behavior_id)
        del self._behaviors[behavior_id]
        return True

    # =========================================================================
    # Lookup
    # =========================================================================

    def get(self, behavior_id: str) -> Behavior | None:
        """Get a behavior by ID.

        Args:
            behavior_id: The behavior ID

        Returns:
            The behavior or None if not found
        """
        return self._behaviors.get(behavior_id)

    def get_by_tier(self, tier: BehaviorTier) -> list[Behavior]:
        """Get all behaviors of a tier.

        Args:
            tier: The behavior tier

        Returns:
            List of behaviors in that tier
        """
        return [self._behaviors[bid] for bid in self._by_tier[tier]]

    def get_active(self) -> list[Behavior]:
        """Get all active behaviors.

        Returns:
            List of behaviors with ACTIVE status
        """
        return [b for b in self._behaviors.values() if b.status == BehaviorStatus.ACTIVE]

    def get_all(self) -> list[Behavior]:
        """Get all registered behaviors.

        Returns:
            List of all behaviors
        """
        return list(self._behaviors.values())

    def list_all(self) -> list[str]:
        """List all behavior IDs.

        Returns:
            List of behavior IDs
        """
        return list(self._behaviors.keys())

    def find_by_trigger(self, query: str, context: dict) -> list[Behavior]:
        """Find behaviors whose triggers match.

        Args:
            query: The user query
            context: Additional context (intent, entities, etc.)

        Returns:
            List of matching behaviors, sorted by trigger priority
        """
        matches = []
        for behavior in self.get_active():
            for trigger in behavior.triggers:
                if self._trigger_matches(trigger, query, context):
                    matches.append(behavior)
                    break

        # Sort by highest trigger priority
        return sorted(
            matches,
            key=lambda b: max((t.priority for t in b.triggers), default=0),
            reverse=True,
        )

    def find_by_action(self, action_name: str) -> list[Behavior]:
        """Find behaviors that provide a specific action.

        Args:
            action_name: The action name to find

        Returns:
            List of behaviors providing this action
        """
        matches = []
        for behavior in self.get_active():
            for action in behavior.actions:
                if action.name == action_name:
                    matches.append(behavior)
                    break
        return matches

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_behavior(self, behavior: Behavior) -> None:
        """Persist a behavior to storage.

        Args:
            behavior: The behavior to save
        """
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)
        path = self._storage_path / f"{behavior.behavior_id}.json"
        with open(path, "w") as f:
            json.dump(self._serialize_behavior(behavior), f, indent=2, default=str)

    def delete_behavior_file(self, behavior_id: str) -> bool:
        """Delete a behavior's storage file.

        Args:
            behavior_id: ID of the behavior to delete

        Returns:
            True if deleted, False if not found
        """
        if not self._storage_path:
            return False

        path = self._storage_path / f"{behavior_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def _load_from_storage(self) -> None:
        """Load behaviors from storage directory."""
        if not self._storage_path or not self._storage_path.exists():
            return

        for path in self._storage_path.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    behavior = self._deserialize_behavior(data)
                    self.register(behavior)
            except (json.JSONDecodeError, KeyError) as e:
                # Log error but continue loading other behaviors
                print(f"Warning: Failed to load behavior from {path}: {e}")

    def _load_from_directory(self, directory: Path, tier: BehaviorTier) -> None:
        """Load behaviors from a directory.

        Args:
            directory: Path to the directory
            tier: Tier to assign to loaded behaviors
        """
        if not directory.exists():
            return

        for path in directory.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    behavior = self._deserialize_behavior(data)
                    behavior.tier = tier
                    self.register(behavior)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load behavior from {path}: {e}")

    # =========================================================================
    # Serialization
    # =========================================================================

    def _serialize_behavior(self, behavior: Behavior) -> dict:
        """Serialize a behavior to a dictionary.

        Args:
            behavior: The behavior to serialize

        Returns:
            Dictionary representation
        """
        data = asdict(behavior)

        # Convert enums to strings
        data["tier"] = behavior.tier.value
        data["status"] = behavior.status.value
        data["activation_scope"] = behavior.activation_scope.value

        # Convert datetime to ISO string
        data["created_at"] = behavior.created_at.isoformat()
        data["updated_at"] = behavior.updated_at.isoformat()

        if behavior.metrics.last_evolved:
            data["metrics"]["last_evolved"] = behavior.metrics.last_evolved.isoformat()

        return data

    def _deserialize_behavior(self, data: dict) -> Behavior:
        """Deserialize a behavior from a dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Behavior instance
        """
        # Import here to avoid circular imports
        from .types import (
            Action,
            ActionExample,
            ActionMetrics,
            ActionParameter,
            ActivationScope,
            BehaviorConstraints,
            BehaviorMetrics,
            BehaviorPrompts,
            BehaviorTestCase,
            TestResults,
            Trigger,
        )

        # Convert enums
        data["tier"] = BehaviorTier(data.get("tier", "application"))
        data["status"] = BehaviorStatus(data.get("status", "draft"))
        data["activation_scope"] = ActivationScope(data.get("activation_scope", "global"))

        # Convert datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Convert nested objects
        if data.get("actions"):
            actions = []
            for a in data["actions"]:
                # Convert parameters
                if a.get("parameters"):
                    a["parameters"] = {
                        k: ActionParameter(**v) if isinstance(v, dict) else v
                        for k, v in a["parameters"].items()
                    }
                # Convert examples
                if a.get("examples"):
                    a["examples"] = [
                        ActionExample(**e) if isinstance(e, dict) else e for e in a["examples"]
                    ]
                actions.append(Action(**a) if isinstance(a, dict) else a)
            data["actions"] = actions

        if data.get("triggers"):
            data["triggers"] = [
                Trigger(**t) if isinstance(t, dict) else t for t in data["triggers"]
            ]

        if data.get("prompts") and isinstance(data["prompts"], dict):
            data["prompts"] = BehaviorPrompts(**data["prompts"])

        if data.get("constraints") and isinstance(data["constraints"], dict):
            data["constraints"] = BehaviorConstraints(**data["constraints"])

        if data.get("metrics") and isinstance(data["metrics"], dict):
            metrics_data = data["metrics"]
            if isinstance(metrics_data.get("last_evolved"), str):
                metrics_data["last_evolved"] = datetime.fromisoformat(
                    metrics_data["last_evolved"]
                )
            if metrics_data.get("action_metrics"):
                metrics_data["action_metrics"] = {
                    k: ActionMetrics(**v) if isinstance(v, dict) else v
                    for k, v in metrics_data["action_metrics"].items()
                }
            data["metrics"] = BehaviorMetrics(**metrics_data)

        if data.get("test_cases"):
            data["test_cases"] = [
                BehaviorTestCase(**tc) if isinstance(tc, dict) else tc
                for tc in data["test_cases"]
            ]

        if data.get("test_results") and isinstance(data["test_results"], dict):
            data["test_results"] = TestResults(**data["test_results"])

        return Behavior(**data)

    # =========================================================================
    # Trigger Matching
    # =========================================================================

    def _trigger_matches(self, trigger: Trigger, query: str, context: dict) -> bool:
        """Check if a trigger matches the query/context.

        Args:
            trigger: The trigger to check
            query: The user query
            context: Additional context

        Returns:
            True if trigger matches
        """
        # Check exclusions first
        for pattern in trigger.exclusion_patterns:
            if self._pattern_matches(pattern, query):
                return False

        # Check positive patterns (any match = True)
        for pattern in trigger.keyword_patterns:
            if self._pattern_matches(pattern, query):
                return True

        for intent in trigger.intent_categories:
            if context.get("intent") == intent:
                return True

        for condition in trigger.context_conditions:
            if self._evaluate_condition(condition, context):
                return True

        # Semantic patterns require LLM evaluation - return True to indicate
        # potential match (actual matching done by activation engine)
        return len(trigger.semantic_patterns) > 0

    def _pattern_matches(self, pattern: str, text: str) -> bool:
        """Check if a regex pattern matches text.

        Args:
            pattern: Regex pattern
            text: Text to match

        Returns:
            True if matches
        """
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False

    def _evaluate_condition(self, condition: str, context: dict) -> bool:
        """Evaluate a context condition.

        Simple evaluator for conditions like "user.role == 'admin'".

        Args:
            condition: Condition string
            context: Context dictionary

        Returns:
            True if condition is met
        """
        # Simple key == value parsing
        if "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                key_path = parts[0].strip()
                expected = parts[1].strip().strip("'\"")

                # Navigate dot-separated path
                value = context
                for key in key_path.split("."):
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        return False

                return str(value) == expected

        return False

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_behaviors": len(self._behaviors),
            "by_tier": {tier.value: len(ids) for tier, ids in self._by_tier.items()},
            "by_status": {
                status.value: len([b for b in self._behaviors.values() if b.status == status])
                for status in BehaviorStatus
            },
            "active_count": len(self.get_active()),
            "total_actions": sum(len(b.actions) for b in self._behaviors.values()),
        }

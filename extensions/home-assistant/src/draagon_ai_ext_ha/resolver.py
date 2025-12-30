"""Entity resolution for natural language references.

This module provides fuzzy matching and resolution of natural language
device references to Home Assistant entity IDs.

Example:
    resolver = EntityResolver(client)
    entity = await resolver.resolve("bedroom lights")
    # Returns: ResolvedEntity(entity_id="light.bedroom", confidence=0.9)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import HomeAssistantClient, EntityState

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""

    entity_id: str
    friendly_name: str
    domain: str
    confidence: float
    state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "friendly_name": self.friendly_name,
            "domain": self.domain,
            "confidence": self.confidence,
            "state": self.state,
        }


class EntityResolver:
    """Resolves natural language references to entity IDs.

    Uses a combination of:
    - Exact ID matching
    - Fuzzy name matching
    - Domain filtering
    - Word overlap scoring

    Example:
        resolver = EntityResolver(client)

        # Simple resolution
        entity = await resolver.resolve("bedroom")
        # Returns best match across all domains

        # Domain-specific resolution
        entity = await resolver.resolve("bedroom", domain="light")
        # Returns best light match
    """

    def __init__(self, client: "HomeAssistantClient") -> None:
        """Initialize resolver.

        Args:
            client: Home Assistant client for fetching entities
        """
        self.client = client

    async def resolve(
        self,
        query: str,
        domain: str | None = None,
        area_id: str | None = None,
    ) -> ResolvedEntity | None:
        """Resolve a natural language query to an entity.

        Args:
            query: Natural language reference (e.g., "bedroom lights")
            domain: Optional domain filter (e.g., "light", "switch")
            area_id: Optional area filter

        Returns:
            ResolvedEntity if found, None otherwise
        """
        if not query:
            return None

        # Check for exact entity_id match
        if "." in query:
            state = await self.client.get_entity(query)
            if state:
                domain_part = query.split(".")[0]
                return ResolvedEntity(
                    entity_id=state.entity_id,
                    friendly_name=state.friendly_name or state.entity_id,
                    domain=domain_part,
                    confidence=1.0,
                    state=state.state,
                )

        # Get all entities
        states = await self.client.get_states()
        if not states:
            return None

        # Filter by domain if specified
        if domain:
            states = [s for s in states if s.entity_id.startswith(f"{domain}.")]

        # Score all entities
        scored = []
        for state in states:
            score = self._score_match(query, state, area_id)
            if score > 0:
                scored.append((state, score))

        if not scored:
            return None

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_state, best_score = scored[0]

        domain_part = best_state.entity_id.split(".")[0]
        return ResolvedEntity(
            entity_id=best_state.entity_id,
            friendly_name=best_state.friendly_name or best_state.entity_id,
            domain=domain_part,
            confidence=min(best_score, 1.0),
            state=best_state.state,
        )

    async def resolve_multiple(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 5,
    ) -> list[ResolvedEntity]:
        """Resolve a query to multiple possible entities.

        Args:
            query: Natural language reference
            domain: Optional domain filter
            limit: Maximum results to return

        Returns:
            List of ResolvedEntity matches
        """
        states = await self.client.get_states()
        if not states:
            return []

        if domain:
            states = [s for s in states if s.entity_id.startswith(f"{domain}.")]

        scored = []
        for state in states:
            score = self._score_match(query, state, area_id=None)
            if score > 0.3:  # Minimum threshold
                scored.append((state, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for state, score in scored[:limit]:
            domain_part = state.entity_id.split(".")[0]
            results.append(ResolvedEntity(
                entity_id=state.entity_id,
                friendly_name=state.friendly_name or state.entity_id,
                domain=domain_part,
                confidence=min(score, 1.0),
                state=state.state,
            ))

        return results

    def _score_match(
        self,
        query: str,
        state: "EntityState",
        area_id: str | None,
    ) -> float:
        """Score how well an entity matches a query.

        Args:
            query: User's query
            state: Entity to score
            area_id: Optional area to boost

        Returns:
            Score from 0.0 to 1.0+
        """
        score = 0.0
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        entity_id = state.entity_id.lower()
        friendly_name = (state.friendly_name or "").lower()

        # Extract words from entity references
        entity_words = set(re.findall(r'\w+', entity_id.replace(".", " ")))
        name_words = set(re.findall(r'\w+', friendly_name))
        all_words = entity_words | name_words

        # Exact entity_id match (after domain)
        entity_suffix = entity_id.split(".")[-1] if "." in entity_id else entity_id
        if query_lower == entity_suffix:
            score += 0.9

        # Exact friendly name match
        if query_lower == friendly_name:
            score += 1.0

        # Query contained in friendly name
        if query_lower in friendly_name:
            score += 0.7

        # Query contained in entity_id
        if query_lower in entity_id:
            score += 0.5

        # Word overlap scoring
        overlap = query_words & all_words
        if overlap:
            overlap_ratio = len(overlap) / len(query_words)
            score += overlap_ratio * 0.6

        # Area boost
        if area_id and area_id.lower() in entity_id:
            score += 0.2

        # Penalize unavailable entities
        if state.state in ("unavailable", "unknown"):
            score *= 0.5

        return score

    async def get_controllable_devices(
        self,
        domains: list[str] | None = None,
    ) -> list["EntityState"]:
        """Get all controllable devices.

        Args:
            domains: Optional list of domains to include

        Returns:
            List of controllable entity states
        """
        controllable_domains = domains or [
            "light", "switch", "fan", "cover", "climate",
            "media_player", "vacuum", "lock", "scene",
        ]

        states = await self.client.get_states()
        return [
            s for s in states
            if any(s.entity_id.startswith(f"{d}.") for d in controllable_domains)
            and s.state not in ("unavailable",)
        ]

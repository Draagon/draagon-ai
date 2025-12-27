"""Prompt Registry for storing and retrieving prompts.

This module provides the PromptRegistry class that manages prompts
stored in Qdrant for versioning, evolution, and retrieval.
"""

import json
import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from .types import (
    Prompt,
    PromptDomain,
    PromptMetadata,
    PromptStatus,
    PromptVersion,
)


class PromptRegistry:
    """Registry for managing versioned prompts in Qdrant.

    The registry stores prompts in Qdrant with the following structure:
    - Collection: draagon_prompts (configurable)
    - Each prompt is a point with:
      - Embedding of the prompt content (for semantic search)
      - Payload containing full prompt data

    Example:
        registry = PromptRegistry(qdrant_config, embedding_provider)
        await registry.initialize()

        # Store a prompt
        await registry.store_prompt(
            domain=PromptDomain.DECISION,
            name="DECISION_PROMPT",
            content="You are the decision engine...",
            description="Core action selection prompt",
        )

        # Get a prompt
        prompt = await registry.get_prompt("decision", "DECISION_PROMPT")

        # Search prompts semantically
        results = await registry.search_prompts("home assistant control")
    """

    def __init__(
        self,
        qdrant_url: str = "http://192.168.168.216:6333",
        collection_name: str = "draagon_prompts",
        embedding_provider: Any = None,
    ):
        """Initialize the registry.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name for prompts
            embedding_provider: Provider for generating embeddings
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = None
        self._cache: dict[str, Prompt] = {}

    async def initialize(self) -> None:
        """Initialize the Qdrant client and create collection if needed."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self._client = AsyncQdrantClient(url=self.qdrant_url)

        # Check if collection exists
        collections = await self._client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            # Get embedding dimension
            dimension = 768
            if self._embedding_provider:
                if hasattr(self._embedding_provider, "embedding_dimension"):
                    dimension = self._embedding_provider.embedding_dimension

            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None

    def _get_cache_key(self, domain: str, name: str) -> str:
        """Get cache key for a prompt."""
        return f"{domain}:{name}"

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        if self._embedding_provider:
            return await self._embedding_provider.embed(text)
        # Return zero vector if no provider (for tests)
        return [0.0] * 768

    def _extract_variables(self, content: str) -> list[str]:
        """Extract template variables from prompt content."""
        # Match {variable_name} patterns
        pattern = r"\{(\w+)\}"
        return list(set(re.findall(pattern, content)))

    async def store_prompt(
        self,
        domain: PromptDomain | str,
        name: str,
        content: str,
        description: str = "",
        version: str = "1.0.0",
        status: PromptStatus = PromptStatus.ACTIVE,
        metadata: PromptMetadata | None = None,
        tags: list[str] | None = None,
    ) -> Prompt:
        """Store a prompt in Qdrant.

        Args:
            domain: Prompt domain
            name: Unique prompt name
            content: Prompt text
            description: Human-readable description
            version: Version string
            status: Initial status
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            The stored Prompt
        """
        from qdrant_client.http.models import PointStruct

        # Normalize domain - keep original string for storage
        domain_str = domain.value if isinstance(domain, PromptDomain) else domain
        if isinstance(domain, str):
            try:
                domain_enum = PromptDomain(domain)
            except ValueError:
                domain_enum = PromptDomain.CUSTOM
        else:
            domain_enum = domain

        # Create or update prompt - use original domain_str for cache/storage
        cache_key = self._get_cache_key(domain_str, name)
        existing = self._cache.get(cache_key)

        if existing:
            # Add new version
            existing.add_version(
                content=content,
                version=version,
                status=status,
                metadata=metadata
                or PromptMetadata(
                    domain=domain_enum,
                    version=version,
                    parent_version=existing.current_version,
                    tags=tags or [],
                ),
            )
            if status == PromptStatus.ACTIVE:
                existing.activate_version(version)
            prompt = existing
        else:
            # Create new prompt
            prompt = Prompt(
                name=name,
                domain=domain_enum,
                description=description,
                current_version=version,
                variables=self._extract_variables(content),
            )
            prompt.versions[version] = PromptVersion(
                version=version,
                content=content,
                status=status,
                metadata=metadata
                or PromptMetadata(domain=domain_enum, version=version, tags=tags or []),
            )

        # Generate embedding
        embedding = await self._get_embedding(content)

        # Store in Qdrant - use domain_str for storage to preserve original domain
        point_id = str(uuid4())
        payload = {
            "record_type": "prompt",
            "domain": domain_str,
            "name": name,
            "version": version,
            "status": status.value,
            "description": description,
            "content": content,
            "variables": prompt.variables,
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
            "prompt_data": prompt.to_dict(),
        }

        await self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )

        # Update cache
        self._cache[cache_key] = prompt

        return prompt

    async def get_prompt(
        self,
        domain: str,
        name: str,
        version: str | None = None,
    ) -> Prompt | None:
        """Get a prompt by domain and name.

        Args:
            domain: Prompt domain
            name: Prompt name
            version: Specific version (None = current)

        Returns:
            Prompt if found, None otherwise
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Check cache first
        cache_key = self._get_cache_key(domain, name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query Qdrant
        query_filter = Filter(
            must=[
                FieldCondition(key="record_type", match=MatchValue(value="prompt")),
                FieldCondition(key="domain", match=MatchValue(value=domain)),
                FieldCondition(key="name", match=MatchValue(value=name)),
            ]
        )

        results = await self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=100,  # Get all versions
            with_payload=True,
        )

        if not results[0]:
            return None

        # Reconstruct prompt from versions
        prompt = None
        for point in results[0]:
            payload = point.payload
            if not prompt:
                prompt = Prompt(
                    name=name,
                    domain=PromptDomain(domain),
                    description=payload.get("description", ""),
                    variables=payload.get("variables", []),
                )

            v = payload.get("version", "1.0.0")
            prompt.versions[v] = PromptVersion(
                version=v,
                content=payload.get("content", ""),
                status=PromptStatus(payload.get("status", "active")),
                metadata=PromptMetadata(
                    domain=PromptDomain(domain),
                    version=v,
                    tags=payload.get("tags", []),
                ),
            )

            # Track current version (active one)
            if payload.get("status") == "active":
                prompt.current_version = v

        if prompt:
            self._cache[cache_key] = prompt

        return prompt

    async def list_prompts(
        self,
        domain: str | None = None,
        status: PromptStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[Prompt]:
        """List prompts with optional filtering.

        Args:
            domain: Filter by domain
            status: Filter by status
            tags: Filter by tags

        Returns:
            List of matching prompts
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Build filter
        conditions = [
            FieldCondition(key="record_type", match=MatchValue(value="prompt"))
        ]

        if domain:
            conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

        if status:
            conditions.append(FieldCondition(key="status", match=MatchValue(value=status.value)))

        query_filter = Filter(must=conditions)

        results = await self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1000,
            with_payload=True,
        )

        # Group by domain:name to deduplicate versions
        prompts_map: dict[str, Prompt] = {}

        for point in results[0]:
            payload = point.payload
            key = self._get_cache_key(
                payload.get("domain", "custom"),
                payload.get("name", "unknown"),
            )

            if key not in prompts_map:
                prompts_map[key] = Prompt(
                    name=payload.get("name", "unknown"),
                    domain=PromptDomain(payload.get("domain", "custom")),
                    description=payload.get("description", ""),
                    variables=payload.get("variables", []),
                )

            prompt = prompts_map[key]
            v = payload.get("version", "1.0.0")
            prompt.versions[v] = PromptVersion(
                version=v,
                content=payload.get("content", ""),
                status=PromptStatus(payload.get("status", "active")),
            )

            if payload.get("status") == "active":
                prompt.current_version = v

        return list(prompts_map.values())

    async def search_prompts(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 5,
    ) -> list[tuple[Prompt, float]]:
        """Search prompts semantically.

        Args:
            query: Search query
            domain: Optional domain filter
            limit: Maximum results

        Returns:
            List of (prompt, score) tuples
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        embedding = await self._get_embedding(query)

        # Build filter
        conditions = [
            FieldCondition(key="record_type", match=MatchValue(value="prompt")),
            FieldCondition(key="status", match=MatchValue(value="active")),
        ]

        if domain:
            conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain)))

        query_filter = Filter(must=conditions)

        results = await self._client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        prompts_with_scores = []
        for point in results.points:
            payload = point.payload
            prompt = Prompt(
                name=payload.get("name", "unknown"),
                domain=PromptDomain(payload.get("domain", "custom")),
                description=payload.get("description", ""),
                variables=payload.get("variables", []),
            )
            v = payload.get("version", "1.0.0")
            prompt.versions[v] = PromptVersion(
                version=v,
                content=payload.get("content", ""),
                status=PromptStatus.ACTIVE,
            )
            prompt.current_version = v
            prompts_with_scores.append((prompt, point.score))

        return prompts_with_scores

    async def update_metrics(
        self,
        domain: str,
        name: str,
        version: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Update usage metrics for a prompt version.

        Args:
            domain: Prompt domain
            name: Prompt name
            version: Version string
            success: Whether the interaction was successful
            latency_ms: Response latency
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        # Find the point
        query_filter = Filter(
            must=[
                FieldCondition(key="record_type", match=MatchValue(value="prompt")),
                FieldCondition(key="domain", match=MatchValue(value=domain)),
                FieldCondition(key="name", match=MatchValue(value=name)),
                FieldCondition(key="version", match=MatchValue(value=version)),
            ]
        )

        results = await self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1,
            with_payload=True,
            with_vectors=True,
        )

        if not results[0]:
            return

        point = results[0][0]
        payload = dict(point.payload)

        # Update metrics
        usage_count = payload.get("usage_count", 0) + 1
        total_success = payload.get("total_success", 0) + (1 if success else 0)
        total_latency = payload.get("total_latency_ms", 0) + latency_ms

        payload["usage_count"] = usage_count
        payload["total_success"] = total_success
        payload["total_latency_ms"] = total_latency
        payload["success_rate"] = total_success / usage_count
        payload["avg_latency_ms"] = total_latency / usage_count

        # Update in Qdrant
        from qdrant_client.http.models import PointStruct

        await self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point.id, vector=point.vector, payload=payload)],
        )

    async def delete_prompt(
        self,
        domain: str,
        name: str,
        version: str | None = None,
    ) -> bool:
        """Delete a prompt or specific version.

        Args:
            domain: Prompt domain
            name: Prompt name
            version: Version to delete (None = all versions)

        Returns:
            True if deleted
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        conditions = [
            FieldCondition(key="record_type", match=MatchValue(value="prompt")),
            FieldCondition(key="domain", match=MatchValue(value=domain)),
            FieldCondition(key="name", match=MatchValue(value=name)),
        ]

        if version:
            conditions.append(FieldCondition(key="version", match=MatchValue(value=version)))

        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions),
        )

        # Clear cache
        cache_key = self._get_cache_key(domain, name)
        self._cache.pop(cache_key, None)

        return True

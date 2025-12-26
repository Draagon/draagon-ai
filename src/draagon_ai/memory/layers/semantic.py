"""Semantic Memory Layer - Structured factual knowledge.

Semantic memory stores:
- Entities (people, places, things)
- Relationships (connections between entities)
- Facts (declarative knowledge)
- Beliefs (agent's reconciled understanding)

Features:
- Entity resolution (merge duplicates, manage aliases)
- Confidence propagation (stated_count boost)
- Community detection (related entities cluster)
- Source tracking (episodic evidence)

Based on research from:
- Zep/Graphiti: Semantic entity extraction
- Mem0: Hybrid vector-graph for facts
- Knowledge graph best practices
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import logging

from ..temporal_nodes import TemporalNode, TemporalEdge, NodeType, EdgeType, MemoryLayer
from ..temporal_graph import TemporalCognitiveGraph, GraphSearchResult
from .base import MemoryLayerBase, LayerConfig

logger = logging.getLogger(__name__)


# Default TTL for semantic memories
DEFAULT_TTL = timedelta(days=90)


@dataclass
class Entity(TemporalNode):
    """An entity in semantic memory.

    Entities represent people, places, things, or concepts
    with their properties and aliases.
    """

    canonical_name: str = ""
    entity_type: str = ""  # person, place, thing, concept, organization
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    # Community detection
    community_id: str | None = None
    community_summary: str | None = None

    # Source tracking
    source_episode_ids: list[str] = field(default_factory=list)


@dataclass
class Fact(TemporalNode):
    """A fact in semantic memory.

    Facts are declarative statements with confidence tracking.
    """

    subject_entity_id: str | None = None
    predicate: str = ""
    object_value: str = ""

    # Verification
    verified: bool = False
    verification_source: str | None = None

    # Source tracking
    source_episode_ids: list[str] = field(default_factory=list)


@dataclass
class Relationship(TemporalNode):
    """A relationship between two entities.

    Relationships connect entities with typed edges.
    """

    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: str = ""  # knows, works_at, lives_in, etc.
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityMatch:
    """Result of entity resolution."""

    entity: Entity
    match_score: float
    match_type: str  # exact, alias, fuzzy


class SemanticMemory(MemoryLayerBase[Entity]):
    """Semantic Memory Layer - Factual knowledge storage.

    Key features:
    - Entity CRUD with resolution
    - Fact storage with confidence
    - Relationship management
    - Community detection
    - Source tracking from episodic layer

    Example:
        semantic = SemanticMemory(graph)

        # Create entity
        doug = await semantic.create_entity(
            name="Doug",
            entity_type="person",
            properties={"role": "user", "location": "Philadelphia"},
        )

        # Add fact about entity
        fact = await semantic.add_fact(
            content="Doug's birthday is March 15",
            subject_entity_id=doug.node_id,
            predicate="birthday",
            object_value="March 15",
        )

        # Create relationship
        rel = await semantic.add_relationship(
            source_id=doug.node_id,
            target_id=philly.node_id,
            relationship_type="lives_in",
        )

        # Resolve entity by name
        matches = await semantic.resolve_entity("Douglas")
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        ttl: timedelta = DEFAULT_TTL,
    ):
        """Initialize semantic memory.

        Args:
            graph: The underlying temporal cognitive graph
            ttl: Time-to-live for items (default 90 days)
        """
        config = LayerConfig(
            max_items=None,
            default_ttl=ttl,
            decay_factor=0.98,  # Slower decay for semantic
            decay_interval=timedelta(days=1),
            importance_threshold=0.85,  # High bar for metacognitive
            access_threshold=10,
            auto_promote=True,
            node_types=[NodeType.ENTITY, NodeType.FACT, NodeType.RELATIONSHIP, NodeType.BELIEF],
        )
        super().__init__(graph, config, MemoryLayer.SEMANTIC)

        # Entity name index for fast resolution
        self._entity_index: dict[str, str] = {}  # lowercase name -> entity_id
        self._alias_index: dict[str, str] = {}  # lowercase alias -> entity_id

    async def add(
        self,
        content: str,
        *,
        node_type: NodeType = NodeType.FACT,
        scope_id: str = "agent:default",
        entities: list[str] | None = None,
        confidence: float = 1.0,
        source_episode_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalNode:
        """Add a semantic memory item.

        For specific item types, prefer:
        - create_entity() for entities
        - add_fact() for facts
        - add_relationship() for relationships

        Args:
            content: The content
            node_type: Type of node
            scope_id: Hierarchical scope
            entities: Extracted entities
            confidence: Confidence level
            source_episode_ids: Episodes this came from
            metadata: Additional metadata

        Returns:
            The created node
        """
        node = await self._graph.add_node(
            content=content,
            node_type=node_type,
            scope_id=scope_id,
            entities=entities,
            confidence=confidence,
            importance=0.7,  # Semantic facts are generally important
            metadata={
                **(metadata or {}),
                "source_episode_ids": source_episode_ids or [],
            },
        )
        return node

    async def get(self, node_id: str) -> TemporalNode | None:
        """Get a semantic item by ID.

        Args:
            node_id: The node ID

        Returns:
            The node or None
        """
        node = await self._graph.get_node(node_id)
        if not node or node.layer != MemoryLayer.SEMANTIC:
            return None
        return node

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        *,
        scope_id: str = "agent:default",
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """Create a new entity.

        Args:
            name: Canonical name
            entity_type: Type of entity
            scope_id: Hierarchical scope
            aliases: Alternative names
            properties: Entity properties
            source_episode_ids: Source episodes
            metadata: Additional metadata

        Returns:
            The created Entity
        """
        # Check for existing entity with same name
        existing = await self.resolve_entity(name, min_score=0.95)
        if existing:
            logger.warning(f"Entity '{name}' may already exist: {existing[0].entity.node_id}")

        node = await self._graph.add_node(
            content=f"{entity_type}: {name}",
            node_type=NodeType.ENTITY,
            scope_id=scope_id,
            entities=[name] + (aliases or []),
            confidence=1.0,
            importance=0.6,
            metadata={
                **(metadata or {}),
                "canonical_name": name,
                "entity_type": entity_type,
                "aliases": aliases or [],
                "properties": properties or {},
                "source_episode_ids": source_episode_ids or [],
            },
        )

        # Update indexes
        self._entity_index[name.lower()] = node.node_id
        for alias in (aliases or []):
            self._alias_index[alias.lower()] = node.node_id

        entity = Entity(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            canonical_name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            properties=properties or {},
            source_episode_ids=source_episode_ids or [],
        )

        logger.debug(f"Created entity: {name} ({entity_type})")
        return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: The entity node ID

        Returns:
            The Entity or None
        """
        node = await self._graph.get_node(entity_id)
        if not node or node.node_type != NodeType.ENTITY:
            return None

        return Entity(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            canonical_name=node.metadata.get("canonical_name", ""),
            entity_type=node.metadata.get("entity_type", ""),
            aliases=node.metadata.get("aliases", []),
            properties=node.metadata.get("properties", {}),
            community_id=node.metadata.get("community_id"),
            community_summary=node.metadata.get("community_summary"),
            source_episode_ids=node.metadata.get("source_episode_ids", []),
        )

    async def resolve_entity(
        self,
        name: str,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> list[EntityMatch]:
        """Resolve an entity name to existing entities.

        Attempts:
        1. Exact match on canonical name
        2. Exact match on aliases
        3. Fuzzy/semantic match

        Args:
            name: Name to resolve
            min_score: Minimum match score
            limit: Maximum matches

        Returns:
            List of potential matches with scores
        """
        matches = []
        name_lower = name.lower()

        # Check exact match in canonical names
        if name_lower in self._entity_index:
            entity_id = self._entity_index[name_lower]
            entity = await self.get_entity(entity_id)
            if entity:
                matches.append(EntityMatch(
                    entity=entity,
                    match_score=1.0,
                    match_type="exact",
                ))
                return matches  # Exact match, no need to continue

        # Check alias index
        if name_lower in self._alias_index:
            entity_id = self._alias_index[name_lower]
            entity = await self.get_entity(entity_id)
            if entity:
                matches.append(EntityMatch(
                    entity=entity,
                    match_score=0.95,
                    match_type="alias",
                ))
                return matches

        # Semantic search for fuzzy matches
        results = await self._graph.search(
            query=name,
            node_types=[NodeType.ENTITY],
            layers=[MemoryLayer.SEMANTIC],
            limit=limit,
            min_score=min_score,
        )

        for result in results:
            entity = await self.get_entity(result.node.node_id)
            if entity and entity.node_id not in [m.entity.node_id for m in matches]:
                matches.append(EntityMatch(
                    entity=entity,
                    match_score=result.score,
                    match_type="fuzzy",
                ))

        return matches

    async def merge_entities(
        self,
        primary_id: str,
        secondary_id: str,
    ) -> Entity | None:
        """Merge two entities into one.

        The primary entity absorbs the secondary's:
        - Aliases
        - Properties (non-conflicting)
        - Source episodes
        - Relationships

        Args:
            primary_id: Entity to keep
            secondary_id: Entity to merge and remove

        Returns:
            The merged Entity or None if either not found
        """
        primary = await self.get_entity(primary_id)
        secondary = await self.get_entity(secondary_id)

        if not primary or not secondary:
            return None

        # Merge aliases
        all_aliases = set(primary.aliases)
        all_aliases.add(secondary.canonical_name)
        all_aliases.update(secondary.aliases)
        all_aliases.discard(primary.canonical_name)

        # Merge properties (primary takes precedence)
        merged_properties = {**secondary.properties, **primary.properties}

        # Merge source episodes
        all_sources = set(primary.source_episode_ids)
        all_sources.update(secondary.source_episode_ids)

        # Update primary
        primary_node = await self._graph.get_node(primary_id)
        if primary_node:
            primary_node.metadata["aliases"] = list(all_aliases)
            primary_node.metadata["properties"] = merged_properties
            primary_node.metadata["source_episode_ids"] = list(all_sources)
            primary_node.entities = [primary.canonical_name] + list(all_aliases)
            primary_node.updated_at = datetime.now()

            # Update indexes
            for alias in all_aliases:
                self._alias_index[alias.lower()] = primary_id

        # Re-point relationships from secondary to primary
        for edge_id, edge in list(self._graph._edges.items()):
            if edge.source_id == secondary_id:
                edge.source_id = primary_id
            if edge.target_id == secondary_id:
                edge.target_id = primary_id

        # Delete secondary
        await self._graph.delete_node(secondary_id)
        self._entity_index.pop(secondary.canonical_name.lower(), None)
        for alias in secondary.aliases:
            if self._alias_index.get(alias.lower()) == secondary_id:
                self._alias_index.pop(alias.lower(), None)

        logger.info(f"Merged entity {secondary.canonical_name} into {primary.canonical_name}")
        return await self.get_entity(primary_id)

    async def add_alias(self, entity_id: str, alias: str) -> bool:
        """Add an alias to an entity.

        Args:
            entity_id: The entity
            alias: New alias

        Returns:
            True if added
        """
        node = await self._graph.get_node(entity_id)
        if not node or node.node_type != NodeType.ENTITY:
            return False

        aliases = node.metadata.get("aliases", [])
        if alias not in aliases:
            aliases.append(alias)
            node.metadata["aliases"] = aliases
            node.entities = [node.metadata.get("canonical_name", "")] + aliases
            self._alias_index[alias.lower()] = entity_id
            node.updated_at = datetime.now()

        return True

    # =========================================================================
    # Fact Operations
    # =========================================================================

    async def add_fact(
        self,
        content: str,
        *,
        scope_id: str = "agent:default",
        subject_entity_id: str | None = None,
        predicate: str = "",
        object_value: str = "",
        confidence: float = 1.0,
        source_episode_ids: list[str] | None = None,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Add a fact to semantic memory.

        Args:
            content: The fact statement
            scope_id: Hierarchical scope
            subject_entity_id: Related entity
            predicate: Relationship predicate
            object_value: Object of the statement
            confidence: Confidence level
            source_episode_ids: Source episodes
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The created Fact
        """
        # Check for existing similar facts (deduplication)
        existing = await self._graph.search(
            query=content,
            node_types=[NodeType.FACT],
            layers=[MemoryLayer.SEMANTIC],
            min_score=0.95,
            limit=1,
        )

        if existing:
            # Reinforce existing fact
            existing_node = existing[0].node
            existing_node.restate()
            logger.debug(f"Restated existing fact: {existing_node.node_id[:8]}...")

            return Fact(
                node_id=existing_node.node_id,
                content=existing_node.content,
                node_type=existing_node.node_type,
                scope_id=existing_node.scope_id,
                embedding=existing_node.embedding,
                event_time=existing_node.event_time,
                ingestion_time=existing_node.ingestion_time,
                valid_from=existing_node.valid_from,
                valid_until=existing_node.valid_until,
                confidence=existing_node.confidence,
                importance=existing_node.importance,
                stated_count=existing_node.stated_count,
                access_count=existing_node.access_count,
                entities=existing_node.entities,
                metadata=existing_node.metadata,
                created_at=existing_node.created_at,
                updated_at=existing_node.updated_at,
                subject_entity_id=existing_node.metadata.get("subject_entity_id"),
                predicate=existing_node.metadata.get("predicate", ""),
                object_value=existing_node.metadata.get("object_value", ""),
                source_episode_ids=existing_node.metadata.get("source_episode_ids", []),
            )

        # Create new fact
        node = await self._graph.add_node(
            content=content,
            node_type=NodeType.FACT,
            scope_id=scope_id,
            entities=entities,
            confidence=confidence,
            importance=0.7,
            metadata={
                **(metadata or {}),
                "subject_entity_id": subject_entity_id,
                "predicate": predicate,
                "object_value": object_value,
                "source_episode_ids": source_episode_ids or [],
            },
        )

        # Link to subject entity
        if subject_entity_id:
            await self._graph.add_edge(
                source_id=subject_entity_id,
                target_id=node.node_id,
                edge_type=EdgeType.HAS,
                label="has_fact",
            )

        fact = Fact(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            subject_entity_id=subject_entity_id,
            predicate=predicate,
            object_value=object_value,
            source_episode_ids=source_episode_ids or [],
        )

        logger.debug(f"Added fact: {content[:50]}...")
        return fact

    async def get_facts_about(
        self,
        entity_id: str,
        predicate: str | None = None,
        limit: int = 20,
    ) -> list[Fact]:
        """Get facts about an entity.

        Args:
            entity_id: The entity
            predicate: Optional filter by predicate
            limit: Maximum facts

        Returns:
            List of facts
        """
        facts = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.FACT:
                continue
            if node.metadata.get("subject_entity_id") != entity_id:
                continue
            if predicate and node.metadata.get("predicate") != predicate:
                continue

            fact = Fact(
                node_id=node.node_id,
                content=node.content,
                node_type=node.node_type,
                scope_id=node.scope_id,
                embedding=node.embedding,
                event_time=node.event_time,
                ingestion_time=node.ingestion_time,
                valid_from=node.valid_from,
                valid_until=node.valid_until,
                confidence=node.confidence,
                importance=node.importance,
                stated_count=node.stated_count,
                access_count=node.access_count,
                entities=node.entities,
                metadata=node.metadata,
                created_at=node.created_at,
                updated_at=node.updated_at,
                subject_entity_id=node.metadata.get("subject_entity_id"),
                predicate=node.metadata.get("predicate", ""),
                object_value=node.metadata.get("object_value", ""),
                source_episode_ids=node.metadata.get("source_episode_ids", []),
            )
            facts.append(fact)

        # Sort by importance
        facts.sort(key=lambda f: (f.importance, f.stated_count), reverse=True)
        return facts[:limit]

    async def supersede_fact(
        self,
        old_fact_id: str,
        new_content: str,
        **kwargs: Any,
    ) -> Fact | None:
        """Supersede an old fact with a new one.

        Args:
            old_fact_id: The fact to supersede
            new_content: New fact content
            **kwargs: Additional properties

        Returns:
            The new Fact or None
        """
        new_node = await self._graph.supersede_node(
            old_node_id=old_fact_id,
            new_content=new_content,
            **kwargs,
        )

        if not new_node:
            return None

        return Fact(
            node_id=new_node.node_id,
            content=new_node.content,
            node_type=new_node.node_type,
            scope_id=new_node.scope_id,
            embedding=new_node.embedding,
            event_time=new_node.event_time,
            ingestion_time=new_node.ingestion_time,
            valid_from=new_node.valid_from,
            valid_until=new_node.valid_until,
            confidence=new_node.confidence,
            importance=new_node.importance,
            stated_count=new_node.stated_count,
            access_count=new_node.access_count,
            entities=new_node.entities,
            metadata=new_node.metadata,
            created_at=new_node.created_at,
            updated_at=new_node.updated_at,
        )

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        *,
        scope_id: str = "agent:default",
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Relationship | None:
        """Add a relationship between entities.

        Args:
            source_entity_id: Source entity
            target_entity_id: Target entity
            relationship_type: Type of relationship
            scope_id: Hierarchical scope
            properties: Relationship properties
            confidence: Confidence level
            metadata: Additional metadata

        Returns:
            The created Relationship or None if entities not found
        """
        # Verify entities exist
        source = await self.get_entity(source_entity_id)
        target = await self.get_entity(target_entity_id)

        if not source or not target:
            logger.warning(f"Cannot create relationship: entities not found")
            return None

        # Create relationship node
        content = f"{source.canonical_name} {relationship_type} {target.canonical_name}"
        node = await self._graph.add_node(
            content=content,
            node_type=NodeType.RELATIONSHIP,
            scope_id=scope_id,
            entities=[source.canonical_name, target.canonical_name],
            confidence=confidence,
            importance=0.6,
            metadata={
                **(metadata or {}),
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "relationship_type": relationship_type,
                "properties": properties or {},
            },
        )

        # Create edge
        await self._graph.add_edge(
            source_id=source_entity_id,
            target_id=target_entity_id,
            edge_type=EdgeType.RELATED_TO,
            label=relationship_type,
            confidence=confidence,
        )

        rel = Relationship(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            properties=properties or {},
        )

        logger.debug(f"Added relationship: {content}")
        return rel

    async def get_relationships(
        self,
        entity_id: str,
        relationship_type: str | None = None,
        direction: str = "both",  # out, in, both
    ) -> list[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: The entity
            relationship_type: Optional type filter
            direction: Which direction to look

        Returns:
            List of relationships
        """
        relationships = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.RELATIONSHIP:
                continue

            source_id = node.metadata.get("source_entity_id")
            target_id = node.metadata.get("target_entity_id")
            rel_type = node.metadata.get("relationship_type")

            # Check direction
            matches = False
            if direction in ("out", "both") and source_id == entity_id:
                matches = True
            if direction in ("in", "both") and target_id == entity_id:
                matches = True

            if not matches:
                continue

            # Check type filter
            if relationship_type and rel_type != relationship_type:
                continue

            rel = Relationship(
                node_id=node.node_id,
                content=node.content,
                node_type=node.node_type,
                scope_id=node.scope_id,
                embedding=node.embedding,
                event_time=node.event_time,
                ingestion_time=node.ingestion_time,
                valid_from=node.valid_from,
                valid_until=node.valid_until,
                confidence=node.confidence,
                importance=node.importance,
                stated_count=node.stated_count,
                access_count=node.access_count,
                entities=node.entities,
                metadata=node.metadata,
                created_at=node.created_at,
                updated_at=node.updated_at,
                source_entity_id=source_id or "",
                target_entity_id=target_id or "",
                relationship_type=rel_type or "",
                properties=node.metadata.get("properties", {}),
            )
            relationships.append(rel)

        return relationships

    # =========================================================================
    # Community Detection (Simplified)
    # =========================================================================

    async def assign_community(
        self,
        entity_id: str,
        community_id: str,
        community_summary: str | None = None,
    ) -> bool:
        """Assign an entity to a community.

        Args:
            entity_id: The entity
            community_id: Community identifier
            community_summary: Optional summary

        Returns:
            True if assigned
        """
        node = await self._graph.get_node(entity_id)
        if not node or node.node_type != NodeType.ENTITY:
            return False

        node.metadata["community_id"] = community_id
        if community_summary:
            node.metadata["community_summary"] = community_summary
        node.updated_at = datetime.now()

        return True

    async def get_community_members(
        self,
        community_id: str,
    ) -> list[Entity]:
        """Get all entities in a community.

        Args:
            community_id: The community

        Returns:
            List of entities
        """
        members = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.ENTITY:
                continue
            if node.metadata.get("community_id") != community_id:
                continue

            entity = await self.get_entity(node_id)
            if entity:
                members.append(entity)

        return members

    def stats(self) -> dict[str, Any]:
        """Get semantic memory statistics.

        Returns:
            Dictionary with stats
        """
        entity_count = 0
        fact_count = 0
        relationship_count = 0
        communities = set()

        for node in self._graph._nodes.values():
            if node.layer != MemoryLayer.SEMANTIC:
                continue

            if node.node_type == NodeType.ENTITY:
                entity_count += 1
                if node.metadata.get("community_id"):
                    communities.add(node.metadata["community_id"])
            elif node.node_type == NodeType.FACT:
                fact_count += 1
            elif node.node_type == NodeType.RELATIONSHIP:
                relationship_count += 1

        return {
            "entity_count": entity_count,
            "fact_count": fact_count,
            "relationship_count": relationship_count,
            "community_count": len(communities),
            "indexed_names": len(self._entity_index),
            "indexed_aliases": len(self._alias_index),
        }

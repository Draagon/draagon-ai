"""Memory Integration - Connects decomposition output to SemanticMemory.

This module provides the bridge between the decomposition pipeline
and the 4-layer cognitive memory system.

The flow is:
1. Natural language input
2. Decomposition → DecompositionResult
3. Integration → Entities, Facts, Relationships stored in SemanticMemory
4. Presuppositions, inferences → Optional storage based on confidence

Based on prototype work in prototypes/implicit_knowledge_graphs/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .types import (
    DecompositionResult,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
)
from .service import DecompositionService, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for memory integration."""

    # What to store
    store_entities: bool = True
    store_facts: bool = True
    store_relationships: bool = True
    store_presuppositions: bool = False  # Only high-confidence
    store_commonsense: bool = False  # Only high-confidence

    # Thresholds for optional storage
    presupposition_threshold: float = 0.9
    commonsense_threshold: float = 0.8

    # Entity resolution
    resolve_existing_entities: bool = True
    merge_duplicate_entities: bool = True
    entity_match_threshold: float = 0.85


@dataclass
class IntegrationResult:
    """Result of integrating decomposition with memory."""

    # Stored items
    entity_ids: list[str]
    fact_ids: list[str]
    relationship_ids: list[str]

    # Mapping from decomposition to memory
    entity_mapping: dict[str, str]  # canonical_name -> entity_id

    # Statistics
    entities_created: int = 0
    entities_merged: int = 0
    facts_created: int = 0
    facts_reinforced: int = 0
    relationships_created: int = 0


class MemoryIntegration:
    """Integrates decomposition results with SemanticMemory.

    This class handles:
    1. Entity resolution (matching decomposed entities to existing ones)
    2. Entity creation (for new entities)
    3. Fact storage (with entity linking)
    4. Relationship creation (between entities)

    Example:
        from draagon_ai.memory.layers.semantic import SemanticMemory
        from draagon_ai.cognition.decomposition import DecompositionService

        # Create services
        decomposer = DecompositionService(llm=my_llm)
        semantic = SemanticMemory(graph)
        integrator = MemoryIntegration(semantic)

        # Process and store
        result = await decomposer.decompose("Doug has 6 cats.")
        integration = await integrator.integrate(result)

        print(f"Created {integration.entities_created} entities")
        print(f"Created {integration.facts_created} facts")
    """

    def __init__(
        self,
        semantic_memory: Any,  # SemanticMemory, using Any to avoid circular import
        config: IntegrationConfig | None = None,
    ):
        """Initialize the integration.

        Args:
            semantic_memory: The SemanticMemory layer to store results in
            config: Configuration options
        """
        self._semantic = semantic_memory
        self._config = config or IntegrationConfig()

    async def integrate(
        self,
        result: DecompositionResult,
        scope_id: str = "agent:default",
        source_episode_ids: list[str] | None = None,
    ) -> IntegrationResult:
        """Integrate decomposition results into semantic memory.

        Args:
            result: The decomposition result to integrate
            scope_id: Hierarchical scope for stored items
            source_episode_ids: Source episode IDs for provenance

        Returns:
            IntegrationResult with stored item IDs and statistics
        """
        integration = IntegrationResult(
            entity_ids=[],
            fact_ids=[],
            relationship_ids=[],
            entity_mapping={},
        )

        # Step 1: Process entities
        if self._config.store_entities:
            await self._process_entities(
                result.entities, scope_id, source_episode_ids, integration
            )

        # Step 2: Process facts (after entities so we can link)
        if self._config.store_facts:
            await self._process_facts(
                result.facts, scope_id, source_episode_ids, integration
            )

        # Step 3: Process relationships
        if self._config.store_relationships:
            await self._process_relationships(
                result.relationships, scope_id, integration
            )

        # Step 4: Optionally store presuppositions as facts
        if self._config.store_presuppositions:
            await self._process_presuppositions(
                result.presuppositions, scope_id, source_episode_ids, integration
            )

        return integration

    async def _process_entities(
        self,
        entities: list[ExtractedEntity],
        scope_id: str,
        source_episode_ids: list[str] | None,
        integration: IntegrationResult,
    ) -> None:
        """Process and store entities."""
        for entity in entities:
            # Skip anaphora that haven't been resolved
            if entity.entity_type == EntityType.ANAPHORA and not entity.resolved_to:
                continue

            # Try to resolve to existing entity
            if self._config.resolve_existing_entities:
                matches = await self._semantic.resolve_entity(
                    entity.canonical_name,
                    min_score=self._config.entity_match_threshold,
                )

                if matches:
                    # Found existing entity
                    existing = matches[0].entity
                    integration.entity_mapping[entity.canonical_name] = existing.node_id
                    integration.entity_ids.append(existing.node_id)

                    # Maybe add alias
                    if entity.text.lower() != entity.canonical_name.lower():
                        await self._semantic.add_alias(existing.node_id, entity.text)

                    logger.debug(
                        f"Resolved '{entity.canonical_name}' to existing entity"
                    )
                    continue

            # Create new entity
            new_entity = await self._semantic.create_entity(
                name=entity.canonical_name,
                entity_type=self._entity_type_to_string(entity.entity_type),
                scope_id=scope_id,
                aliases=[entity.text] if entity.text != entity.canonical_name else None,
                properties={
                    "synset_id": entity.synset_id,
                    "wikidata_qid": entity.wikidata_qid,
                    "definition": entity.definition,
                    **entity.properties,
                },
                source_episode_ids=source_episode_ids,
            )

            integration.entity_mapping[entity.canonical_name] = new_entity.node_id
            integration.entity_ids.append(new_entity.node_id)
            integration.entities_created += 1

            logger.debug(f"Created entity: {entity.canonical_name}")

    async def _process_facts(
        self,
        facts: list[ExtractedFact],
        scope_id: str,
        source_episode_ids: list[str] | None,
        integration: IntegrationResult,
    ) -> None:
        """Process and store facts."""
        for fact in facts:
            # Find subject entity ID
            subject_entity_id = integration.entity_mapping.get(fact.subject_text)

            stored_fact = await self._semantic.add_fact(
                content=fact.content,
                scope_id=scope_id,
                subject_entity_id=subject_entity_id,
                predicate=fact.predicate,
                object_value=fact.object_value,
                confidence=fact.confidence,
                source_episode_ids=source_episode_ids,
                metadata={
                    "source_text": fact.source_text,
                    "temporal_qualifier": fact.temporal_qualifier,
                },
            )

            integration.fact_ids.append(stored_fact.node_id)

            # Check if this was a new fact or reinforced existing
            if stored_fact.stated_count > 1:
                integration.facts_reinforced += 1
            else:
                integration.facts_created += 1

            logger.debug(f"Stored fact: {fact.content[:50]}...")

    async def _process_relationships(
        self,
        relationships: list[ExtractedRelationship],
        scope_id: str,
        integration: IntegrationResult,
    ) -> None:
        """Process and store relationships."""
        for rel in relationships:
            # Get entity IDs
            source_id = integration.entity_mapping.get(rel.source_text)
            target_id = integration.entity_mapping.get(rel.target_text)

            if not source_id or not target_id:
                logger.warning(
                    f"Skipping relationship: entities not found "
                    f"({rel.source_text} -> {rel.target_text})"
                )
                continue

            stored_rel = await self._semantic.add_relationship(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relationship_type=rel.relationship_type,
                scope_id=scope_id,
                properties=rel.properties,
                confidence=rel.confidence,
            )

            if stored_rel:
                integration.relationship_ids.append(stored_rel.node_id)
                integration.relationships_created += 1

                logger.debug(
                    f"Created relationship: {rel.source_text} "
                    f"-[{rel.relationship_type}]-> {rel.target_text}"
                )

    async def _process_presuppositions(
        self,
        presuppositions: list,  # list[Presupposition]
        scope_id: str,
        source_episode_ids: list[str] | None,
        integration: IntegrationResult,
    ) -> None:
        """Process and store high-confidence presuppositions as facts."""
        for presup in presuppositions:
            if presup.confidence < self._config.presupposition_threshold:
                continue

            # Store presupposition as a fact with special metadata
            stored = await self._semantic.add_fact(
                content=presup.content,
                scope_id=scope_id,
                confidence=presup.confidence,
                source_episode_ids=source_episode_ids,
                metadata={
                    "source_type": "presupposition",
                    "trigger_type": presup.trigger_type.value,
                    "trigger_text": presup.trigger_text,
                    "cancellable": presup.cancellable,
                },
            )

            integration.fact_ids.append(stored.node_id)
            integration.facts_created += 1

    def _entity_type_to_string(self, entity_type: EntityType) -> str:
        """Convert EntityType enum to semantic memory entity_type string."""
        mapping = {
            EntityType.INSTANCE: "person",  # Most instances are people/orgs
            EntityType.CLASS: "concept",
            EntityType.NAMED_CONCEPT: "concept",
            EntityType.ROLE: "role",
            EntityType.ANAPHORA: "reference",
            EntityType.GENERIC: "generic",
        }
        return mapping.get(entity_type, "thing")


class DecompositionMemoryService:
    """High-level service combining decomposition and memory storage.

    This is the main entry point for processing natural language
    and storing the results in semantic memory.

    Example:
        service = DecompositionMemoryService(
            llm=my_llm,
            semantic_memory=my_semantic,
        )

        result = await service.process_and_store(
            "Doug forgot his keys again.",
            scope_id="user:assistant:doug",
        )

        print(f"Stored {result.entities_created} entities")
        print(f"Stored {result.facts_created} facts")
    """

    def __init__(
        self,
        llm: LLMProvider,
        semantic_memory: Any,
        decomposition_config: Any = None,
        integration_config: IntegrationConfig | None = None,
    ):
        """Initialize the combined service.

        Args:
            llm: LLM provider for decomposition
            semantic_memory: SemanticMemory layer for storage
            decomposition_config: Optional decomposition config
            integration_config: Optional integration config
        """
        self._decomposer = DecompositionService(llm=llm, config=decomposition_config)
        self._integrator = MemoryIntegration(
            semantic_memory=semantic_memory,
            config=integration_config,
        )
        self._semantic = semantic_memory

    async def process_and_store(
        self,
        text: str,
        scope_id: str = "agent:default",
        source_episode_ids: list[str] | None = None,
        context: str | None = None,
    ) -> IntegrationResult:
        """Process natural language and store results in memory.

        Args:
            text: The natural language input
            scope_id: Hierarchical scope for stored items
            source_episode_ids: Source episode IDs for provenance
            context: Optional context for disambiguation

        Returns:
            IntegrationResult with stored item IDs and statistics
        """
        # Step 1: Decompose
        decomposition = await self._decomposer.decompose(text, context=context)

        # Step 2: Integrate with memory
        result = await self._integrator.integrate(
            decomposition,
            scope_id=scope_id,
            source_episode_ids=source_episode_ids,
        )

        logger.info(
            f"Processed '{text[:50]}...': "
            f"{result.entities_created} entities, "
            f"{result.facts_created} facts, "
            f"{result.relationships_created} relationships"
        )

        return result

    async def decompose_only(
        self,
        text: str,
        context: str | None = None,
    ) -> DecompositionResult:
        """Decompose without storing (for inspection/debugging).

        Args:
            text: The natural language input
            context: Optional context for disambiguation

        Returns:
            DecompositionResult without storage
        """
        return await self._decomposer.decompose(text, context=context)

"""Decomposition Service - Integrates decomposition with memory.

This service:
1. Decomposes natural language into structured knowledge
2. Stores extracted entities, facts, and relationships in SemanticMemory
3. Uses memory context to improve disambiguation

Based on prototype work in prototypes/implicit_knowledge_graphs/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .types import (
    CommonsenseInference,
    CommonsenseRelation,
    DecompositionResult,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    ExtractedRelationship,
    InterpretationBranch,
    ModalityInfo,
    ModalType,
    NegationInfo,
    Polarity,
    Presupposition,
    SemanticRole,
    TemporalInfo,
    Tense,
    Aspect,
    TriggerType,
)

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict],
        model_tier: str = "standard",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Send a chat request to the LLM."""
        ...


class MemoryProvider(Protocol):
    """Protocol for memory providers (optional context enhancement)."""

    async def search(self, query: str, limit: int = 5) -> list[Any]:
        """Search memory for context."""
        ...


@dataclass
class DecompositionConfig:
    """Configuration for decomposition service."""

    # Feature toggles
    extract_entities: bool = True
    extract_facts: bool = True
    extract_relationships: bool = True
    extract_semantic_roles: bool = True
    extract_presuppositions: bool = True
    extract_commonsense: bool = True
    extract_temporal: bool = True
    extract_modality: bool = True

    # Thresholds
    min_confidence: float = 0.5
    min_branch_weight: float = 0.1

    # Performance
    max_inferences: int = 20


class DecompositionService:
    """Service for decomposing natural language into structured knowledge.

    This service uses LLM prompts (following the LLM-first architecture)
    to extract semantic structure from natural language input.

    Example:
        service = DecompositionService(llm=my_llm)
        result = await service.decompose("Doug has 6 cats.")

        # Result contains:
        # - entities: [Doug (INSTANCE), cats (CLASS)]
        # - facts: [Doug has 6 cats]
        # - semantic_roles: [predicate=has, ARG0=Doug, ARG1=6 cats]
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        config: DecompositionConfig | None = None,
    ):
        """Initialize the decomposition service.

        Args:
            llm: LLM provider for semantic analysis
            memory: Optional memory provider for context enhancement
            config: Configuration options
        """
        self._llm = llm
        self._memory = memory
        self._config = config or DecompositionConfig()

    async def decompose(
        self,
        text: str,
        context: str | None = None,
    ) -> DecompositionResult:
        """Decompose natural language into structured knowledge.

        Args:
            text: The natural language input
            context: Optional context for disambiguation

        Returns:
            DecompositionResult with extracted knowledge
        """
        start_time = time.time()

        # Initialize result
        result = DecompositionResult(source_text=text)

        # Phase 0: Entity extraction and classification
        if self._config.extract_entities:
            result.entities = await self._extract_entities(text, context)

        # Phase 1a: Semantic role extraction
        if self._config.extract_semantic_roles:
            result.semantic_roles = await self._extract_semantic_roles(text)

        # Phase 1b: Fact and relationship extraction
        if self._config.extract_facts:
            result.facts = await self._extract_facts(text, result.entities)

        if self._config.extract_relationships:
            result.relationships = await self._extract_relationships(
                text, result.entities
            )

        # Phase 1c: Presupposition detection
        if self._config.extract_presuppositions:
            result.presuppositions = await self._extract_presuppositions(text)

        # Phase 1d: Commonsense inference
        if self._config.extract_commonsense:
            result.commonsense_inferences = await self._extract_commonsense(
                text, result.entities
            )

        # Phase 1e: Temporal/modal analysis
        if self._config.extract_temporal:
            result.temporal = await self._extract_temporal(text)

        if self._config.extract_modality:
            result.modality, result.negation = await self._extract_modality(text)

        # Create interpretation branches
        result.branches = await self._create_branches(result)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    async def _extract_entities(
        self,
        text: str,
        context: str | None,
    ) -> list[ExtractedEntity]:
        """Extract and classify entities from text."""
        prompt = f"""Analyze this text and extract all entities (people, places, things, concepts).

Text: "{text}"
{f'Context: {context}' if context else ''}

For each entity, classify as:
- INSTANCE: Specific named entity (Doug, Apple Inc., Paris)
- CLASS: Category of things (person, cat, bank)
- NAMED_CONCEPT: Named abstract concept (Christmas, Agile)
- ROLE: Relational role (CEO, wife, author)
- ANAPHORA: Pronoun needing resolution (he, she, it, they)
- GENERIC: Generic reference (someone, everyone)

Respond with XML:
<entities>
  <entity>
    <text>exact text from input</text>
    <canonical_name>normalized name</canonical_name>
    <entity_type>INSTANCE|CLASS|NAMED_CONCEPT|ROLE|ANAPHORA|GENERIC</entity_type>
    <confidence>0.0-1.0</confidence>
  </entity>
</entities>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_entities(response)

    async def _extract_semantic_roles(self, text: str) -> list[SemanticRole]:
        """Extract predicate-argument structures."""
        prompt = f"""Extract semantic roles (predicate-argument structures) from this text.

Text: "{text}"

For each predicate (verb), identify:
- ARG0: Agent/experiencer
- ARG1: Theme/patient
- ARG2: Destination/beneficiary
- ARGM-TMP: Temporal
- ARGM-LOC: Location
- ARGM-CAU: Cause
- ARGM-MNR: Manner

Respond with XML:
<roles>
  <role>
    <predicate>the verb</predicate>
    <role_type>ARG0|ARG1|ARG2|ARGM-*</role_type>
    <filler>the argument text</filler>
    <confidence>0.0-1.0</confidence>
  </role>
</roles>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_semantic_roles(response)

    async def _extract_facts(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedFact]:
        """Extract factual statements."""
        entity_list = ", ".join(e.canonical_name for e in entities)

        prompt = f"""Extract factual statements from this text.

Text: "{text}"
Entities found: {entity_list}

A fact is a statement that can be true or false.
Format as subject-predicate-object triples.

Respond with XML:
<facts>
  <fact>
    <subject>subject entity</subject>
    <predicate>relationship/property</predicate>
    <object>object value</object>
    <confidence>0.0-1.0</confidence>
  </fact>
</facts>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_facts(response, text)

    async def _extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        """Extract relationships between entities."""
        if len(entities) < 2:
            return []

        entity_list = ", ".join(e.canonical_name for e in entities)

        prompt = f"""Extract relationships between entities in this text.

Text: "{text}"
Entities: {entity_list}

Identify how entities are related to each other.

Respond with XML:
<relationships>
  <relationship>
    <source>source entity</source>
    <target>target entity</target>
    <type>relationship type (e.g., owns, works_at, lives_in)</type>
    <confidence>0.0-1.0</confidence>
  </relationship>
</relationships>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_relationships(response)

    async def _extract_presuppositions(self, text: str) -> list[Presupposition]:
        """Extract presuppositions from trigger words."""
        prompt = f"""Identify presuppositions in this text.

Text: "{text}"

Presuppositions are implicit assumptions that must be true for the sentence to make sense.

Common triggers:
- FACTIVE: "forgot", "knew", "regret" (presuppose complement is true)
- ITERATIVE: "again", "still", "anymore" (presuppose previous occurrence)
- CHANGE_OF_STATE: "stopped", "started", "began" (presuppose prior state)
- DEFINITE: "the X" (presupposes X exists)
- POSSESSIVE: "X's Y" (presupposes X has Y)

Respond with XML:
<presuppositions>
  <presupposition>
    <content>what is presupposed</content>
    <trigger_type>FACTIVE|ITERATIVE|CHANGE_OF_STATE|DEFINITE|POSSESSIVE</trigger_type>
    <trigger_text>the word that triggered it</trigger_text>
    <confidence>0.0-1.0</confidence>
  </presupposition>
</presuppositions>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_presuppositions(response)

    async def _extract_commonsense(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[CommonsenseInference]:
        """Extract commonsense inferences."""
        # Only generate rich inferences for INSTANCE entities
        has_instance = any(e.entity_type == EntityType.INSTANCE for e in entities)

        prompt = f"""Generate commonsense inferences from this text.

Text: "{text}"

Use ATOMIC-style relations:
- xIntent: Why the agent does this
- xNeed: What the agent needs to do this
- xEffect: Effect on the agent
- xWant: What the agent wants afterward
- xReact: Agent's emotional reaction
- oEffect: Effect on others
- oReact: Others' emotional reaction

{'Generate detailed inferences since specific people are mentioned.' if has_instance else 'Generate general inferences only.'}

Respond with XML:
<inferences>
  <inference>
    <relation>xIntent|xNeed|xEffect|xWant|xReact|oEffect|oReact</relation>
    <head>the event/state</head>
    <tail>what is inferred</tail>
    <confidence>0.0-1.0</confidence>
  </inference>
</inferences>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        inferences = self._parse_commonsense(response)
        return inferences[: self._config.max_inferences]

    async def _extract_temporal(self, text: str) -> TemporalInfo | None:
        """Extract temporal information."""
        prompt = f"""Analyze the temporal aspects of this text.

Text: "{text}"

Identify:
- Tense: past, present, future, past_perfect, present_perfect, future_perfect
- Aspect: simple, progressive, perfect, state, activity, achievement

Respond with XML:
<temporal>
  <tense>past|present|future|past_perfect|present_perfect|future_perfect</tense>
  <aspect>simple|progressive|perfect|state|activity</aspect>
  <reference_type>absolute|relative|deictic</reference_type>
  <reference_value>specific time reference if any</reference_value>
  <confidence>0.0-1.0</confidence>
</temporal>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_temporal(response)

    async def _extract_modality(
        self,
        text: str,
    ) -> tuple[ModalityInfo | None, NegationInfo | None]:
        """Extract modality and negation."""
        prompt = f"""Analyze modality and negation in this text.

Text: "{text}"

Modality types:
- none: No modal markers
- epistemic: might, may, could (possibility)
- deontic: must, should, ought (obligation)
- dynamic: can, able to (ability)
- hypothetical: would, if...then

Respond with XML:
<analysis>
  <modality>
    <type>none|epistemic|deontic|dynamic|hypothetical</type>
    <marker>the modal word if any</marker>
    <certainty>0.0-1.0</certainty>
  </modality>
  <negation>
    <is_negated>true|false</is_negated>
    <cue>negation word if any</cue>
    <polarity>positive|negative</polarity>
  </negation>
</analysis>"""

        response = await self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_modality(response)

    async def _create_branches(
        self,
        result: DecompositionResult,
    ) -> list[InterpretationBranch]:
        """Create interpretation branches for ambiguous inputs."""
        # For now, create a single primary branch
        # Full branching based on WSD alternatives will be added later

        branch = InterpretationBranch(
            interpretation=f"Primary interpretation of: {result.source_text[:50]}...",
            confidence=1.0,
            memory_support=0.0,
            final_weight=1.0,
            entity_interpretations={
                e.text: e.entity_type.value for e in result.entities
            },
            supporting_evidence=[
                f"{len(result.facts)} facts extracted",
                f"{len(result.semantic_roles)} semantic roles",
            ],
        )

        return [branch]

    # =========================================================================
    # Parsing Helpers
    # =========================================================================

    def _parse_entities(self, response: str) -> list[ExtractedEntity]:
        """Parse entity extraction response."""
        entities = []

        # Simple XML parsing (production would use proper XML parser)
        import re

        entity_pattern = r"<entity>(.*?)</entity>"
        for match in re.finditer(entity_pattern, response, re.DOTALL):
            entity_xml = match.group(1)

            text = self._extract_tag(entity_xml, "text") or ""
            canonical = self._extract_tag(entity_xml, "canonical_name") or text
            type_str = self._extract_tag(entity_xml, "entity_type") or "CLASS"
            conf_str = self._extract_tag(entity_xml, "confidence") or "0.8"

            try:
                entity_type = EntityType(type_str.lower())
            except ValueError:
                entity_type = EntityType.CLASS

            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.8

            if text and confidence >= self._config.min_confidence:
                entities.append(
                    ExtractedEntity(
                        text=text,
                        canonical_name=canonical,
                        entity_type=entity_type,
                        confidence=confidence,
                    )
                )

        return entities

    def _parse_semantic_roles(self, response: str) -> list[SemanticRole]:
        """Parse semantic role response."""
        roles = []
        import re

        role_pattern = r"<role>(.*?)</role>"
        for match in re.finditer(role_pattern, response, re.DOTALL):
            role_xml = match.group(1)

            predicate = self._extract_tag(role_xml, "predicate") or ""
            role_type = self._extract_tag(role_xml, "role_type") or "ARG0"
            filler = self._extract_tag(role_xml, "filler") or ""
            conf_str = self._extract_tag(role_xml, "confidence") or "0.8"

            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.8

            if predicate and filler and confidence >= self._config.min_confidence:
                roles.append(
                    SemanticRole(
                        predicate=predicate,
                        role=role_type,
                        filler=filler,
                        confidence=confidence,
                    )
                )

        return roles

    def _parse_facts(
        self, response: str, source_text: str
    ) -> list[ExtractedFact]:
        """Parse fact extraction response."""
        facts = []
        import re

        fact_pattern = r"<fact>(.*?)</fact>"
        for match in re.finditer(fact_pattern, response, re.DOTALL):
            fact_xml = match.group(1)

            subject = self._extract_tag(fact_xml, "subject") or ""
            predicate = self._extract_tag(fact_xml, "predicate") or ""
            obj = self._extract_tag(fact_xml, "object") or ""
            conf_str = self._extract_tag(fact_xml, "confidence") or "0.8"

            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.8

            if subject and predicate and confidence >= self._config.min_confidence:
                facts.append(
                    ExtractedFact(
                        content=f"{subject} {predicate} {obj}".strip(),
                        subject_text=subject,
                        predicate=predicate,
                        object_value=obj,
                        confidence=confidence,
                        source_text=source_text,
                    )
                )

        return facts

    def _parse_relationships(self, response: str) -> list[ExtractedRelationship]:
        """Parse relationship extraction response."""
        relationships = []
        import re

        rel_pattern = r"<relationship>(.*?)</relationship>"
        for match in re.finditer(rel_pattern, response, re.DOTALL):
            rel_xml = match.group(1)

            source = self._extract_tag(rel_xml, "source") or ""
            target = self._extract_tag(rel_xml, "target") or ""
            rel_type = self._extract_tag(rel_xml, "type") or ""
            conf_str = self._extract_tag(rel_xml, "confidence") or "0.8"

            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.8

            if source and target and rel_type and confidence >= self._config.min_confidence:
                relationships.append(
                    ExtractedRelationship(
                        source_text=source,
                        target_text=target,
                        relationship_type=rel_type,
                        confidence=confidence,
                    )
                )

        return relationships

    def _parse_presuppositions(self, response: str) -> list[Presupposition]:
        """Parse presupposition response."""
        presuppositions = []
        import re

        presup_pattern = r"<presupposition>(.*?)</presupposition>"
        for match in re.finditer(presup_pattern, response, re.DOTALL):
            presup_xml = match.group(1)

            content = self._extract_tag(presup_xml, "content") or ""
            type_str = self._extract_tag(presup_xml, "trigger_type") or "DEFINITE"
            trigger = self._extract_tag(presup_xml, "trigger_text") or ""
            conf_str = self._extract_tag(presup_xml, "confidence") or "0.8"

            try:
                trigger_type = TriggerType(type_str.lower())
            except ValueError:
                trigger_type = TriggerType.DEFINITE

            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.8

            if content and confidence >= self._config.min_confidence:
                presuppositions.append(
                    Presupposition(
                        content=content,
                        trigger_type=trigger_type,
                        trigger_text=trigger,
                        confidence=confidence,
                    )
                )

        return presuppositions

    def _parse_commonsense(self, response: str) -> list[CommonsenseInference]:
        """Parse commonsense inference response."""
        inferences = []
        import re

        inf_pattern = r"<inference>(.*?)</inference>"
        for match in re.finditer(inf_pattern, response, re.DOTALL):
            inf_xml = match.group(1)

            relation_str = self._extract_tag(inf_xml, "relation") or "xEffect"
            head = self._extract_tag(inf_xml, "head") or ""
            tail = self._extract_tag(inf_xml, "tail") or ""
            conf_str = self._extract_tag(inf_xml, "confidence") or "0.7"

            try:
                relation = CommonsenseRelation(relation_str)
            except ValueError:
                relation = CommonsenseRelation.X_EFFECT

            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = 0.7

            if head and tail and confidence >= self._config.min_confidence:
                inferences.append(
                    CommonsenseInference(
                        relation=relation,
                        head=head,
                        tail=tail,
                        confidence=confidence,
                    )
                )

        return inferences

    def _parse_temporal(self, response: str) -> TemporalInfo | None:
        """Parse temporal response."""
        import re

        temporal_match = re.search(r"<temporal>(.*?)</temporal>", response, re.DOTALL)
        if not temporal_match:
            return None

        temporal_xml = temporal_match.group(1)

        tense_str = self._extract_tag(temporal_xml, "tense") or "present"
        aspect_str = self._extract_tag(temporal_xml, "aspect") or "simple"
        ref_type = self._extract_tag(temporal_xml, "reference_type")
        ref_value = self._extract_tag(temporal_xml, "reference_value")
        conf_str = self._extract_tag(temporal_xml, "confidence") or "0.8"

        try:
            tense = Tense(tense_str.lower())
        except ValueError:
            tense = Tense.PRESENT

        try:
            aspect = Aspect(aspect_str.lower())
        except ValueError:
            aspect = Aspect.SIMPLE

        try:
            confidence = float(conf_str)
        except ValueError:
            confidence = 0.8

        return TemporalInfo(
            tense=tense,
            aspect=aspect,
            reference_type=ref_type,
            reference_value=ref_value,
            confidence=confidence,
        )

    def _parse_modality(
        self, response: str
    ) -> tuple[ModalityInfo | None, NegationInfo | None]:
        """Parse modality and negation response."""
        import re

        modality = None
        negation = None

        # Parse modality
        modality_match = re.search(r"<modality>(.*?)</modality>", response, re.DOTALL)
        if modality_match:
            mod_xml = modality_match.group(1)
            type_str = self._extract_tag(mod_xml, "type") or "none"
            marker = self._extract_tag(mod_xml, "marker")
            cert_str = self._extract_tag(mod_xml, "certainty") or "1.0"

            try:
                modal_type = ModalType(type_str.lower())
            except ValueError:
                modal_type = ModalType.NONE

            try:
                certainty = float(cert_str)
            except ValueError:
                certainty = 1.0

            modality = ModalityInfo(
                modal_type=modal_type,
                modal_marker=marker,
                certainty=certainty,
            )

        # Parse negation
        negation_match = re.search(r"<negation>(.*?)</negation>", response, re.DOTALL)
        if negation_match:
            neg_xml = negation_match.group(1)
            is_negated_str = self._extract_tag(neg_xml, "is_negated") or "false"
            cue = self._extract_tag(neg_xml, "cue")
            polarity_str = self._extract_tag(neg_xml, "polarity") or "positive"

            is_negated = is_negated_str.lower() == "true"

            try:
                polarity = Polarity(polarity_str.lower())
            except ValueError:
                polarity = Polarity.POSITIVE

            negation = NegationInfo(
                is_negated=is_negated,
                negation_cue=cue,
                polarity=polarity,
            )

        return modality, negation

    def _extract_tag(self, xml: str, tag: str) -> str | None:
        """Extract content from an XML tag."""
        import re

        match = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

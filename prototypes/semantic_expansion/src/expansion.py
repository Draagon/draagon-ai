"""Semantic Frame Expansion Service.

Provides deep semantic expansion of statements:
1. Extract semantic triples (subject-predicate-object)
2. Identify presuppositions (implicit assumptions)
3. Generate implications (likely consequences)
4. Resolve ambiguities and create variations
5. Score variations by cognitive factors

Based on research from:
- Frame Semantics (FrameNet)
- ATOMIC/COMET commonsense reasoning
- Presupposition extraction

Usage:
    >>> service = SemanticExpansionService(llm=my_llm)
    >>> variants = await service.expand("He prefers tea in the morning", inputs)
    >>> print(variants[0].frame.triples)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from semantic_types import (
    Ambiguity,
    ExpansionVariant,
    Implication,
    Presupposition,
    SemanticFrame,
    SemanticTriple,
    WordSense,
)
from wsd import WordSenseDisambiguator

if TYPE_CHECKING:
    pass


# =============================================================================
# Protocols
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Any:
        """Send chat messages and get response.

        Returns:
            Either a string directly, or a ChatResponse object with .content attribute.
        """
        ...


def _extract_llm_content(response: Any) -> str:
    """Extract string content from LLM response.

    Handles both direct string returns (from simple/mock providers)
    and ChatResponse objects (from real providers like GroqLLM).

    Args:
        response: The LLM response (str or object with .content)

    Returns:
        The response content as a string.
    """
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    # Fallback: try to convert to string
    return str(response)


# =============================================================================
# Expansion Input
# =============================================================================


@dataclass
class EntityInfo:
    """Information about an entity."""

    entity_type: str  # "PERSON", "PLACE", "THING", etc.
    aliases: list[str] = field(default_factory=list)
    synset_id: str | None = None


@dataclass
class ExpansionInput:
    """All inputs for generating semantic expansion variations.

    Collects context from multiple sources:
    - Immediate conversation context
    - Memory layers (working, episodic, semantic, metacognitive)
    - Belief system
    - Commonsense knowledge
    """

    # A. Contextual inputs
    immediate_context: list[str] = field(default_factory=list)  # Recent utterances
    session_entities: dict[str, EntityInfo] = field(default_factory=dict)  # Known entities
    active_topic: str | None = None
    current_time: datetime = field(default_factory=datetime.now)

    # B. Memory layer inputs (simplified representations)
    working_observations: list[dict[str, Any]] = field(default_factory=list)
    episodic_summaries: list[str] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)
    metacognitive_patterns: list[str] = field(default_factory=list)

    # C. Belief system inputs
    relevant_beliefs: list[str] = field(default_factory=list)
    entity_relationships: list[tuple[str, str, str]] = field(default_factory=list)  # (entity, rel, entity)
    known_preferences: list[str] = field(default_factory=list)

    # D. Commonsense inputs (from ATOMIC/ConceptNet)
    atomic_inferences: dict[str, list[str]] = field(default_factory=dict)  # relation_type -> inferences

    def get_context_text(self) -> str:
        """Get combined context as text for LLM prompt."""
        parts = []

        if self.immediate_context:
            parts.append("Recent conversation:\n" + "\n".join(f"- {c}" for c in self.immediate_context[-3:]))

        if self.session_entities:
            entity_strs = [f"{name} ({info.entity_type})" for name, info in self.session_entities.items()]
            parts.append("Known entities: " + ", ".join(entity_strs))

        if self.semantic_facts:
            parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in self.semantic_facts[:5]))

        if self.relevant_beliefs:
            parts.append("Relevant beliefs:\n" + "\n".join(f"- {b}" for b in self.relevant_beliefs[:5]))

        return "\n\n".join(parts) if parts else "No additional context available."


# =============================================================================
# Semantic Expansion Service
# =============================================================================


class SemanticExpansionService:
    """Expand statements into full semantic frames with variations.

    The core service that:
    1. Uses LLM to extract semantic frames from statements
    2. Identifies ambiguities and generates resolution options
    3. Creates multiple interpretation variants
    4. Scores variants by cognitive plausibility
    """

    # LLM prompt for semantic frame extraction
    EXTRACTION_PROMPT = """Analyze this statement and extract its full semantic meaning.

Statement: "{statement}"

Context:
{context}

Extract the following in XML format:

<semantic_frame>
    <triples>
        <triple>
            <subject>Entity performing action or having property</subject>
            <predicate>RELATION_TYPE (e.g., PREFERS, LIKES, IS_A, HAS, BELIEVES)</predicate>
            <object>Target entity or value</object>
            <context>Optional context like temporal="morning" or location="home"</context>
        </triple>
    </triples>

    <presuppositions>
        <presupposition type="existential|factive|lexical">
            What must be true for this statement to make sense
        </presupposition>
    </presuppositions>

    <implications>
        <implication type="pragmatic|logical|commonsense" confidence="0.0-1.0">
            What this statement implies or suggests
        </implication>
    </implications>

    <negations>
        <negation>What this statement explicitly rules out</negation>
    </negations>

    <ambiguities>
        <ambiguity type="reference|word_sense|scope|temporal">
            <text>The ambiguous part</text>
            <possibilities>
                <possibility>Option 1</possibility>
                <possibility>Option 2</possibility>
            </possibilities>
        </ambiguity>
    </ambiguities>

    <open_questions>
        <question>Information that would help clarify</question>
    </open_questions>

    <frame_type>ASSERTION|REQUEST|QUESTION|COMMAND</frame_type>
    <confidence>0.0-1.0</confidence>
</semantic_frame>
"""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        wsd: WordSenseDisambiguator | None = None,
    ):
        """Initialize the semantic expansion service.

        Args:
            llm: LLM provider for frame extraction
            wsd: Word sense disambiguator (optional)
        """
        self.llm = llm
        self.wsd = wsd or WordSenseDisambiguator()
        self.variation_generator = VariationGenerator()

    async def expand(
        self,
        statement: str,
        inputs: ExpansionInput | None = None,
        max_variations: int = 3,
    ) -> list[ExpansionVariant]:
        """Expand a statement into semantic frame variants.

        Args:
            statement: The statement to expand
            inputs: Contextual inputs for variation generation
            max_variations: Maximum number of variants to generate

        Returns:
            List of ExpansionVariants, sorted by combined score
        """
        inputs = inputs or ExpansionInput()

        # Step 1: Extract base semantic frame
        base_frame = await self._extract_frame(statement, inputs)

        # Step 2: Perform word sense disambiguation
        if self.wsd:
            word_senses = await self.wsd.disambiguate_sentence(statement)
            base_frame.word_senses = word_senses

        # Step 3: Generate variations for ambiguities
        variants = await self.variation_generator.generate_variations(
            base_frame, inputs, max_variations
        )

        return variants

    async def _extract_frame(
        self,
        statement: str,
        inputs: ExpansionInput,
    ) -> SemanticFrame:
        """Extract semantic frame using LLM.

        Args:
            statement: The statement to analyze
            inputs: Context for extraction

        Returns:
            SemanticFrame with extracted information
        """
        if not self.llm:
            # Return minimal frame without LLM
            return SemanticFrame(
                original_text=statement,
                triples=[],
                confidence=0.5,
                extraction_method="none",
            )

        # Build prompt
        context = inputs.get_context_text()
        prompt = self.EXTRACTION_PROMPT.format(
            statement=statement,
            context=context,
        )

        # Call LLM
        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Parse response (handle both str and ChatResponse)
        return self._parse_frame_response(_extract_llm_content(response), statement)

    def _parse_frame_response(
        self,
        response: str,
        original_text: str,
    ) -> SemanticFrame:
        """Parse LLM response into SemanticFrame."""
        frame = SemanticFrame(original_text=original_text)

        # Parse triples
        triple_matches = re.findall(
            r"<triple>\s*"
            r"<subject>([^<]+)</subject>\s*"
            r"<predicate>([^<]+)</predicate>\s*"
            r"<object>([^<]+)</object>\s*"
            r"(?:<context>([^<]*)</context>)?\s*"
            r"</triple>",
            response,
            re.DOTALL
        )

        for match in triple_matches:
            subject, predicate, obj, context_str = match
            context = {}
            if context_str:
                # Parse context like temporal="morning"
                ctx_matches = re.findall(r'(\w+)="([^"]+)"', context_str)
                context = dict(ctx_matches)

            frame.triples.append(SemanticTriple(
                subject=subject.strip(),
                predicate=predicate.strip(),
                object=obj.strip(),
                context=context,
            ))

        # Parse presuppositions
        presup_matches = re.findall(
            r'<presupposition\s+type="([^"]+)">\s*([^<]+)\s*</presupposition>',
            response,
            re.DOTALL
        )
        for ptype, content in presup_matches:
            frame.presuppositions.append(Presupposition(
                content=content.strip(),
                presupposition_type=ptype,
            ))

        # Parse implications
        impl_matches = re.findall(
            r'<implication\s+type="([^"]+)"\s+confidence="([^"]+)">\s*([^<]+)\s*</implication>',
            response,
            re.DOTALL
        )
        for itype, conf, content in impl_matches:
            frame.implications.append(Implication(
                content=content.strip(),
                implication_type=itype,
                confidence=float(conf) if conf else 0.8,
            ))

        # Parse negations
        negation_matches = re.findall(r"<negation>([^<]+)</negation>", response)
        frame.negations = [n.strip() for n in negation_matches]

        # Parse ambiguities
        amb_matches = re.findall(
            r'<ambiguity\s+type="([^"]+)">\s*'
            r"<text>([^<]+)</text>\s*"
            r"<possibilities>(.*?)</possibilities>\s*"
            r"</ambiguity>",
            response,
            re.DOTALL
        )
        for atype, text, poss_block in amb_matches:
            possibilities = re.findall(r"<possibility>([^<]+)</possibility>", poss_block)
            frame.ambiguities.append(Ambiguity(
                text=text.strip(),
                ambiguity_type=atype,
                possibilities=[p.strip() for p in possibilities],
            ))

        # Parse open questions
        question_matches = re.findall(r"<question>([^<]+)</question>", response)
        frame.open_questions = [q.strip() for q in question_matches]

        # Parse frame type
        frame_type_match = re.search(r"<frame_type>([^<]+)</frame_type>", response)
        if frame_type_match:
            frame.frame_type = frame_type_match.group(1).strip()

        # Parse confidence
        conf_match = re.search(r"<confidence>([0-9.]+)</confidence>", response)
        if conf_match:
            frame.confidence = float(conf_match.group(1))

        frame.extraction_method = "llm"
        return frame


# =============================================================================
# Variation Generator
# =============================================================================


class VariationGenerator:
    """Generate interpretation variants from semantic frames.

    Creates multiple interpretations based on:
    - Ambiguity resolutions
    - Context assumptions
    - Cognitive factor weighting
    """

    def __init__(self):
        """Initialize the variation generator."""
        pass

    async def generate_variations(
        self,
        base_frame: SemanticFrame,
        inputs: ExpansionInput,
        max_variations: int = 3,
    ) -> list[ExpansionVariant]:
        """Generate interpretation variants.

        Args:
            base_frame: The base semantic frame
            inputs: Contextual inputs for weighting
            max_variations: Maximum variants to generate

        Returns:
            List of ExpansionVariants, sorted by combined score
        """
        # If no ambiguities, return single variant
        if not base_frame.ambiguities:
            variant = self._create_variant(
                base_frame,
                resolution_choices={},
                inputs=inputs,
            )
            return [variant]

        # Generate variants for each ambiguity resolution combination
        variants = []

        # Generate primary variant (most likely resolutions)
        primary = await self._generate_primary_variant(base_frame, inputs)
        variants.append(primary)

        # Generate alternative variants
        for amb in base_frame.ambiguities:
            if len(variants) >= max_variations:
                break

            for i, poss in enumerate(amb.possibilities[1:], 1):  # Skip first (used in primary)
                if len(variants) >= max_variations:
                    break

                # Create variant with this alternative resolution
                choices = primary.resolution_choices.copy()
                choices[amb.text] = poss

                alt_frame = self._apply_resolution(base_frame, amb.text, poss)
                alt_variant = self._create_variant(
                    alt_frame,
                    resolution_choices=choices,
                    inputs=inputs,
                )
                variants.append(alt_variant)

        # Score all variants
        for variant in variants:
            self._compute_cognitive_scores(variant, inputs)

        # Sort by combined score
        variants.sort(key=lambda v: v.combined_score, reverse=True)

        return variants

    async def _generate_primary_variant(
        self,
        base_frame: SemanticFrame,
        inputs: ExpansionInput,
    ) -> ExpansionVariant:
        """Generate the primary (most likely) variant.

        Uses first resolution option for each ambiguity,
        which should be the most contextually likely.
        """
        resolution_choices = {}

        for amb in base_frame.ambiguities:
            if amb.possibilities:
                # Choose best resolution based on context
                best = self._choose_best_resolution(amb, inputs)
                resolution_choices[amb.text] = best
                amb.resolution = best

        return self._create_variant(
            base_frame,
            resolution_choices=resolution_choices,
            inputs=inputs,
        )

    def _choose_best_resolution(
        self,
        ambiguity: Ambiguity,
        inputs: ExpansionInput,
    ) -> str:
        """Choose the best resolution for an ambiguity based on context."""
        if not ambiguity.possibilities:
            return ""

        # Check if any possibility matches a known entity
        for poss in ambiguity.possibilities:
            if poss in inputs.session_entities:
                return poss

        # Check recent context for mentions
        context_text = " ".join(inputs.immediate_context).lower()
        for poss in ambiguity.possibilities:
            if poss.lower() in context_text:
                return poss

        # Default to first option
        return ambiguity.possibilities[0]

    def _apply_resolution(
        self,
        frame: SemanticFrame,
        ambiguous_text: str,
        resolution: str,
    ) -> SemanticFrame:
        """Apply a resolution to create a new frame variant.

        Replaces the ambiguous text with the resolution in all triples.
        """
        # Create new frame with resolved triples
        new_frame = SemanticFrame(
            original_text=frame.original_text,
            triples=[],
            presuppositions=frame.presuppositions.copy(),
            implications=frame.implications.copy(),
            negations=frame.negations.copy(),
            ambiguities=[a for a in frame.ambiguities if a.text != ambiguous_text],
            open_questions=frame.open_questions.copy(),
            word_senses=frame.word_senses.copy(),
            frame_type=frame.frame_type,
            confidence=frame.confidence * 0.9,  # Slightly lower for alternatives
            extraction_method=frame.extraction_method,
        )

        # Apply resolution to triples
        for triple in frame.triples:
            new_triple = SemanticTriple(
                subject=triple.subject.replace(ambiguous_text, resolution)
                if ambiguous_text.lower() in triple.subject.lower() else triple.subject,
                predicate=triple.predicate,
                object=triple.object.replace(ambiguous_text, resolution)
                if ambiguous_text.lower() in triple.object.lower() else triple.object,
                context=triple.context.copy(),
                confidence=triple.confidence,
            )
            new_frame.triples.append(new_triple)

        return new_frame

    def _create_variant(
        self,
        frame: SemanticFrame,
        resolution_choices: dict[str, str],
        inputs: ExpansionInput,
    ) -> ExpansionVariant:
        """Create an ExpansionVariant from a frame."""
        return ExpansionVariant(
            variant_id=str(uuid.uuid4()),
            frame=frame,
            resolution_choices=resolution_choices,
            context_assumptions=[
                f"Resolved '{amb}' as '{res}'"
                for amb, res in resolution_choices.items()
            ],
            base_confidence=frame.confidence,
        )

    def _compute_cognitive_scores(
        self,
        variant: ExpansionVariant,
        inputs: ExpansionInput,
    ) -> None:
        """Compute cognitive plausibility scores for a variant.

        Updates the variant's weight fields based on:
        - Recency: Recent context matches
        - Memory support: Memory layer evidence
        - Belief consistency: Alignment with beliefs
        - Commonsense: ATOMIC inference support
        """
        # Recency score: How well does this match recent context?
        variant.recency_weight = self._score_recency(variant, inputs)

        # Working memory score
        variant.working_memory_weight = self._score_working_memory(variant, inputs)

        # Episodic memory score
        variant.episodic_memory_weight = self._score_episodic_memory(variant, inputs)

        # Semantic memory score
        variant.semantic_memory_weight = self._score_semantic_memory(variant, inputs)

        # Belief consistency score
        variant.belief_weight = self._score_belief_consistency(variant, inputs)

        # Commonsense score
        variant.commonsense_weight = self._score_commonsense(variant, inputs)

        # Metacognitive score (calibration)
        variant.metacognitive_weight = self._score_metacognitive(variant, inputs)

    def _score_recency(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on recency of context mentions."""
        if not inputs.immediate_context:
            return 0.5

        # Check if variant entities appear in recent context
        entities = variant.frame.get_entities()
        recent_text = " ".join(inputs.immediate_context[-3:]).lower()

        matches = sum(1 for e in entities if e.lower() in recent_text)
        return min(1.0, 0.4 + (matches * 0.2))

    def _score_working_memory(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on working memory observations."""
        if not inputs.working_observations:
            return 0.5

        # Check for relevant observations
        entities = variant.frame.get_entities()
        matches = 0
        for obs in inputs.working_observations:
            content = obs.get("content", "").lower()
            for entity in entities:
                if entity.lower() in content:
                    matches += 1
                    break

        return min(1.0, 0.4 + (matches * 0.15))

    def _score_episodic_memory(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on episodic memory summaries."""
        if not inputs.episodic_summaries:
            return 0.5

        entities = variant.frame.get_entities()
        all_summaries = " ".join(inputs.episodic_summaries).lower()

        matches = sum(1 for e in entities if e.lower() in all_summaries)
        return min(1.0, 0.4 + (matches * 0.2))

    def _score_semantic_memory(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on semantic memory facts."""
        if not inputs.semantic_facts:
            return 0.5

        # Check if variant aligns with known facts
        entities = variant.frame.get_entities()
        all_facts = " ".join(inputs.semantic_facts).lower()

        # Check for entity mentions
        entity_matches = sum(1 for e in entities if e.lower() in all_facts)

        # Check for potential conflicts
        conflicts = 0
        for triple in variant.frame.triples:
            triple_text = triple.to_text().lower()
            # Look for negation patterns
            for fact in inputs.semantic_facts:
                if triple.subject.lower() in fact.lower():
                    # Check if fact contradicts triple
                    if "not" in fact.lower() or "doesn't" in fact.lower():
                        if triple.object.lower() in fact.lower():
                            conflicts += 1

        score = 0.4 + (entity_matches * 0.15) - (conflicts * 0.2)
        return max(0.0, min(1.0, score))

    def _score_belief_consistency(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on alignment with beliefs."""
        if not inputs.relevant_beliefs:
            return 0.5

        # Check for alignment vs conflict
        all_beliefs = " ".join(inputs.relevant_beliefs).lower()
        entities = variant.frame.get_entities()

        alignments = 0
        for entity in entities:
            if entity.lower() in all_beliefs:
                alignments += 1

        return min(1.0, 0.4 + (alignments * 0.15))

    def _score_commonsense(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on commonsense inference alignment."""
        if not inputs.atomic_inferences:
            return 0.5

        # Check if implications align with ATOMIC inferences
        total_inferences = sum(len(v) for v in inputs.atomic_inferences.values())
        if total_inferences == 0:
            return 0.5

        # Simple check: do any implications appear in atomic inferences?
        all_inferences = " ".join(
            inf for infs in inputs.atomic_inferences.values() for inf in infs
        ).lower()

        matches = 0
        for impl in variant.frame.implications:
            if any(word in all_inferences for word in impl.content.lower().split()):
                matches += 1

        return min(1.0, 0.4 + (matches * 0.15))

    def _score_metacognitive(self, variant: ExpansionVariant, inputs: ExpansionInput) -> float:
        """Score based on metacognitive calibration."""
        if not inputs.metacognitive_patterns:
            return 0.5

        # Check for patterns that might affect this interpretation
        all_patterns = " ".join(inputs.metacognitive_patterns).lower()

        # Look for warnings about interpretation types
        if "misinterpret" in all_patterns or "often wrong" in all_patterns:
            # Check if this variant type is mentioned
            for amb in variant.frame.ambiguities:
                if amb.ambiguity_type.lower() in all_patterns:
                    return 0.3  # Lower confidence for this type

        return 0.5  # Neutral by default


# =============================================================================
# Convenience Functions
# =============================================================================


async def expand_statement(
    statement: str,
    llm: LLMProvider | None = None,
    context: list[str] | None = None,
    entities: dict[str, str] | None = None,
) -> list[ExpansionVariant]:
    """Convenience function to expand a statement.

    Args:
        statement: The statement to expand
        llm: LLM provider (optional)
        context: Recent utterances (optional)
        entities: Known entities {name: type} (optional)

    Returns:
        List of ExpansionVariants
    """
    service = SemanticExpansionService(llm=llm)

    inputs = ExpansionInput(
        immediate_context=context or [],
        session_entities={
            name: EntityInfo(entity_type=etype)
            for name, etype in (entities or {}).items()
        },
    )

    return await service.expand(statement, inputs)

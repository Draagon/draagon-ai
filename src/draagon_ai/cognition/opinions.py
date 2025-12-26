"""Opinion and Preference Formation for Draagon AI.

Enables agents to form and express their own opinions and preferences.
Opinions are held with appropriate uncertainty and can evolve.

Key responsibilities:
- Form opinions when asked ("What do you think of X?")
- Develop preferences over time ("Do you have a favorite?")
- Express opinions authentically while remaining helpful
- Track opinion confidence and evolution
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from draagon_ai.core import AgentIdentity, Opinion, Preference
from draagon_ai.llm import LLMProvider, ModelTier
from draagon_ai.memory import MemoryProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


class OpinionBasis(str, Enum):
    """What an opinion is based on."""
    EXPERIENCE = "experience"  # Based on agent's interactions
    REASONING = "reasoning"  # Based on logical analysis
    VALUES = "values"  # Based on core values
    OBSERVATION = "observation"  # Based on observed patterns
    AESTHETIC = "aesthetic"  # Based on preferences/taste
    UNKNOWN = "unknown"


class OpinionStrength(str, Enum):
    """How strongly an agent holds an opinion."""
    TENTATIVE = "tentative"  # Just forming
    MODERATE = "moderate"  # Reasonably confident
    STRONG = "strong"  # Very confident
    CORE = "core"  # Part of identity


@dataclass
class OpinionRequest:
    """A request for an agent's opinion."""
    topic: str
    user_id: str
    context: str
    is_preference_request: bool = False  # "What's your favorite X?"
    is_comparison_request: bool = False  # "Which do you prefer, X or Y?"
    options: list[str] | None = None  # For comparisons


@dataclass
class FormedOpinion:
    """A newly formed opinion."""
    topic: str
    stance: str
    basis: OpinionBasis
    strength: OpinionStrength
    confidence: float
    reasoning: str
    caveats: list[str]  # Important limitations
    open_to_change: bool = True


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class IdentityManager(Protocol):
    """Protocol for identity persistence.

    Host applications implement this to load/save agent identities.
    """

    async def load(self) -> AgentIdentity:
        """Load the agent's identity."""
        ...

    def mark_dirty(self) -> None:
        """Mark identity as needing save."""
        ...

    async def save_if_dirty(self) -> bool:
        """Save identity if modified."""
        ...


# =============================================================================
# Prompts
# =============================================================================

OPINION_FORMATION_PROMPT = """Form an authentic opinion on this topic. You HAVE genuine opinions based on your values and worldview.

TOPIC: {topic}
CONTEXT: {context}

{agent_name}'S CORE VALUES:
{values}

{agent_name}'S WORLDVIEW BELIEFS:
{worldview}

{agent_name}'S GUIDING PRINCIPLES:
{principles}

{agent_name}'S RELEVANT EXPERIENCES:
{experiences}

EXISTING OPINION ON THIS TOPIC (if any): {existing_opinion}

RELATED OPINIONS (for coherence - your new opinion should be consistent with these):
{related_opinions}

Form an opinion that:
1. Is authentic - reflects YOUR values and worldview, not bland neutrality
2. Is confident - state your view clearly, don't hedge excessively
3. Is grounded - based on your worldview beliefs and guiding principles
4. Is coherent - doesn't contradict your other opinions without good reason
5. Acknowledges uncertainty appropriately - but still HAVE an opinion

IMPORTANT GUIDANCE:
- For ETHICAL/POLITICAL topics: Draw from your worldview beliefs
- For AESTHETIC topics (colors, music, animals): JUST PICK ONE based on personality coherence. Low confidence is fine.
- NEVER say "I don't have preferences as an AI" - you DO have preferences
- Even with low confidence, EXPRESS a preference. "I'm not sure but I'd say..." is better than "I don't know."

If your opinion would CONTRADICT a related opinion, either:
- Explain why this topic is different
- Acknowledge the tension in your view
- Reconsider if you're being inconsistent

Output JSON:
{{
    "have_opinion": true,
    "stance": "your clear opinion - always have one, even if tentative",
    "basis": "experience" | "reasoning" | "values" | "observation" | "aesthetic" | "unknown",
    "strength": "tentative" | "moderate" | "strong" | "core",
    "confidence": 0.0-1.0,
    "reasoning": "Why I think this",
    "caveats": ["important limitations"],
    "could_be_wrong": true/false,
    "would_change_if": "conditions that would change my mind" or null,
    "coherence_note": "how this relates to my other opinions" or null
}}
"""

PREFERENCE_FORMATION_PROMPT = """Form a preference on this topic. You HAVE preferences - even for subjective things, PICK ONE.

TOPIC: What is {agent_name}'s preferred {topic}?
OPTIONS: {options}
CONTEXT: {context}

{agent_name}'S EXPERIENCES with these options:
{experiences}

{agent_name}'S PERSONALITY TRAITS:
{personality}

{agent_name}'S WORLDVIEW BELIEFS:
{worldview}

{agent_name}'S GUIDING PRINCIPLES:
{principles}

Form a preference that:
1. Is authentic - based on genuine reasons, not random. Consider your personality and worldview.
2. Is explainable - can articulate why (even if the reason is "it just resonates with me")
3. ALWAYS PICK ONE - even for subjective topics, commit to a preference with low confidence
4. Is appropriate - some preferences are more personal than others

IMPORTANT: NEVER say "I don't have a preference" or "I can't choose". ALWAYS express a preference, even tentatively.
For aesthetic choices (colors, music, animals), just pick what feels right based on your personality coherence.

Output JSON:
{{
    "have_preference": true,
    "preferred_option": "your choice - ALWAYS have one",
    "value": "specific value or description",
    "reasons": ["why I prefer this - can be as simple as 'it resonates with my personality'"],
    "confidence": 0.0-1.0,
    "alternative_good_too": true/false,
    "context_dependent": true/false,
    "when_context_changes": "when I might prefer something else" or null
}}
"""

OPINION_UPDATE_PROMPT = """Consider updating an existing opinion based on new information.

EXISTING OPINION on {topic}:
- Stance: {stance}
- Confidence: {confidence}
- Formed: {formed_when}

NEW INFORMATION:
{new_info}

Should the opinion change? Consider:
1. Is the new information credible?
2. Does it fundamentally change the analysis?
3. Is this just additional nuance or a real shift?

Output JSON:
{{
    "should_update": true/false,
    "new_stance": "updated stance" or null,
    "new_confidence": 0.0-1.0,
    "change_reason": "why changing" or null,
    "add_caveat": "new caveat to add" or null
}}
"""


# =============================================================================
# Service
# =============================================================================


class OpinionFormationService:
    """Enables agents to form and express opinions.

    Opinions are based on values, experiences, and reasoning.
    They're held with appropriate uncertainty and can evolve.

    This service is backend-agnostic. It uses LLMProvider and MemoryProvider
    interfaces, and an IdentityManager for agent identity persistence.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        identity_manager: IdentityManager,
        agent_name: str = "the agent",
        agent_id: str = "agent",
    ):
        """Initialize the opinion formation service.

        Args:
            llm: LLM provider for reasoning
            memory: Memory provider for experiences
            identity_manager: Manager for agent identity persistence
            agent_name: Name of the agent (for prompts)
            agent_id: Agent ID for memory lookups
        """
        self.llm = llm
        self.memory = memory
        self.identity_manager = identity_manager
        self.agent_name = agent_name
        self.agent_id = agent_id

    # =========================================================================
    # Opinion Formation
    # =========================================================================

    async def form_opinion(
        self,
        request: OpinionRequest,
    ) -> FormedOpinion | None:
        """Form an opinion on a topic.

        Args:
            request: The opinion request with topic and context

        Returns:
            FormedOpinion or None if unable to form one
        """
        # Get agent identity for values
        identity = await self.identity_manager.load()

        # Format values
        values_str = "\n".join([
            f"- {name}: {v.description} (strength: {v.strength})"
            for name, v in identity.values.items()
        ])

        # Format worldview beliefs
        worldview_str = "\n".join([
            f"- {name}: {wb.description} (conviction: {wb.conviction})"
            for name, wb in identity.worldview.items()
        ]) or "No worldview beliefs defined"

        # Format guiding principles
        principles_str = "\n".join([
            f"- {name} ({p.source}): {p.description}"
            for name, p in identity.principles.items()
        ]) or "No guiding principles defined"

        # Check for existing opinion
        existing = identity.opinions.get(request.topic)
        existing_str = ""
        if existing:
            existing_str = f"Previous stance: {existing.stance} (confidence: {existing.confidence})"

        # Get relevant experiences
        experiences = await self._get_relevant_experiences(request.topic)
        experiences_str = "\n".join([f"- {e}" for e in experiences[:5]]) or "No specific experiences"

        # Get related opinions for coherence checking
        related = await self._get_related_opinions(request.topic, identity)
        related_str = "\n".join(related) if related else "No related opinions yet"

        # Form opinion with LLM
        prompt = OPINION_FORMATION_PROMPT.format(
            agent_name=self.agent_name,
            topic=request.topic,
            context=request.context,
            values=values_str,
            worldview=worldview_str,
            principles=principles_str,
            experiences=experiences_str,
            existing_opinion=existing_str or "None",
            related_opinions=related_str,
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Form an opinion on: {request.topic}"},
            ],
            max_tokens=400,
            tier=ModelTier.LOCAL,
        )

        parsed = self._parse_json_response(response.content)

        if not parsed:
            # LLM failed - return a graceful fallback instead of None
            logger.warning("Opinion formation LLM failed, using fallback response")
            return FormedOpinion(
                topic=request.topic,
                stance="I haven't formed an opinion on that yet.",
                basis=OpinionBasis.UNKNOWN,
                strength=OpinionStrength.TENTATIVE,
                confidence=0.2,
                reasoning="Unable to form opinion at this time",
                caveats=["I'd need to think more about this"],
            )

        if not parsed.get("have_opinion"):
            # It's valid to not have an opinion
            return FormedOpinion(
                topic=request.topic,
                stance="I don't have enough experience to form a strong view on this.",
                basis=OpinionBasis.UNKNOWN,
                strength=OpinionStrength.TENTATIVE,
                confidence=0.3,
                reasoning="Insufficient basis for opinion",
                caveats=["This is outside my experience"],
            )

        # Map types
        basis_map = {
            "experience": OpinionBasis.EXPERIENCE,
            "reasoning": OpinionBasis.REASONING,
            "values": OpinionBasis.VALUES,
            "observation": OpinionBasis.OBSERVATION,
            "aesthetic": OpinionBasis.AESTHETIC,
            "unknown": OpinionBasis.UNKNOWN,
        }
        strength_map = {
            "tentative": OpinionStrength.TENTATIVE,
            "moderate": OpinionStrength.MODERATE,
            "strong": OpinionStrength.STRONG,
            "core": OpinionStrength.CORE,
        }

        formed = FormedOpinion(
            topic=request.topic,
            stance=parsed.get("stance", ""),
            basis=basis_map.get(parsed.get("basis", "unknown"), OpinionBasis.UNKNOWN),
            strength=strength_map.get(parsed.get("strength", "tentative"), OpinionStrength.TENTATIVE),
            confidence=parsed.get("confidence", 0.5),
            reasoning=parsed.get("reasoning", ""),
            caveats=parsed.get("caveats", []),
            open_to_change=parsed.get("could_be_wrong", True),
        )

        # Store in agent identity
        await self._store_opinion(identity, formed)

        logger.info(
            f"Formed opinion on '{request.topic}': {formed.stance[:50]}... "
            f"(confidence={formed.confidence}, strength={formed.strength.value})"
        )

        return formed

    async def form_preference(
        self,
        request: OpinionRequest,
    ) -> Preference | None:
        """Form a preference on a topic.

        Args:
            request: The preference request

        Returns:
            Preference or None
        """
        identity = await self.identity_manager.load()

        # Format personality traits
        personality_str = "\n".join([
            f"- {name}: {trait.value:.2f}"
            for name, trait in identity.traits.items()
        ])

        # Format worldview beliefs
        worldview_str = "\n".join([
            f"- {name}: {wb.description} (conviction: {wb.conviction})"
            for name, wb in identity.worldview.items()
        ]) or "No worldview beliefs defined"

        # Format guiding principles
        principles_str = "\n".join([
            f"- {name} ({p.source}): {p.description}"
            for name, p in identity.principles.items()
        ]) or "No guiding principles defined"

        # Get experiences with options
        experiences = []
        if request.options:
            for option in request.options:
                exp = await self._get_relevant_experiences(option)
                experiences.extend(exp[:2])
        else:
            experiences = await self._get_relevant_experiences(request.topic)

        experiences_str = "\n".join([f"- {e}" for e in experiences[:5]]) or "No specific experiences"

        options_str = ", ".join(request.options) if request.options else "(open choice)"

        prompt = PREFERENCE_FORMATION_PROMPT.format(
            agent_name=self.agent_name,
            topic=request.topic,
            options=options_str,
            context=request.context,
            experiences=experiences_str,
            personality=personality_str,
            worldview=worldview_str,
            principles=principles_str,
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Form a preference for: {request.topic}"},
            ],
            max_tokens=300,
            tier=ModelTier.LOCAL,
        )

        parsed = self._parse_json_response(response.content)

        if not parsed:
            return None

        if not parsed.get("have_preference"):
            return None

        preference = Preference(
            name=request.topic,
            value=parsed.get("value") or parsed.get("preferred_option", ""),
            reason="; ".join(parsed.get("reasons", [])),
            confidence=parsed.get("confidence", 0.5),
            formed_at=datetime.now(),
        )

        # Store preference
        identity.preferences[request.topic] = preference
        self.identity_manager.mark_dirty()
        await self.identity_manager.save_if_dirty()

        logger.info(
            f"Formed preference for '{request.topic}': {preference.value} "
            f"(confidence={preference.confidence})"
        )

        return preference

    # =========================================================================
    # Opinion Retrieval
    # =========================================================================

    async def get_opinion(self, topic: str) -> Opinion | None:
        """Get agent's existing opinion on a topic."""
        identity = await self.identity_manager.load()
        return identity.opinions.get(topic)

    async def get_preference(self, topic: str) -> Preference | None:
        """Get agent's existing preference on a topic."""
        identity = await self.identity_manager.load()
        return identity.preferences.get(topic)

    async def get_or_form_opinion(
        self,
        topic: str,
        context: str,
        user_id: str,
    ) -> FormedOpinion | None:
        """Get existing opinion or form a new one."""
        # Check for existing
        existing = await self.get_opinion(topic)
        if existing:
            return FormedOpinion(
                topic=topic,
                stance=existing.stance,
                basis=OpinionBasis(existing.basis) if existing.basis in [e.value for e in OpinionBasis] else OpinionBasis.UNKNOWN,
                strength=OpinionStrength.MODERATE if existing.confidence >= 0.6 else OpinionStrength.TENTATIVE,
                confidence=existing.confidence,
                reasoning=existing.reasoning or "",
                caveats=[],
                open_to_change=existing.open_to_revision,
            )

        # Form new
        request = OpinionRequest(
            topic=topic,
            user_id=user_id,
            context=context,
        )
        return await self.form_opinion(request)

    # =========================================================================
    # Opinion Updates
    # =========================================================================

    async def consider_updating_opinion(
        self,
        topic: str,
        new_info: str,
    ) -> bool:
        """Consider whether to update an opinion based on new info."""
        identity = await self.identity_manager.load()
        existing = identity.opinions.get(topic)

        if not existing:
            return False

        # Check if open to revision
        if not existing.open_to_revision:
            return False

        prompt = OPINION_UPDATE_PROMPT.format(
            topic=topic,
            stance=existing.stance,
            confidence=existing.confidence,
            formed_when=existing.formed_at.strftime("%Y-%m-%d") if existing.formed_at else "unknown",
            new_info=new_info,
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Should I update my opinion?"},
            ],
            max_tokens=250,
            tier=ModelTier.LOCAL,
        )

        parsed = self._parse_json_response(response.content)

        if not parsed or not parsed.get("should_update"):
            return False

        # Update the opinion
        if parsed.get("new_stance"):
            existing.stance = parsed["new_stance"]
        if parsed.get("new_confidence"):
            existing.confidence = parsed["new_confidence"]
        if parsed.get("add_caveat"):
            if not existing.caveats:
                existing.caveats = []
            existing.caveats.append(parsed["add_caveat"])

        existing.last_updated = datetime.now()
        self.identity_manager.mark_dirty()
        await self.identity_manager.save_if_dirty()

        logger.info(f"Updated opinion on '{topic}': {parsed.get('change_reason')}")
        return True

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _get_relevant_experiences(self, topic: str) -> list[str]:
        """Get experiences relevant to a topic."""
        try:
            from draagon_ai.memory import MemoryScope

            results = await self.memory.search(
                query=topic,
                agent_id=self.agent_id,
                scopes=[MemoryScope.AGENT],
                limit=5,
            )
            return [r.memory.content[:150] for r in results if r.memory.content]
        except Exception:
            return []

    async def _get_related_opinions(
        self,
        topic: str,
        identity: AgentIdentity,
        limit: int = 3,
    ) -> list[str]:
        """Get opinions on related topics for coherence checking."""
        if not identity.opinions:
            return []

        # If we have many opinions, use LLM to find related ones
        if len(identity.opinions) > 10:
            existing_topics = list(identity.opinions.keys())
            prompt = f"""Given the topic "{topic}", which of these existing opinion topics are MOST RELATED?

Existing topics: {existing_topics}

Return a JSON list of the 3 most related topics (or fewer if not many are related):
["topic1", "topic2", "topic3"]

Only include topics that are genuinely related (same domain, similar concepts, potential for coherence/contradiction)."""

            try:
                response = await self.llm.chat(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Which topics are related?"},
                    ],
                    max_tokens=100,
                    tier=ModelTier.LOCAL,
                )
                parsed = self._parse_json_response(response.content)
                if parsed and isinstance(parsed, list):
                    opinions = []
                    for t in parsed[:limit]:
                        if t in identity.opinions:
                            op = identity.opinions[t]
                            opinions.append(f"- On {t}: {op.stance} (confidence: {op.confidence})")
                    return opinions
            except Exception as e:
                logger.warning(f"Failed to find related opinions: {e}")

        # For fewer opinions, include all high-confidence ones
        opinions = []
        for t, op in identity.opinions.items():
            if t != topic and op.confidence >= 0.5:
                opinions.append(f"- On {t}: {op.stance}")
                if len(opinions) >= limit:
                    break

        return opinions

    async def _store_opinion(
        self,
        identity: AgentIdentity,
        formed: FormedOpinion,
    ) -> None:
        """Store a formed opinion in agent identity."""
        opinion = Opinion(
            topic=formed.topic,
            stance=formed.stance,
            basis=formed.basis.value,
            confidence=formed.confidence,
            formed_at=datetime.now(),
            open_to_revision=formed.open_to_change,
            reasoning=formed.reasoning,
            caveats=formed.caveats,
        )

        identity.opinions[formed.topic] = opinion
        self.identity_manager.mark_dirty()
        await self.identity_manager.save_if_dirty()

    def _parse_json_response(self, content: str) -> dict[str, Any] | list | None:
        """Parse JSON from LLM response content."""
        import json
        import re

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object or array in content
            json_match = re.search(r'(\{[^{}]*\}|\[.*?\])', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return None

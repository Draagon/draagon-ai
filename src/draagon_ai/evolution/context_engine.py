"""ACE-Style Context Evolution Engine.

This module implements the Agentic Context Engineering approach:
- Generate → Reflect → Curate → Evolve pattern
- Grow-and-refine mechanism for contexts
- Self-improving context management

Based on the ACE Framework (Stanford, 2025).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
import logging
import hashlib

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Execute a chat completion."""
        ...

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Execute a chat completion expecting JSON output."""
        ...


@dataclass
class ContextCandidate:
    """A candidate context/prompt generated during evolution."""

    content: str
    source: str  # "generated", "refined", "merged"
    parent_id: str | None = None
    generation: int = 0
    effectiveness: float = 0.0
    similarity_to_existing: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def id(self) -> str:
        """Generate unique ID from content hash."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class ContextEvaluation:
    """Evaluation of a context candidate."""

    candidate_id: str
    effectiveness: float  # 0.0 - 1.0
    clarity: float  # 0.0 - 1.0
    completeness: float  # 0.0 - 1.0
    reasoning: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


@dataclass
class InteractionFeedback:
    """Feedback from an interaction that informs evolution."""

    query: str
    response: str
    success: bool
    user_correction: str | None = None
    quality_score: float = 0.5
    context_used: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Result of context evolution."""

    success: bool
    evolved_context: str | None = None
    candidates_generated: int = 0
    candidates_accepted: int = 0
    candidates_merged: int = 0
    improvement_score: float = 0.0
    error: str | None = None


# Templates for context evolution
GENERATE_TEMPLATE = """Based on these successful interaction patterns, generate a new context instruction.

SUCCESSFUL PATTERNS:
{patterns}

CURRENT CONTEXT:
{current_context}

TASK: Generate a new context instruction that:
1. Incorporates lessons from the successful patterns
2. Maintains the core purpose of the current context
3. Adds specific guidance for similar future situations

OUTPUT the new context instruction only, no explanation."""


REFLECT_TEMPLATE = """Evaluate this context instruction for effectiveness.

CONTEXT TO EVALUATE:
```
{context}
```

EVALUATION CRITERIA:
1. Clarity: Are instructions clear and unambiguous?
2. Completeness: Does it cover necessary cases?
3. Effectiveness: Would it produce good responses?

OUTPUT JSON:
{{
  "effectiveness": 0.0-1.0,
  "clarity": 0.0-1.0,
  "completeness": 0.0-1.0,
  "strengths": ["list", "of", "strengths"],
  "weaknesses": ["list", "of", "weaknesses"],
  "reasoning": "brief explanation"
}}"""


CURATE_MERGE_TEMPLATE = """Merge these two context instructions into an improved version.

CONTEXT A:
```
{context_a}
```

CONTEXT B:
```
{context_b}
```

TASK: Create a merged context that:
1. Combines the strongest elements from both
2. Resolves any conflicts coherently
3. Removes redundancy
4. Maintains clarity

OUTPUT the merged context only, no explanation."""


CURATE_REFINE_TEMPLATE = """Refine this context instruction based on feedback.

CURRENT CONTEXT:
```
{context}
```

IDENTIFIED WEAKNESSES:
{weaknesses}

TASK: Refine the context to address the weaknesses while preserving strengths.

OUTPUT the refined context only, no explanation."""


class ContextEvolutionEngine:
    """ACE-inspired context evolution engine.

    Implements the Generate → Reflect → Curate → Evolve pattern:
    1. GENERATE: Create candidate contexts from successful patterns
    2. REFLECT: Evaluate candidates for effectiveness
    3. CURATE: Merge/refine to grow or improve
    4. EVOLVE: Update the active context

    Key Innovation: Contexts grow and refine over time based on
    interaction feedback, rather than being static.
    """

    def __init__(
        self,
        llm: LLMProvider,
        similarity_threshold: float = 0.85,
        min_effectiveness: float = 0.3,
        max_context_length: int = 8000,
    ):
        """Initialize evolution engine.

        Args:
            llm: LLM provider for generation and evaluation
            similarity_threshold: Threshold for considering contexts similar
            min_effectiveness: Minimum effectiveness to keep a candidate
            max_context_length: Maximum allowed context length
        """
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.min_effectiveness = min_effectiveness
        self.max_context_length = max_context_length

        # Evolution state
        self._candidates: list[ContextCandidate] = []
        self._evaluations: dict[str, ContextEvaluation] = {}
        self._evolution_count = 0

    async def evolve(
        self,
        current_context: str,
        feedback: list[InteractionFeedback],
    ) -> EvolutionResult:
        """Run full evolution cycle on a context.

        Args:
            current_context: The current context to evolve
            feedback: Recent interaction feedback

        Returns:
            EvolutionResult with evolved context
        """
        if not feedback:
            return EvolutionResult(
                success=False,
                error="No feedback provided for evolution",
            )

        logger.info(f"Starting context evolution with {len(feedback)} feedback items")

        try:
            # Phase 1: GENERATE candidates from successful patterns
            candidates = await self._generate_candidates(current_context, feedback)
            logger.info(f"Generated {len(candidates)} candidates")

            # Phase 2: REFLECT - evaluate each candidate
            evaluations = await self._reflect_on_candidates(candidates)
            logger.info(f"Evaluated {len(evaluations)} candidates")

            # Phase 3: CURATE - grow (merge) or refine
            curated = await self._curate_contexts(
                current_context,
                candidates,
                evaluations,
            )
            logger.info(f"Curated to {len(curated)} contexts")

            # Phase 4: EVOLVE - select best as new context
            if not curated:
                return EvolutionResult(
                    success=True,
                    evolved_context=current_context,
                    candidates_generated=len(candidates),
                    candidates_accepted=0,
                    improvement_score=0.0,
                )

            # Select best candidate
            best = max(curated, key=lambda c: c.effectiveness)

            # Calculate improvement
            original_eval = await self._evaluate_single(current_context)
            improvement = best.effectiveness - original_eval.effectiveness

            self._evolution_count += 1

            return EvolutionResult(
                success=True,
                evolved_context=best.content,
                candidates_generated=len(candidates),
                candidates_accepted=len([c for c in curated if c.source != "merged"]),
                candidates_merged=len([c for c in curated if c.source == "merged"]),
                improvement_score=improvement,
            )

        except Exception as e:
            logger.error(f"Context evolution failed: {e}")
            return EvolutionResult(success=False, error=str(e))

    async def _generate_candidates(
        self,
        current_context: str,
        feedback: list[InteractionFeedback],
    ) -> list[ContextCandidate]:
        """Generate candidate contexts from successful patterns.

        Phase 1: GENERATE
        """
        candidates = []

        # Extract successful patterns
        successful = [f for f in feedback if f.success and f.quality_score >= 0.7]

        if not successful:
            logger.info("No successful patterns found, using all feedback")
            successful = feedback[:5]

        # Format patterns for prompt
        patterns = "\n".join(
            f"- Query: {f.query}\n  Response: {f.response[:200]}..."
            for f in successful[:5]
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert context engineer."},
                {"role": "user", "content": GENERATE_TEMPLATE.format(
                    patterns=patterns,
                    current_context=current_context[:4000],
                )},
            ]

            result = await self.llm.chat(
                messages,
                max_tokens=2000,
                temperature=0.7,
            )

            if result and result.get("content"):
                content = result["content"].strip()
                if len(content) <= self.max_context_length:
                    candidates.append(ContextCandidate(
                        content=content,
                        source="generated",
                        generation=self._evolution_count,
                    ))

        except Exception as e:
            logger.warning(f"Candidate generation failed: {e}")

        # Also generate variants with different temperatures
        for temp in [0.5, 0.9]:
            try:
                messages = [
                    {"role": "system", "content": "You are an expert context engineer."},
                    {"role": "user", "content": GENERATE_TEMPLATE.format(
                        patterns=patterns,
                        current_context=current_context[:4000],
                    )},
                ]

                result = await self.llm.chat(
                    messages,
                    max_tokens=2000,
                    temperature=temp,
                )

                if result and result.get("content"):
                    content = result["content"].strip()
                    if len(content) <= self.max_context_length:
                        candidates.append(ContextCandidate(
                            content=content,
                            source="generated",
                            generation=self._evolution_count,
                            metadata={"temperature": temp},
                        ))

            except Exception as e:
                logger.debug(f"Variant generation failed: {e}")

        return candidates

    async def _reflect_on_candidates(
        self,
        candidates: list[ContextCandidate],
    ) -> list[ContextEvaluation]:
        """Evaluate each candidate for effectiveness.

        Phase 2: REFLECT
        """
        evaluations = []

        for candidate in candidates:
            eval_result = await self._evaluate_single(candidate.content)
            eval_result.candidate_id = candidate.id
            evaluations.append(eval_result)

            # Update candidate effectiveness
            candidate.effectiveness = eval_result.effectiveness

            # Store for later reference
            self._evaluations[candidate.id] = eval_result

        return evaluations

    async def _evaluate_single(self, context: str) -> ContextEvaluation:
        """Evaluate a single context."""
        try:
            messages = [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": REFLECT_TEMPLATE.format(
                    context=context[:4000]
                )},
            ]

            result = await self.llm.chat_json(
                messages,
                max_tokens=500,
                temperature=0.1,
            )

            if result and result.get("parsed"):
                parsed = result["parsed"]
                return ContextEvaluation(
                    candidate_id="",
                    effectiveness=float(parsed.get("effectiveness", 0.5)),
                    clarity=float(parsed.get("clarity", 0.5)),
                    completeness=float(parsed.get("completeness", 0.5)),
                    reasoning=parsed.get("reasoning", ""),
                    strengths=parsed.get("strengths", []),
                    weaknesses=parsed.get("weaknesses", []),
                )

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

        return ContextEvaluation(
            candidate_id="",
            effectiveness=0.5,
            clarity=0.5,
            completeness=0.5,
        )

    async def _curate_contexts(
        self,
        current_context: str,
        candidates: list[ContextCandidate],
        evaluations: list[ContextEvaluation],
    ) -> list[ContextCandidate]:
        """Curate candidates by merging similar ones and refining weak ones.

        Phase 3: CURATE (Grow-and-Refine)
        """
        curated = []

        # Filter by minimum effectiveness
        effective = [
            c for c in candidates
            if c.effectiveness >= self.min_effectiveness
        ]

        if not effective:
            logger.info("No candidates met minimum effectiveness")
            return []

        # Calculate similarity to each other
        for i, c1 in enumerate(effective):
            for c2 in effective[i+1:]:
                sim = self._calculate_similarity(c1.content, c2.content)
                c1.similarity_to_existing = max(c1.similarity_to_existing, sim)
                c2.similarity_to_existing = max(c2.similarity_to_existing, sim)

        # Group similar candidates for merging
        merged_groups = []
        used = set()

        for c in effective:
            if c.id in used:
                continue

            group = [c]
            used.add(c.id)

            for other in effective:
                if other.id in used:
                    continue
                sim = self._calculate_similarity(c.content, other.content)
                if sim >= self.similarity_threshold:
                    group.append(other)
                    used.add(other.id)

            if len(group) > 1:
                merged_groups.append(group)
            else:
                curated.append(c)

        # Merge similar groups
        for group in merged_groups:
            merged = await self._merge_contexts(group)
            if merged:
                curated.append(merged)

        # Refine candidates with weaknesses
        for c in curated:
            eval_result = self._evaluations.get(c.id)
            if eval_result and eval_result.weaknesses:
                refined = await self._refine_context(c, eval_result.weaknesses)
                if refined and refined.effectiveness > c.effectiveness:
                    # Replace with refined version
                    curated = [refined if x.id == c.id else x for x in curated]

        return curated

    async def _merge_contexts(
        self,
        contexts: list[ContextCandidate],
    ) -> ContextCandidate | None:
        """Merge similar contexts into one."""
        if len(contexts) < 2:
            return contexts[0] if contexts else None

        # Sort by effectiveness and take top 2
        sorted_ctx = sorted(contexts, key=lambda c: c.effectiveness, reverse=True)
        ctx_a, ctx_b = sorted_ctx[0], sorted_ctx[1]

        try:
            messages = [
                {"role": "system", "content": "You are an expert context engineer."},
                {"role": "user", "content": CURATE_MERGE_TEMPLATE.format(
                    context_a=ctx_a.content[:3000],
                    context_b=ctx_b.content[:3000],
                )},
            ]

            result = await self.llm.chat(
                messages,
                max_tokens=2000,
                temperature=0.3,
            )

            if result and result.get("content"):
                content = result["content"].strip()
                if len(content) <= self.max_context_length:
                    # Evaluate merged result
                    eval_result = await self._evaluate_single(content)

                    return ContextCandidate(
                        content=content,
                        source="merged",
                        parent_id=ctx_a.id,
                        generation=self._evolution_count,
                        effectiveness=eval_result.effectiveness,
                    )

        except Exception as e:
            logger.warning(f"Context merge failed: {e}")

        # Return best if merge fails
        return ctx_a

    async def _refine_context(
        self,
        candidate: ContextCandidate,
        weaknesses: list[str],
    ) -> ContextCandidate | None:
        """Refine a context to address weaknesses."""
        try:
            messages = [
                {"role": "system", "content": "You are an expert context engineer."},
                {"role": "user", "content": CURATE_REFINE_TEMPLATE.format(
                    context=candidate.content[:4000],
                    weaknesses="\n".join(f"- {w}" for w in weaknesses),
                )},
            ]

            result = await self.llm.chat(
                messages,
                max_tokens=2000,
                temperature=0.3,
            )

            if result and result.get("content"):
                content = result["content"].strip()
                if len(content) <= self.max_context_length:
                    # Evaluate refined result
                    eval_result = await self._evaluate_single(content)

                    return ContextCandidate(
                        content=content,
                        source="refined",
                        parent_id=candidate.id,
                        generation=self._evolution_count,
                        effectiveness=eval_result.effectiveness,
                    )

        except Exception as e:
            logger.warning(f"Context refinement failed: {e}")

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get evolution statistics."""
        return {
            "evolution_count": self._evolution_count,
            "candidates_in_memory": len(self._candidates),
            "evaluations_in_memory": len(self._evaluations),
            "similarity_threshold": self.similarity_threshold,
            "min_effectiveness": self.min_effectiveness,
        }

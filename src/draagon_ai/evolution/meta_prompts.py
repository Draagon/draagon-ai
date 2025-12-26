"""Self-Referential Meta Prompts.

This module implements meta prompts that generate/improve other prompts.
Key innovation: Meta prompts can improve themselves!

Based on:
- Promptbreeder paper (Sakana AI)
- ACE Framework (Stanford)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
import logging
import random

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


@dataclass
class MetaPrompt:
    """A prompt that generates/improves other prompts.

    Self-referential: MetaPrompts can improve other MetaPrompts!
    """

    prompt_id: str
    content: str
    target_type: str  # "mutation", "crossover", "evaluation", "meta"

    # Fitness tracking
    fitness: float = 0.5
    usage_count: int = 0
    success_count: int = 0

    # Lineage
    parent_id: str | None = None
    generation: int = 0

    # Self-reference
    can_mutate_self: bool = True
    mutation_history: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count

    def record_usage(self, successful: bool) -> None:
        """Record a usage of this meta prompt."""
        self.usage_count += 1
        if successful:
            self.success_count += 1
        self.updated_at = datetime.now()


# Default mutation prompts - these evolve over time!
# IMPORTANT: Balance of expansion and compression to avoid shrinkage bias
DEFAULT_MUTATION_PROMPTS = [
    # Clarity mutations (neutral on length)
    MetaPrompt(
        prompt_id="mutation_clarity_1",
        content="Improve this prompt by making the instructions clearer and more specific.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_clarity_2",
        content="Restructure this prompt to put the most important instructions first.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_format",
        content="Improve the output format instructions for more consistent responses.",
        target_type="mutation",
    ),

    # Expansion mutations (add detail)
    MetaPrompt(
        prompt_id="mutation_example",
        content="Add a concrete example that demonstrates handling the identified failure case.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_edge_case",
        content="Add edge case handling for situations where the current prompt fails.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_expand",
        content="Expand the instructions for the action that failed in the failure cases.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_guidance",
        content="Add explicit guidance for choosing between similar actions.",
        target_type="mutation",
    ),

    # Compression mutations (remove redundancy - preserve features!)
    MetaPrompt(
        prompt_id="mutation_compress_1",
        content="Remove ONLY truly redundant instructions, but preserve all actions and examples.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_compress_2",
        content="Consolidate similar instructions without removing any capabilities.",
        target_type="mutation",
    ),

    # Robustness mutations
    MetaPrompt(
        prompt_id="mutation_fallback",
        content="Add fallback behavior when the primary action doesn't apply.",
        target_type="mutation",
    ),
    MetaPrompt(
        prompt_id="mutation_strengthen",
        content="Strengthen the instructions to prevent the failure case from recurring.",
        target_type="mutation",
    ),
]


# Templates for prompt operations
MUTATION_TEMPLATE = """You are mutating a prompt to improve its performance.

MUTATION INSTRUCTION:
{mutation_instruction}

CURRENT PROMPT:
```
{current_prompt}
```

FAILURE CASES TO ADDRESS:
{failure_cases}

CRITICAL CONSTRAINTS:
1. PRESERVE all actions listed in "AVAILABLE ACTIONS" - do NOT remove any
2. PRESERVE all placeholders like {{question}}, {{context}}, {{user_id}}
3. PRESERVE the output format structure (XML tags)
4. PRESERVE examples if the original has them
5. Only IMPROVE, do not remove capabilities

OUTPUT the improved prompt only. Do not include any explanation or markdown formatting.
The output should be ready to use directly as a prompt."""


CROSSOVER_TEMPLATE = """Combine the best elements of these two prompts into a new, improved prompt.

PROMPT A (Parent 1):
```
{parent1}
```

PROMPT B (Parent 2):
```
{parent2}
```

INSTRUCTIONS:
1. Identify the strongest elements from each prompt
2. Combine them coherently without duplication
3. Ensure the result is well-structured and clear
4. Keep the same output format as the original prompts

OUTPUT the combined prompt only. Do not include any explanation."""


META_EVOLUTION_TEMPLATE = """You are improving a mutation instruction used to evolve other prompts.

CURRENT MUTATION INSTRUCTION:
"{current_instruction}"

This instruction is used to generate variations of AI prompts. We want to evolve it
to produce better mutations that lead to higher-performing prompts.

TASK: Generate an improved version of this mutation instruction that:
1. Is more specific about what to change
2. Leads to more effective prompt improvements
3. Maintains the general intent but refines the approach

OUTPUT only the new mutation instruction, no explanation. Keep it to 1-2 sentences."""


@dataclass
class MutationResult:
    """Result of a mutation operation."""

    success: bool
    mutated_prompt: str | None = None
    meta_prompt_used: MetaPrompt | None = None
    error: str | None = None


class MetaPromptManager:
    """Manages meta prompts and their evolution.

    Key features:
    - Selection based on fitness (roulette wheel)
    - Self-referential evolution (meta prompts evolve themselves)
    - Tracking of usage and success
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        initial_prompts: list[MetaPrompt] | None = None,
    ):
        """Initialize manager.

        Args:
            llm: LLM provider for evolution
            initial_prompts: Initial set of meta prompts
        """
        self.llm = llm
        self.mutation_prompts = list(initial_prompts or DEFAULT_MUTATION_PROMPTS)
        self._evolution_count = 0

    def select_mutation_prompt(self) -> MetaPrompt:
        """Select a mutation prompt using fitness-weighted selection.

        Uses roulette wheel selection based on success rate.
        """
        if not self.mutation_prompts:
            raise ValueError("No mutation prompts available")

        # Calculate selection probabilities based on fitness
        total_fitness = sum(mp.success_rate + 0.1 for mp in self.mutation_prompts)

        r = random.random() * total_fitness
        cumulative = 0.0

        for mp in self.mutation_prompts:
            cumulative += mp.success_rate + 0.1  # +0.1 to avoid zero probability
            if cumulative >= r:
                return mp

        # Fallback to last (shouldn't happen)
        return self.mutation_prompts[-1]

    async def mutate(
        self,
        prompt: str,
        failure_cases: list[dict[str, Any]] | None = None,
        meta_prompt: MetaPrompt | None = None,
    ) -> MutationResult:
        """Mutate a prompt using a meta prompt.

        Args:
            prompt: The prompt to mutate
            failure_cases: List of failure case dicts with query/issue
            meta_prompt: Specific meta prompt to use (or select one)

        Returns:
            MutationResult with the mutated prompt
        """
        if not self.llm:
            return MutationResult(success=False, error="No LLM provider")

        # Select meta prompt if not provided
        if meta_prompt is None:
            meta_prompt = self.select_mutation_prompt()

        # Format failure cases
        failure_text = "None identified"
        if failure_cases:
            failure_text = "\n".join(
                f"- Query: {fc.get('query', 'unknown')}\n  Issue: {fc.get('issue', 'unspecified')}"
                for fc in failure_cases[:3]
            )

        mutation_prompt = MUTATION_TEMPLATE.format(
            mutation_instruction=meta_prompt.content,
            current_prompt=prompt[:6000],  # Truncate if very long
            failure_cases=failure_text,
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": mutation_prompt},
            ]

            result = await self.llm.chat(
                messages,
                max_tokens=4000,
                temperature=0.7,
            )

            if result and result.get("content"):
                mutated = result["content"].strip()

                # Remove markdown code blocks if present
                if mutated.startswith("```"):
                    lines = mutated.split("\n")
                    mutated = "\n".join(
                        lines[1:-1] if lines[-1] == "```" else lines[1:]
                    )

                return MutationResult(
                    success=True,
                    mutated_prompt=mutated,
                    meta_prompt_used=meta_prompt,
                )

        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            return MutationResult(success=False, error=str(e))

        return MutationResult(success=False, error="Empty response")

    async def crossover(
        self,
        parent1: str,
        parent2: str,
    ) -> MutationResult:
        """Perform crossover between two prompts.

        Args:
            parent1: First parent prompt
            parent2: Second parent prompt

        Returns:
            MutationResult with offspring prompt
        """
        if not self.llm:
            return MutationResult(success=False, error="No LLM provider")

        crossover_prompt = CROSSOVER_TEMPLATE.format(
            parent1=parent1[:4000],
            parent2=parent2[:4000],
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": crossover_prompt},
            ]

            result = await self.llm.chat(
                messages,
                max_tokens=4000,
                temperature=0.5,
            )

            if result and result.get("content"):
                child = result["content"].strip()
                if child:
                    return MutationResult(
                        success=True,
                        mutated_prompt=child,
                    )

        except Exception as e:
            logger.warning(f"Crossover failed: {e}")
            return MutationResult(success=False, error=str(e))

        return MutationResult(success=False, error="Empty response")

    async def evolve_meta_prompts(self) -> int:
        """Evolve the mutation prompts themselves (self-referential!).

        Returns:
            Number of meta prompts evolved
        """
        if not self.llm:
            return 0

        logger.info("Evolving mutation prompts (self-referential)")

        evolved_count = 0
        new_prompts = []

        for mp in self.mutation_prompts:
            if not mp.can_mutate_self:
                new_prompts.append(mp)
                continue

            try:
                messages = [
                    {"role": "system", "content": "You improve prompt engineering instructions."},
                    {"role": "user", "content": META_EVOLUTION_TEMPLATE.format(
                        current_instruction=mp.content
                    )},
                ]

                result = await self.llm.chat(
                    messages,
                    max_tokens=200,
                    temperature=0.7,
                )

                if result and result.get("content"):
                    evolved_content = result["content"].strip()

                    # Basic validation
                    if 10 < len(evolved_content) < 300:
                        # Create evolved version
                        evolved_mp = MetaPrompt(
                            prompt_id=f"{mp.prompt_id}_v{mp.generation + 1}",
                            content=evolved_content,
                            target_type=mp.target_type,
                            parent_id=mp.prompt_id,
                            generation=mp.generation + 1,
                            can_mutate_self=mp.can_mutate_self,
                            mutation_history=mp.mutation_history + [mp.content],
                        )
                        new_prompts.append(evolved_mp)
                        evolved_count += 1
                    else:
                        new_prompts.append(mp)
                else:
                    new_prompts.append(mp)

            except Exception as e:
                logger.warning(f"Failed to evolve meta prompt {mp.prompt_id}: {e}")
                new_prompts.append(mp)

        self.mutation_prompts = new_prompts
        self._evolution_count += 1

        logger.info(f"Evolved {evolved_count}/{len(self.mutation_prompts)} meta prompts")
        return evolved_count

    def record_result(
        self,
        meta_prompt: MetaPrompt,
        successful: bool,
    ) -> None:
        """Record the result of using a meta prompt.

        Args:
            meta_prompt: The meta prompt that was used
            successful: Whether the mutation was successful
        """
        # Find and update the meta prompt
        for mp in self.mutation_prompts:
            if mp.prompt_id == meta_prompt.prompt_id:
                mp.record_usage(successful)
                break

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about meta prompts."""
        return {
            "total_prompts": len(self.mutation_prompts),
            "evolution_count": self._evolution_count,
            "prompts": [
                {
                    "id": mp.prompt_id,
                    "generation": mp.generation,
                    "usage_count": mp.usage_count,
                    "success_rate": mp.success_rate,
                    "content_preview": mp.content[:50] + "...",
                }
                for mp in self.mutation_prompts
            ],
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize meta prompts for persistence."""
        return {
            "evolution_count": self._evolution_count,
            "prompts": [
                {
                    "prompt_id": mp.prompt_id,
                    "content": mp.content,
                    "target_type": mp.target_type,
                    "fitness": mp.fitness,
                    "usage_count": mp.usage_count,
                    "success_count": mp.success_count,
                    "parent_id": mp.parent_id,
                    "generation": mp.generation,
                    "can_mutate_self": mp.can_mutate_self,
                    "mutation_history": mp.mutation_history,
                }
                for mp in self.mutation_prompts
            ],
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Load meta prompts from serialized data."""
        self._evolution_count = data.get("evolution_count", 0)

        prompts_data = data.get("prompts", [])
        if prompts_data:
            self.mutation_prompts = [
                MetaPrompt(
                    prompt_id=p["prompt_id"],
                    content=p["content"],
                    target_type=p["target_type"],
                    fitness=p.get("fitness", 0.5),
                    usage_count=p.get("usage_count", 0),
                    success_count=p.get("success_count", 0),
                    parent_id=p.get("parent_id"),
                    generation=p.get("generation", 0),
                    can_mutate_self=p.get("can_mutate_self", True),
                    mutation_history=p.get("mutation_history", []),
                )
                for p in prompts_data
            ]

"""Probabilistic expansion of ambiguous messages.

Expands messages into multiple interpretation branches with
probability scores, enabling beam search exploration.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..decomposition.graph import SemanticGraph, GraphNode, NodeType

from .context import RecencyWindow


logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Send chat messages and get response."""
        ...


@dataclass
class InterpretationBranch:
    """A single interpretation of ambiguous input."""

    branch_id: str
    interpretation: str  # Natural language description
    probability: float   # 0.0 - 1.0
    graph: SemanticGraph  # Semantic representation
    reasoning: str       # Why this interpretation
    semantic_structure: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        interpretation: str,
        probability: float,
        reasoning: str,
        semantic_structure: dict[str, Any] | None = None,
    ) -> InterpretationBranch:
        """Create a new branch with generated ID."""
        branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        graph = SemanticGraph()

        # Build graph from semantic structure
        if semantic_structure:
            cls._build_graph(graph, semantic_structure)

        return cls(
            branch_id=branch_id,
            interpretation=interpretation,
            probability=probability,
            graph=graph,
            reasoning=reasoning,
            semantic_structure=semantic_structure or {},
        )

    @staticmethod
    def _build_graph(graph: SemanticGraph, structure: dict[str, Any]) -> None:
        """Build graph nodes from semantic structure."""
        # Create subject node
        if "subject" in structure:
            subj = graph.create_node(
                structure["subject"],
                NodeType.INSTANCE,
            )

        # Create predicate node
        if "predicate" in structure:
            pred_info = structure["predicate"]
            if isinstance(pred_info, dict):
                pred = graph.create_node(
                    pred_info.get("text", "action"),
                    NodeType.EVENT,
                    synset_id=pred_info.get("synset"),
                )
            else:
                pred = graph.create_node(str(pred_info), NodeType.EVENT)

            # Link subject to predicate
            if "subject" in structure:
                graph.create_edge(subj.node_id, pred.node_id, "agent_of")

        # Create object node
        if "object" in structure:
            obj = graph.create_node(
                structure["object"],
                NodeType.INSTANCE,
            )

            # Link predicate to object
            if "predicate" in structure:
                graph.create_edge(pred.node_id, obj.node_id, "patient")


@dataclass
class ExpansionResult:
    """Result of probabilistic expansion."""

    original_text: str
    original_graph: SemanticGraph | None
    branches: list[InterpretationBranch]
    ambiguity_type: str  # "referential", "semantic", "pragmatic", "none"
    expansion_time_ms: float = 0.0

    @property
    def is_ambiguous(self) -> bool:
        """Check if multiple interpretations were generated."""
        return len(self.branches) > 1

    @property
    def top_branch(self) -> InterpretationBranch | None:
        """Get highest probability branch."""
        if not self.branches:
            return None
        return max(self.branches, key=lambda b: b.probability)

    def branches_above_threshold(self, threshold: float = 0.1) -> list[InterpretationBranch]:
        """Get branches above probability threshold."""
        return [b for b in self.branches if b.probability >= threshold]


# XML prompt for probabilistic expansion
EXPANSION_PROMPT = """Analyze this message and generate possible interpretations with probabilities.

MESSAGE: {message}

RECENT CONTEXT:
{context}

CURRENT SEMANTIC UNDERSTANDING:
{semantics}

Generate 1-4 interpretations based on what the message likely means.
Consider:
- Pronouns and what they refer to (anaphora resolution)
- Multiple word senses (e.g., "got" = received/retrieved/understood)
- Pragmatic meaning (what the speaker intends vs literal meaning)
- Context from recent conversation

For each interpretation:
1. Describe what the user likely means
2. Assign probability (all probabilities must sum to 1.0)
3. Provide semantic structure (subject, predicate with synset, object)
4. Explain your reasoning

Output format:

<response>
<ambiguity_type>referential|semantic|pragmatic|none</ambiguity_type>
<interpretations>
<interpretation probability="0.XX">
<description>Plain English interpretation</description>
<semantic_structure>
<subject>The agent/experiencer</subject>
<predicate synset="word.v.NN">The action/state</predicate>
<object>The patient/theme</object>
</semantic_structure>
<reasoning>Why this interpretation makes sense given context</reasoning>
</interpretation>
</interpretations>
</response>"""


class ProbabilisticExpander:
    """Expands ambiguous messages into interpretation branches.

    Uses an LLM to analyze messages in context and generate
    multiple possible interpretations with probability scores.
    """

    def __init__(
        self,
        llm: LLMProvider,
        min_probability: float = 0.05,
        max_branches: int = 4,
    ):
        self.llm = llm
        self.min_probability = min_probability
        self.max_branches = max_branches

    async def expand(
        self,
        message: str,
        message_graph: SemanticGraph | None = None,
        recency_context: RecencyWindow | None = None,
    ) -> ExpansionResult:
        """
        Expand message into multiple interpretations with probabilities.

        Args:
            message: The message to interpret
            message_graph: Pre-extracted semantic graph (if available)
            recency_context: Recent conversation context

        Returns:
            ExpansionResult with branches and metadata
        """
        import time
        start = time.perf_counter()

        # Build context summary
        context_summary = recency_context.to_summary() if recency_context else "No recent context."

        # Build semantics summary
        if message_graph:
            semantics_summary = self._graph_to_text(message_graph)
        else:
            semantics_summary = "No pre-extracted semantics."

        # Build prompt
        prompt = EXPANSION_PROMPT.format(
            message=message,
            context=context_summary,
            semantics=semantics_summary,
        )

        # Call LLM
        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temp for more consistent parsing
                max_tokens=1500,
            )

            # Parse response
            branches, ambiguity_type = self._parse_response(response)

        except Exception as e:
            logger.warning(f"Expansion failed: {e}, falling back to single branch")
            branches = [
                InterpretationBranch.create(
                    interpretation=message,
                    probability=1.0,
                    reasoning="Fallback: expansion failed",
                )
            ]
            ambiguity_type = "none"

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExpansionResult(
            original_text=message,
            original_graph=message_graph,
            branches=branches,
            ambiguity_type=ambiguity_type,
            expansion_time_ms=elapsed_ms,
        )

    def _graph_to_text(self, graph: SemanticGraph) -> str:
        """Convert graph to text summary for LLM."""
        lines = []

        # List entities
        entities = [n for n in graph.iter_nodes() if n.node_type == NodeType.INSTANCE]
        if entities:
            lines.append(f"Entities: {', '.join(n.canonical_name for n in entities[:10])}")

        # List relationships
        for edge in list(graph.iter_edges())[:10]:
            source = graph.get_node(edge.source_node_id)
            target = graph.get_node(edge.target_node_id)
            if source and target:
                lines.append(f"  {source.canonical_name} -[{edge.relation_type}]-> {target.canonical_name}")

        return "\n".join(lines) if lines else "Empty graph"

    def _parse_response(
        self,
        response: str
    ) -> tuple[list[InterpretationBranch], str]:
        """Parse LLM response into branches."""
        branches = []
        ambiguity_type = "none"

        # Extract ambiguity type
        amb_match = re.search(r'<ambiguity_type>(\w+)</ambiguity_type>', response)
        if amb_match:
            ambiguity_type = amb_match.group(1)

        # Extract interpretations
        interp_pattern = r'<interpretation probability="([\d.]+)">(.*?)</interpretation>'
        for match in re.finditer(interp_pattern, response, re.DOTALL):
            prob = float(match.group(1))
            content = match.group(2)

            # Extract fields
            desc_match = re.search(r'<description>(.*?)</description>', content, re.DOTALL)
            reason_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)

            description = desc_match.group(1).strip() if desc_match else "Unknown"
            reasoning = reason_match.group(1).strip() if reason_match else ""

            # Extract semantic structure
            struct = {}
            subj_match = re.search(r'<subject>(.*?)</subject>', content)
            if subj_match:
                struct["subject"] = subj_match.group(1).strip()

            pred_match = re.search(r'<predicate(?:\s+synset="([^"]*)")?>(.*?)</predicate>', content)
            if pred_match:
                struct["predicate"] = {
                    "synset": pred_match.group(1),
                    "text": pred_match.group(2).strip(),
                }

            obj_match = re.search(r'<object>(.*?)</object>', content)
            if obj_match:
                struct["object"] = obj_match.group(1).strip()

            # Create branch
            if prob >= self.min_probability:
                branch = InterpretationBranch.create(
                    interpretation=description,
                    probability=prob,
                    reasoning=reasoning,
                    semantic_structure=struct,
                )
                branches.append(branch)

        # Ensure we have at least one branch
        if not branches:
            branches = [
                InterpretationBranch.create(
                    interpretation="Direct interpretation of message",
                    probability=1.0,
                    reasoning="No alternative interpretations found",
                )
            ]

        # Normalize probabilities
        total = sum(b.probability for b in branches)
        if total > 0:
            for b in branches:
                b.probability = b.probability / total

        # Sort by probability
        branches.sort(key=lambda b: b.probability, reverse=True)

        # Limit branches
        branches = branches[:self.max_branches]

        return branches, ambiguity_type

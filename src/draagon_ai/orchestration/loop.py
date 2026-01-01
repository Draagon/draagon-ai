"""Main agent loop.

The agent loop ties together all components to process queries:
1. Gather context (memories, knowledge)
2. Make decision (which action to take)
3. Execute action (using tools)
4. Synthesize response (format for output)
5. Post-processing (learning, metrics)

Supports two execution modes:
- Simple: Single decision → action → response (fast path)
- ReAct: Multi-step reasoning with THOUGHT → ACTION → OBSERVATION loop

ReAct Pattern (REQ-002-01):
    loop:
        THOUGHT: "I need to check the user's calendar for conflicts"
        ACTION: search_calendar(days=7)
        OBSERVATION: [3 events found]

        THOUGHT: "Now I see there's an overlap on Tuesday..."
        ACTION: get_event_details(event_id="...")
        OBSERVATION: {details}

        THOUGHT: "I have enough information to answer"
        FINAL_ANSWER: "You have a conflict on Tuesday at 3pm..."
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
import asyncio
import logging
import sys

from ..behaviors import Action, Behavior
from .protocols import LLMMessage, LLMProvider, MemoryProvider
from .decision import DecisionContext, DecisionEngine, DecisionResult
from .execution import ActionExecutor, ActionResult

# Lazy import to avoid circular dependencies
SemanticContextService = None

logger = logging.getLogger(__name__)


def _get_semantic_context_service():
    """Lazy import of SemanticContextService to avoid circular imports."""
    global SemanticContextService
    if SemanticContextService is None:
        try:
            from .semantic_context import SemanticContextService as SCS
            SemanticContextService = SCS
        except ImportError:
            pass
    return SemanticContextService


# Require Python 3.11+ for asyncio.timeout and asyncio.Barrier
if sys.version_info < (3, 11):
    raise RuntimeError(
        f"draagon-ai requires Python 3.11+, but you are running {sys.version_info.major}.{sys.version_info.minor}. "
        "Please upgrade Python."
    )
async_timeout = asyncio.timeout


class LoopMode(Enum):
    """Execution mode for the agent loop."""

    SIMPLE = "simple"  # Single-step: decision → action → response
    REACT = "react"  # Multi-step: THOUGHT → ACTION → OBSERVATION loop
    AUTO = "auto"  # Automatically detect based on query complexity


class StepType(Enum):
    """Type of step in a ReAct trace."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReActStep:
    """A single step in a ReAct reasoning trace.

    Captures the THOUGHT, ACTION, or OBSERVATION at each iteration.
    """

    type: StepType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    # For ACTION steps
    action_name: str | None = None
    action_args: dict[str, Any] = field(default_factory=dict)

    # For OBSERVATION steps
    observation_success: bool = True
    observation_error: str | None = None


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop.

    Controls execution mode, iteration limits, and timeouts.
    """

    # Execution mode
    mode: LoopMode = LoopMode.AUTO

    # ReAct loop settings
    max_iterations: int = 10
    iteration_timeout_seconds: float = 30.0

    # Auto-mode detection settings
    # threshold=0.33 means 1+ keyword match triggers REACT
    # This ensures sensor queries (solar, battery, etc.) use multi-step reasoning
    complexity_threshold: float = 0.33  # Query complexity for auto-mode

    # Debug settings
    log_thought_traces: bool = True

    # ==========================================================================
    # Semantic Context Integration (Phase 2)
    # ==========================================================================
    # Enable semantic context enrichment via ReasoningLoop
    use_semantic_context: bool = False  # Off by default until stable

    # Neo4j connection for semantic graph (required if use_semantic_context=True)
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    semantic_instance_id: str = "default"

    # Context retrieval settings
    max_semantic_facts: int = 10
    max_semantic_memories: int = 5
    semantic_context_depth: int = 2

    # Complexity keywords that suggest multi-step reasoning
    complexity_keywords: list[str] = field(
        default_factory=lambda: [
            # Explicit multi-step indicators
            "and then",
            "after that",
            "also",
            "first",
            "next",
            "finally",
            # Analysis keywords
            "check",
            "compare",
            "analyze",
            # Home Assistant queries - often need entity discovery
            "how much",
            "what's my",
            "is the",
            "solar",
            "battery",
            "energy",
            "power",
            "water",
            "temperature",
            "thermostat",
            "charger",
        ]
    )


@dataclass
class AgentContext:
    """Context for an agent interaction."""

    # Identity
    user_id: str
    session_id: str = ""
    conversation_id: str = ""

    # Location/Area
    area_id: str | None = None
    device_id: str | None = None

    # Conversation state
    conversation_history: list[dict] = field(default_factory=list)
    pending_details: str | None = None

    # ReAct observations (accumulated during multi-step reasoning)
    observations: list[str] = field(default_factory=list)

    # Metadata
    debug: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, observation: str) -> None:
        """Add an observation to the context for multi-step reasoning.

        Args:
            observation: The observation to add
        """
        self.observations.append(observation)

    def clear_observations(self) -> None:
        """Clear all observations (call at start of new query)."""
        self.observations.clear()

    def get_observations_text(self) -> str:
        """Get observations as formatted text for prompts.

        Returns:
            Formatted observations string
        """
        if not self.observations:
            return ""
        return "\n".join(
            f"Observation {i + 1}: {obs}" for i, obs in enumerate(self.observations)
        )


@dataclass
class AgentResponse:
    """Response from agent processing."""

    # Main response
    response: str
    success: bool = True

    # Details (for "tell me more")
    full_response: str | None = None

    # What happened
    action_taken: str = ""
    tool_results: list[ActionResult] = field(default_factory=list)

    # Decision details
    decision: DecisionResult | None = None

    # ReAct reasoning trace (REQ-002-01)
    react_steps: list[ReActStep] = field(default_factory=list)
    loop_mode: LoopMode = LoopMode.SIMPLE
    iterations_used: int = 0

    # Timing
    latency_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)

    # Memory operations
    memories_stored: list[str] = field(default_factory=list)

    # Debug info
    debug_info: dict[str, Any] = field(default_factory=dict)

    def add_react_step(
        self,
        step_type: StepType,
        content: str,
        duration_ms: float = 0.0,
        action_name: str | None = None,
        action_args: dict | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> ReActStep:
        """Add a ReAct step to the reasoning trace.

        Args:
            step_type: Type of step (THOUGHT, ACTION, OBSERVATION, FINAL_ANSWER)
            content: Content of the step
            duration_ms: Duration of this step in milliseconds
            action_name: For ACTION steps, the action name
            action_args: For ACTION steps, the action arguments
            success: For OBSERVATION steps, whether the action succeeded
            error: For OBSERVATION steps, any error message

        Returns:
            The created ReActStep
        """
        step = ReActStep(
            type=step_type,
            content=content,
            duration_ms=duration_ms,
            action_name=action_name,
            action_args=action_args or {},
            observation_success=success,
            observation_error=error,
        )
        self.react_steps.append(step)
        return step

    def get_thought_trace(self) -> list[dict[str, Any]]:
        """Get the thought trace as a list of dicts for debugging/logging.

        Returns:
            List of step dictionaries with type, content, timing
        """
        return [
            {
                "step": i + 1,
                "type": step.type.value,
                "content": step.content,
                "timestamp": step.timestamp.isoformat(),
                "duration_ms": step.duration_ms,
                "action_name": step.action_name,
                "action_args": step.action_args,
                "success": step.observation_success,
                "error": step.observation_error,
            }
            for i, step in enumerate(self.react_steps)
        ]


class AgentLoop:
    """The main agent processing loop.

    This class orchestrates the entire agent interaction:
    query -> context -> decision -> execution -> synthesis -> response

    Supports two execution modes (REQ-002-01):
    - Simple: Single decision → action → response (fast path)
    - ReAct: Multi-step THOUGHT → ACTION → OBSERVATION loop

    It is designed to be used by the Agent class but can also
    be used directly for more control.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        decision_engine: DecisionEngine | None = None,
        action_executor: ActionExecutor | None = None,
        config: AgentLoopConfig | None = None,
    ):
        """Initialize the agent loop.

        Args:
            llm: LLM provider
            memory: Optional memory provider
            decision_engine: Optional decision engine (created if not provided)
            action_executor: Optional action executor
            config: Optional loop configuration (defaults to AgentLoopConfig())
        """
        self.llm = llm
        self.memory = memory
        self.decision_engine = decision_engine or DecisionEngine(llm)
        self.action_executor = action_executor
        self.config = config or AgentLoopConfig()

        # Semantic context service (lazy-initialized)
        self._semantic_context_service = None
        self._semantic_context_initialized = False

    @property
    def semantic_context_service(self):
        """Get or create the semantic context service (lazy init).

        Returns:
            SemanticContextService instance or None if not configured
        """
        if not self._semantic_context_initialized:
            self._semantic_context_initialized = True

            if not self.config.use_semantic_context:
                return None

            SCS = _get_semantic_context_service()
            if SCS is None:
                logger.warning("SemanticContextService not available")
                return None

            try:
                from .semantic_context import SemanticContextConfig

                scs_config = SemanticContextConfig(
                    neo4j_uri=self.config.neo4j_uri,
                    neo4j_user=self.config.neo4j_user,
                    neo4j_password=self.config.neo4j_password,
                    instance_id=self.config.semantic_instance_id,
                    max_facts=self.config.max_semantic_facts,
                    max_memories=self.config.max_semantic_memories,
                    context_depth=self.config.semantic_context_depth,
                )

                self._semantic_context_service = SCS(
                    llm=self.llm,
                    memory_provider=self.memory,
                    config=scs_config,
                )
                logger.info("SemanticContextService initialized for AgentLoop")

            except Exception as e:
                logger.warning(f"Failed to initialize SemanticContextService: {e}")
                self._semantic_context_service = None

        return self._semantic_context_service

    async def process(
        self,
        query: str,
        behavior: Behavior,
        context: AgentContext,
        assistant_intro: str = "",
        mode_override: LoopMode | None = None,
    ) -> AgentResponse:
        """Process a query through the full agent loop.

        Dispatches to either simple (single-step) or ReAct (multi-step) mode
        based on configuration or query complexity.

        Args:
            query: User's query
            behavior: Behavior defining available actions
            context: Agent context (user, session, history)
            assistant_intro: Personality/identity introduction
            mode_override: Optional mode override (ignores config)

        Returns:
            AgentResponse with the result
        """
        start_time = datetime.now()

        # Clear observations from previous queries
        context.clear_observations()

        # Determine which mode to use
        mode = mode_override or self.config.mode
        if mode == LoopMode.AUTO:
            mode = self._detect_complexity(query)

        if context.debug:
            logger.debug(f"Processing query in {mode.value} mode: {query[:50]}...")

        # Dispatch to appropriate handler
        if mode == LoopMode.REACT:
            return await self._run_react(
                query=query,
                behavior=behavior,
                context=context,
                assistant_intro=assistant_intro,
                start_time=start_time,
            )
        else:
            return await self._run_simple(
                query=query,
                behavior=behavior,
                context=context,
                assistant_intro=assistant_intro,
                start_time=start_time,
            )

    def _detect_complexity(self, query: str) -> LoopMode:
        """Detect if a query requires multi-step reasoning.

        Uses keyword matching to detect queries that likely need
        multiple steps (searches, comparisons, etc.).

        Args:
            query: The user's query

        Returns:
            REACT for complex queries, SIMPLE for simple ones
        """
        query_lower = query.lower()

        # Check for complexity keywords
        keyword_matches = sum(
            1 for kw in self.config.complexity_keywords if kw in query_lower
        )

        # Calculate complexity score (0-1)
        complexity = min(keyword_matches / 3, 1.0)

        if complexity >= self.config.complexity_threshold:
            return LoopMode.REACT
        return LoopMode.SIMPLE

    async def _run_simple(
        self,
        query: str,
        behavior: Behavior,
        context: AgentContext,
        assistant_intro: str,
        start_time: datetime,
    ) -> AgentResponse:
        """Run the simple (single-step) agent loop.

        This is the original process() logic for fast single-step queries.

        Args:
            query: User's query
            behavior: Behavior defining available actions
            context: Agent context
            assistant_intro: Personality/identity introduction
            start_time: When processing started

        Returns:
            AgentResponse with the result
        """
        response = AgentResponse(
            response="", started_at=start_time, loop_mode=LoopMode.SIMPLE
        )

        try:
            # 1. Gather context
            gathered_context, memory_count = await self._gather_context(query, context)
            if context.debug:
                response.debug_info["gathered_context"] = gathered_context
                response.debug_info["memories_found"] = memory_count

            # 2. Make decision
            decision_context = DecisionContext(
                user_id=context.user_id,
                assistant_intro=assistant_intro,
                conversation_history=self._format_history(context.conversation_history),
                pending_details=context.pending_details,
                gathered_context=gathered_context,
                area_id=context.area_id,
            )

            decision = await self.decision_engine.decide(
                behavior=behavior,
                query=query,
                context=decision_context,
            )
            response.decision = decision
            response.action_taken = decision.action

            if context.debug:
                response.debug_info["decision"] = {
                    "action": decision.action,
                    "reasoning": decision.reasoning,
                    "model_tier": decision.model_tier,
                }

            # 3. Handle direct answers
            if decision.action == "answer" and decision.answer:
                response.response = decision.answer
                response.success = True
                response.latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Process memory update if present
                if decision.memory_update:
                    await self._process_memory_update(decision.memory_update, context)
                    response.memories_stored.append(
                        decision.memory_update.get("content", "")
                    )

                return response

            # 4. Execute action(s)
            if self.action_executor:
                execution_context = {
                    "user_id": context.user_id,
                    "area_id": context.area_id,
                    "pending_details": context.pending_details,
                }

                result = await self.action_executor.execute(
                    action_name=decision.action,
                    args=decision.args,
                    behavior=behavior,
                    context=execution_context,
                )
                response.tool_results.append(result)

                # Handle direct answers from actions
                if result.direct_answer:
                    response.response = result.direct_answer
                    response.success = result.success
                else:
                    # Synthesize response from tool results
                    response.response = await self._synthesize_response(
                        query=query,
                        behavior=behavior,
                        tool_results=[result],
                        assistant_intro=assistant_intro,
                        context=context,
                    )
                    response.success = result.success

                # Execute additional actions if present
                if decision.additional_actions:
                    for action_name in decision.additional_actions:
                        add_result = await self.action_executor.execute(
                            action_name=action_name,
                            args={},
                            behavior=behavior,
                            context=execution_context,
                        )
                        response.tool_results.append(add_result)
            else:
                # No executor - just return the decision answer
                response.response = decision.answer or "I processed your request."
                response.success = True

            # 5. Process memory update if present
            if decision.memory_update:
                await self._process_memory_update(decision.memory_update, context)
                response.memories_stored.append(
                    decision.memory_update.get("content", "")
                )

        except Exception as e:
            response.response = f"I encountered an error: {str(e)}"
            response.success = False
            if context.debug:
                response.debug_info["error"] = str(e)

        response.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        return response

    async def _gather_context(
        self,
        query: str,
        context: AgentContext,
    ) -> tuple[str, int]:
        """Gather context (memories, knowledge) for the query.

        Uses semantic context enrichment if configured, otherwise falls back
        to basic memory search.

        Args:
            query: User's query
            context: Agent context

        Returns:
            Tuple of (formatted context string, memory count)
        """
        context_parts = []
        total_items = 0

        # Check for additional context from adapter (e.g., self-knowledge)
        additional_context = context.metadata.get("additional_context")
        if additional_context:
            context_parts.append(additional_context)

        # ==========================================================================
        # Phase 2: Semantic Context via ReasoningLoop
        # ==========================================================================
        # Try semantic context first if configured - this uses the knowledge graph
        # to find relevant facts, entities, and relationships
        if self.semantic_context_service is not None:
            try:
                semantic_ctx = await self.semantic_context_service.enrich(
                    query=query,
                    user_id=context.user_id,
                )

                if not semantic_ctx.is_empty():
                    # Add semantic context to prompt
                    prompt_context = semantic_ctx.to_prompt_context()
                    if prompt_context:
                        context_parts.append(prompt_context)
                        total_items += semantic_ctx.context_nodes_found

                    logger.debug(
                        f"Semantic context enriched query with {semantic_ctx.context_nodes_found} nodes "
                        f"in {semantic_ctx.retrieval_time_ms:.1f}ms"
                    )

                # If semantic context already searched memory, we're done
                if semantic_ctx.used_memory_provider:
                    if context_parts:
                        return "\n\n".join(context_parts), total_items
                    return "No relevant context found.", 0

            except Exception as e:
                logger.warning(f"Semantic context enrichment failed: {e}")
                # Fall through to basic memory search

        # ==========================================================================
        # Fallback: Basic Memory Search
        # ==========================================================================
        if not self.memory:
            if context_parts:
                return "\n\n".join(context_parts), total_items
            return "No context available.", 0

        # Contextualize query for better RAG search
        search_query = self._contextualize_for_search(query, context)

        try:
            results = await self.memory.search(
                query=search_query,
                user_id=context.user_id,
                limit=5,
            )

            if results:
                memory_lines = []
                for r in results:
                    memory_lines.append(f"[{r.memory_type}] {r.content}")
                context_parts.append("\n".join(memory_lines))
                total_items += len(results)

            if not context_parts:
                return "No relevant memories found.", 0

            return "\n\n".join(context_parts), total_items

        except Exception as e:
            if context_parts:
                return "\n\n".join(context_parts), total_items
            return f"Error gathering context: {e}", 0

    def _format_history(self, history: list[dict]) -> str:
        """Format conversation history for the prompt.

        Args:
            history: List of conversation turns

        Returns:
            Formatted history string
        """
        if not history:
            return "No previous conversation."

        lines = []
        for turn in history[-5:]:  # Last 5 turns
            user = turn.get("user", "")
            assistant = turn.get("assistant", "")
            if user:
                lines.append(f"User: {user}")
            if assistant:
                lines.append(f"Assistant: {assistant}")

        return "\n".join(lines)

    def _contextualize_for_search(
        self,
        query: str,
        context: AgentContext,
    ) -> str:
        """Contextualize query for better RAG search.

        For multi-turn conversations, replace pronouns and references with
        explicit entities from conversation history.

        Args:
            query: User's query
            context: Agent context with conversation history

        Returns:
            Contextualized query for search
        """
        # If no history, return original query
        if not context.conversation_history:
            return query

        # Check if query has pronouns or references that need context
        # Simple heuristic: short queries with pronouns/references
        pronouns = ["it", "that", "this", "they", "them", "he", "she", "its"]
        references = ["next", "previous", "again", "more", "same", "other"]

        query_lower = query.lower()
        has_pronoun = any(f" {p} " in f" {query_lower} " for p in pronouns)
        has_reference = any(r in query_lower for r in references)

        if not has_pronoun and not has_reference:
            return query

        # Build context from recent history for inline expansion
        # Look at the last turn to understand what pronouns refer to
        last_turn = context.conversation_history[-1]
        last_user = last_turn.get("user", "")
        last_assistant = last_turn.get("assistant", "")

        # Simple pronoun replacement based on last context
        expanded_query = query

        # If user said "what about X" after a topic, expand to full question
        if query_lower.startswith("what about"):
            # Try to find the main topic from last question
            if "event" in last_user.lower() or "calendar" in last_user.lower():
                topic = "events"
            elif "light" in last_user.lower() or "bedroom" in last_user.lower():
                topic = "lights"
            elif "weather" in last_user.lower():
                topic = "weather"
            else:
                topic = None

            if topic:
                # "What about next week?" -> "What events do I have next week?"
                rest = query[len("what about"):].strip().rstrip("?")
                expanded_query = f"What {topic} do I have {rest}?"
                logger.debug(f"Contextualized: '{query}' -> '{expanded_query}'")

        # If query refers to "it" or "that", try to find the subject
        elif has_pronoun:
            # Look for key nouns in last exchange
            for entity in ["bedroom lights", "lights", "calendar", "timer", "event"]:
                if entity in last_user.lower() or entity in last_assistant.lower():
                    # Replace "it" with the entity
                    for p in pronouns:
                        expanded_query = expanded_query.replace(f" {p} ", f" the {entity} ")
                        expanded_query = expanded_query.replace(f" {p}.", f" the {entity}.")
                        expanded_query = expanded_query.replace(f" {p}?", f" the {entity}?")
                    logger.debug(f"Contextualized: '{query}' -> '{expanded_query}'")
                    break

        return expanded_query

    async def _synthesize_response(
        self,
        query: str,
        behavior: Behavior,
        tool_results: list[ActionResult],
        assistant_intro: str,
        context: AgentContext,
    ) -> str:
        """Synthesize a response from tool results.

        Args:
            query: Original query
            behavior: Behavior with synthesis prompt
            tool_results: Results from tool execution
            assistant_intro: Personality introduction
            context: Agent context

        Returns:
            Synthesized response
        """
        if not behavior.prompts:
            # No synthesis prompt - use formatted results
            if tool_results:
                return tool_results[0].formatted_result
            return "I completed your request."

        # Format tool results
        results_text = ""
        for result in tool_results:
            if result.success:
                results_text += f"{result.action_name}: {result.formatted_result}\n"
            else:
                results_text += f"{result.action_name}: Error - {result.error}\n"

        # Build synthesis prompt
        prompt = behavior.prompts.synthesis_prompt.format(
            assistant_intro=assistant_intro,
            user_id=context.user_id,
            question=query,
            tool_results=results_text,
        )

        # Get LLM response
        messages = [
            LLMMessage(role="user", content=prompt),
        ]

        response = await self.llm.chat(messages)

        # Parse structured response (XML or JSON)
        content = response.content.strip()

        # Try XML first (preferred format per LLM-First architecture)
        if "<synthesis>" in content or "<answer>" in content:
            try:
                import re
                import xml.etree.ElementTree as ET

                # Extract XML block if wrapped in markdown
                xml_match = re.search(r"<synthesis>.*?</synthesis>", content, re.DOTALL)
                if xml_match:
                    xml_str = xml_match.group(0)
                    root = ET.fromstring(xml_str)
                    answer = root.find("answer")
                    if answer is not None and answer.text:
                        return answer.text.strip()
                # Try bare <answer> tag
                answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                if answer_match:
                    return answer_match.group(1).strip()
            except Exception:
                pass

        # Fall back to JSON parsing for backward compatibility
        if content.startswith("{"):
            try:
                import json

                data = json.loads(content)
                return data.get("answer", content)
            except Exception:
                pass

        return content

    async def _process_memory_update(
        self,
        memory_update: dict[str, Any],
        context: AgentContext,
    ) -> None:
        """Process a memory update from the decision.

        Args:
            memory_update: Memory update data
            context: Agent context
        """
        if not self.memory:
            logger.warning("[MEMORY] No memory provider, cannot store update")
            return

        content = memory_update.get("content")
        if not content:
            logger.warning("[MEMORY] No content in memory update")
            return

        try:
            logger.info(f"[MEMORY] Storing: {content[:100]}... (type={memory_update.get('type', 'fact')})")
            await self.memory.store(
                content=content,
                user_id=context.user_id,
                memory_type=memory_update.get("type", "fact"),
                entities=memory_update.get("entities", []),
                importance=memory_update.get("confidence", 0.9),
            )
            logger.info(f"[MEMORY] Successfully stored memory for {context.user_id}")
        except Exception as e:
            # Log but don't fail the response
            logger.error(f"[MEMORY] Failed to store: {e}")

    async def _run_react(
        self,
        query: str,
        behavior: Behavior,
        context: AgentContext,
        assistant_intro: str,
        start_time: datetime,
    ) -> AgentResponse:
        """Run the ReAct (multi-step reasoning) agent loop.

        Implements the ReAct pattern:
        1. THOUGHT: Reason about what to do next
        2. ACTION: Execute the decided action
        3. OBSERVATION: Record the result
        4. Repeat until FINAL_ANSWER or max_iterations

        Args:
            query: User's query
            behavior: Behavior defining available actions
            context: Agent context
            assistant_intro: Personality/identity introduction
            start_time: When processing started

        Returns:
            AgentResponse with the result and thought trace
        """
        response = AgentResponse(
            response="", started_at=start_time, loop_mode=LoopMode.REACT
        )

        # 1. Gather initial context
        gathered_context, memory_count = await self._gather_context(query, context)
        if context.debug:
            response.debug_info["gathered_context"] = gathered_context
            response.debug_info["memories_found"] = memory_count

        # ReAct loop
        for iteration in range(self.config.max_iterations):
            response.iterations_used = iteration + 1
            iteration_start = datetime.now()

            try:
                # Apply per-iteration timeout
                async with async_timeout(self.config.iteration_timeout_seconds):
                    # THOUGHT: Decide what to do next
                    thought_start = datetime.now()
                    decision_context = DecisionContext(
                        user_id=context.user_id,
                        assistant_intro=assistant_intro,
                        conversation_history=self._format_history(
                            context.conversation_history
                        ),
                        pending_details=context.pending_details,
                        gathered_context=self._build_react_context(
                            gathered_context, context
                        ),
                        area_id=context.area_id,
                    )

                    decision = await self.decision_engine.decide(
                        behavior=behavior,
                        query=query,
                        context=decision_context,
                    )
                    thought_duration = (
                        datetime.now() - thought_start
                    ).total_seconds() * 1000

                    # Record the THOUGHT step
                    response.add_react_step(
                        step_type=StepType.THOUGHT,
                        content=decision.reasoning or "Deciding action...",
                        duration_ms=thought_duration,
                    )

                    if self.config.log_thought_traces:
                        logger.debug(
                            f"THOUGHT [{iteration + 1}]: {decision.reasoning}"
                        )

                    # Check for FINAL_ANSWER
                    if decision.action == "answer" and decision.answer:
                        response.add_react_step(
                            step_type=StepType.FINAL_ANSWER,
                            content=decision.answer,
                        )

                        if self.config.log_thought_traces:
                            logger.debug(f"FINAL_ANSWER: {decision.answer}")

                        response.response = decision.answer
                        response.decision = decision
                        response.action_taken = "answer"
                        response.success = True

                        # Process any memory updates
                        if decision.memory_update:
                            await self._process_memory_update(
                                decision.memory_update, context
                            )
                            response.memories_stored.append(
                                decision.memory_update.get("content", "")
                            )

                        response.latency_ms = (
                            datetime.now() - start_time
                        ).total_seconds() * 1000

                        if context.debug:
                            response.debug_info["thought_trace"] = (
                                response.get_thought_trace()
                            )

                        return response

                    # ACTION: Execute the decided action
                    action_start = datetime.now()
                    response.add_react_step(
                        step_type=StepType.ACTION,
                        content=f"{decision.action}({decision.args})",
                        action_name=decision.action,
                        action_args=decision.args,
                    )

                    if self.config.log_thought_traces:
                        logger.debug(
                            f"ACTION [{iteration + 1}]: {decision.action}({decision.args})"
                        )

                    if self.action_executor:
                        execution_context = {
                            "user_id": context.user_id,
                            "area_id": context.area_id,
                            "pending_details": context.pending_details,
                        }

                        result = await self.action_executor.execute(
                            action_name=decision.action,
                            args=decision.args,
                            behavior=behavior,
                            context=execution_context,
                        )
                        action_duration = (
                            datetime.now() - action_start
                        ).total_seconds() * 1000
                        response.tool_results.append(result)

                        # OBSERVATION: Record the result
                        observation_content = (
                            result.formatted_result
                            if result.success
                            else f"Error: {result.error}"
                        )
                        response.add_react_step(
                            step_type=StepType.OBSERVATION,
                            content=observation_content,
                            duration_ms=action_duration,
                            success=result.success,
                            error=result.error,
                        )

                        # Add observation to context for next iteration
                        context.add_observation(observation_content)

                        if self.config.log_thought_traces:
                            logger.debug(
                                f"OBSERVATION [{iteration + 1}]: {observation_content[:100]}..."
                            )
                    else:
                        # No executor - log warning
                        logger.warning("No action executor available for ReAct loop")
                        context.add_observation("No action executor available")

                    # Record decision for later
                    response.decision = decision
                    response.action_taken = decision.action

            except asyncio.TimeoutError:
                logger.warning(
                    f"ReAct iteration {iteration + 1} timed out after "
                    f"{self.config.iteration_timeout_seconds}s"
                )
                response.add_react_step(
                    step_type=StepType.OBSERVATION,
                    content=f"Timeout after {self.config.iteration_timeout_seconds}s",
                    success=False,
                    error="timeout",
                )
                context.add_observation("Previous step timed out")
                continue

            except Exception as e:
                logger.error(f"ReAct iteration {iteration + 1} error: {e}")
                response.add_react_step(
                    step_type=StepType.OBSERVATION,
                    content=f"Error: {str(e)}",
                    success=False,
                    error=str(e),
                )
                context.add_observation(f"Error: {str(e)}")
                # Continue to next iteration to try to recover
                continue

        # Max iterations reached without final answer
        logger.warning(
            f"ReAct loop reached max iterations ({self.config.max_iterations}) "
            f"without final answer"
        )

        response.response = (
            f"I couldn't complete this task in {self.config.max_iterations} steps. "
            f"Here's what I found: {context.get_observations_text()}"
        )
        response.success = False

        response.latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        if context.debug:
            response.debug_info["thought_trace"] = response.get_thought_trace()
            response.debug_info["max_iterations_reached"] = True

        return response

    def _build_react_context(
        self,
        gathered_context: str,
        context: AgentContext,
    ) -> str:
        """Build context string for ReAct including observations.

        Args:
            gathered_context: Initial context from memory search
            context: Agent context with accumulated observations

        Returns:
            Combined context string
        """
        parts = [gathered_context]

        observations_text = context.get_observations_text()
        if observations_text:
            parts.append("\n\n## Previous Observations in this reasoning chain:")
            parts.append(observations_text)

        return "\n".join(parts)

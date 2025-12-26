"""Multi-Agent Orchestrator for AGI-Lite architecture.

The orchestrator coordinates multiple agents working together:
- Sequential: Agents execute in order, passing context
- Parallel: Agents execute simultaneously (Phase C.4)
- Handoff: One agent delegates to another (Phase C.4)
- Collaborative: Agents share working memory (Phase C.4)

Phase C.1 implements sequential mode only.

Based on research from:
- Multi-agent orchestration patterns
- LangGraph and CrewAI architectures
- Microsoft AutoGen patterns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
import asyncio
import logging
import uuid

from .learning_channel import (
    LearningChannel,
    Learning,
    LearningType,
    LearningScope,
    get_learning_channel,
)

logger = logging.getLogger(__name__)


class OrchestrationMode(str, Enum):
    """How agents are coordinated."""

    SEQUENTIAL = "sequential"    # One after another
    PARALLEL = "parallel"        # All at once (C.4)
    HANDOFF = "handoff"          # Explicit delegation (C.4)
    COLLABORATIVE = "collaborative"  # Shared memory (C.4)


class AgentRole(str, Enum):
    """Common agent roles in multi-agent systems."""

    PRIMARY = "primary"          # Main responding agent
    RESEARCHER = "researcher"    # Gathers information
    CRITIC = "critic"            # Reviews and critiques
    EXECUTOR = "executor"        # Takes actions
    PLANNER = "planner"          # Plans multi-step tasks
    SPECIALIST = "specialist"    # Domain expert


@dataclass
class AgentSpec:
    """Specification for an agent in the orchestration.

    This is a lightweight reference to an agent, not the full agent.
    The actual agent implementation is provided by the host application.
    """

    agent_id: str
    name: str = ""
    role: AgentRole = AgentRole.PRIMARY
    description: str = ""

    # Scope for learning contributions
    contributes_to_scope: str = "context"

    # Execution hints
    timeout_seconds: float = 30.0
    max_retries: int = 1
    required: bool = True  # Fail task if this agent fails

    # Conditions (evaluated by orchestrator)
    run_condition: str | None = None  # Expression like "prev.success == True"


@dataclass
class TaskContext:
    """Context passed between agents during orchestration."""

    # Task identity
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Original request
    query: str = ""
    user_id: str = ""
    session_id: str | None = None

    # Accumulated context from previous agents
    agent_outputs: dict[str, Any] = field(default_factory=dict)

    # Shared working memory for this task
    working_memory: dict[str, Any] = field(default_factory=dict)

    # Learnings discovered during the task
    learnings: list[Learning] = field(default_factory=list)

    # Execution state
    current_agent_index: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Flags
    should_stop: bool = False
    stop_reason: str | None = None

    def add_learning(
        self,
        content: str,
        learning_type: LearningType,
        source_agent_id: str,
        **kwargs: Any,
    ) -> Learning:
        """Add a learning discovered during task execution.

        Args:
            content: What was learned
            learning_type: Type of learning
            source_agent_id: Agent that learned it
            **kwargs: Additional learning properties

        Returns:
            The created Learning
        """
        learning = Learning(
            content=content,
            learning_type=learning_type,
            source_agent_id=source_agent_id,
            **kwargs,
        )
        self.learnings.append(learning)
        return learning


@dataclass
class AgentResult:
    """Result from a single agent's execution."""

    agent_id: str
    success: bool
    output: Any = None
    error: str | None = None

    # Learnings from this agent
    learnings: list[Learning] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = field(default_factory=datetime.now)

    @property
    def duration_ms(self) -> float:
        """Execution duration in milliseconds."""
        return (self.completed_at - self.started_at).total_seconds() * 1000


@dataclass
class OrchestratorResult:
    """Result from the full orchestration."""

    task_id: str
    success: bool
    final_output: Any = None
    error: str | None = None

    # All agent results
    agent_results: list[AgentResult] = field(default_factory=list)

    # Aggregated learnings
    learnings: list[Learning] = field(default_factory=list)

    # Context at completion
    final_context: TaskContext | None = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = field(default_factory=datetime.now)

    @property
    def duration_ms(self) -> float:
        """Total orchestration duration in milliseconds."""
        return (self.completed_at - self.started_at).total_seconds() * 1000


# Type for agent execution function
AgentExecutor = Callable[[AgentSpec, TaskContext], Awaitable[AgentResult]]


class MultiAgentOrchestrator:
    """Orchestrates multiple agents working on a task.

    Phase C.1: Sequential mode only.

    Example:
        orchestrator = MultiAgentOrchestrator()

        # Define agent pipeline
        agents = [
            AgentSpec(agent_id="researcher", role=AgentRole.RESEARCHER),
            AgentSpec(agent_id="writer", role=AgentRole.PRIMARY),
            AgentSpec(agent_id="critic", role=AgentRole.CRITIC),
        ]

        # Execute pipeline
        result = await orchestrator.orchestrate(
            agents=agents,
            context=TaskContext(query="Write about AI"),
            executor=my_agent_executor,
        )
    """

    def __init__(
        self,
        learning_channel: LearningChannel | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            learning_channel: Channel for sharing learnings
        """
        self._channel = learning_channel or get_learning_channel()

    async def orchestrate(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        executor: AgentExecutor,
        mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
    ) -> OrchestratorResult:
        """Execute an orchestration.

        Args:
            agents: List of agents to orchestrate
            context: Initial task context
            executor: Function to execute each agent
            mode: Orchestration mode

        Returns:
            Orchestration result
        """
        if mode == OrchestrationMode.SEQUENTIAL:
            return await self._sequential(agents, context, executor)
        elif mode == OrchestrationMode.PARALLEL:
            return await self._parallel(agents, context, executor)
        elif mode == OrchestrationMode.HANDOFF:
            return await self._handoff(agents, context, executor)
        elif mode == OrchestrationMode.COLLABORATIVE:
            return await self._collaborative(agents, context, executor)
        else:
            raise ValueError(f"Unknown orchestration mode: {mode}")

    async def _sequential(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Execute agents sequentially.

        Each agent receives the output of previous agents in the context.
        If an agent fails and is required, the pipeline stops.

        Args:
            agents: Agents to execute in order
            context: Task context
            executor: Agent execution function

        Returns:
            Orchestration result
        """
        logger.info(f"Starting sequential orchestration: task={context.task_id}, agents={len(agents)}")

        result = OrchestratorResult(
            task_id=context.task_id,
            success=True,
        )

        for i, agent in enumerate(agents):
            context.current_agent_index = i

            # Check run condition
            if agent.run_condition and not self._evaluate_condition(agent.run_condition, context):
                logger.info(f"Skipping agent {agent.agent_id}: condition not met")
                continue

            # Check if we should stop
            if context.should_stop:
                logger.info(f"Stopping orchestration: {context.stop_reason}")
                break

            # Execute agent with timeout and retries
            agent_result = await self._execute_with_retry(
                agent,
                context,
                executor,
            )

            result.agent_results.append(agent_result)

            # Store output in context for next agent
            context.agent_outputs[agent.agent_id] = agent_result.output

            # Collect learnings
            result.learnings.extend(agent_result.learnings)

            # Handle failure
            if not agent_result.success:
                if agent.required:
                    result.success = False
                    result.error = f"Required agent {agent.agent_id} failed: {agent_result.error}"
                    logger.error(result.error)
                    break
                else:
                    logger.warning(f"Optional agent {agent.agent_id} failed: {agent_result.error}")

        # Finalize
        result.completed_at = datetime.now()
        context.completed_at = datetime.now()
        result.final_context = context

        # Set final output to last successful agent's output
        for agent_result in reversed(result.agent_results):
            if agent_result.success and agent_result.output is not None:
                result.final_output = agent_result.output
                break

        # Broadcast learnings
        await self._broadcast_learnings(result.learnings)

        logger.info(
            f"Sequential orchestration complete: task={context.task_id}, "
            f"success={result.success}, duration={result.duration_ms:.1f}ms"
        )

        return result

    async def _execute_with_retry(
        self,
        agent: AgentSpec,
        context: TaskContext,
        executor: AgentExecutor,
    ) -> AgentResult:
        """Execute an agent with timeout and retry logic.

        Args:
            agent: Agent to execute
            context: Task context
            executor: Execution function

        Returns:
            Agent result
        """
        last_error = None

        for attempt in range(agent.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    executor(agent, context),
                    timeout=agent.timeout_seconds,
                )

                if result.success:
                    return result

                last_error = result.error

            except asyncio.TimeoutError:
                last_error = f"Agent {agent.agent_id} timed out after {agent.timeout_seconds}s"
                logger.warning(f"Attempt {attempt + 1}: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}: Agent {agent.agent_id} error: {e}")

            # Wait before retry (exponential backoff)
            if attempt < agent.max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))

        # All retries failed
        return AgentResult(
            agent_id=agent.agent_id,
            success=False,
            error=last_error,
        )

    async def _broadcast_learnings(self, learnings: list[Learning]) -> None:
        """Broadcast learnings through the learning channel.

        Args:
            learnings: Learnings to broadcast
        """
        for learning in learnings:
            try:
                await self._channel.broadcast(learning)
            except Exception as e:
                logger.error(f"Failed to broadcast learning: {e}")

    def _evaluate_condition(self, condition: str, context: TaskContext) -> bool:
        """Evaluate a run condition expression safely.

        Supports simple expressions like:
        - "prev.success == True"
        - "agent_outputs.researcher != None"
        - "prev.output.status == 'complete'"

        Uses a safe expression parser instead of eval() to prevent code injection.

        Args:
            condition: Condition expression
            context: Task context

        Returns:
            True if condition is met
        """
        # Build evaluation context
        eval_context: dict[str, Any] = {
            "context": context,
            "agent_outputs": context.agent_outputs,
            "working_memory": context.working_memory,
        }

        # Add "prev" as last agent result
        if context.agent_outputs:
            last_key = list(context.agent_outputs.keys())[-1]
            eval_context["prev"] = context.agent_outputs.get(last_key)

        try:
            return self._safe_eval(condition, eval_context)
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return True  # Default to running

    def _safe_eval(self, expression: str, context: dict[str, Any]) -> bool:
        """Safely evaluate a simple boolean expression.

        Supports:
        - Comparisons: ==, !=, <, >, <=, >=
        - Literals: True, False, None, integers, strings (single or double quoted)
        - Attribute access: obj.attr, obj.attr.subattr
        - Dict access: dict.key, dict['key']
        - Boolean operators: and, or, not

        Does NOT support:
        - Function calls
        - List/dict comprehensions
        - Arbitrary code execution

        Args:
            expression: Expression to evaluate
            context: Variable context

        Returns:
            Boolean result

        Raises:
            ValueError: If expression is malformed or unsafe
        """
        import re

        # Normalize whitespace
        expression = expression.strip()

        # Handle boolean operators (lowest precedence, evaluate left to right)
        # Split on ' and ' or ' or ' while respecting string literals
        and_parts = self._split_preserving_strings(expression, " and ")
        if len(and_parts) > 1:
            return all(self._safe_eval(part.strip(), context) for part in and_parts)

        or_parts = self._split_preserving_strings(expression, " or ")
        if len(or_parts) > 1:
            return any(self._safe_eval(part.strip(), context) for part in or_parts)

        # Handle 'not' prefix
        if expression.startswith("not "):
            return not self._safe_eval(expression[4:].strip(), context)

        # Handle parentheses
        if expression.startswith("(") and expression.endswith(")"):
            # Check if these are matching parens
            depth = 0
            for i, c in enumerate(expression):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth == 0 and i < len(expression) - 1:
                    break  # Found matching paren before end
            else:
                # Parens match the whole expression
                return self._safe_eval(expression[1:-1].strip(), context)

        # Handle comparisons
        comparisons = [
            ("==", lambda a, b: a == b),
            ("!=", lambda a, b: a != b),
            ("<=", lambda a, b: a <= b),
            (">=", lambda a, b: a >= b),
            ("<", lambda a, b: a < b),
            (">", lambda a, b: a > b),
        ]

        for op, func in comparisons:
            # Split on operator, but not inside strings
            parts = self._split_preserving_strings(expression, op)
            if len(parts) == 2:
                left = self._resolve_value(parts[0].strip(), context)
                right = self._resolve_value(parts[1].strip(), context)
                return func(left, right)

        # No comparison found - evaluate as truthy value
        return bool(self._resolve_value(expression, context))

    def _split_preserving_strings(self, text: str, delimiter: str) -> list[str]:
        """Split text on delimiter, but preserve string literals.

        Args:
            text: Text to split
            delimiter: Delimiter to split on

        Returns:
            List of parts
        """
        parts = []
        current = ""
        in_string = None
        i = 0

        while i < len(text):
            # Check for string delimiters
            if text[i] in "\"'" and (i == 0 or text[i-1] != "\\"):
                if in_string is None:
                    in_string = text[i]
                elif in_string == text[i]:
                    in_string = None

            # Check for delimiter (only if not in string)
            if in_string is None and text[i:i+len(delimiter)] == delimiter:
                parts.append(current)
                current = ""
                i += len(delimiter)
                continue

            current += text[i]
            i += 1

        parts.append(current)
        return parts

    def _resolve_value(self, token: str, context: dict[str, Any]) -> Any:
        """Resolve a token to its value.

        Handles:
        - Literals: True, False, None, integers, floats, quoted strings
        - Variable access: name, name.attr, name.attr.subattr
        - Dict access: name['key'] or name["key"]

        Args:
            token: Token to resolve
            context: Variable context

        Returns:
            Resolved value

        Raises:
            ValueError: If token is invalid
        """
        import re

        token = token.strip()

        # Boolean literals
        if token == "True":
            return True
        if token == "False":
            return False
        if token == "None":
            return None

        # Integer literal
        if re.match(r"^-?\d+$", token):
            return int(token)

        # Float literal
        if re.match(r"^-?\d+\.\d+$", token):
            return float(token)

        # String literal (single or double quotes)
        if (token.startswith("'") and token.endswith("'")) or \
           (token.startswith('"') and token.endswith('"')):
            return token[1:-1]

        # Variable/attribute access (e.g., prev.success, agent_outputs.researcher.status)
        # Also handle dict access like agent_outputs['key']
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*", token):
            return self._resolve_path(token, context)

        raise ValueError(f"Invalid token: {token}")

    def _resolve_path(self, path: str, context: dict[str, Any]) -> Any:
        """Resolve a dotted path like 'prev.success' or 'agent_outputs["key"]'.

        Args:
            path: Dotted path
            context: Variable context

        Returns:
            Resolved value
        """
        import re

        # Parse the path into segments
        # Handle both . access and ['key'] access
        segments = []
        current = ""
        i = 0

        while i < len(path):
            if path[i] == ".":
                if current:
                    segments.append(("attr", current))
                    current = ""
                i += 1
            elif path[i] == "[":
                if current:
                    segments.append(("attr", current))
                    current = ""
                # Find closing bracket
                j = i + 1
                in_string = None
                while j < len(path):
                    if path[j] in "\"'" and (j == i + 1 or path[j-1] != "\\"):
                        if in_string is None:
                            in_string = path[j]
                        elif in_string == path[j]:
                            in_string = None
                    elif path[j] == "]" and in_string is None:
                        break
                    j += 1
                key = path[i+1:j]
                # Remove quotes if present
                if (key.startswith("'") and key.endswith("'")) or \
                   (key.startswith('"') and key.endswith('"')):
                    key = key[1:-1]
                segments.append(("key", key))
                i = j + 1
            else:
                current += path[i]
                i += 1

        if current:
            segments.append(("attr", current))

        # Resolve the segments
        if not segments:
            raise ValueError(f"Empty path: {path}")

        # First segment must be a context variable
        first_type, first_name = segments[0]
        if first_name not in context:
            raise ValueError(f"Unknown variable: {first_name}")

        value = context[first_name]

        # Resolve remaining segments
        for seg_type, seg_name in segments[1:]:
            if value is None:
                return None  # Short-circuit on None

            if seg_type == "attr":
                if hasattr(value, seg_name):
                    value = getattr(value, seg_name)
                elif isinstance(value, dict) and seg_name in value:
                    value = value[seg_name]
                else:
                    raise ValueError(f"Cannot access '{seg_name}' on {type(value).__name__}")
            elif seg_type == "key":
                if isinstance(value, dict):
                    value = value.get(seg_name)
                else:
                    raise ValueError(f"Cannot use key access on {type(value).__name__}")

        return value

    # =========================================================================
    # Phase C.4 Stubs
    # =========================================================================

    async def _parallel(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Execute agents in parallel (Phase C.4 stub).

        Currently falls back to sequential.
        """
        logger.warning("Parallel mode not implemented in C.1, falling back to sequential")
        return await self._sequential(agents, context, executor)

    async def _handoff(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Execute with explicit handoff (Phase C.4 stub).

        Currently falls back to sequential.
        """
        logger.warning("Handoff mode not implemented in C.1, falling back to sequential")
        return await self._sequential(agents, context, executor)

    async def _collaborative(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        executor: AgentExecutor,
    ) -> OrchestratorResult:
        """Execute with shared working memory (Phase C.4 stub).

        Currently falls back to sequential.
        """
        logger.warning("Collaborative mode not implemented in C.1, falling back to sequential")
        return await self._sequential(agents, context, executor)


# Convenience function for simple single-agent execution
async def execute_single_agent(
    agent: AgentSpec,
    query: str,
    user_id: str,
    executor: AgentExecutor,
    **kwargs: Any,
) -> AgentResult:
    """Execute a single agent (convenience wrapper).

    Args:
        agent: Agent to execute
        query: User query
        user_id: User ID
        executor: Execution function
        **kwargs: Additional context

    Returns:
        Agent result
    """
    context = TaskContext(
        query=query,
        user_id=user_id,
        **kwargs,
    )

    orchestrator = MultiAgentOrchestrator()
    result = await orchestrator.orchestrate(
        agents=[agent],
        context=context,
        executor=executor,
    )

    if result.agent_results:
        return result.agent_results[0]

    return AgentResult(
        agent_id=agent.agent_id,
        success=False,
        error="No agent results",
    )

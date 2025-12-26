"""Main agent loop.

The agent loop ties together all components to process queries:
1. Gather context (memories, knowledge)
2. Make decision (which action to take)
3. Execute action (using tools)
4. Synthesize response (format for output)
5. Post-processing (learning, metrics)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..behaviors import Action, Behavior
from .protocols import LLMMessage, LLMProvider, MemoryProvider
from .decision import DecisionContext, DecisionEngine, DecisionResult
from .execution import ActionExecutor, ActionResult


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

    # Metadata
    debug: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


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

    # Timing
    latency_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)

    # Memory operations
    memories_stored: list[str] = field(default_factory=list)

    # Debug info
    debug_info: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """The main agent processing loop.

    This class orchestrates the entire agent interaction:
    query -> context -> decision -> execution -> synthesis -> response

    It is designed to be used by the Agent class but can also
    be used directly for more control.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        decision_engine: DecisionEngine | None = None,
        action_executor: ActionExecutor | None = None,
    ):
        """Initialize the agent loop.

        Args:
            llm: LLM provider
            memory: Optional memory provider
            decision_engine: Optional decision engine (created if not provided)
            action_executor: Optional action executor
        """
        self.llm = llm
        self.memory = memory
        self.decision_engine = decision_engine or DecisionEngine(llm)
        self.action_executor = action_executor

    async def process(
        self,
        query: str,
        behavior: Behavior,
        context: AgentContext,
        assistant_intro: str = "",
    ) -> AgentResponse:
        """Process a query through the full agent loop.

        Args:
            query: User's query
            behavior: Behavior defining available actions
            context: Agent context (user, session, history)
            assistant_intro: Personality/identity introduction

        Returns:
            AgentResponse with the result
        """
        start_time = datetime.now()
        response = AgentResponse(response="", started_at=start_time)

        try:
            # 1. Gather context
            gathered_context = await self._gather_context(query, context)
            if context.debug:
                response.debug_info["gathered_context"] = gathered_context

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
    ) -> str:
        """Gather context (memories, knowledge) for the query.

        Args:
            query: User's query
            context: Agent context

        Returns:
            Formatted context string
        """
        if not self.memory:
            return "No context available."

        try:
            results = await self.memory.search(
                query=query,
                user_id=context.user_id,
                limit=5,
            )

            if not results:
                return "No relevant memories found."

            lines = []
            for r in results:
                lines.append(f"[{r.memory_type}] {r.content}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error gathering context: {e}"

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

        # Parse JSON response if present
        content = response.content.strip()
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
            return

        content = memory_update.get("content")
        if not content:
            return

        try:
            await self.memory.store(
                content=content,
                user_id=context.user_id,
                memory_type=memory_update.get("type", "fact"),
                entities=memory_update.get("entities", []),
                importance=memory_update.get("confidence", 0.9),
            )
        except Exception:
            # Log but don't fail the response
            pass

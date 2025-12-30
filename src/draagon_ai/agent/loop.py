"""Main agent loop with Roxy patterns.

This module provides the core agent loop that orchestrates:
- Fast-path routing for common queries
- Query contextualization for multi-turn conversations
- Context gathering (RAG, cross-chat, episodic)
- LLM decision making with multi-modal context
- Tool execution with undo support
- Async post-response reflection and learning
- Memory reinforcement
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from draagon_ai.conversation import (
    ConversationManager,
    ConversationManagerConfig,
    ConversationState,
    UndoableAction,
)
from draagon_ai.reflection import ReflectionService
from draagon_ai.retrieval import HybridRetriever, QueryContextualizer

from .fast_router import FastRouter, FastRouteHandler
from .models import (
    AgentRequest,
    AgentResponse,
    DebugInfo,
    FastRouteResult,
    ModelTier,
    ToolCallInfo,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat request to the LLM."""
        ...


class ToolExecutor(Protocol):
    """Protocol for tool executors."""

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Execute a tool and return the result."""
        ...


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop.

    Attributes:
        enable_fast_router: Enable fast-path routing
        enable_contextualization: Enable query contextualization
        enable_retrieval: Enable RAG context gathering
        enable_reflection: Enable post-response reflection
        enable_learning: Enable autonomous learning
        enable_memory_reinforcement: Enable memory importance boosting
        max_history_messages: Maximum conversation history length
        conversation_timeout_minutes: Conversation expiry timeout
        reflection_async: Run reflection asynchronously
        model_fast: Model for fast-path queries
        model_standard: Model for standard queries
        model_complex: Model for complex reasoning
    """

    enable_fast_router: bool = True
    enable_contextualization: bool = True
    enable_retrieval: bool = True
    enable_reflection: bool = True
    enable_learning: bool = True
    enable_memory_reinforcement: bool = True
    max_history_messages: int = 20
    conversation_timeout_minutes: int = 15
    reflection_async: bool = True
    model_fast: str = "llama-3.1-8b-instant"
    model_standard: str = "openai/gpt-oss-20b"
    model_complex: str = "llama-3.3-70b-versatile"


class DecisionHandler(ABC):
    """Abstract base class for handling LLM decisions.

    Implement this to create custom decision handlers for specific
    action types.
    """

    @property
    @abstractmethod
    def action_type(self) -> str:
        """Action type this handler processes."""
        pass

    @abstractmethod
    async def handle(
        self,
        decision: dict[str, Any],
        request: AgentRequest,
        context: dict[str, Any],
    ) -> AgentResponse:
        """Handle the decision and return a response."""
        pass


class AgentLoop:
    """Main agent loop orchestrating the cognitive pipeline.

    The agent loop implements a multi-phase processing pipeline:

    1. **Fast-path check** - Skip expensive operations for simple queries
    2. **Parallel init** - Sentiment, mode detection, profile updates
    3. **Query contextualization** - Rewrite ambiguous queries
    4. **Context gathering** - RAG, cross-chat, episodic memory
    5. **Decision making** - LLM determines action and response
    6. **Action execution** - Execute tools, gather results
    7. **Response synthesis** - Format response for output
    8. **Async post-processing** - Reflection, learning, reinforcement

    Usage:
        loop = AgentLoop(
            llm=groq_client,
            retriever=hybrid_retriever,
            config=AgentLoopConfig(),
        )

        # Register handlers
        loop.register_fast_handler(TimeHandler())
        loop.register_decision_handler(CalendarHandler())

        # Process request
        response = await loop.process(AgentRequest(
            query="What's on my calendar?",
            user_id="user-123",
        ))
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: HybridRetriever | None = None,
        contextualizer: QueryContextualizer | None = None,
        reflection: ReflectionService | None = None,
        tool_executor: ToolExecutor | None = None,
        config: AgentLoopConfig | None = None,
    ):
        """Initialize the agent loop.

        Args:
            llm: LLM provider for chat completions
            retriever: Hybrid retriever for context gathering
            contextualizer: Query contextualizer for multi-turn
            reflection: Reflection service for quality assessment
            tool_executor: Tool executor for action execution
            config: Configuration options
        """
        self.llm = llm
        self.retriever = retriever
        self.contextualizer = contextualizer
        self.reflection = reflection
        self.tool_executor = tool_executor
        self.config = config or AgentLoopConfig()

        # Initialize components
        self._fast_router = FastRouter()
        self._conversations = ConversationManager(
            config=ConversationManagerConfig(
                max_history=self.config.max_history_messages,
                timeout_seconds=self.config.conversation_timeout_minutes * 60,
            )
        )
        self._decision_handlers: dict[str, DecisionHandler] = {}

        # Callbacks for extensibility
        self._on_response_callbacks: list[Callable] = []
        self._on_learning_callbacks: list[Callable] = []

    # =========================================================================
    # Handler Registration
    # =========================================================================

    def register_fast_handler(self, handler: FastRouteHandler) -> None:
        """Register a fast-path route handler."""
        self._fast_router.register_handler(handler)

    def register_decision_handler(self, handler: DecisionHandler) -> None:
        """Register a decision handler for an action type."""
        self._decision_handlers[handler.action_type] = handler
        logger.debug(f"Registered decision handler: {handler.action_type}")

    def on_response(self, callback: Callable) -> None:
        """Register a callback for post-response processing."""
        self._on_response_callbacks.append(callback)

    def on_learning(self, callback: Callable) -> None:
        """Register a callback for learning events."""
        self._on_learning_callbacks.append(callback)

    # =========================================================================
    # Main Processing Pipeline
    # =========================================================================

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a request through the agent loop.

        This is the main entry point for processing user queries.

        Args:
            request: The agent request to process

        Returns:
            AgentResponse with the result
        """
        start_time = time.time()
        debug_info = DebugInfo() if request.debug else None

        try:
            # Get or create conversation state
            conversation = self._conversations.get_or_create(
                request.conversation_id or f"{request.user_id}_default"
            )

            # Update device/area context
            if request.area_id:
                conversation.current_area_id = request.area_id
            if request.device_id:
                conversation.current_device_id = request.device_id

            # Build context dict for handlers
            context = self._build_context(request, conversation)

            # Phase 1: Fast-path routing
            if self.config.enable_fast_router:
                fast_result = await self._try_fast_route(request, context, debug_info)
                if fast_result:
                    response = fast_result.to_agent_response(debug_info)
                    await self._post_process(request, response, conversation, fast_result.skip_reflection)
                    return response

            # Phase 2: Parallel initialization
            parallel_results = await self._parallel_init(request, conversation, debug_info)

            # Phase 3: Query contextualization
            query, contextualized = await self._contextualize_query(
                request.query, conversation, debug_info
            )

            # Phase 4: Context gathering (RAG)
            retrieved_context, memory_ids = await self._gather_context(
                query, request.user_id, debug_info
            )

            # Phase 5: Make decision
            decision = await self._make_decision(
                query=query,
                context=retrieved_context,
                conversation=conversation,
                request=request,
                debug_info=debug_info,
            )

            # Phase 6: Execute action
            response = await self._execute_action(
                decision=decision,
                request=request,
                context=context,
                debug_info=debug_info,
            )

            # Track memory IDs for reinforcement
            if memory_ids:
                conversation.metadata["retrieved_memory_ids"] = memory_ids

            # Phase 7: Post-process (async reflection, learning)
            await self._post_process(request, response, conversation)

            # Record timing
            if debug_info:
                debug_info.timings["total_ms"] = int((time.time() - start_time) * 1000)

            return response

        except Exception as e:
            logger.exception(f"Agent loop error: {e}")
            return AgentResponse(
                response="I encountered an error processing your request.",
                success=False,
                debug=debug_info,
                metadata={"error": str(e)},
            )

    # =========================================================================
    # Phase 1: Fast-Path Routing
    # =========================================================================

    async def _try_fast_route(
        self,
        request: AgentRequest,
        context: dict[str, Any],
        debug_info: DebugInfo | None,
    ) -> FastRouteResult | None:
        """Try to route through fast-path handlers."""
        start = time.time()

        result = await self._fast_router.try_route(request.query, context)

        if debug_info:
            debug_info.timings["fast_route_ms"] = int((time.time() - start) * 1000)
            if result:
                debug_info.fast_path = result.route_type

        return result

    # =========================================================================
    # Phase 2: Parallel Initialization
    # =========================================================================

    async def _parallel_init(
        self,
        request: AgentRequest,
        conversation: ConversationState,
        debug_info: DebugInfo | None,
    ) -> dict[str, Any]:
        """Run parallel initialization tasks.

        This includes sentiment detection, mode detection, and
        profile updates that don't depend on each other.
        """
        # Placeholder for parallel tasks
        # Applications can extend this by subclassing
        return {}

    # =========================================================================
    # Phase 3: Query Contextualization
    # =========================================================================

    async def _contextualize_query(
        self,
        query: str,
        conversation: ConversationState,
        debug_info: DebugInfo | None,
    ) -> tuple[str, bool]:
        """Contextualize query for multi-turn conversations.

        Rewrites ambiguous queries like "What about next week?"
        into standalone queries like "What events do I have next week?"

        Returns:
            Tuple of (standalone_query, was_contextualized)
        """
        if not self.config.enable_contextualization or not self.contextualizer:
            return query, False

        if not conversation.history:
            return query, False

        start = time.time()

        try:
            result = await self.contextualizer.contextualize(
                query=query,
                history=conversation.history,
            )

            if debug_info:
                debug_info.timings["contextualize_ms"] = int((time.time() - start) * 1000)
                if result.was_rewritten:
                    debug_info.contextualized = True
                    debug_info.original_query = query
                    debug_info.standalone_query = result.standalone_query
                    debug_info.context_intent = result.intent

            return result.standalone_query, result.was_rewritten

        except Exception as e:
            logger.error(f"Contextualization failed: {e}")
            return query, False

    # =========================================================================
    # Phase 4: Context Gathering (RAG)
    # =========================================================================

    async def _gather_context(
        self,
        query: str,
        user_id: str,
        debug_info: DebugInfo | None,
    ) -> tuple[str, list[str]]:
        """Gather context via RAG.

        Returns:
            Tuple of (context_string, memory_ids)
        """
        if not self.config.enable_retrieval or not self.retriever:
            return "", []

        start = time.time()

        try:
            result = await self.retriever.retrieve(
                query=query,
                user_id=user_id,
                k=10,
                min_relevance=0.5,
            )

            # Build context string
            context_parts = []
            memory_ids = []

            for doc in result.documents:
                context_parts.append(doc.content)
                if doc.id:
                    memory_ids.append(doc.id)

            if debug_info:
                debug_info.timings["rag_ms"] = int((time.time() - start) * 1000)
                debug_info.memory_found = len([d for d in result.documents if d.metadata.get("type") != "knowledge"])
                debug_info.knowledge_found = len([d for d in result.documents if d.metadata.get("type") == "knowledge"])
                if result.crag_stats:
                    debug_info.crag_enabled = True
                    debug_info.crag_grading = result.crag_stats

            return "\n\n".join(context_parts), memory_ids

        except Exception as e:
            logger.error(f"Context gathering failed: {e}")
            return "", []

    # =========================================================================
    # Phase 5: Decision Making
    # =========================================================================

    async def _make_decision(
        self,
        query: str,
        context: str,
        conversation: ConversationState,
        request: AgentRequest,
        debug_info: DebugInfo | None,
    ) -> dict[str, Any]:
        """Make a decision about how to respond.

        This is the core LLM call that determines:
        - What action to take (answer, search, tool, etc.)
        - What response to give
        - What model tier to use

        Returns:
            Decision dict with action, answer, reasoning, model_tier
        """
        start = time.time()

        # Format conversation history
        history_str = self._format_history(conversation)

        # Build decision prompt
        # This is a placeholder - applications should override or configure
        messages = [
            {
                "role": "system",
                "content": self._build_decision_prompt(
                    query=query,
                    context=context,
                    history=history_str,
                    request=request,
                ),
            },
            {"role": "user", "content": query},
        ]

        # Call LLM
        response = await self.llm.chat(
            messages=messages,
            model=self.config.model_standard,
        )

        if debug_info:
            debug_info.timings["decision_ms"] = int((time.time() - start) * 1000)
            debug_info.router_used = True
            debug_info.llm_calls += 1

        # Parse decision from response
        decision = self._parse_decision(response)

        if debug_info:
            debug_info.router_decision = decision
            debug_info.action = decision.get("action", "answer")
            debug_info.model_tier = decision.get("model_tier", "standard")

        return decision

    def _build_decision_prompt(
        self,
        query: str,
        context: str,
        history: str,
        request: AgentRequest,
    ) -> str:
        """Build the decision prompt.

        Override this method to customize decision prompting.
        """
        parts = [
            "You are a helpful AI assistant.",
            "",
            "## Context",
            context if context else "(No relevant context found)",
            "",
            "## Conversation History",
            history if history else "(No prior conversation)",
            "",
            "## Instructions",
            "Analyze the user's query and provide a helpful response.",
            "Be concise and direct.",
        ]
        return "\n".join(parts)

    def _parse_decision(self, response: str) -> dict[str, Any]:
        """Parse decision from LLM response.

        Override to implement custom parsing (e.g., XML, JSON).
        """
        return {
            "action": "answer",
            "answer": response,
            "model_tier": "standard",
        }

    def _format_history(self, conversation: ConversationState) -> str:
        """Format conversation history for the decision prompt."""
        if not conversation.history:
            return ""

        parts = []
        for turn in conversation.history[-5:]:  # Last 5 turns
            if "user" in turn:
                parts.append(f"User: {turn['user']}")
            if "assistant" in turn:
                parts.append(f"Assistant: {turn['assistant']}")
            # Include tool results for multi-turn references
            if "tool_results" in turn:
                for tr in turn["tool_results"]:
                    result_preview = str(tr.get("result", ""))[:500]
                    parts.append(f"[{tr['tool']} result: {result_preview}]")

        return "\n".join(parts)

    # =========================================================================
    # Phase 6: Action Execution
    # =========================================================================

    async def _execute_action(
        self,
        decision: dict[str, Any],
        request: AgentRequest,
        context: dict[str, Any],
        debug_info: DebugInfo | None,
    ) -> AgentResponse:
        """Execute the decided action.

        Delegates to registered decision handlers or uses defaults.
        """
        action = decision.get("action", "answer")
        tool_calls: list[ToolCallInfo] = []

        # Check for registered handler
        handler = self._decision_handlers.get(action)
        if handler:
            return await handler.handle(decision, request, context)

        # Default action handling
        if action == "answer":
            response = decision.get("answer", "I'm not sure how to help with that.")

            # Check for condensed response with pending details
            pending_details = None
            if len(response) > 300:
                # Could implement condensation here
                pending_details = response

            return AgentResponse(
                response=response,
                success=True,
                action=action,
                tool_calls=tool_calls,
                debug=debug_info,
                pending_details=pending_details,
            )

        elif action == "tool_call" and self.tool_executor:
            tool_name = decision.get("tool")
            tool_args = decision.get("tool_args", {})

            start = time.time()
            try:
                result = await self.tool_executor.execute(tool_name, tool_args, context)
                tool_calls.append(ToolCallInfo(
                    tool=tool_name,
                    args=tool_args,
                    result=result,
                    elapsed_ms=int((time.time() - start) * 1000),
                    success=True,
                ))
            except Exception as e:
                tool_calls.append(ToolCallInfo(
                    tool=tool_name,
                    args=tool_args,
                    error=str(e),
                    elapsed_ms=int((time.time() - start) * 1000),
                    success=False,
                ))

            # Generate response based on tool result
            response_text = decision.get("answer", "Done.")

            return AgentResponse(
                response=response_text,
                success=all(tc.success for tc in tool_calls),
                action=action,
                tool_calls=tool_calls,
                debug=debug_info,
            )

        else:
            return AgentResponse(
                response=decision.get("answer", "I'm not sure how to help with that."),
                success=True,
                action=action,
                debug=debug_info,
            )

    # =========================================================================
    # Phase 7: Post-Processing
    # =========================================================================

    async def _post_process(
        self,
        request: AgentRequest,
        response: AgentResponse,
        conversation: ConversationState,
        skip_reflection: bool = False,
    ) -> None:
        """Post-process after response generation.

        This runs async tasks like reflection, learning, and
        memory reinforcement without blocking the response.
        """
        # Add turn to conversation history
        tool_calls = None
        if response.tool_calls:
            tool_calls = [tc.to_dict() for tc in response.tool_calls]
        conversation.add_turn(
            user=request.query,
            assistant=response.response,
            tool_calls=tool_calls,
        )

        # Store pending details for "tell me more"
        if response.pending_details:
            conversation.pending_details = response.pending_details

        # Run callbacks
        for callback in self._on_response_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(request, response, conversation)
                else:
                    callback(request, response, conversation)
            except Exception as e:
                logger.error(f"Response callback failed: {e}")

        # Skip async tasks for fast routes that don't need them
        if skip_reflection:
            return

        # Fire-and-forget async tasks
        if self.config.reflection_async:
            asyncio.create_task(self._async_post_process(request, response, conversation))
        else:
            await self._async_post_process(request, response, conversation)

    async def _async_post_process(
        self,
        request: AgentRequest,
        response: AgentResponse,
        conversation: ConversationState,
    ) -> None:
        """Async post-processing tasks (fire-and-forget)."""
        try:
            # Reflection
            if self.config.enable_reflection and self.reflection:
                await self._run_reflection(request, response)

            # Learning callbacks
            if self.config.enable_learning:
                for callback in self._on_learning_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(request, response, conversation)
                        else:
                            callback(request, response, conversation)
                    except Exception as e:
                        logger.error(f"Learning callback failed: {e}")

            # Memory reinforcement
            if self.config.enable_memory_reinforcement:
                memory_ids = conversation.metadata.get("retrieved_memory_ids", [])
                if memory_ids:
                    await self._reinforce_memories(memory_ids)

        except Exception as e:
            logger.error(f"Async post-processing failed: {e}")

    async def _run_reflection(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> None:
        """Run reflection service to assess quality."""
        if not self.reflection:
            return

        try:
            result = await self.reflection.reflect(
                interaction_id=request.conversation_id or "unknown",
                query=request.query,
                response=response.response,
                action=response.action,
                tool_calls=[tc.to_dict() for tc in response.tool_calls],
            )

            if result.issues:
                logger.info(f"Reflection found {len(result.issues)} issues")

        except Exception as e:
            logger.error(f"Reflection failed: {e}")

    async def _reinforce_memories(self, memory_ids: list[str]) -> None:
        """Reinforce memories that were used in the response.

        This boosts the importance of memories that are actually
        useful, preventing them from decaying.
        """
        # Placeholder - implement with memory service
        logger.debug(f"Would reinforce {len(memory_ids)} memories")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_context(
        self,
        request: AgentRequest,
        conversation: ConversationState,
    ) -> dict[str, Any]:
        """Build context dict for handlers."""
        return {
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "area_id": request.area_id or conversation.current_area_id,
            "device_id": request.device_id or conversation.current_device_id,
            "timezone": request.timezone,
            "conversation": conversation,
            "metadata": request.metadata,
        }

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def cleanup_expired(self) -> list[str]:
        """Clean up expired conversations."""
        expired = self._conversations.cleanup_expired()
        return [e.conversation_id for e in expired]

    def get_stats(self) -> dict[str, Any]:
        """Get loop statistics."""
        return {
            "config": {
                "enable_fast_router": self.config.enable_fast_router,
                "enable_contextualization": self.config.enable_contextualization,
                "enable_retrieval": self.config.enable_retrieval,
                "enable_reflection": self.config.enable_reflection,
            },
            "fast_router": self._fast_router.get_stats(),
            "decision_handlers": list(self._decision_handlers.keys()),
            "conversations": self._conversations.get_stats(),
        }

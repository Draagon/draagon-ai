# draagon-ai Architecture - Revised Design

**Date:** 2025-12-27
**Status:** Design Document (for discussion)

---

## Executive Summary

This document clarifies the proper architecture for draagon-ai and implementation projects (roxy-voice-assistant, work-assistant, etc.). The key insight is that **most of what was called "roxy-assistant" is actually generic assistant functionality that belongs in draagon-ai**.

---

## The Correct Layering

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            draagon-ai (CORE)                                 │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           MEMORY SYSTEM                                 │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │   Working    │ │   Episodic   │ │   Semantic   │ │ Metacognitive│  │ │
│  │  │   Memory     │ │   Memory     │ │   Memory     │ │   Memory     │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │         ▲               ▲               ▲               ▲              │ │
│  │         └───────────────┴───────────────┴───────────────┘              │ │
│  │                    MemoryPromotion (auto-promotion)                     │ │
│  │                    HierarchicalScopes (WORLD/CONTEXT/AGENT/USER/SESSION)│ │
│  │                    TemporalCognitiveGraph (bi-temporal nodes)           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          ORCHESTRATION                                  │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │    Agent     │ │  AgentLoop   │ │  Decision    │ │   Action     │  │ │
│  │  │              │ │  (ReAct?)    │ │  Engine      │ │   Executor   │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │  ┌──────────────┐ ┌──────────────────────────────────────────────────┐│ │
│  │  │  Multi-Agent │ │  Autonomous Agent (background cognitive)         ││ │
│  │  │  Orchestrator│ │  Guardrails, Self-Monitoring                     ││ │
│  │  └──────────────┘ └──────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           COGNITION                                     │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │   Belief     │ │   Learning   │ │  Curiosity   │ │   Opinion    │  │ │
│  │  │ Reconciliation│ │   Service   │ │   Engine     │ │  Formation   │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │  ┌──────────────┐ ┌──────────────────────────────────────────────────┐│ │
│  │  │  Proactive   │ │  Identity Manager (Self, traits, values)         ││ │
│  │  │  Questions   │ │                                                   ││ │
│  │  └──────────────┘ └──────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           MCP LAYER                                     │ │
│  │  ┌──────────────────────────────┐ ┌──────────────────────────────────┐│ │
│  │  │  MCP Client                  │ │  MCP Server                      ││ │
│  │  │  (use external tools)        │ │  (expose memory to other apps)   ││ │
│  │  └──────────────────────────────┘ └──────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         PROMPT EVOLUTION                                │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────────┐│ │
│  │  │   Prompt     │ │ Promptbreeder│ │  Fitness Evaluation              ││ │
│  │  │   Registry   │ │  Evolution   │ │  (train/holdout split)           ││ │
│  │  └──────────────┘ └──────────────┘ └──────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         EXTENSIONS                                      │ │
│  │  (Optional capabilities via entry points)                              │ │
│  │  • Home Assistant tools                                                 │ │
│  │  • Google Calendar tools                                                │ │
│  │  • Slack integration                                                    │ │
│  │  • etc.                                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Implements protocols
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMPLEMENTATION (roxy-voice-assistant)                     │
│                                                                              │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌────────────────────────┐ │
│  │    ADAPTERS         │ │  PERSONALITY CONFIG  │ │  CHANNEL-SPECIFIC     │ │
│  │                     │ │                     │ │                        │ │
│  │  QdrantMemoryStore  │ │  roxy_persona.yaml  │ │  Voice/TTS             │ │
│  │  GroqLLMProvider    │ │  - name: "Roxy"     │ │  Home Assistant        │ │
│  │  OllamaEmbeddings   │ │  - values: {...}    │ │  Wyoming Protocol      │ │
│  │                     │ │  - traits: {...}    │ │                        │ │
│  └─────────────────────┘ └─────────────────────┘ └────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         APPLICATION SHELL                               ││
│  │  FastAPI + routes + main.py                                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What's Generic vs Implementation-Specific

### Generic (draagon-ai core)

| Component | Why It's Generic |
|-----------|------------------|
| Memory system | Any assistant needs memory layers, scopes, promotion |
| Orchestrator/Agent | Any assistant needs decision → execution → response loop |
| Cognitive services | Any assistant benefits from learning, beliefs, curiosity |
| Autonomous agent | Any assistant could have background processing |
| Identity/Self model | Any assistant has an identity (the *content* is specific) |
| MCP client/server | Standard protocol for tool integration |
| Prompt evolution | Any prompt system benefits from evolution |

### Implementation-Specific

| Component | Why It's Specific |
|-----------|-------------------|
| Adapters (Qdrant, Groq) | Different deployments use different backends |
| Personality content | "Roxy's values" vs "Work Assistant's values" |
| Channel integration | Voice, Slack, CLI are different channels |
| Application shell | FastAPI vs Discord bot vs CLI |

---

## ReAct vs Simple Tool-Use

### Current State (Simple Tool-Use)

Roxy's current orchestrator does:
```
1. Contextualize query
2. LLM decides action + args (single call)
3. Execute tool
4. Synthesize response
```

This is **single-step tool-use** - one LLM call, one action.

### ReAct Pattern (Reasoning + Acting)

ReAct allows **multi-step reasoning with explicit thought traces**:

```
loop:
    THOUGHT: "I need to check the user's calendar for conflicts"
    ACTION: search_calendar(days=7)
    OBSERVATION: [3 events found]

    THOUGHT: "Now I see there's an overlap on Tuesday. Let me check details."
    ACTION: get_event_details(event_id="...")
    OBSERVATION: {details}

    THOUGHT: "I have enough information to answer"
    FINAL_ANSWER: "You have a conflict on Tuesday at 3pm..."
```

### Why ReAct is Better

1. **Complex multi-step tasks** - Can break down reasoning
2. **Explainable** - Thought traces show reasoning
3. **Self-correction** - Can reconsider after observations
4. **Better for agents** - More human-like reasoning

### Implementation

draagon-ai's `orchestration/loop.py` could support both:

```python
class AgentLoop:
    async def run(self, query: str, context: AgentContext) -> AgentResponse:
        if self.config.use_react:
            return await self._run_react(query, context)
        else:
            return await self._run_simple(query, context)

    async def _run_react(self, query: str, context: AgentContext) -> AgentResponse:
        """ReAct loop with explicit reasoning."""
        steps = []

        while True:
            # THOUGHT: Reason about what to do next
            thought = await self._think(query, context, steps)
            steps.append(thought)

            if thought.is_final_answer:
                return self._synthesize(thought.answer, steps)

            # ACTION: Execute the decided action
            result = await self._act(thought.action)
            steps.append(result)

            # Add observation to context for next iteration
            context.add_observation(result)
```

---

## MCP Server Design

### The Vision (from IDEAS.txt)

> "can I replace [Claude] with my own version that can do the same things as claude but it as my assistants personality and my memories"

This implies:
1. Multiple apps (Claude Code, VS Code, mobile) share the same memory
2. The memory is the "hub" that connects everything
3. Each app can use its own LLM but shares context

### Two MCP Patterns

**Pattern A: Memory MCP Server** (Recommended for shared memory)
```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Claude Code  │   │    VS Code    │   │  Mobile App   │
│  (Claude LLM) │   │  (Copilot)    │   │  (Roxy)       │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        │ MCP               │ MCP               │ MCP
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Memory MCP Server    │
                │  (draagon-ai)         │
                │                       │
                │  Tools:               │
                │  • memory.store       │
                │  • memory.search      │
                │  • memory.list        │
                │  • beliefs.reconcile  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │       Qdrant          │
                │   (shared storage)    │
                └───────────────────────┘
```

In this pattern:
- Memory is the shared resource
- Each app uses its own LLM but stores/retrieves from shared memory
- Claude Code learns something → Roxy knows it too
- Scopes control access (private, shared, system)

**Pattern B: Assistant MCP Server** (For delegating tasks)
```
┌───────────────┐
│  Claude Code  │
│  (Claude LLM) │
└───────┬───────┘
        │
        │ "ask_roxy: What's on my calendar tomorrow?"
        │
        ▼
┌───────────────────────┐
│  Roxy MCP Server      │
│  (full assistant)     │
│                       │
│  Tools:               │
│  • roxy.ask           │
│  • roxy.remember      │
│  • roxy.home_control  │
└───────────────────────┘
```

In this pattern:
- Roxy is a full assistant that other agents can call
- Claude delegates entire tasks to Roxy
- Roxy maintains its own reasoning and personality

### Recommendation

**Use Pattern A (Memory MCP Server) for the shared memory use case.** This aligns better with:
- "My assistants personality and my memories"
- Each app keeping its own LLM/personality
- Just sharing the knowledge base

Pattern B is useful when you want to delegate to a specialized agent (e.g., "Ask my home assistant to turn off the lights").

### Memory MCP Server Implementation

```python
# draagon_ai/mcp/server.py

class MemoryMCPServer:
    """MCP server exposing memory operations."""

    tools = [
        MCPTool(
            name="memory.store",
            description="Store a memory in the shared knowledge base",
            input_schema={
                "content": {"type": "string"},
                "memory_type": {"type": "string", "enum": ["fact", "skill", "insight"]},
                "scope": {"type": "string", "enum": ["private", "shared", "system"]},
            },
        ),
        MCPTool(
            name="memory.search",
            description="Search the shared knowledge base",
            input_schema={
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
        ),
        MCPTool(
            name="memory.list_recent",
            description="List recent memories",
            input_schema={
                "memory_type": {"type": "string", "optional": True},
                "limit": {"type": "integer", "default": 10},
            },
        ),
    ]

    def __init__(self, memory: LayeredMemoryProvider):
        self.memory = memory

    async def handle_store(self, content: str, memory_type: str, scope: str) -> str:
        """Handle memory.store tool call."""
        # Validate caller permissions based on scope
        # Store in the memory system
        result = await self.memory.store(content, memory_type, scope)
        return f"Stored memory {result.id}"
```

---

## Current Gap Analysis

### What Roxy Has vs Should Have

| Component | Current | Should Be |
|-----------|---------|-----------|
| Memory | Roxy's simple MemoryService | draagon-ai LayeredMemoryProvider |
| Orchestrator | Roxy's orchestrator.py | draagon-ai Agent/AgentLoop |
| Cognitive services | Mix of duplicates and adapters | All from draagon-ai via adapters |
| Autonomous agent | Local + new extension | Extension only |
| Prompts | Roxy's prompts.py | draagon-ai prompt registry |

### Migration Path

**Phase 1: Memory Migration** (High Value)
1. Replace Roxy's MemoryService with draagon-ai LayeredMemoryProvider
2. Configure with Qdrant backend
3. Enable memory promotion, layers, scopes
4. This alone would dramatically improve memory quality

**Phase 2: Orchestrator Migration**
1. Replace Roxy's orchestrator.py with draagon-ai Agent
2. Use draagon-ai's AgentLoop (with ReAct support)
3. Roxy becomes just a configuration of the generic agent

**Phase 3: Full Cognitive Integration**
1. Remove all duplicate cognitive services from Roxy
2. Use draagon-ai services via adapters (already have some)
3. Roxy becomes thin: voice handling + personality config

**Phase 4: MCP Server**
1. Implement Memory MCP Server in draagon-ai
2. Claude Code can connect and share memories
3. Other apps can integrate

---

## Summary of Design Decisions

1. **"roxy-assistant" is really "draagon-ai assistant"** - Move all generic assistant functionality to draagon-ai core

2. **Roxy should use draagon-ai's memory system** - The 4-layer cognitive memory with promotion is much better than the current flat store

3. **Roxy should use draagon-ai's orchestrator** - With optional ReAct support for complex reasoning

4. **Memory MCP Server is the right design** - For sharing knowledge across apps while each keeps its own LLM

5. **Autonomous agent should be core, not extension** - It's a fundamental assistant capability

6. **Implementations (roxy, work-assistant) are thin** - Just adapters, personality config, and channel-specific code

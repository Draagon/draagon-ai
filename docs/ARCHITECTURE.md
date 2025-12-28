# Draagon AI Architecture

**Last Updated:** 2025-12-27
**Version:** 0.2.0

---

## Vision: The AI Assistant Ecosystem

The Draagon AI architecture enables a **shared AI assistant ecosystem** where:

1. **Central Memory** - All applications share the same memories about users
2. **Shared Personality** - The assistant's personality is consistent across apps
3. **Multi-App Integration** - Works in voice assistants, IDEs, mobile apps, etc.
4. **Local + Central** - Apps can act locally while sharing central state
5. **Multi-Tenant** - Personal, household, and work memories can coexist

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Applications (Thin Clients)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐   │
│  │ roxy-voice  │  │ roxy-code   │  │ roxy-mobile │  │ roxy-desktop │   │
│  │ (HA voice)  │  │ (VS Code)   │  │ (mobile app)│  │ (system tray)│   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘   │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                   │                                     │
│                          MCP Protocol (or API)                          │
│                                   │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│                          roxy-server                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     roxy-assistant                               │   │
│  │  (Roxy's personality, prompts, cognitive style)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│                    draagon-ai (core + extensions)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐   │
│  │    memory    │  │  cognition   │  │  autonomous  │  │   tools   │   │
│  │   (layers)   │  │  (beliefs)   │  │  (background)│  │   (MCP)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Definitions

### 1. draagon-ai (Core)

The foundational framework providing:

| Component | Purpose |
|-----------|---------|
| `memory/` | Layered memory (Working/Episodic/Semantic/Metacognitive) |
| `llm/` | Multi-provider LLM abstraction (Groq, Ollama, Claude) |
| `cognition/` | Belief reconciliation, learning, curiosity, opinions |
| `tools/` | MCP client for tool integration |
| `personality/` | Archetype system for agent personalities |
| `extensions/` | Plugin system for optional capabilities |
| `testing/` | Testing framework for applications |

**Key Design Principle:** Core provides **protocols/interfaces** that applications implement via adapters.

### 2. draagon-ai Extensions

Optional capabilities that integrate via the extension system:

| Extension | Purpose | Status |
|-----------|---------|--------|
| `draagon-ai-autonomous` | Background agent, guardrails, self-monitoring | **NEW** |
| `draagon-ai-storytelling` | Narrative generation for games | Exists |
| `draagon-ai-homeassistant` | Smart home control | Planned |

Extensions provide:
- Services (singletons for the application)
- Behaviors (agent capabilities)
- Tools (MCP-style tools)
- Prompt domains (specialized prompts)
- MCP servers (external tool access)

### 3. roxy-assistant (Personality Layer)

Roxy's specific personality, NOT the generic framework:

| Component | Purpose |
|-----------|---------|
| `prompts/` | Roxy's DECISION_PROMPT, SYNTHESIS_PROMPT, etc. |
| `values/` | Roxy's values (truth-seeking, epistemic humility) |
| `traits/` | Roxy's default personality traits |
| `style/` | Roxy's voice, tone, response patterns |
| `domains/` | Roxy's expertise (home automation, personal assistant) |

**Key Design Principle:** The FRAMEWORK is in draagon-ai, the PERSONALITY is here.

### 4. roxy-voice-assistant (Implementation)

This specific deployment for Home Assistant voice:

| Component | Purpose |
|-----------|---------|
| `integrations/homeassistant/` | HA entity resolution, services, satellites |
| `voice/` | TTS optimization, response condensation |
| `api/` | FastAPI endpoints, dashboard |
| `tools/` | HA tools, calendar, commands |

---

## Extension System

Extensions are discovered via Python entry points and configured via `draagon.yaml`:

```yaml
# draagon.yaml
extensions:
  autonomous:
    enabled: true
    config:
      cycle_minutes: 30
      daily_budget: 20
      self_monitoring: true

  homeassistant:
    enabled: true
    config:
      url: "http://192.168.168.206:8123"
```

### Creating an Extension

```python
from draagon_ai.extensions import Extension, ExtensionInfo

class AutonomousExtension(Extension):
    @property
    def info(self) -> ExtensionInfo:
        return ExtensionInfo(
            name="autonomous",
            version="0.1.0",
            description="Background cognitive processes",
            requires_extensions=[],
        )

    def initialize(self, config: dict) -> None:
        self._service = AutonomousAgentService(config)

    def get_services(self) -> dict[str, Any]:
        return {"autonomous_agent": self._service}

    def get_behaviors(self) -> list:
        return [AutonomousBehavior()]
```

---

## Memory Architecture

### Layered Memory (C.2)

```
┌─────────────────────────────────────────────────────────┐
│                    Working Memory                        │
│  (Session-scoped, limited capacity, high activation)     │
├─────────────────────────────────────────────────────────┤
│                    Episodic Memory                       │
│  (Events, episodes, chronological, time-decaying)        │
├─────────────────────────────────────────────────────────┤
│                    Semantic Memory                       │
│  (Entities, facts, relationships, stable)                │
├─────────────────────────────────────────────────────────┤
│                  Metacognitive Memory                    │
│  (Skills, strategies, insights, behaviors)               │
└─────────────────────────────────────────────────────────┘
```

### Scopes for Multi-Tenant Access

| Scope | Description | Example |
|-------|-------------|---------|
| WORLD | Shared facts | "Paris is the capital of France" |
| CONTEXT | Shared within context | Household WiFi password |
| AGENT | Agent's private memories | Roxy's learned preferences |
| USER | Per-user within agent | Doug's birthday |
| SESSION | Current conversation | What we just discussed |

---

## Adapter Pattern

Applications use **adapters** to connect their implementations to draagon-ai protocols:

```python
# Roxy's Qdrant memory wrapped to implement draagon-ai protocol
from roxy.adapters import RoxyMemoryAdapter
from roxy.services.memory import MemoryService

memory_service = MemoryService()  # Roxy's Qdrant implementation
memory_adapter = RoxyMemoryAdapter(memory_service)  # Implements MemoryProvider

# Now draagon-ai services can use it
from draagon_ai import LearningService
learning = LearningService(memory=memory_adapter, llm=llm_adapter)
```

---

## MCP Integration for Ecosystem

For the multi-app ecosystem vision, Roxy can expose itself as an MCP server:

```
┌─────────────────────┐     ┌─────────────────────┐
│    Claude Code      │────▶│    roxy-mcp-server  │
│    (MCP Client)     │     │    (MCP Server)     │
└─────────────────────┘     └──────────┬──────────┘
                                       │
┌─────────────────────┐               │
│    VS Code Plugin   │────▶──────────┤
│    (MCP Client)     │               │
└─────────────────────┘               │
                                       ▼
                            ┌─────────────────────┐
                            │   Shared Memory     │
                            │   (Qdrant/Postgres) │
                            └─────────────────────┘
```

MCP Tools Roxy could expose:
- `roxy.remember` - Store a memory
- `roxy.recall` - Search memories
- `roxy.ask` - Ask Roxy something
- `roxy.calendar` - Calendar operations
- `roxy.home` - Home automation

---

## Current State vs Target

### What's Already in draagon-ai

✅ Memory abstraction (MemoryProvider, layered memory)
✅ LLM abstraction (LLMProvider, multi-tier)
✅ Cognition (BeliefReconciliation, Learning, Curiosity, Opinions)
✅ Personality (Archetypes)
✅ Extensions system
✅ MCP tools integration
✅ Prompt system
✅ Testing framework

### What Needs to Move TO draagon-ai

| Component | Current Location | Target |
|-----------|-----------------|--------|
| Autonomous Agent | roxy/services/autonomous_agent.py | draagon-ai extension |
| Circuit Breakers | roxy/services/circuit_breaker.py | draagon-ai core |
| HTTP Client | roxy/services/http.py | draagon-ai core |

### What Should Stay in roxy-voice-assistant

| Component | Why |
|-----------|-----|
| Home Assistant service | HA-specific integration |
| TTS optimizer | Voice-specific |
| Calendar service | Google-specific OAuth |
| Command execution | Security-sensitive local |
| Dashboard | UI for this deployment |

### Duplicates to Remove from Roxy

These exist in roxy/services/ but should use draagon-ai versions via adapters:

| Roxy Service | draagon-ai Equivalent | Action |
|--------------|----------------------|--------|
| belief_reconciliation.py | draagon_ai.cognition.BeliefReconciliationService | Delete, use adapter |
| curiosity_engine.py | draagon_ai.cognition.CuriosityEngine | Delete, use adapter |
| learning.py | draagon_ai.cognition.LearningService | Delete, use adapter |
| opinion_formation.py | draagon_ai.cognition.OpinionFormationService | Delete, use adapter |
| proactive_questions.py | draagon_ai.cognition.ProactiveQuestionTimingService | Delete, use adapter |

---

## Migration Phases

### Phase 1: Autonomous Agent in Core ✅ COMPLETE

1. ✅ Moved autonomous agent to `draagon_ai.orchestration.autonomous` (core)
2. ✅ Protocol-based design with 5 provider protocols
3. ✅ Created adapters in Roxy (`roxy/adapters/autonomous_adapter.py`)
4. ✅ Created factory in Roxy that uses core directly
5. ✅ Updated Roxy main.py and dashboard to use the factory

**Core Location:** `src/draagon_ai/orchestration/autonomous/`
**Roxy Adapters:** `src/roxy/adapters/autonomous_adapter.py`, `autonomous_factory.py`

### Phase 2: Clean Up Duplicates (Next)

1. Verify adapters/factory.py works correctly
2. Delete duplicate services from roxy/services/
3. Update all imports to use adapters
4. Run tests to verify

### Phase 3: Add Core Utilities (Future)

1. Move circuit breaker to draagon-ai core
2. Move HTTP client to draagon-ai core
3. These are generic, not Roxy-specific

### Phase 4: MCP Server (Future)

1. Create roxy-mcp-server package
2. Expose memory, calendar, home tools via MCP
3. Enable Claude Code and other apps to use Roxy

### Phase 5: Multi-App Support (Future)

1. Factor out roxy-assistant as shared personality
2. Create roxy-code for VS Code
3. Create roxy-mobile for mobile apps
4. All share memory via MCP or direct connection

---

## Key Design Decisions

### 1. Adapters over Rewrites

Don't rewrite Roxy's working services. Wrap them with adapters so draagon-ai can use them.

### 2. Extensions for Optional Features

Autonomous agent is an extension, not core. Not all draagon-ai apps need background processing.

### 3. Personality is Separate from Framework

draagon-ai provides the "how" (cognition, memory, learning).
roxy-assistant provides the "who" (Roxy's voice, values, style).

### 4. MCP for Ecosystem Integration

Use MCP protocol for multi-app integration. It's standard, works with Claude, and handles the "local + central" problem.

---

## File Structure: Target State

```
draagon-ai/
├── src/draagon_ai/
│   ├── core/           # Core types, context
│   ├── memory/         # Layered memory system
│   ├── llm/            # LLM abstraction
│   ├── cognition/      # Belief, learning, curiosity, opinions
│   ├── extensions/     # Extension system
│   ├── tools/          # MCP integration
│   ├── personality/    # Archetypes
│   └── testing/        # Test framework

roxy-voice-assistant/
├── src/roxy/
│   ├── adapters/       # Wrap Roxy services for draagon-ai protocols
│   │   ├── llm_adapter.py
│   │   ├── memory_adapter.py
│   │   └── factory.py
│   ├── personality/    # Roxy's specific personality (future: roxy-assistant)
│   │   ├── prompts/
│   │   ├── values.py
│   │   └── traits.py
│   ├── integrations/   # Implementation-specific
│   │   ├── homeassistant/
│   │   ├── calendar/
│   │   └── voice/
│   ├── services/       # Roxy's core services (that aren't duplicates)
│   │   ├── memory.py   # Qdrant implementation
│   │   ├── groq.py     # Groq LLM
│   │   └── ...
│   ├── api/            # FastAPI routes
│   └── agent/          # Orchestrator
```

---

## Notes for Future Work

1. **Roxy as MCP Server** - Could expose all Roxy capabilities via MCP for Claude Code integration
2. **Shared Memory** - Use Qdrant or PostgreSQL with proper scoping for multi-app access
3. **Work vs Personal** - Support layered contexts (personal + household + work)
4. **ReAct Patterns** - Investigate graph-based agentic approaches mentioned in IDEAS.txt

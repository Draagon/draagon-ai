# draagon-ai Project Scope

## What draagon-ai IS

### Core Framework

A Python framework for building AI agents with cognitive capabilities:

1. **Agent Orchestration**
   - ReAct loop implementation (THOUGHT → ACTION → OBSERVATION)
   - Decision engine with LLM integration
   - Action executor with tool registry
   - Multi-agent sequential and parallel orchestration

2. **Tool System**
   - `@tool` decorator for declarative tool registration
   - Tool discovery and auto-registration
   - MCP (Model Context Protocol) client integration
   - Structured parameters with validation

3. **Memory Architecture**
   - 4-layer cognitive memory (working, episodic, semantic, metacognitive)
   - Automatic promotion between layers
   - Attention weighting and decay
   - Memory provider protocol for custom backends

4. **Cognitive Services**
   - Learning service (skill/fact extraction)
   - Belief reconciliation (observation → belief)
   - Curiosity engine (knowledge gap detection)
   - Opinion formation (authentic agent perspectives)

5. **Multi-Agent Capabilities**
   - Sequential orchestration (Phase C.1)
   - Parallel orchestration with shared memory (Phase C.4)
   - Learning channel for cross-agent communication
   - Transactive memory (expertise routing)

6. **Prompt Evolution** (Experimental)
   - Promptbreeder-style genetic optimization
   - Safety guards against overfitting
   - Capability preservation validation

## What draagon-ai IS NOT

### Not Provided

1. **LLM Implementations**
   - No built-in OpenAI/Anthropic/Groq clients
   - Framework provides `LLMProvider` protocol
   - Host applications implement their preferred provider

2. **Vector Database**
   - No built-in Qdrant/Pinecone/Chroma integration
   - Framework provides `MemoryProvider` protocol
   - Host applications implement their preferred backend

3. **Application Logic**
   - No domain-specific tools (home automation, calendar, etc.)
   - Framework provides infrastructure
   - Host applications define behaviors and tools

4. **User Interface**
   - No CLI, web UI, or API server
   - Framework is a library, not an application
   - Host applications provide user interfaces

5. **Deployment Infrastructure**
   - No Docker containers or Kubernetes manifests
   - No cloud provider integrations
   - Host applications handle deployment

6. **Production Monitoring**
   - No built-in metrics/logging services
   - Framework provides hooks for observability
   - Host applications integrate with their monitoring

## Boundary Examples

### In Scope

```python
# Agent orchestration - YES
from draagon_ai import Agent, AgentConfig
agent = Agent(config, llm_provider, memory_provider)
response = await agent.run("What time is it?")

# Tool registration - YES
from draagon_ai.tools import tool

@tool(name="get_time", description="Get current time")
async def get_time(args, **context):
    return {"time": datetime.now().isoformat()}

# Memory architecture - YES
from draagon_ai.memory import WorkingMemory
memory = WorkingMemory(graph, session_id="123")
await memory.add("User prefers Celsius", attention_weight=0.8)

# Cognitive services - YES
from draagon_ai.cognition import BeliefReconciliationService
service = BeliefReconciliationService(llm)
belief = await service.reconcile(observations)
```

### Out of Scope

```python
# LLM client - NO (host app provides)
# from draagon_ai import OpenAI  # Does not exist

# Vector database - NO (host app provides)
# from draagon_ai import QdrantMemory  # Does not exist

# Domain tools - NO (host app provides)
# from draagon_ai.tools import smart_home  # Does not exist

# Web server - NO (host app provides)
# from draagon_ai import run_server  # Does not exist
```

## Reference Implementation

**Roxy Voice Assistant** (`../roxy-voice-assistant/`) demonstrates:

- LLM provider implementation (Groq, OpenAI adapters)
- Memory provider implementation (Qdrant backend)
- Domain tools (Home Assistant, calendar, weather)
- Voice interface (speech-to-text, text-to-speech)
- Deployment (Docker, systemd)

## Scope Evolution

### Phase 0 (Completed)
- ReAct loop
- Decision engine
- Action executor
- Tool registry
- 4-layer memory
- Sequential multi-agent

### Phase 1 (Current)
- @tool decorator with discovery
- Cognitive service integration
- Learning channel activation

### Phase 2 (Planned)
- Parallel orchestration
- Shared cognitive working memory
- Belief reconciliation across agents
- Transactive memory

### Phase 3 (Future)
- Production hardening
- Performance optimization
- PyPI distribution
- Comprehensive documentation

## Non-Goals

These are explicitly NOT planned:

1. **Becoming an LLM abstraction layer** (use LiteLLM)
2. **Competing with LangChain for chains/pipelines**
3. **Building a vector database** (use Qdrant, Chroma)
4. **Creating an agent marketplace**
5. **Providing pre-built domain agents**

## Success Metrics

### Framework Adoption
- 3+ production applications using draagon-ai
- 100+ GitHub stars within 6 months
- Active community contributions

### Technical Excellence
- 90%+ test coverage
- Sub-100ms decision latency (excluding LLM)
- Zero semantic regex patterns

### Cognitive Leadership
- Beat SOTA on MultiAgentBench
- Published research on cognitive architecture
- Referenced in academic papers

---

**Document Status**: Active
**Last Updated**: 2025-12-30

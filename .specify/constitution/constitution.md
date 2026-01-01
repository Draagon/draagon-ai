# draagon-ai Constitution

## Project Charter

**Name:** draagon-ai
**Type:** Agentic AI Framework for Cognitive Assistants
**Version:** 0.1.0
**Date:** 2025-12-30
**Status:** Active Development

## Mission Statement

Provide the definitive framework for building AI agents with genuine cognitive capabilities—not just memory, but beliefs, opinions, curiosity, and metacognition. Enable developers to create agents that think, learn, and collaborate like intelligent beings.

**Proven in Production:** draagon-ai powers Roxy Voice Assistant, a production home automation voice assistant serving real users with 4-layer cognitive memory, autonomous learning, and belief reconciliation.

## Vision

Be the cognitive architecture that powers the next generation of AI assistants. Where other frameworks provide "agents with memory," draagon-ai provides **agents with minds**.

**Not vaporware.** This architecture is battle-tested in production, handling real-world ambiguity, user corrections, knowledge conflicts, and autonomous learning—all without semantic regex patterns.

## Core Values

### 1. LLM-First Architecture (ABSOLUTE)
**The LLM handles ALL semantic understanding.** Never use regex or keyword patterns for semantic tasks. The LLM is the brain—let it think.

**This is non-negotiable.** Regex-based semantic detection is the #1 failure mode of "agentic" systems. It breaks with:
- Typos and speech-to-text errors
- Paraphrasing and synonyms
- Negation and sarcasm
- Context-dependent meaning
- Natural language variation

```
NEVER: regex patterns for intent detection
ALWAYS: LLM semantic analysis
```

**Production Evidence:** Roxy handles "tell me more", "elaborate", "go on", "what else?", "the full list"—all semantically, no hardcoded triggers. Works with ANY phrasing because the LLM understands intent.

### 2. Cognitive Authenticity
Agents should have genuine cognitive capabilities, not simulations:
- **Beliefs** that can be reconciled when conflicting
- **Opinions** that evolve based on experience
- **Curiosity** that drives knowledge-seeking behavior
- **Memory** that consolidates like human cognition

### 3. XML Output Format (MANDATORY)
All LLM prompts use XML output, never JSON. XML is:
- **Better for streaming** - Parse incrementally as tokens arrive
- **Fewer escaping issues** - JSON requires escaping quotes, backslashes, newlines
- **More robust parsing** - Malformed XML is easier to recover from
- **Clearer nesting** - Element names make structure self-documenting

**Production Evidence:** Roxy processes 1000+ queries/day with XML parsing. Zero JSON escaping errors. Streaming TTS works because XML parses incrementally.

### 4. Protocol-Based Design
Use Python `Protocol` abstractions for all integrations. Host applications implement protocols; the framework provides implementations.

### 5. Pragmatic Async (NOT Async-First)
Use async when it provides real benefit. Keep synchronous code simple.

**Use async for:**
- External I/O (LLM calls, database queries, HTTP requests)
- Concurrent operations (parallel agents, tool batching)
- Background tasks that shouldn't block user response (learning, consolidation)

**Keep synchronous:**
- Pure computation and data transformation
- Configuration and initialization
- Simple utilities and helpers
- Getters, builders, validators

**Rationale:** Async adds complexity. Every `async def` forces callers to be async (viral). Only pay that cost when you get real concurrency or non-blocking I/O benefits.

### 6. Research-Grounded Development
Every architectural decision is backed by research:
- **Cognitive psychology** - Baddeley's Working Memory Model, Miller's Law (7±2)
- **Multi-agent systems** - MAST taxonomy, MultiAgentBench, Anthropic research
- **Metacognition** - ICML 2025 position paper on intrinsic learning
- **Transactive memory** - Wegner (1987) on expertise routing
- **Epistemic logic** - BDI (Beliefs, Desires, Intentions) agent architectures

### 7. Test Outcomes, Not Processes (CRITICAL)
**Goal: Build the smartest AI, NOT pass tests.**

Tests should validate **OUTCOMES** (correct answer, good UX), not **PROCESSES** (specific tool usage).

```
GOOD TEST: "User gets correct weather info"
BAD TEST:  "Agent MUST use tool 'get_weather' exactly"
```

**Why This Matters:**
- If Roxy answers correctly using knowledge vs. web search, both are valid
- Intelligence means choosing the best approach, not following scripts
- Tests that enforce rigid behavior reduce agent intelligence

**Production Evidence:** Roxy's test suite validates outcomes. If the user gets what they want, the method doesn't matter. This philosophy led to better autonomous decision-making.

## System Boundaries

### In Scope

- **Agent Orchestration**: ReAct loops, decision engine, action execution
- **Tool System**: @tool decorator, registry, MCP integration
- **Memory Architecture**: 4-layer cognitive memory with promotion
- **Cognitive Services**: Learning, beliefs, curiosity, opinions
- **Multi-Agent**: Parallel orchestration, shared memory, belief reconciliation
- **Transactive Memory**: Expertise tracking, query routing
- **Prompt Evolution**: Genetic optimization with safety guards

### Out of Scope

- **LLM Providers**: Framework provides protocols; host apps provide implementations
- **Vector Databases**: Memory providers are protocol-based; implementations external
- **Application Logic**: Framework provides infrastructure; apps provide behavior
- **UI/UX**: No frontend components
- **Deployment**: No containerization or orchestration

## Technical Constraints

### Must Have
- Python 3.11+ compatibility
- Async for I/O operations (LLM, database, HTTP)
- Zero required external services (all optional)
- Protocol-based extensibility
- Comprehensive type hints

### Must Avoid
- **Regex for semantic understanding** (except security blocklists, TTS transforms, entity IDs)
- **JSON output from LLM prompts** (XML only)
- **Sync I/O in hot paths** (use async for LLM/database/HTTP)
- **Unnecessary async** (don't make pure functions async)
- **Hard dependencies on specific LLM providers** (protocol-based)
- **Breaking changes without major version bump** (semantic versioning)
- **Hardcoded trigger phrases** (LLM understands naturally)
- **Test-specific prompt hacking** (general principles only)
- **Binary confidence decisions** (graduated thresholds)

## Non-Negotiable Principles

### 1. Never Pattern-Match Semantics
```python
# FORBIDDEN
if re.match(r"actually|no,|wrong", user_input):
    handle_correction()

# REQUIRED
correction = await llm.detect_correction(user_input)
if correction.is_correction:
    handle_correction(correction)
```

### 2. Always Use XML for LLM Output
```python
# FORBIDDEN
prompt = "Return JSON: {\"action\": \"...\"}"

# REQUIRED
prompt = """Return XML:
<response>
    <action>...</action>
</response>"""
```

### 3. Beliefs Are Not Memories
```python
# FORBIDDEN - Treating observations as facts
memory.store(user_said)

# REQUIRED - Observations become beliefs through reconciliation
observation = belief_service.create_observation(user_said)
belief = await belief_service.reconcile(observation)
```

### 4. Confidence-Based Actions
```python
# FORBIDDEN - Binary decisions
if should_do_action:
    do_action()

# REQUIRED - Graduated confidence
if confidence > 0.9:
    do_action()
elif confidence > 0.7:
    do_action_with_monitoring()
elif confidence > 0.5:
    confirm_then_do_action()
else:
    ask_for_clarification()
```

## Success Criteria

### Framework Success
- **Production Adoption**: 3+ production applications using draagon-ai ✅ (Roxy in production)
- **Benchmark Leadership**: Beat SOTA on MultiAgentBench coordination tasks (target: 55%+ vs 36%)
- **Research Publication**: Published paper on cognitive architecture
- **Community Growth**: Active open-source community with contributions

### Technical Success
- **Test Coverage**: 90%+ coverage across all modules
- **Performance**: Sub-100ms decision latency (excluding LLM calls)
- **Constitution Compliance**: Zero semantic regex patterns in codebase ✅ (Roxy: 0 regex for semantics)
- **Protocol Coverage**: All integrations via Protocols ✅

### Cognitive Success (Production-Validated)
- **Belief Coherence**: Agents form coherent beliefs across sessions ✅ (Roxy: RoxySelf persistent)
- **Curiosity-Driven Learning**: Knowledge gaps drive question queuing ✅ (Roxy: max 3/day, 30min gaps)
- **Opinion Evolution**: Opinions evolve based on experience ✅ (Roxy: debate_persistence trait)
- **Conflict Resolution**: Multi-agent systems resolve belief conflicts gracefully
- **Autonomous Learning**: Post-interaction skill/fact extraction ✅ (Roxy: 1000+ learned facts)
- **Correction Verification**: Trust-but-verify user corrections ✅ (Roxy: verifies via web search)

## Production Lessons Learned

These principles are derived from running Roxy Voice Assistant in production:

### 1. LLM-First Is Non-Negotiable
**Lesson:** Every attempt to use regex for semantic understanding failed. Speech-to-text errors, paraphrasing, and natural variation break pattern matching.

**Evidence:** Roxy's "tell me more" feature works with 20+ different phrasings because the LLM understands intent. No hardcoded triggers needed.

### 2. XML Parsing Is Robust
**Lesson:** JSON escaping causes 90% of parsing errors. XML just works.

**Evidence:** 1000+ queries/day with zero JSON escaping issues. Streaming TTS parses XML incrementally without waiting for complete response.

### 3. Outcome Testing Enables Intelligence
**Lesson:** Tests that enforce specific tool usage reduce agent intelligence. Test outcomes, not processes.

**Evidence:** When we relaxed "must use tool X" tests to "must return correct answer," Roxy made smarter decisions about when to use knowledge vs. web search.

### 4. Pragmatic Async Prevents Latency
**Lesson:** Background tasks (learning, consolidation, reflection) should never block user responses. But not everything needs to be async.

**Evidence:** Roxy's learning service runs post-response. Users get answers in <2s; learning happens in background. Pure computation stays synchronous for simplicity.

### 5. Belief Reconciliation Handles Conflicts
**Lesson:** Users contradict themselves. Users contradict each other. Treat statements as observations, not facts.

**Evidence:** Roxy's trust-but-verify system catches incorrect user corrections by web-searching verifiable facts while trusting personal preferences.

### 6. Progressive Profiling > Blocking Onboarding
**Lesson:** Never block users for profile setup. Gather info opportunistically through natural conversation.

**Evidence:** Roxy asks for user name during 1st and every 3rd interaction—non-blocking, non-annoying. PINs captured when user says "set my PIN to X."

## Governance

### Change Process
1. **Constitution changes**: Require explicit approval and production validation
2. **Architecture changes**: Require specification, review, and benchmark testing
3. **API changes**: Require deprecation period for breaking changes
4. **Implementation changes**: Follow testing principles and outcome validation

### Documentation Requirements
- All public APIs documented with examples
- All cognitive behaviors explained with research citations
- Production evidence for architectural decisions
- Examples from real-world usage (Roxy)

### Review Criteria
Before merging changes, verify:
- [ ] Zero semantic regex patterns introduced
- [ ] All LLM prompts use XML output
- [ ] Async used appropriately (I/O yes, pure computation no)
- [ ] Tests validate outcomes, not processes
- [ ] Protocol-based for all integrations

---

**Document Status**: Active - Production-Validated
**Last Updated**: 2025-12-30
**Production Evidence**: Roxy Voice Assistant (1000+ queries/day, 0 semantic regex)
**Review Schedule**: Quarterly or after major production learnings

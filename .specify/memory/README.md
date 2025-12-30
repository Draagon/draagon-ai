# draagon-ai Project Memory

This directory contains references to key project documents and artifacts for AI development agents.

## Key Documents

### Constitution (Start Here)
- [Constitution](../constitution/constitution.md) - Project charter and non-negotiable principles
- [Design Principles](../constitution/principles.md) - LLM-first, 4-layer memory, beliefs, etc.
- [Scope](../constitution/scope.md) - What draagon-ai is and isn't
- [Testing Principles](../constitution/testing-principles.md) - Cognitive testing philosophy

### Architecture
- [CLAUDE.md](../../CLAUDE.md) - Main project context and architecture overview
- [Cognitive Swarm Spec](../../docs/specs/COGNITIVE_SWARM_ARCHITECTURE.md) - Phase 2 detailed specification

### Reference Implementation
- [Roxy Voice Assistant](../../../roxy-voice-assistant/) - Production implementation example

## Cognitive Architecture Summary

### 4-Layer Memory
```
Layer 4: Metacognitive (Permanent)
├─ Self-knowledge, capabilities
└─ Learned patterns

Layer 3: Semantic (6 months TTL)
├─ Facts, skills, preferences
└─ Consolidated beliefs

Layer 2: Episodic (2 weeks TTL)
├─ Conversation summaries
└─ Interaction patterns

Layer 1: Working (5 minutes TTL)
├─ Current context (7±2 items)
└─ Attention-weighted
```

### Belief Pipeline
```
User Statement → UserObservation → BeliefReconciliation → AgentBelief → Memory
```

### Key Modules
- `src/draagon_ai/orchestration/` - Agent loop, decision, execution
- `src/draagon_ai/memory/` - 4-layer cognitive memory
- `src/draagon_ai/cognition/` - Learning, beliefs, curiosity, opinions
- `src/draagon_ai/tools/` - @tool decorator and registry

## Non-Negotiable Rules

1. **No semantic regex** - LLM handles all semantic understanding
2. **XML output** - All LLM prompts use XML, never JSON
3. **Protocol-based** - All integrations via Python Protocols
4. **Beliefs, not memories** - User statements become observations → beliefs

## Current Phase

**Phase 1: Cognitive Integration** (In Progress)
- Wiring cognitive services into orchestration
- @tool decorator with discovery
- Post-response learning
- Belief-aware decisions

**Next: Phase 2: Cognitive Swarm**
- Shared working memory
- Parallel agent execution
- Transactive memory
- Metacognitive reflection

## Quick Reference

### Running Tests
```bash
pytest tests/ -v
pytest tests/cognitive/ -v  # Cognitive behavior tests
```

### Key Benchmarks
- MultiAgentBench Werewolf: Target 55%+ (SOTA: 36%)
- MemoryAgentBench Conflict: Target 75%+ (SOTA: 50%)
- GAIA Level 3: Target 55%+ (SOTA: 40%)

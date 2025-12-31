# Semantic Expansion Prototype - Claude Context

**Status:** Experimental
**Last Updated:** 2025-12-31

---

## Overview

This prototype explores **two-pass semantic understanding** - a technique for improving how the agent interprets ambiguous natural language by leveraging memory context before and after semantic expansion.

**Hypothesis:** By querying memory BEFORE expansion (to resolve pronouns, get context) AND AFTER expansion (to find supporting/contradicting evidence), we can achieve more accurate semantic understanding than single-pass approaches.

---

## Key Concepts

### Two-Pass Architecture

```
Input: "She said the bass was great"
        │
        ▼
┌─────────────────────────────────┐
│  Pass 1: Pre-Expansion          │
│  • Query memory for "she"       │
│  • Get recent context           │
│  • Resolve pronouns             │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Semantic Expansion             │
│  • WSD: bass = fish or music?   │
│  • Generate semantic variants   │
│  • Score by context             │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Pass 2: Post-Expansion         │
│  • Query memory with entities   │
│  • Find supporting evidence     │
│  • Detect contradictions        │
└─────────────────────────────────┘
        │
        ▼
    Best interpretation + evidence
```

### Word Sense Disambiguation (WSD)

Uses Lesk algorithm with WordNet to disambiguate homonyms:
- "bank" (financial) vs "bank" (river)
- "bass" (fish) vs "bass" (music)

Located in: `src/wsd.py`

### Semantic Frame Expansion

Extracts structured semantic frames from text:
- Subject, predicate, object
- Entities with types
- Ambiguities with possible senses

Located in: `src/expansion.py`

### Memory Integration

The `TwoPassSemanticOrchestrator` in `src/integration.py`:
1. `PreExpansionRetriever` - Gets context before expansion
2. `PostExpansionRetriever` - Finds evidence after expansion
3. `NaturalLanguageGenerator` - Produces human-readable output

---

## File Structure

```
src/
├── __init__.py           # Package exports
├── semantic_types.py     # Type definitions (WordSense, SemanticFrame, etc.)
├── wsd.py                # Word Sense Disambiguation (Lesk algorithm)
├── expansion.py          # Semantic frame expansion service
└── integration.py        # Two-pass orchestrator, memory integration

tests/
├── conftest.py                        # Path setup for imports
├── test_semantic_expansion.py         # Core functionality tests
├── test_two_pass_integration.py       # Integration tests
├── test_semantic_correctness.py       # Correctness validation
├── test_breaking_point.py             # Adversarial/stress tests
├── test_chaos_and_evolution.py        # Chaos engineering tests
└── test_real_provider_integration.py  # Real LLM integration tests

docs/
├── research/             # Background concepts, prior art
├── requirements/         # FR-006, FR-007 requirement docs
├── specs/                # Technical architecture specs
└── findings/             # Experiment results, learnings
```

---

## Important Patterns

### LLM Content Extraction

The prototype handles both `str` and `ChatResponse` returns from LLM providers:

```python
def _extract_llm_content(response: Any) -> str:
    """Extract string content from LLM response."""
    if isinstance(response, str):
        return response
    if hasattr(response, 'content'):
        return response.content
    return str(response)
```

### Memory Protocol Compatibility

Uses standard draagon-ai `MemoryProvider` protocol plus optional `search_by_entities()`:

```python
# Standard search
results = await memory.search(query, limit=10)

# Entity-based search (if available)
if hasattr(memory, 'search_by_entities'):
    results = await memory.search_by_entities(entities, limit=10)
```

### Direct Imports

Prototype uses direct imports (not relative) with `conftest.py` setting up sys.path:

```python
# In prototype code
from semantic_types import WordSense, SemanticFrame
from wsd import WordSenseDisambiguator
from expansion import SemanticExpansionService
```

---

## Running Tests

```bash
# From prototype directory
cd prototypes/semantic_expansion

# Run all tests
python3 -m pytest tests/ -v

# Run with real Groq LLM
GROQ_API_KEY=your_key python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_two_pass_integration.py -v
```

---

## Integration Readiness

**Current Status:** NOT ready for core integration

**Blocking Issues:**
1. Not wired into AgentLoop or DecisionEngine
2. Needs explicit integration point design
3. Memory reinforcement needs orchestrator-level changes

**To Graduate:**
1. Design integration hook (pre-decision? post-query?)
2. Wire into AgentLoop with feature flag
3. Run regression tests on Roxy
4. Document performance impact

---

## Related Documentation

- `docs/requirements/FR-006-word-sense-disambiguation.md`
- `docs/requirements/FR-007-semantic-expansion-service.md`
- `docs/specs/SEMANTIC_EXPANSION_ARCHITECTURE.md`
- `docs/research/SEMANTIC_EXPANSION_CONCEPT.md`

---

**End of Prototype CLAUDE.md**

# Semantic Expansion Prototype

## Hypothesis

What if we deeply expand user statements into rich semantic representations BEFORE storing in memory, and use two-pass retrieval (pre-expansion context + post-expansion evidence) for better understanding?

## Core Concepts

### 1. Word Sense Disambiguation (WSD)
Resolve ambiguous words to canonical meanings using WordNet synsets.
- "bank" (financial) vs "bank" (river)
- Uses Lesk algorithm + optional LLM fallback

### 2. Semantic Frame Expansion
Extract rich semantic structure from statements:
- **Triples**: subject-predicate-object relationships
- **Presuppositions**: implicit assumptions
- **Implications**: likely consequences
- **Ambiguities**: unresolved references

### 3. Two-Pass Memory Integration
- **Pass 1 (Pre-Expansion)**: Query memory BEFORE expansion to resolve pronouns and get context
- **Expansion**: Use LLM with context to produce semantic frames
- **Pass 2 (Post-Expansion)**: Query again with resolved entities for evidence
- **Re-scoring**: Weight variants by memory support

### 4. Memory Reinforcement Learning
- Boost memories that contribute to correct responses
- Demote memories that lead to errors
- Layer promotion/demotion based on importance thresholds

## Status

- [x] Initial implementation (~3,500 lines)
- [x] WSD with Lesk algorithm working
- [x] Semantic frame extraction working
- [x] Two-pass orchestration working
- [x] Memory reinforcement learning implemented
- [x] 140/142 tests passing
- [ ] Tested with real LLM (Groq) - needs API key
- [ ] Tested with real memory (Qdrant) - needs embeddings
- [ ] Benchmarked against baseline
- [ ] Integration path designed

## Key Files

```
src/
├── __init__.py          # Exports
├── types.py             # Data structures (WordSense, SemanticFrame, etc.)
├── wsd.py               # Word Sense Disambiguation
├── expansion.py         # Semantic frame extraction
└── integration.py       # Two-pass orchestrator, NLG, conflict detection

tests/
├── test_semantic_expansion.py      # Unit tests
├── test_two_pass_integration.py    # Integration tests
├── test_semantic_correctness.py    # Correctness verification
├── test_breaking_point.py          # Stress tests
├── test_chaos_and_evolution.py     # Edge cases
└── test_real_provider_integration.py # Real provider tests
```

## Usage Example

```python
from prototypes.semantic_expansion.src import (
    TwoPassSemanticOrchestrator,
    MockMemoryProvider,  # Or real provider
)

# Create orchestrator
memory = MockMemoryProvider()
await memory.store("Doug has 3 cats", entities=["Doug", "cats"])

orchestrator = TwoPassSemanticOrchestrator(
    memory=memory,
    llm=my_llm,  # Or None for heuristics
)

# Process a statement
result = await orchestrator.process("He loves his pets")

print(result.response_text)
print(f"Variants: {len(result.variants)}")
print(f"Conflicts: {len(result.conflicts)}")
```

## Dependencies

- `nltk` (optional, for WordNet WSD)
- Any LLM provider matching the `LLMProvider` protocol
- Any memory provider matching the `MemoryProvider` protocol

## Findings

### What Works Well
1. WSD with Lesk algorithm is fast and reasonably accurate
2. Two-pass retrieval catches more relevant context
3. LLM-based conflict detection is more robust than heuristics
4. Memory reinforcement creates a natural "memory fitness" signal

### Open Questions
1. Should semantic frames be stored, or computed on-demand?
2. How to handle long-form content (paragraphs vs single sentences)?
3. Performance impact of two passes vs one?
4. How to integrate with existing AgentLoop without breaking changes?

### Integration Considerations
When ready to integrate into core draagon-ai:
1. Add `TwoPassSemanticOrchestrator` as optional processor in AgentLoop
2. Modify `LayeredMemoryProvider.store()` to optionally expand before storing
3. Add `search_by_entities()` method to MemoryProvider protocol
4. Wire reinforcement into response feedback loop

---

**Status:** 🧪 Experimental - Not yet integrated into core

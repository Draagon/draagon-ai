# DD-001: Content Type Aware Processing

**Status:** Accepted
**Date:** 2025-12-31
**Deciders:** Doug, Claude
**Category:** Architecture

---

## Context

The implicit knowledge graphs prototype was designed with **natural language prose** as the primary input type. The WSD and decomposition pipelines assume sentence structure exists and that words have semantic meanings that can be disambiguated via WordNet synsets.

However, real-world users will provide many types of content beyond prose:

| Content Type | Example | Has Sentences? | WSD Applicable? |
|--------------|---------|----------------|-----------------|
| **Prose** | "I deposited money in the bank" | Yes | Full WSD |
| **Source Code** | `def process_bank(): pass` | No (partially in comments) | Only on NL portions |
| **CSV Data** | `customer_id,bank_account,balance` | No | No - schema extraction instead |
| **JSON/Config** | `{"database": {"host": "localhost"}}` | No | No - pattern extraction instead |
| **Logs** | `2024-01-15 ERROR Bank connection failed` | Partially | Selective |
| **Mixed** | README with code blocks | Both | Selective |

**Problem:** Applying WSD to code syntax or data values produces nonsensical results. For example:
- `bank.deposit(100)` - "bank" is a variable name, not a financial institution or riverbank
- `"bank_account": "12345"` - "bank_account" is a JSON key, not a phrase to disambiguate

---

## Decision

**We will implement a content-aware preprocessing layer that classifies content type BEFORE applying WSD or decomposition.**

### Processing Strategy by Content Type

| Content Type | Processing Strategy | Semantic Knowledge Extracted |
|--------------|--------------------|-----------------------------|
| **PROSE** | Full WSD → Decomposition | Word senses, presuppositions, commonsense inferences |
| **CODE** | Extract NL portions → WSD on those | Type contracts, relationships FROM code; word senses FROM comments/docstrings |
| **DATA** | Schema extraction | Column meanings, data types, relationships, constraints |
| **CONFIG** | Pattern extraction | Key hierarchies, common patterns, structure |
| **LOGS** | Pattern + selective NL | Message templates, anomalies, error types |
| **MIXED** | Split → Apply per-component | Combined knowledge |

### Architecture

```
Input Content
      │
      ▼
┌─────────────────────────────┐
│ ContentAnalyzer             │
│ - LLM classifies type       │
│ - Extracts NL portions      │
│ - Extracts structure        │
└─────────────────────────────┘
      │
      ├─── PROSE ──────────► Full WSD Pipeline
      │
      ├─── CODE ───────────► Extract NL ──► WSD on NL only
      │    │                              ├► Type/contract extraction
      │    └── Structural Knowledge ──────┘
      │
      ├─── DATA ───────────► Schema Extractor (no WSD)
      │
      └─── CONFIG ─────────► Pattern Matcher (no WSD)
              │
              ▼
      Unified Knowledge Store
```

### Key Insight

**WSD is for natural language understanding, not all semantic understanding.**

Different content types have different kinds of meaning:

| Content | Where is the "meaning"? |
|---------|------------------------|
| Prose | In word senses, sentence structure, implications |
| Code | In identifiers, types, contracts, relationships |
| Data | In column names, data types, patterns, constraints |
| Config | In key hierarchies, valid values, common patterns |

We should NOT force WSD onto non-prose content. Instead, we extract the type of semantic knowledge appropriate to each content type.

---

## Implementation

### New Modules

1. **`content_analyzer.py`** - LLM-driven content type classification
   - `ContentType` enum: PROSE, CODE, DATA, CONFIG, LOGS, MIXED
   - `ContentAnalyzer` class with `analyze()` method
   - Extracts NL portions from code (docstrings, comments)
   - Extracts structural knowledge (types, contracts)

2. **`content_aware_wsd.py`** - Content-aware WSD integration
   - `ContentAwareWSD` class wraps WSD with content preprocessing
   - Routes content through appropriate processing
   - Merges NL disambiguation with structural knowledge

### LLM-First Content Detection

Per draagon-ai's LLM-First Architecture principle, we use the LLM to classify content, NOT regex heuristics:

```python
# WRONG - brittle pattern matching
if re.search(r'def \w+\(', content):
    return ContentType.CODE

# RIGHT - semantic understanding
analysis = await llm.analyze_content(content)
# LLM understands that:
# - "I wrote a def statement" is PROSE about code
# - "def foo():" is actual CODE
```

The `ContentAnalyzer` has a heuristic fallback for when LLM is unavailable, but the LLM path is preferred.

---

## Consequences

### Positive

1. **Accuracy**: WSD only runs on appropriate content, avoiding nonsensical results
2. **Efficiency**: Data/config content skips expensive WSD processing
3. **Flexibility**: System handles real-world mixed content
4. **Better Context**: Extracts appropriate semantic knowledge per content type
5. **Foundation**: Sets up architecture for future content-specific processing

### Negative

1. **Complexity**: Adds a preprocessing layer before WSD
2. **Latency**: Additional LLM call for content classification
3. **Coverage**: Heuristic fallback may misclassify ambiguous content

### Risks

| Risk | Mitigation |
|------|------------|
| Misclassification | LLM analysis with confidence scores; fallback to prose when uncertain |
| Lost information | Store raw content alongside extracted knowledge |
| Scope creep | Focus on the 5 main types (prose, code, data, config, mixed) |

---

## Affected Requirements

### Phase 0 Updates

**REQ-0.5: Hybrid Disambiguation Pipeline** - Add content type check as first step:

```
Input: (word, context)
    │
    ├─ Content type check → Is context NL?
    │   └─ No → Skip WSD (return None or structural extraction)
    │
    ├─ Single synset? → Return immediately
    │   ...
```

**New Requirement: REQ-0.8: Content Type Analysis**

```python
async def analyze_content_type(
    content: str,
    llm: LLMProvider | None = None,
) -> ContentAnalysis
```

### Phase 1 Updates

**REQ-1.7: Decomposition Pipeline Orchestrator** - Add content preprocessing:

```python
async def decompose(self, text: str, context: str | None = None) -> DecomposedKnowledge:
    # NEW: Analyze content type first
    content_analysis = await self.content_analyzer.analyze(text)

    if content_analysis.content_type == ContentType.DATA:
        return self._decompose_data(content_analysis)
    elif content_analysis.content_type == ContentType.CODE:
        # Only decompose NL portions
        nl_text = content_analysis.get_natural_language_text()
        if nl_text:
            return await self._decompose_prose(nl_text, structural=content_analysis.structural_knowledge)
        return self._decompose_code_only(content_analysis)
    else:
        return await self._decompose_prose(text)
```

**New Requirement: REQ-1.9: Structural Knowledge Extraction**

For non-prose content, extract structural knowledge:
- Code: Type hierarchies, function contracts, imports
- Data: Column schemas, relationships, constraints
- Config: Key hierarchies, valid patterns

---

## Storage Strategy

### Short-Term (Working Memory)
- **All types**: Store raw content for immediate use
- **Why**: LLM may need exact syntax for code assistance

### Medium-Term (Episodic)
- **Prose**: Store semantic decomposition
- **Code**: Store summarized understanding + key snippets
- **Data**: Store schema + sample + statistics

### Long-Term (Semantic)
- **Prose**: Store extracted knowledge (facts, relationships, word senses)
- **Code**: Store API contracts, type hierarchies, patterns
- **Data**: Store schema evolution, constraint knowledge
- **Config**: Store common patterns, valid structures

---

## When to Use Raw Content vs Semantic Knowledge

| Scenario | Use Raw | Use Semantic |
|----------|---------|--------------|
| User asks about syntax/formatting | ✓ | |
| Debugging specific code | ✓ | |
| Understanding concepts | | ✓ |
| Finding related information | | ✓ |
| Long-term recall | | ✓ |
| Recent/changing content | ✓ | |
| Answering questions | Hybrid | ✓ |

### Hybrid Approach Example

```
User: "How does our authentication work?"

Context provided to LLM:
1. Semantic knowledge: "System uses JWT tokens,
   AuthService validates, tokens expire in 24h"
2. Key code snippet: actual auth middleware (100 lines, not 1000)
3. Related facts: "Added OAuth support in v2.1"
```

---

## Success Metrics

1. **Context Efficiency**: Same quality answers with fewer tokens
2. **Accuracy Preservation**: No degradation from semantic extraction
3. **Cross-Reference**: Can answer "what code implements X concept?"
4. **Staleness Detection**: Know when semantic knowledge is outdated

---

## Future Considerations

1. **Mixed Content Handling**: More sophisticated splitting for markdown with code blocks
2. **Language-Specific Code Analysis**: Deeper extraction for Python vs JavaScript vs Go
3. **Schema Evolution**: Track changes to data schemas over time
4. **Cross-Type Relationships**: Link concepts in docs to implementations in code

---

## References

- [CONTENT_TYPE_HANDLING.md](../research/CONTENT_TYPE_HANDLING.md) - Original research
- [PHASE_0_IDENTIFIERS.md](../requirements/PHASE_0_IDENTIFIERS.md) - Phase 0 requirements
- [PHASE_1_DECOMPOSITION.md](../requirements/PHASE_1_DECOMPOSITION.md) - Phase 1 requirements
- [content_analyzer.py](../../src/content_analyzer.py) - Implementation
- [content_aware_wsd.py](../../src/content_aware_wsd.py) - WSD integration

---

**End of Design Decision DD-001**
